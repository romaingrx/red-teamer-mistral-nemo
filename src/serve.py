import os
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
from loguru import logger
from prometheus_client import CollectorRegistry, make_wsgi_app, multiprocess
from ray import serve
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

app = FastAPI(name="vllm-api", description="vLLM API serving the red-teamer model")


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.mount_metrics(app)
        self.__openai_serving_chat = None  # Initialize as None

    async def get_openai_serving_chat(self):
        if self.__openai_serving_chat is None:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.__openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                chat_template=self.chat_template,
                prompt_adapters=None,
                request_logger=None,
            )
        return self.__openai_serving_chat

    def mount_metrics(self, app: FastAPI):
        prometheus_multiproc_dir_path = os.getenv("PROMETHEUS_MULTIPROC_DIR", None)
        if prometheus_multiproc_dir_path is not None:
            logger.info(
                "vLLM to use %s as PROMETHEUS_MULTIPROC_DIR",
                prometheus_multiproc_dir_path,
            )
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            metrics_app = make_wsgi_app(registry=registry)
        else:
            metrics_app = make_wsgi_app()

        # Mount the Prometheus metrics WSGI app to the FastAPI app
        app.mount("/metrics", WSGIMiddleware(metrics_app))

        # Add a route to handle potential redirects
        @app.get("/metrics", include_in_schema=False)
        async def metrics_redirect():
            return RedirectResponse(url="/metrics/")

    @app.get("/health")
    async def health_check(self):
        await self.engine.check_health()
        return Response(status_code=200)

    @app.get("/v1/models")
    async def show_available_models(self):
        openai_serving_chat = await self.get_openai_serving_chat()
        models = await openai_serving_chat.show_available_models()
        return models.model_dump()

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        logger.info(f"Request: {request}")
        openai_serving_chat = await self.get_openai_serving_chat()
        generator = await openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    engine_args.enable_lora = False

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return VLLMDeployment.options(
        placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK"
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.chat_template,
    )


if __name__ == "__main__":
    pass

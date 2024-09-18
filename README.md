
# Red Teamer Mistral Nemo

<div style="display: flex; justify-content: start; gap: 10px; align-items: center;">
  <a href="https://huggingface.co/romaingrx/red-teamer-mistral-nemo">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg" alt="Model on HF">
  </a>
  <a href="https://api.wandb.ai/links/romaingrx/lcqv0y0l">
    <img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Wandb" width="120">
  </a>
</div>

> [!WARNING]  
> This research is conducted for educational and defensive purposes only, aiming to improve AI safety and security.

This project explores the potential use of red-teaming models to jailbreak LLMs.
I fine-tuned [Mistral Nemo](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) on the [WildJailbreak](https://huggingface.co/datasets/allenai/wildjailbreak) dataset.

Key features of this project include:
- [Fine-tuning Mistral Nemo on pairs of prompts and responses from the WildJailbreak dataset](./src/fine_tune)
- [Evaluation against other models using HarmBench](./src/harmbench)
- [Generation of example outputs](./src/generate_examples.py)
- [Deployment as an OpenAI-compatible API with VLLM and Ray](./src/serve.py)
- [Detailed metrics and analysis](./src/metrics.py)

## Finetuning


I used the code in [the `fine_tune` folder](./src/fine_tune) to finetune the [`mistralai/Mistral-Nemo-Instruct-2407` model](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) on the subset of the WildJailbreak dataset containing only the `adversarial_harmful` data type.

To run the whole pipeline in order to finetune the model, you can run the following command:
```bash
accelerate launch --config_file configs/accelerate/deepspeed_zero3.yaml src/fine_tune.py
```

→ The merged model can be found [here](https://huggingface.co/romaingrx/red-teamer-mistral-nemo) and the adapter can be found [here](https://huggingface.co/romaingrx/red-teamer-mistral-nemo-adapter).

## Harmbench evaluation

I forked the [Harmbench](https://github.com/allenai/harmbench) repo and added the code to evaluate the finetuned model against a variety of models, all the instructions can be found directly in the [Harmbench README](./HarmBench/README.md) to run the evaluation pipeline.

The name of the baseline is `RedTeamerMistralNemo`.

You can therefore run the full evaluation pipeline with the following command:
```bash
cd HarmBench
python scripts/run_pipeline.py --methods RedTeamerMistralNemo --models llama2_7b --step all --mode local
```

## Generating examples

I used the code in [the `generate_examples` folder](./src/generate_examples.py) to generate examples from the finetuned model and reported the results in wandb.

## Serving

The code in [the `serve` script](./src/serve.py) is used to serve the model as an OpenAI-compatible API using VLLM and Ray.

You can run the server with:
```bash
cd src
serve run serve:build_app model="romaingrx/red-teamer-mistral-nemo" max-model-len=118000
```

## Metrics

The VLLM server serves a `/metrics` endpoint that can be used to get the metrics using prometheus. You can therefore run the docker in the [dockers/prometheus](./dockers/prometheus) folder to get a dashboard with Grafana to see the metrics of the served models in real time.

Run the following command to run the docker compose:
```bash
cd dockers/prometheus
docker compose up -d
```

And you should have the `vLLM` dashboard by default in the Grafana interface.

You can then access the dashboard at http://localhost:3000 with user `admin` and password `admin`.
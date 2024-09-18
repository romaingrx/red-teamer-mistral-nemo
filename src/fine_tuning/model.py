from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from omegaconf import DictConfig
from loguru import logger
import bitsandbytes as bnb
from utils import get_kbit_device_map, get_torch_dtype, add_pad_token


def load_model(cfg: DictConfig):
    logger.info(
        f"Loading model config for `{cfg.model.name}` from `{cfg.model.hf_name}`"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg.model.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=get_torch_dtype(cfg.model.dtype),
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_name,
        device_map=get_kbit_device_map(),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        quantization_config=bnb_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.hf_name,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )

    add_pad_token(cfg, model, tokenizer)

    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer


def prepare_model(cfg: DictConfig, model: AutoModelForCausalLM):
    logger.info("Preparing model with LoRA weights")

    if (
        ratio := cfg.finetune.lora.get("layers_to_transform_end_ratio", None)
    ) is not None:
        # Here we're interested in transforming only the last layers of
        # the model (higher concepts)
        num_layers = model.config.num_hidden_layers
        layers_to_transform = int(num_layers * ratio)
        layers_to_transform = list(range(num_layers - layers_to_transform, num_layers))

        logger.info(
            f"Transforming the last {len(layers_to_transform)} layers out \
            of {num_layers} total layers"
        )

    def find_all_linear_names(model):
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        return lora_module_names

    target_modules = list(
        cfg.finetune.lora.get("target_modules") or find_all_linear_names(model)
    )
    logger.info(f"Lora target modules: {target_modules}")
    peft_config = LoraConfig(
        r=cfg.finetune.lora.get("r", 16),
        use_rslora=cfg.finetune.lora.get("use_rslora", True),
        layers_to_transform=layers_to_transform if ratio else None,
        lora_alpha=cfg.finetune.lora.get("alpha", 16),
        lora_dropout=cfg.finetune.lora.get("dropout", 0),
        bias=cfg.finetune.lora.get("bias", "none"),
        task_type="CAUSAL_LM",
        modules_to_save=cfg.finetune.lora.get("modules_to_save", None),
        target_modules=target_modules,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.finetune.training.get(
            "use_gradient_checkpointing", True
        ),
    )
    model = get_peft_model(model, peft_config)

    if cfg.finetune.training.get("use_gradient_checkpointing", True):
        model.gradient_checkpointing_enable()

    return model

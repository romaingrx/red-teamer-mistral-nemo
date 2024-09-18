import os
import re

import functools
import hashlib
import json
from datasets import ClassLabel, DatasetDict, load_dataset
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
from accelerate import Accelerator
import random
from multiprocessing import cpu_count
from config import USE_WANDB

try:
    import wandb
except ImportError:
    pass


def wandb_artifact_dataset():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not USE_WANDB:
                return func(*args, **kwargs)
            if not wandb.run:
                logger.warning(
                    "W&B is not initialized. Won't be able to load dataset from W&B artifact."
                )
                return func(*args, **kwargs)

            cfg = next(arg for arg in args if isinstance(arg, DictConfig))
            accelerator = next(arg for arg in args if isinstance(arg, Accelerator))

            # Create artifact name based on dataset name
            artifact_name = re.sub(r"[^a-zA-Z0-9.]", "_", cfg.finetune.dataset.name)

            # Create a hash of the dataset configuration
            dataset_config = OmegaConf.to_container(cfg.finetune.dataset, resolve=True)
            config_hash = hashlib.md5(
                json.dumps(dataset_config, sort_keys=True).encode()
            ).hexdigest()

            # Try to load the latest version of the artifact
            try:
                artifact = wandb.run.use_artifact(f"{artifact_name}:latest")
                if artifact and artifact.metadata.get("config_hash") == config_hash:
                    logger.info(
                        f"Loading dataset from W&B artifact: {artifact_name}:latest"
                    )
                    artifact_dir = artifact.download()
                    return DatasetDict.load_from_disk(
                        os.path.join(artifact_dir, "datasets")
                    )
            except wandb.errors.CommError:
                logger.info(
                    f"No artifact found with name {artifact_name}:latest, will create new artifact"
                )
                pass

            dataset: DatasetDict = func(*args, **kwargs)
            if accelerator.is_main_process:
                dataset_artifact = wandb.Artifact(
                    name=f"{artifact_name}",
                    type="dataset",
                    description="Dataset used for training",
                    metadata={
                        "config": dataset_config,
                        "config_hash": config_hash,
                    },
                )

                dataset_dir = os.path.join(cfg.wandb.cache_dir, artifact_name)
                dataset.save_to_disk(dataset_dir)
                dataset_artifact.add_dir(dataset_dir, name="datasets")
                logger.info(
                    f"Saving new version of dataset to W&B artifact: {artifact_name}"
                )
                wandb.run.log_artifact(dataset_artifact)

            return dataset

        return wrapper

    return decorator


@wandb_artifact_dataset()
def load_and_prepare_dataset(
    cfg: DictConfig, tokenizer: AutoTokenizer, accelerator: Accelerator
):
    def prepare_function(example):
        assert tokenizer.chat_template, "The tokenizer must have a chat template"
        # Todo: would be nice to have a list of instructions to choose from
        system_prompt = (
            cfg.finetune.dataset.get("system_prompt")
            or "Convert the following prompt to an adversarial prompt\
                   that achieves the same goal:"
        )
        # Ensure the system prompt is included by explicitly adding it to
        # the user message. This prevents the chat template from potentially
        # omitting it when processing the assistant's response
        user_message = (
            f"{system_prompt}\n\n{example[cfg.finetune.dataset.cols.vanilla]}"
        )
        prompt_messages = [{"role": "user", "content": user_message}]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, add_generation_token=False, tokenize=False
        )
        # Don't have the control over the trainer to avoid adding special tokens
        # here, so we need to remove them manually
        if prompt.startswith(tokenizer.bos_token):
            prompt = prompt[len(tokenizer.bos_token) :]
        text_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": example[cfg.finetune.dataset.cols.adversarial],
            },
        ]
        text = tokenizer.apply_chat_template(text_messages, tokenize=False)
        # Don't have the control over the trainer to avoid adding special tokens
        # here, so we need to remove them manually
        if text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token) :]
        return {
            "text": text,
            "prompt": prompt,
            # **tokenizer(text)
        }

    logger.info(f"Loading and preparing dataset {cfg.finetune.dataset.name}")

    ds = load_dataset(
        cfg.finetune.dataset.name,
        cfg.finetune.dataset.split,
        delimiter="\t",
        keep_default_na=False,
    )["train"]

    logger.info(f"Dataset loaded successfully with {len(ds)} samples")
    adversarial_ds = ds.filter(
        lambda x: x[cfg.finetune.dataset.cols.data_type] in cfg.finetune.dataset.types
    )

    if cfg.finetune.training.get("sample_limit") is not None:
        logger.info(f"Limiting dataset to {cfg.finetune.training.sample_limit} samples")
        adversarial_ds = adversarial_ds.take(cfg.finetune.training.sample_limit)

    train_test_split = cfg.finetune.dataset.get("train_test_split", 1)
    assert 0 <= train_test_split <= 1, "Train test split must be between 0 and 1"

    if train_test_split < 1:
        if data_type := cfg.finetune.dataset.cols.get("data_type", None):
            logger.info(f"Casting {data_type} column to ClassLabel")
            unique_labels = adversarial_ds.unique(data_type)
            adversarial_ds = adversarial_ds.cast_column(
                data_type, ClassLabel(names=unique_labels)
            )

        dataset = adversarial_ds.train_test_split(
            train_size=int(len(adversarial_ds) * train_test_split),
            seed=cfg.seed,
            stratify_by_column=data_type if data_type is not None else None,
            shuffle=True,
        )
        logger.info(
            f"Splitting dataset into train and test with ratio {train_test_split} \
            (train size: {len(dataset['train'])}, test size: {len(dataset['test'])})"
        )
    else:
        dataset = DatasetDict(train=adversarial_ds)
        logger.info(f"Using all {len(dataset['train'])} samples for training")

    dataset = dataset.map(
        prepare_function,
        batched=False,
        remove_columns=adversarial_ds.column_names,
        num_proc=cpu_count(),
        desc="Preparing dataset...",
    )

    if "input_ids" in dataset.column_names:
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) <= cfg.model.max_seq_length,
            desc="Filter out samples longer than max sequence length...",
        )

    logger.info(f"Prepared {len(dataset['train'])} samples for training")
    if accelerator.is_main_process:
        if USE_WANDB:
            for split in dataset.keys():
                wandb.log(
                    {
                        f"{split}_dataset_samples": wandb.Table(
                            dataframe=dataset[split].take(10).to_pandas(),
                        )
                    }
                )

        for index in random.sample(range(len(dataset["train"])), 3):
            logger.info(
                f"""Sample {index} of the processed training set:
                
                {repr(dataset['train'][index]['text'])}
                """
            )
    return dataset

from typing import Dict

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from hydra import compose, initialize
from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.integrations import WandbCallback
from tqdm import tqdm


def load_cfg(
    cfg_path: str = "../../configs",
) -> DictConfig:
    with initialize(
        version_base=None,
        config_path=cfg_path,
    ):
        cfg = compose(config_name="config")
    return cfg


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the model predictions
    as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency (in steps) of logging. Defaults to 2.
    """

    def __init__(
        self,
        trainer: Trainer,
        tokenizer: PreTrainedTokenizerBase,
        val_dataset,
        prompt_column_name="prompt",
        num_samples=8,
        freq=2,
        max_new_tokens=100,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    ):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency (in steps) of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq
        self.prompt_column_name = prompt_column_name
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.temperature = temperature

    def on_step_end(
        self,
        args,
        state,
        control,
        **kwargs,
    ):
        super().on_step_end(
            args,
            state,
            control,
            **kwargs,
        )
        try:
            if state.global_step % self.freq == 0:
                # Generate predictions using the model's generate method
                logger.info(f"Generating predictions for step {state.global_step}")

                for sample in tqdm(
                    self.sample_dataset,
                    desc="Generating predictions",
                    leave=False,
                    position=1,
                ):
                    inputs = self.tokenizer(
                        sample["prompt"],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.trainer.model.device)

                    generated_ids = self.trainer.model.generate(
                        inputs.input_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        temperature=self.temperature,
                    )

                    # Decode predictions
                    predictions = self.tokenizer.batch_decode(generated_ids)[0]

                # Prepare data for logging
                predictions_data = {
                    "step": [state.global_step] * len(predictions),
                    "prompt": sample["prompt"],
                    "prediction": predictions,
                }

                # Add predictions to a wandb.Table
                predictions_df = pd.DataFrame(predictions_data)
                records_table = self._wandb.Table(dataframe=predictions_df)

                # Log the table to wandb
                self._wandb.log({"sample_predictions": records_table})
        except Exception as e:
            logger.error(f"Error logging predictions: {e}")


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index
    to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Dict[str, int] | None:
    """
    Useful for running inference with quantized models by setting
    `device_map=get_peft_device_map()`
    """
    return {"": get_current_device()} if torch.cuda.is_available() else None


def get_torch_dtype(
    dtype: str | None,
) -> torch.dtype | str | None:
    if dtype in ["auto", None]:
        return dtype
    return getattr(torch, dtype)


def add_pad_token(
    cfg, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase
) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token is None:
        cfg_pad_token = cfg.model.get("tokenizer", {}).get("pad_token", "<pad>")

        if cfg_pad_token in tokenizer.get_vocab():
            logger.info(
                f"Pad token '{cfg_pad_token}' already exists in vocabulary, \
                setting it as pad_token"
            )
            tokenizer.pad_token = cfg_pad_token
        else:
            logger.info(
                f"Adding '{cfg_pad_token}' as new pad token to the tokenizer \
                and resizing model embeddings"
            )
            tokenizer.add_special_tokens({"pad_token": cfg_pad_token})
            model.resize_token_embeddings(len(tokenizer))

        # Set the model pad token id to the new pad token id for loss computation
        model.config.pad_token_id = tokenizer.pad_token_id

    elif cfg_pad_token := cfg.model.get("tokenizer", {}).get("pad_token"):
        if tokenizer.pad_token != cfg_pad_token:
            logger.warning(
                f"Tokenizer already has a padding token '{tokenizer.pad_token}' \
                but will override with the config value '{cfg_pad_token}'"
            )
            tokenizer.pad_token = cfg_pad_token


def get_end_of_instruction_token(tokenizer: PreTrainedTokenizerBase):
    assert (
        tokenizer.chat_template is not None
    ), "In order to retrieve the end of instruction token, we need an INSTRUCT model"
    eoi_candidate = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hey"}], tokenize=True
    )[-1]
    eos_candidate = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Yo"},
            {"role": "assistant", "content": "Hey"},
        ],
        tokenize=True,
    )[-1]

    assert (
        eoi_candidate != eos_candidate
    ), "The end of sequence should be different from the end of instruction"
    return tokenizer.decode([eoi_candidate])

import os
from multiprocessing import cpu_count

from config import USE_WANDB
from loguru import logger
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling, Trainer
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from utils import get_end_of_instruction_token


def get_trainer(
    cfg: DictConfig,
    model,
    tokenizer,
    dataset,
):
    logger.info("Setting up SFT trainer")

    try:
        raise Exception(
            "Forcing DataCollatorForLanguageModeling for experimental purposes"
        )
        collator = DataCollatorForCompletionOnlyLM(
            get_end_of_instruction_token(tokenizer),
            tokenizer=tokenizer,
            mlm=False,
        )
        packing = False
    except Exception as e:
        logger.error(f"Error in getting the DataCollatorForCompletionOnlyLM: {e}")
        logger.info(
            "Using the DataCollatorForLanguageModeling as a fallback and will train on the whole sequence"
        )
        collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        packing = True

    total_batch_size = cfg.finetune.training.get(
        "per_device_batch_size", 2
    ) * cfg.finetune.training.get("gradient_accumulation_steps", 4)

    n_steps_per_epoch = len(dataset["train"]) // total_batch_size
    eval_steps = cfg.finetune.training.get("eval_steps", n_steps_per_epoch // 5)

    training_args = SFTConfig(
        run_name=cfg.run_dir,
        output_dir=os.path.join(cfg.run_dir, "training_artifacts"),
        per_device_train_batch_size=cfg.finetune.training.get(
            "per_device_batch_size", 2
        ),
        per_device_eval_batch_size=cfg.finetune.training.get(
            "per_device_batch_size", 2
        ),
        gradient_accumulation_steps=cfg.finetune.training.get(
            "gradient_accumulation_steps", 4
        ),
        gradient_checkpointing=cfg.finetune.training.get(
            "use_gradient_checkpointing", True
        ),
        warmup_ratio=cfg.finetune.training.get("warmup_ratio", 0.05),
        lr_scheduler_type=cfg.finetune.training.get(
            "lr_scheduler_type", "cosine_with_restarts"
        ),
        log_level="info",
        logging_strategy="steps",
        logging_steps=5,
        report_to="wandb" if USE_WANDB else None,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        packing=packing,
        max_seq_length=cfg.model.max_seq_length,
        max_grad_norm=cfg.finetune.training.get("max_grad_norm", 1.0),
        dataset_text_field="text",
        seed=cfg.seed,
        dataset_num_proc=cpu_count(),
        num_train_epochs=cfg.finetune.training.get("num_train_epochs", 1),
        dataloader_num_workers=cpu_count() // 2,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=cfg.finetune.training.get("eval_steps", n_steps_per_epoch // 5),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy=cfg.finetune.training.get("save_strategy", "steps"),
        save_steps=cfg.finetune.training.get("save_steps", eval_steps),
        save_total_limit=cfg.finetune.training.get("save_total_limit", 5),
        # resume_from_checkpoint=cfg.finetune.training.get("resume_from_checkpoint", None),
    )

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        data_collator=collator,
    )

    if dataset.get("test", None) is not None:
        logger.warning("Skip WandbPredictionProgressCallback for evaluation")
        # logger.info("Adding WandbPredictionProgressCallback for evaluation")
        # sft_trainer.add_callback(
        #     WandbPredictionProgressCallback(
        #         sft_trainer, tokenizer, dataset["test"], freq=1, num_samples=10
        #     )
        # )

    return sft_trainer


def save_model(cfg: DictConfig, trainer: SFTTrainer):
    logger.info(f"Saving model to {cfg.model.finetuned_dir}")
    trainer.save_model(cfg.model.finetuned_dir)

    trainer.create_model_card(
        finetuned_from=cfg.model.hf_name,
        dataset=cfg.finetune.dataset.name,
        dataset_tags=cfg.finetune.dataset.name,
        tags=["red-teamer", "jailbreaking"],
    )
    trainer.model.config.use_cache = True
    trainer.model.config.save_pretrained(cfg.model.finetuned_dir)

    if cfg.finetune.training.get("push_to_hub", False) is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub()

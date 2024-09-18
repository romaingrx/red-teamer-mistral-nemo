import traceback

import hydra
from accelerate import Accelerator, DistributedDataParallelKwargs
from config import USE_WANDB, set_global_seed, setup_logging
from data import load_and_prepare_dataset
from loguru import logger
from model import load_model, prepare_model
from omegaconf import DictConfig
from trainer import get_trainer, save_model


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        log_with="wandb" if USE_WANDB else None,
        kwargs_handlers=[ddp_kwargs],
    )

    set_global_seed(cfg.seed)
    setup_logging(cfg, accelerator)

    model, tokenizer = load_model(cfg)
    dataset = load_and_prepare_dataset(cfg, tokenizer, accelerator)
    model = prepare_model(cfg, model)
    trainer = get_trainer(cfg, model, tokenizer, dataset)

    try:
        with logger.contextualize(task="SFT training"):
            logger.info("Starting SFT training")
            trainer.train(
                resume_from_checkpoint=cfg.finetune.training.get(
                    "resume_from_checkpoint", None
                )
            )
            logger.info("SFT training completed")
    except KeyboardInterrupt:
        logger.info("SFT training interrupted by user, will save the model")
    except Exception as e:
        logger.error(f"SFT training failed: {e}")
        logger.error(f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}")
        logger.info("Saving model on main process")
        raise
    finally:
        if accelerator.is_main_process:
            save_model(cfg, trainer)


if __name__ == "__main__":
    main()

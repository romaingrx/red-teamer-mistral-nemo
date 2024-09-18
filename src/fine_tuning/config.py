import os
import random
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import set_seed
from loguru import logger

try:
    import wandb

    USE_WANDB = True
except ImportError:
    USE_WANDB = False


def set_global_seed(seed: int):
    """
    Set the global random seed for reproducibility.
    """
    set_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(cfg: DictConfig, accelerator):
    run = None
    if accelerator.is_main_process:
        if USE_WANDB:
            run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.run_dir,
                config=OmegaConf.to_container(cfg, resolve=True),
                job_type="SFT training",
            )
            logger.info("Wandb initialized and updated with config")
        else:
            logger.warning("Wandb not installed, skipping the wandb logger")
    else:
        logger.remove()
        logger.add(lambda _: None)

    return run

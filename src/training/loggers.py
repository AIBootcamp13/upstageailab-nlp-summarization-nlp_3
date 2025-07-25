from pytorch_lightning.loggers import CSVLogger, WandbLogger, logger

from src import config


def get_loggers(experiment_name: str) -> list[logger.Logger]:
    csv_logger = CSVLogger(
        save_dir=config.LOG_ROOT_DIR,
        name=experiment_name,
    )
    wandb_logger = WandbLogger(
        project=config.WANDB_PROJECT,
        entity=config.WANDB_ENTITY,
        name=experiment_name,
        save_dir=config.RAW_DATA_DIR,
        log_model=False,
    )
    return [csv_logger, wandb_logger]

import hydra
import lightning
import wandb

from src.config import CONFIGS_DIR
from src.config.schemas import Config
from src.training.callbacks import get_callbacks
from src.training.loggers import get_loggers
from src.utils.helper import add_timestamp_prefix, fix_random_seed
from src.utils.log import get_logger


@hydra.main(config_path=str(CONFIGS_DIR), config_name="config")
def main(config: Config):
    experiment_name = add_timestamp_prefix(config.experiment_name)

    logger = get_logger(f"train-{experiment_name}")
    logger.info("-" * 80)
    logger.info("문서 이미지 분류기 학습")
    logger.info("-" * 80)

    # Random seed 설정
    fix_random_seed(config.seed)

    # DataModule 준비
    data_module = ...  # TODO implement it
    logger.info("DataModule 준비 완료")

    # 손실함수 정의
    # TODO implement it

    # model 정의
    model = ...  # TODO implement it
    logger.info("model 정의 완료")

    # callback 준비
    callbacks = get_callbacks(experiment_name)

    # logger 준비
    loggers = get_loggers(experiment_name)

    # trainer 준비
    trainer = lightning.Trainer(
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        logger=loggers,
    )
    logger.info("trainer 정의 완료")

    # 학습!
    if config.training.checkpoint_path:
        logger.info("체크 포인트에서 학습 시작")
        trainer.fit(model, data_module, ckpt_path=config.training.checkpoint_path)
    else:
        logger.info("학습 시작")
        trainer.fit(model, data_module)

    wandb.finish()


if __name__ == "__main__":
    main()

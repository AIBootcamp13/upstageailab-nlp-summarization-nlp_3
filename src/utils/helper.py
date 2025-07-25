from datetime import datetime

import pytorch_lightning as pl

from src import config


DATETIME_FORMAT = "%Y%m%d%H%M%S"


def fix_random_seed(seed_num: int):
    """모든 랜덤 시드를 고정하여 실험의 재현성을 보장하는 함수"""
    pl.seed_everything(seed_num)


def add_timestamp_prefix(filename: str) -> str:
    timestamp = datetime.now(config.TIMEZONE).strftime(DATETIME_FORMAT)
    return f"{timestamp}_{filename}"

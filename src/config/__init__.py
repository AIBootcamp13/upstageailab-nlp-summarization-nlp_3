import os
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


load_dotenv()

TIMEZONE: ZoneInfo = ZoneInfo("Asia/Seoul")
_ROOT_MARKER = "pyproject.toml"


def find_project_root() -> Path:
    """
    pyproject.toml 파일을 기준으로 루트 디렉토리를 찾는다.
    :return: Path: 프로젝트 루트 디렉토리 경로
    """

    current_path = Path().resolve()

    while current_path != current_path.parent:
        if (current_path / _ROOT_MARKER).exists():
            return current_path

        current_path = current_path.parent

    raise FileNotFoundError("프로젝트 루트 디렉토리를 찾을 수 없습니다.")


def get_env_or_raise(key: str) -> str:
    env_value = os.getenv(key)
    if env_value is None:
        raise ValueError(f"{key} not set in environment variables")
    return env_value


# %% directory

ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CONFIGS_DIR = ROOT_DIR / "configs"

PREDICTION_DIR = RAW_DATA_DIR / "predictions"
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = RAW_DATA_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_ROOT_DIR = ROOT_DIR / "logs"
LOG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

# %% 환경변수

# Weights & Biases
WANDB_API_KEY = get_env_or_raise("WANDB_API_KEY")
WANDB_PROJECT = get_env_or_raise("WANDB_PROJECT")
WANDB_ENTITY = get_env_or_raise("WANDB_ENTITY")

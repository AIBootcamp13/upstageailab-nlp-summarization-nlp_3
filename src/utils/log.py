import json
import logging
from datetime import datetime
from pathlib import Path

from src import config


DATE_FORMAT = "%Y-%m-%d"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

FILE_BACKUP_COUNT: int = 10
FILE_MAX_BYTES: int = 10 * 1024 * 1024  # default: 10MB


class JsonFormatter(logging.Formatter):
    """JSON 형식으로 로그를 포맷팅하는 클래스"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, DATETIME_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        # extra fields 추가
        if hasattr(record, "extras"):
            log_data.update(record.extras)

        # request 관련 정보가 있다면 추가
        request_attrs = ("request_id", "method", "path", "client_host", "request_body")
        for attr in request_attrs:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        # exception 정보가 있다면 추가
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    if logger.handlers:  # 이미 설정됨
        return logger

    log_file_path = _get_log_path(logger_name)
    _setup_logger(
        logger=logger,
        log_level=logging.DEBUG,
        log_file_path=log_file_path,
        log_file_max_bytes=FILE_MAX_BYTES,
        log_file_backup_count=FILE_BACKUP_COUNT,
    )
    return logger


def _get_log_path(logger_name: str) -> Path:
    """로그 파일의 전체 경로 반환"""
    # 날짜별로 log 가 모이도록
    date = datetime.now(config.TIMEZONE).strftime(DATE_FORMAT)
    current_log_dir: Path = config.LOG_ROOT_DIR / date
    current_log_dir.mkdir(parents=True, exist_ok=True)
    return current_log_dir / f"{logger_name}.log"


def _setup_logger(
    logger: logging.Logger,
    log_level: int,
    log_file_path: Path,
    log_file_max_bytes: int,
    log_file_backup_count: int,
) -> None:
    """logger 설정(콘솔 및 파일)

    Args:
        logger: 설정할 로거 인스턴스
        log_level: 로그 레벨
        log_file_path: 로그 파일 경로
        log_file_max_bytes: 각 로그 파일의 최대 크기
        log_file_backup_count: 보관할 백업 파일 수
    """
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s] %(message)s",
        datefmt=DATETIME_FORMAT,
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file_path),
        maxBytes=log_file_max_bytes,
        backupCount=log_file_backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

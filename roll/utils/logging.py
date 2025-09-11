import logging
import os
import sys
from typing import Optional


def is_roll_debug_mode():
    return os.getenv("ROLL_DEBUG", os.getenv("RAY_PROFILING", "0")) == "1"


class CustomFormatter(logging.Formatter):
    def format(self, record):
        record.__dict__["RANK"] = os.environ.get("RANK", "0")
        record.__dict__["WORLD_SIZE"] = os.environ.get("WORLD_SIZE", "1")
        record.__dict__["WORKER_NAME"] = os.environ.get("WORKER_NAME", "DRIVER")
        return super(CustomFormatter, self).format(record)


def reset_file_logger_handler(_logger, log_dir, formatter, WORKER_NAME=None):
    for handler in _logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            _logger.removeHandler(handler)
            handler.close()
    global logger_log_dir
    logger_log_dir = log_dir
    if WORKER_NAME == None:
        WORKER_NAME = os.environ.get("WORKER_NAME", "DRIVER")
    log_path = os.path.join(
        log_dir, f"log_rank_{WORKER_NAME}_{os.environ.get('RANK', '0')}_" f"{os.environ.get('WORLD_SIZE', '1')}.log"
    )
    try:
        log_dir_path = os.path.dirname(log_path)
        if log_dir_path:
            os.makedirs(log_dir_path, exist_ok=True)
            print(f"Created or verified log directory: {log_dir_path}")
    except Exception as e:
        print(f"Warning: Failed to create log directory: {e}")
        log_path = os.path.join("./output/logs", os.path.basename(log_path))
        os.makedirs("./output/logs", exist_ok=True)
    try:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)
        print(f"Added logging to file: {os.path.abspath(log_path)}")
    except Exception as e:
        print(f"Warning: Unexpected error creating log file: {e}")


logger: Optional[logging.Logger] = None
logger_log_dir: Optional[str] = None


def get_logger() -> logging.Logger:
    r"""
    Gets a standard logger with a stream handler to stdout.
    """
    formatter = CustomFormatter(
        fmt=f"[%(asctime)s] [%(filename)s (%(lineno)d)] [%(levelname)s] "
        f"[%(WORKER_NAME)s %(RANK)s / %(WORLD_SIZE)s]"
        f"[PID {os.getpid()}] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log_dir = os.environ.get("ROLL_LOG_DIR", "./output/logs")
    global logger, logger_log_dir
    if logger is not None:
        if logger_log_dir == log_dir:
            return logger
        else:
            reset_file_logger_handler(logger, log_dir, formatter)
    _logger_name = (
        f"log_rank_{os.environ.get('WORKER_NAME', 'DRIVER')}_{os.environ.get('RANK', '0')}_"
        f"{os.environ.get('WORLD_SIZE', '1')}"
    )
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.INFO)
    stream_handler_exists = any(handler.get_name() == _logger_name for handler in _logger.handlers)

    if not stream_handler_exists:
        print(f"add logger: {_logger_name}")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.set_name(_logger_name)
        _logger.addHandler(handler)

    reset_file_logger_handler(_logger, log_dir, formatter)

    logger = _logger
    return _logger

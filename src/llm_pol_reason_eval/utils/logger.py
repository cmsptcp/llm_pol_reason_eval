import logging
import os

def get_logger(name: str):
    logger = logging.getLogger(name)
    log_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
    logger.setLevel(log_level)
    if not logger.handlers:
        _handler = logging.StreamHandler()
        _formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        _handler.setFormatter(_formatter)
        logger.addHandler(_handler)
    return logger
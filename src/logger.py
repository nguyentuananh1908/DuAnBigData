"""
Structured Logging utility for the Big Data Pipeline.
Outputs both to daily rotating log files and terminal stream with clear formatting.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from src.config import LOGS_DIR

def get_logger(name: str = "BigDataPipeline") -> logging.Logger:
    """
    Returns a configured logger instance that writes to both console and a daily log file.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Formatters
    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # File Handler (Daily log file)
    current_date = datetime.now().strftime("%Y%m%d")
    log_file = LOGS_DIR / f"pipeline_{current_date}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger

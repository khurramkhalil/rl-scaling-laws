"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    name: str = "rl_scaling_laws",
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_dir: Directory to save log files. If None, only console logging.
        log_level: Logging level (e.g., logging.INFO, logging.DEBUG).
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "train.log")
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rl_scaling_laws") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)

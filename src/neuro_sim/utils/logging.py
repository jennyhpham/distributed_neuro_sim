"""Logging utilities for neuro_sim."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "neuro_sim",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files (if None, no file logging)
        log_level: Logging level
        console: Whether to add console handler

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter(
        "%(levelname)s - %(message)s",
    )
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


"""
logger.py
---------
Configures and returns a reusable logger for the project.
Logs are written both to file and to the console.
"""

import logging
import os
from pathlib import Path

# Directory to store log files
LOG_DIR: Path = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger with file and console handlers.

    Args:
        name (str): The name of the logger / module.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

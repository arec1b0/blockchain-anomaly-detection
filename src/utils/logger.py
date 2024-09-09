"""
logger.py

This module provides centralized logging functionality for the application. It ensures
that all modules have consistent and structured logging. Logs can be output to both 
the console and a file for long-term storage and analysis.

Adheres to the Single Responsibility Principle (SRP) by handling only logging-related operations.
"""

import logging
import os
from src.utils.config import LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger instance with the specified name.

    :param name: The name of the logger, typically the module name.
    :return: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Ensure no duplicate handlers are added
    if not logger.hasHandlers():
        # Console handler for output to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

        # File handler for output to a log file
        log_directory = "logs"
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)

        file_handler = logging.FileHandler(f"{log_directory}/app.log")
        file_handler.setLevel(logging.INFO)

        # Formatter for structured logging output
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Attach the formatter to both handlers
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Attach handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

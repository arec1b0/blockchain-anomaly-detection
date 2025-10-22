"""
config.py

This module contains global configuration parameters loaded from environment variables.
It ensures centralized management of key configurations used across different modules.

All configuration values are loaded from environment variables with sensible defaults.
Adheres to the principle of Single Responsibility by isolating all configuration data.
"""

import os
from typing import Any, Dict


class Config:
    """Configuration class that loads all settings from environment variables."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        # Etherscan API Configuration
        self.API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
        self.BASE_URL = os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/api")
        self.ETHERSCAN_ADDRESS = os.getenv("ETHERSCAN_ADDRESS", "")

        # Request Configuration
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "10"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_BACKOFF = int(os.getenv("RETRY_BACKOFF", "2"))

        # Environment Settings
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

        # Dask Configuration
        self.USE_DASK = os.getenv("USE_DASK", "false").lower() == "true"
        self.DASK_N_WORKERS = int(os.getenv("DASK_N_WORKERS", "4"))
        self.DASK_THREADS_PER_WORKER = int(os.getenv("DASK_THREADS_PER_WORKER", "1"))
        self.DASK_MEMORY_LIMIT = os.getenv("DASK_MEMORY_LIMIT", "auto")

        # Sentry Configuration
        self.SENTRY_DSN = os.getenv("SENTRY_DSN", "")
        self.SENTRY_ENABLED = os.getenv("SENTRY_ENABLED", "false").lower() == "true"
        self.SENTRY_ENVIRONMENT = os.getenv("SENTRY_ENVIRONMENT", self.ENVIRONMENT)
        self.SENTRY_TRACES_SAMPLE_RATE = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0"))

    def validate(self) -> bool:
        """
        Validate that required configuration values are set.

        :return: True if configuration is valid, False otherwise.
        :raises: ValueError if required configuration is missing.
        """
        errors = []

        if not self.API_KEY:
            errors.append("ETHERSCAN_API_KEY is required but not set")

        if self.REQUEST_TIMEOUT <= 0:
            errors.append("REQUEST_TIMEOUT must be positive")

        if self.MAX_RETRIES < 0:
            errors.append("MAX_RETRIES must be non-negative")

        if self.RETRY_BACKOFF <= 0:
            errors.append("RETRY_BACKOFF must be positive")

        if self.SENTRY_ENABLED and not self.SENTRY_DSN:
            errors.append("SENTRY_DSN is required when SENTRY_ENABLED is true")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        :return: Dictionary containing all configuration values.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        }


# Global configuration instance
_config = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    :return: Configuration object.
    """
    global _config
    if _config is None:
        _config = Config()
    return _config


# Backward compatibility - expose individual values as module-level variables
# These will be initialized when the module is first imported
_config_instance = get_config()

API_KEY = _config_instance.API_KEY
BASE_URL = _config_instance.BASE_URL
REQUEST_TIMEOUT = _config_instance.REQUEST_TIMEOUT
MAX_RETRIES = _config_instance.MAX_RETRIES
RETRY_BACKOFF = _config_instance.RETRY_BACKOFF
ENVIRONMENT = _config_instance.ENVIRONMENT
LOG_LEVEL = _config_instance.LOG_LEVEL

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
    """
    Configuration class that loads all settings from environment variables.

    This class centralizes the management of configuration parameters, making it
    easy to access and validate settings used throughout the application.

    Attributes:
        API_KEY (str): The Etherscan API key.
        BASE_URL (str): The base URL for the Etherscan API.
        ETHERSCAN_ADDRESS (str): The Ethereum address to monitor.
        REQUEST_TIMEOUT (int): The timeout for HTTP requests in seconds.
        MAX_RETRIES (int): The maximum number of retries for failed requests.
        RETRY_BACKOFF (int): The backoff factor for retries.
        ENVIRONMENT (str): The application environment (e.g., 'development', 'production').
        LOG_LEVEL (str): The logging level for the application.
        USE_DASK (bool): A flag to enable or disable Dask for parallel processing.
        DASK_N_WORKERS (int): The number of Dask workers.
        DASK_THREADS_PER_WORKER (int): The number of threads per Dask worker.
        DASK_MEMORY_LIMIT (str): The memory limit for each Dask worker.
        SENTRY_DSN (str): The DSN for Sentry error tracking.
        SENTRY_ENABLED (bool): A flag to enable or disable Sentry.
        SENTRY_ENVIRONMENT (str): The Sentry environment.
        SENTRY_TRACES_SAMPLE_RATE (float): The traces sample rate for Sentry.
    """

    def __init__(self):
        """
        Initializes the configuration from environment variables.
        """
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

        # Database Configuration
        self.DATABASE_HOST = os.getenv("DATABASE_HOST", "localhost")
        self.DATABASE_PORT = int(os.getenv("DATABASE_PORT", "5432"))
        self.DATABASE_NAME = os.getenv("DATABASE_NAME", "blockchain_anomaly")
        self.DATABASE_USER = os.getenv("DATABASE_USER", "anomaly_user")
        self.DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "")
        
        # Construct DATABASE_URL if not provided directly
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            database_url = (
                f"postgresql://{self.DATABASE_USER}:{self.DATABASE_PASSWORD}"
                f"@{self.DATABASE_HOST}:{self.DATABASE_PORT}/{self.DATABASE_NAME}"
            )
        self.DATABASE_URL = database_url
        
        # Database Pool Configuration
        self.DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "20"))
        self.DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "10"))
        self.DATABASE_POOL_TIMEOUT = int(os.getenv("DATABASE_POOL_TIMEOUT", "30"))
        self.DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

    def validate(self) -> bool:
        """
        Validates that the required configuration values are set.

        Returns:
            bool: True if the configuration is valid.

        Raises:
            ValueError: If a required configuration is missing or invalid.
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

        if not self.DATABASE_PASSWORD:
            errors.append("DATABASE_PASSWORD is required")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the configuration to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing all configuration values.
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
    Gets the global configuration instance.

    This function ensures that the configuration is loaded only once and returns
    the same instance on subsequent calls.

    Returns:
        Config: The global configuration object.
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

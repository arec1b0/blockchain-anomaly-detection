"""
config.py

This module contains global configuration parameters such as API keys and other
important settings. It ensures centralized management of key configurations used across
different modules of the application.

Adheres to the principle of Single Responsibility by isolating all configuration data.
"""

# Etherscan API Key - Used to authenticate requests to the Etherscan API
API_KEY = "YOUR_ETHERSCAN_API_KEY"

# Base URL for Etherscan API
BASE_URL = "https://api.etherscan.io/api"

# Timeout settings for API requests
REQUEST_TIMEOUT = 10  # seconds

# Retry settings in case of request failure
MAX_RETRIES = 3  # Number of times to retry on failure
RETRY_BACKOFF = 2  # Backoff multiplier for retries

# Environment settings
ENVIRONMENT = "development"  # Change to "production" in production environment

# Logging level (can be "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
LOG_LEVEL = "INFO"

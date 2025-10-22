"""
etherscan_api.py

This module is responsible for interacting with the Etherscan API to retrieve blockchain
transaction data. It includes methods for fetching transactions by address and handling
common errors like API rate limits.

Adheres to the Single Responsibility Principle (SRP) by isolating the logic for interacting
with the API, and follows best practices for error handling and logging.
"""

import requests
from requests.exceptions import HTTPError
from src.utils.logger import get_logger
from src.utils.config import get_config
import time
import re

# Initialize logger
logger = get_logger(__name__)


class EtherscanAPI:
    """
    EtherscanAPI handles communication with the Etherscan API to fetch transaction data.
    """

    def __init__(self, api_key=None, base_url=None):
        """
        Initializes the EtherscanAPI class with the API key and base URL.

        :param api_key: API key for accessing the Etherscan API.
        :param base_url: The base URL for the Etherscan API.
        :raises ValueError: If API key or base URL is invalid.
        """
        config = get_config()
        self.api_key = api_key if api_key is not None else config.API_KEY
        self.base_url = base_url if base_url is not None else config.BASE_URL

        if not self.api_key:
            raise ValueError("API key is required")
        if not self.base_url:
            raise ValueError("Base URL is required")
        if not isinstance(self.api_key, str) or len(self.api_key) == 0:
            raise ValueError("API key must be a non-empty string")

        logger.info("EtherscanAPI initialized successfully")

    def get_transactions(self, address: str, start_block: int = 0,
                         end_block: int = 99999999, sort: str = 'asc'):
        """
        Fetches the list of transactions for a given Ethereum address from Etherscan API.

        :param address: The Ethereum wallet address to fetch transactions for.
        :param start_block: The starting block number for the transaction history.
        :param end_block: The ending block number for the transaction history.
        :param sort: The order in which to sort the transactions ('asc' or 'desc').
        :return: List of transactions in JSON format or None if an error occurs.
        :raises ValueError: If input parameters are invalid.
        """
        # Validate inputs
        if not address or not isinstance(address, str):
            raise ValueError("Address must be a non-empty string")

        # Validate Ethereum address format (basic check)
        if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
            raise ValueError(f"Invalid Ethereum address format: {address}")

        if not isinstance(start_block, int) or start_block < 0:
            raise ValueError("start_block must be a non-negative integer")

        if not isinstance(end_block, int) or end_block < 0:
            raise ValueError("end_block must be a non-negative integer")

        if start_block > end_block:
            raise ValueError("start_block cannot be greater than end_block")

        if sort not in ['asc', 'desc']:
            raise ValueError("sort must be either 'asc' or 'desc'")

        config = get_config()
        url = (f"{self.base_url}?module=account&action=txlist&address={address}"
               f"&startblock={start_block}&endblock={end_block}&sort={sort}"
               f"&apikey={self.api_key}")

        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(f"Requesting transactions for address {address} "
                            f"from block {start_block} to {end_block}")
                response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, dict):
                    raise ValueError("Invalid response format from API")

                if data.get('status') == '1':
                    logger.info(f"Successfully retrieved {len(data['result'])} "
                                f"transactions for address {address}")
                    return data['result']
                else:
                    error_msg = data.get('message', 'Unknown error')
                    logger.error(f"Error in API response: {error_msg}")
                    return None

            except HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
            except requests.Timeout:
                logger.error(f"Request timed out after {config.REQUEST_TIMEOUT} seconds.")
            except ValueError as val_err:
                logger.error(f"Validation error: {val_err}")
                raise
            except Exception as err:
                logger.error(f"An unexpected error occurred: {err}")

            if attempt < config.MAX_RETRIES - 1:  # Don't sleep on the last attempt
                logger.warning(f"Retrying in {config.RETRY_BACKOFF ** attempt} seconds...")
                time.sleep(config.RETRY_BACKOFF ** attempt)

        logger.error(f"Failed to fetch transactions for address {address} "
                     f"after {config.MAX_RETRIES} attempts.")
        return None

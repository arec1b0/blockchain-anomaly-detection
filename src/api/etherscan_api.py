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
from src.utils.config import API_KEY, BASE_URL, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF
import time

# Initialize logger
logger = get_logger(__name__)


class EtherscanAPI:
    """
    EtherscanAPI handles communication with the Etherscan API to fetch transaction data.
    """

    def __init__(self, api_key=API_KEY, base_url=BASE_URL):
        """
        Initializes the EtherscanAPI class with the API key and base URL.

        :param api_key: API key for accessing the Etherscan API.
        :param base_url: The base URL for the Etherscan API.
        """
        self.api_key = api_key
        self.base_url = base_url

    def get_transactions(self, address: str, start_block: int = 0,
                         end_block: int = 99999999, sort: str = 'asc'):
        """
        Fetches the list of transactions for a given Ethereum address from Etherscan API.

        :param address: The Ethereum wallet address to fetch transactions for.
        :param start_block: The starting block number for the transaction history.
        :param end_block: The ending block number for the transaction history.
        :param sort: The order in which to sort the transactions ('asc' or 'desc').
        :return: List of transactions in JSON format or None if an error occurs.
        """
        url = (f"{self.base_url}?module=account&action=txlist&address={address}"
               f"&startblock={start_block}&endblock={end_block}&sort={sort}"
               f"&apikey={self.api_key}")

        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Requesting transactions for address {address} "
                            f"from block {start_block} to {end_block}")
                response = requests.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()

                if data['status'] == '1':
                    logger.info(f"Successfully retrieved {len(data['result'])} "
                                f"transactions for address {address}")
                    return data['result']
                else:
                    logger.error(f"Error in API response: {data['message']}")
                    return None

            except HTTPError as http_err:
                logger.error(f"HTTP error occurred: {http_err}")
            except requests.Timeout:
                logger.error(f"Request timed out after {REQUEST_TIMEOUT} seconds.")
            except Exception as err:
                logger.error(f"An unexpected error occurred: {err}")

            logger.warning(f"Retrying in {RETRY_BACKOFF ** attempt} seconds...")
            time.sleep(RETRY_BACKOFF ** attempt)

        logger.error(f"Failed to fetch transactions for address {address} "
                     f"after {MAX_RETRIES} attempts.")
        return None

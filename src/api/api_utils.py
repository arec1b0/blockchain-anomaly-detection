"""
api_utils.py

This module provides utility functions to assist with API requests, such as handling rate limits,
validating responses, and retry logic. These utilities enhance the robustness and scalability of API interactions.

Adheres to the Single Responsibility Principle (SRP) by focusing only on auxiliary API-related tasks.
"""

import time
import requests
from requests.exceptions import HTTPError
from src.utils.logger import get_logger
from src.utils.config import MAX_RETRIES, RETRY_BACKOFF

# Initialize logger
logger = get_logger(__name__)


def handle_api_rate_limit(response):
    """
    Handles rate limiting for API requests. If the rate limit is exceeded (status code 429), 
    it pauses the execution for the specified amount of time.

    :param response: The response object from the API request.
    :return: None, but pauses execution if rate limit is exceeded.
    """
    if response.status_code == 429:  # Too Many Requests
        retry_after = int(response.headers.get('Retry-After', 60))  # Default to 60 seconds if header is missing
        logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
        time.sleep(retry_after)


def validate_response(response):
    """
    Validates the API response. If the response is not successful (non-2xx status code),
    it logs the error and raises an exception.

    :param response: The response object from the API request.
    :return: Parsed JSON data if valid, otherwise raises an HTTPError.
    """
    try:
        response.raise_for_status()
        logger.info("API response successfully validated.")
        return response.json()
    except HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise http_err
    except Exception as err:
        logger.error(f"An unexpected error occurred while validating response: {err}")
        raise err


def retry_request(func, *args, **kwargs):
    """
    Retry wrapper for functions making API requests. It attempts to execute the function and retries
    in case of failure up to MAX_RETRIES, using exponential backoff.

    :param func: The function to be retried.
    :param args: Positional arguments to pass to the function.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The result of the function if successful, otherwise raises an exception.
    """
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempt {attempt + 1} of {MAX_RETRIES}")
            return func(*args, **kwargs)
        except (HTTPError, requests.Timeout) as e:
            logger.error(f"Error occurred: {e}. Retrying in {RETRY_BACKOFF ** attempt} seconds...")
            time.sleep(RETRY_BACKOFF ** attempt)
        except Exception as err:
            logger.error(f"An unexpected error occurred: {err}")
            raise err

    logger.error(f"Failed after {MAX_RETRIES} attempts.")
    raise Exception(f"Max retries exceeded for function {func.__name__}")

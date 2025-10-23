"""
sentry.py

This module handles Sentry integration for error tracking and monitoring.
It provides functions to initialize Sentry and capture exceptions.

Sentry integration is optional and controlled by environment variables.
"""

import sentry_sdk
from typing import Optional
from src.utils.logger import get_logger
from src.utils.config import get_config

# Initialize logger
logger = get_logger(__name__)

# Global flag to track if Sentry is initialized
_sentry_initialized = False


def init_sentry() -> bool:
    """
    Initializes the Sentry SDK for error tracking.

    This function configures and initializes the Sentry SDK based on the
    environment variables. Sentry will not be initialized if it is disabled
    or the DSN is not provided.

    Returns:
        bool: True if Sentry was successfully initialized, False otherwise.
    """
    global _sentry_initialized

    if _sentry_initialized:
        logger.info("Sentry already initialized")
        return True

    config = get_config()

    if not config.SENTRY_ENABLED:
        logger.info("Sentry is disabled via configuration")
        return False

    if not config.SENTRY_DSN:
        logger.warning("Sentry is enabled but DSN is not configured")
        return False

    try:
        sentry_sdk.init(
            dsn=config.SENTRY_DSN,
            environment=config.SENTRY_ENVIRONMENT,
            traces_sample_rate=config.SENTRY_TRACES_SAMPLE_RATE,
            # Set release version if available
            release=f"blockchain-anomaly-detection@0.1.0",
            # Enable automatic breadcrumbs
            attach_stacktrace=True,
            # Send default PII (Personally Identifiable Information)
            send_default_pii=False,
        )
        _sentry_initialized = True
        logger.info(f"Sentry initialized successfully for environment: {config.SENTRY_ENVIRONMENT}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False


def capture_exception(error: Exception, context: Optional[dict] = None) -> Optional[str]:
    """
    Captures an exception and sends it to Sentry.

    Args:
        error (Exception): The exception to capture.
        context (Optional[dict]): An optional dictionary with additional context.
            Defaults to None.

    Returns:
        Optional[str]: The event ID if the exception was successfully sent to Sentry,
            None otherwise.
    """
    if not _sentry_initialized:
        logger.debug("Sentry not initialized, skipping exception capture")
        return None

    try:
        # Add custom context if provided
        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_context(key, value)
                event_id = sentry_sdk.capture_exception(error)
        else:
            event_id = sentry_sdk.capture_exception(error)

        logger.debug(f"Exception captured by Sentry with event ID: {event_id}")
        return event_id
    except Exception as e:
        logger.error(f"Failed to capture exception in Sentry: {e}")
        return None


def capture_message(message: str, level: str = 'info', context: Optional[dict] = None) -> Optional[str]:
    """
    Captures a message and sends it to Sentry.

    Args:
        message (str): The message to capture.
        level (str): The severity level of the message (e.g., 'debug', 'info',
            'warning', 'error', 'fatal'). Defaults to 'info'.
        context (Optional[dict]): An optional dictionary with additional context.
            Defaults to None.

    Returns:
        Optional[str]: The event ID if the message was successfully sent to Sentry,
            None otherwise.
    """
    if not _sentry_initialized:
        logger.debug("Sentry not initialized, skipping message capture")
        return None

    try:
        # Add custom context if provided
        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_context(key, value)
                event_id = sentry_sdk.capture_message(message, level=level)
        else:
            event_id = sentry_sdk.capture_message(message, level=level)

        logger.debug(f"Message captured by Sentry with event ID: {event_id}")
        return event_id
    except Exception as e:
        logger.error(f"Failed to capture message in Sentry: {e}")
        return None


def set_user(user_id: Optional[str] = None, email: Optional[str] = None,
             username: Optional[str] = None, **kwargs) -> None:
    """
    Sets the user information for the Sentry context.

    Args:
        user_id (Optional[str]): The user ID. Defaults to None.
        email (Optional[str]): The user's email address. Defaults to None.
        username (Optional[str]): The username. Defaults to None.
        **kwargs: Additional user attributes.
    """
    if not _sentry_initialized:
        logger.debug("Sentry not initialized, skipping user context")
        return

    try:
        user_data = {}
        if user_id:
            user_data['id'] = user_id
        if email:
            user_data['email'] = email
        if username:
            user_data['username'] = username
        user_data.update(kwargs)

        sentry_sdk.set_user(user_data)
        logger.debug("User context set in Sentry")
    except Exception as e:
        logger.error(f"Failed to set user context in Sentry: {e}")


def add_breadcrumb(message: str, category: str = 'default',
                   level: str = 'info', data: Optional[dict] = None) -> None:
    """
    Adds a breadcrumb to Sentry for better error context.

    Args:
        message (str): The breadcrumb message.
        category (str): The category of the breadcrumb. Defaults to 'default'.
        level (str): The severity level of the breadcrumb. Defaults to 'info'.
        data (Optional[dict]): An optional dictionary with additional data.
            Defaults to None.
    """
    if not _sentry_initialized:
        logger.debug("Sentry not initialized, skipping breadcrumb")
        return

    try:
        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except Exception as e:
        logger.error(f"Failed to add breadcrumb in Sentry: {e}")


def close_sentry(timeout: int = 2) -> None:
    """
    Flushes and closes the Sentry client.

    Args:
        timeout (int): The timeout in seconds to wait for pending events to be sent.
            Defaults to 2.
    """
    global _sentry_initialized

    if not _sentry_initialized:
        logger.debug("Sentry not initialized, nothing to close")
        return

    try:
        sentry_sdk.flush(timeout=timeout)
        _sentry_initialized = False
        logger.info("Sentry client closed")
    except Exception as e:
        logger.error(f"Failed to close Sentry client: {e}")

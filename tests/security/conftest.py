"""
Shared fixtures for security tests.

Provides reusable test fixtures for authentication, authorization,
API keys, rate limiting, and audit logging tests.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.auth.jwt_handler import jwt_handler
from src.auth.user_manager import UserManager, User, get_user_manager
from src.auth.api_key_manager import APIKeyManager, get_api_key_manager
from src.middleware.rate_limiter import RateLimiter, InMemoryRateLimiter
from src.audit.audit_logger import AuditLogger, get_audit_logger


@pytest.fixture
def fresh_user_manager():
    """Create a fresh UserManager instance for each test."""
    manager = UserManager()
    # Clear default users for clean testing
    manager._users.clear()
    manager._email_index.clear()
    return manager


@pytest.fixture
def test_user(fresh_user_manager):
    """Create a test user."""
    return fresh_user_manager.create_user(
        email="test@example.com",
        password="testpass123",
        roles=["user"]
    )


@pytest.fixture
def test_admin(fresh_user_manager):
    """Create a test admin user."""
    return fresh_user_manager.create_user(
        email="admin@example.com",
        password="adminpass123",
        roles=["user", "admin"]
    )


@pytest.fixture
def test_access_token(test_user):
    """Create a valid access token for test user."""
    return jwt_handler.create_access_token(
        user_id=test_user.id,
        email=test_user.email,
        roles=test_user.roles
    )


@pytest.fixture
def test_admin_token(test_admin):
    """Create a valid access token for admin user."""
    return jwt_handler.create_access_token(
        user_id=test_admin.id,
        email=test_admin.email,
        roles=test_admin.roles
    )


@pytest.fixture
def test_refresh_token(test_user):
    """Create a valid refresh token for test user."""
    return jwt_handler.create_refresh_token(user_id=test_user.id)


@pytest.fixture
def expired_token(test_user):
    """Create an expired access token."""
    return jwt_handler.create_access_token(
        user_id=test_user.id,
        email=test_user.email,
        roles=test_user.roles,
        expires_delta=timedelta(seconds=-1)  # Already expired
    )


@pytest.fixture
def fresh_api_key_manager():
    """Create a fresh APIKeyManager instance for each test."""
    manager = APIKeyManager()
    manager._api_keys.clear()
    manager._key_index.clear()
    return manager


@pytest.fixture
def test_api_key(fresh_api_key_manager, test_user):
    """Create a test API key."""
    api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
        user_id=test_user.id,
        name="Test API Key"
    )
    # Store plain key on the object for testing
    api_key_obj._plain_key = plain_key
    return api_key_obj


@pytest.fixture
def expired_api_key(fresh_api_key_manager, test_user):
    """Create an expired API key."""
    api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
        user_id=test_user.id,
        name="Expired Key",
        expires_days=0  # Expires immediately
    )
    api_key_obj._plain_key = plain_key
    return api_key_obj


@pytest.fixture
def fresh_rate_limiter():
    """Create a fresh in-memory rate limiter for each test."""
    return InMemoryRateLimiter()


@pytest.fixture
def fresh_audit_logger():
    """Create a fresh AuditLogger instance for each test."""
    logger = AuditLogger(max_size=100)
    logger._logs.clear()
    return logger


@pytest.fixture
def mock_request():
    """Create a mock FastAPI Request object."""
    request = Mock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-client"}
    request.url.path = "/test"
    request.method = "GET"
    return request


@pytest.fixture
def auth_headers(test_access_token):
    """Create authentication headers with valid token."""
    return {"Authorization": f"Bearer {test_access_token}"}


@pytest.fixture
def admin_headers(test_admin_token):
    """Create authentication headers with admin token."""
    return {"Authorization": f"Bearer {test_admin_token}"}


@pytest.fixture
def api_key_headers(test_api_key):
    """Create API key authentication headers."""
    return {"X-API-Key": test_api_key._plain_key}


@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing."""
    return {
        "hash": "0x123abc",
        "value": 1.5,
        "gas": 21000.0,
        "gasPrice": 50.0,
        "from": "0xSender",
        "to": "0xReceiver",
        "blockNumber": 12345,
        "timestamp": "2024-01-01T00:00:00Z"
    }


@pytest.fixture
def batch_transaction_data():
    """Batch of transaction data for testing."""
    return {
        "transactions": [
            {
                "hash": f"0x{i}",
                "value": float(i * 10),
                "gas": 21000.0 + i * 100,
                "gasPrice": 50.0 + i,
                "from": f"0xSender{i}",
                "to": f"0xReceiver{i}",
                "blockNumber": 12345 + i,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            for i in range(10)
        ]
    }


# Utility functions for tests

def create_user_payload(email: str = "newuser@example.com",
                        password: str = "securepass123") -> Dict[str, str]:
    """Create a user registration payload."""
    return {
        "email": email,
        "password": password,
        "confirm_password": password
    }


def create_login_payload(email: str = "test@example.com",
                        password: str = "testpass123") -> Dict[str, str]:
    """Create a login payload."""
    return {
        "email": email,
        "password": password
    }


def decode_token_unsafe(token: str) -> Dict[str, Any]:
    """Decode token without verification (for testing)."""
    import jwt as pyjwt
    return pyjwt.decode(token, options={"verify_signature": False})

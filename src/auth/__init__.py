"""
Authentication module for the blockchain anomaly detection system.

This module provides:
- JWT token generation and validation
- User authentication and management
- API key management for service accounts
- Password hashing and verification
"""

from src.auth.jwt_handler import (
    jwt_handler,
    get_current_user,
    get_current_active_user,
    require_role,
    JWTHandler
)
from src.auth.user_manager import UserManager
from src.auth.api_key_manager import APIKeyManager

__all__ = [
    'jwt_handler',
    'get_current_user',
    'get_current_active_user',
    'require_role',
    'JWTHandler',
    'UserManager',
    'APIKeyManager',
]

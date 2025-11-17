"""
API Key management for service-to-service authentication.

This module provides API key functionality for:
- Service accounts and automated systems
- CI/CD pipelines
- Third-party integrations
- Long-lived authentication tokens

API keys are more suitable than JWT tokens for:
- Automated systems (no user interaction)
- Long-running processes
- Simpler integration (no token refresh needed)
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from src.auth.jwt_handler import jwt_handler
from src.auth.user_manager import get_user_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIKey:
    """
    API Key model (in-memory for Phase 1).

    In Phase 2, this will be replaced with SQLAlchemy model.
    """

    def __init__(
        self,
        id: str,
        user_id: str,
        name: str,
        hashed_key: str,
        prefix: str,
        is_active: bool = True,
        expires_at: Optional[datetime] = None,
        created_at: datetime = None,
        last_used: datetime = None
    ):
        self.id = id
        self.user_id = user_id
        self.name = name
        self.hashed_key = hashed_key
        self.prefix = prefix  # First 8 chars for identification
        self.is_active = is_active
        self.expires_at = expires_at
        self.created_at = created_at or datetime.utcnow()
        self.last_used = last_used

    def to_dict(self, include_key: bool = False) -> Dict[str, Any]:
        """
        Convert API key to dictionary.

        Args:
            include_key: If True, include plain key (only on creation)
        """
        data = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "prefix": self.prefix,
            "is_active": self.is_active,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }
        return data


class APIKeyManager:
    """
    Manages API keys for service-to-service authentication.

    API keys are:
    - Long-lived (optional expiration)
    - Hashed in storage (like passwords)
    - Prefixed for easy identification
    - Associated with a user account
    """

    def __init__(self):
        """Initialize API key manager with in-memory storage."""
        self._api_keys: Dict[str, APIKey] = {}
        self._prefix_index: Dict[str, list] = {}  # prefix -> [key_ids]

    def create_api_key(
        self,
        user_id: str,
        name: str,
        expires_days: Optional[int] = None
    ) -> tuple[APIKey, str]:
        """
        Create a new API key.

        Args:
            user_id: User ID who owns this API key
            name: Descriptive name for the API key
            expires_days: Days until expiration (None = no expiration)

        Returns:
            tuple[APIKey, str]: Created API key object and plain key
                                (plain key is only returned once!)

        Example:
            >>> manager = APIKeyManager()
            >>> api_key, plain_key = manager.create_api_key(
            ...     user_id="123",
            ...     name="CI/CD Pipeline",
            ...     expires_days=90
            ... )
            >>> print(f"API Key: {plain_key}")
            >>> # Store plain_key securely - it won't be shown again!
        """
        # Generate secure random key
        # Format: sk_<32 hex chars> (similar to Stripe, OpenAI)
        random_part = secrets.token_hex(32)
        plain_key = f"sk_{random_part}"

        # Hash the key for storage
        hashed_key = jwt_handler.hash_password(plain_key)

        # Extract prefix (first 8 chars for lookup)
        prefix = plain_key[:8]

        # Calculate expiration
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        # Create API key
        key_id = str(uuid.uuid4())
        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name,
            hashed_key=hashed_key,
            prefix=prefix,
            is_active=True,
            expires_at=expires_at,
            created_at=datetime.utcnow()
        )

        # Store API key
        self._api_keys[key_id] = api_key

        # Index by prefix for faster lookup
        if prefix not in self._prefix_index:
            self._prefix_index[prefix] = []
        self._prefix_index[prefix].append(key_id)

        logger.info(
            f"Created API key '{name}' for user {user_id} "
            f"(expires: {expires_at.isoformat() if expires_at else 'never'})"
        )

        return api_key, plain_key

    def validate_api_key(self, key: str) -> Optional[str]:
        """
        Validate API key and return associated user ID.

        Args:
            key: API key string (e.g., "sk_...")

        Returns:
            str: User ID if key is valid, None otherwise

        Example:
            >>> manager = APIKeyManager()
            >>> user_id = manager.validate_api_key("sk_abc123...")
            >>> if user_id:
            ...     print(f"Valid key for user: {user_id}")
        """
        # Extract prefix
        if not key.startswith("sk_"):
            logger.warning("Invalid API key format (missing sk_ prefix)")
            return None

        prefix = key[:8]

        # Get potential keys with this prefix
        key_ids = self._prefix_index.get(prefix, [])
        if not key_ids:
            logger.warning(f"No API keys found with prefix: {prefix}...")
            return None

        # Try to verify against all keys with this prefix
        for key_id in key_ids:
            api_key = self._api_keys.get(key_id)
            if not api_key:
                continue

            # Skip inactive keys
            if not api_key.is_active:
                continue

            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                logger.warning(f"API key expired: {api_key.name} (ID: {key_id})")
                # Automatically deactivate expired keys
                api_key.is_active = False
                continue

            # Verify key hash
            if jwt_handler.verify_password(key, api_key.hashed_key):
                # Update last used timestamp
                api_key.last_used = datetime.utcnow()

                logger.info(f"API key validated: {api_key.name} (user: {api_key.user_id})")
                return api_key.user_id

        logger.warning(f"Invalid API key: {prefix}...")
        return None

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """
        Get API key by ID.

        Args:
            key_id: API key ID

        Returns:
            APIKey: API key object, or None if not found
        """
        return self._api_keys.get(key_id)

    def list_api_keys(self, user_id: str) -> list:
        """
        List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            list: List of API key dictionaries
        """
        user_keys = [
            key for key in self._api_keys.values()
            if key.user_id == user_id
        ]
        return [key.to_dict() for key in user_keys]

    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Revoke (deactivate) an API key.

        Args:
            key_id: API key ID
            user_id: User ID (for authorization check)

        Returns:
            bool: True if key was revoked successfully

        Example:
            >>> manager.revoke_api_key(key_id="123", user_id="user456")
        """
        api_key = self._api_keys.get(key_id)
        if not api_key:
            logger.warning(f"Attempted to revoke non-existent API key: {key_id}")
            return False

        # Check ownership
        if api_key.user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to revoke API key owned by {api_key.user_id}"
            )
            return False

        # Deactivate key
        api_key.is_active = False
        logger.info(f"Revoked API key: {api_key.name} (ID: {key_id})")
        return True

    def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """
        Permanently delete an API key.

        Args:
            key_id: API key ID
            user_id: User ID (for authorization check)

        Returns:
            bool: True if key was deleted successfully
        """
        api_key = self._api_keys.get(key_id)
        if not api_key:
            return False

        # Check ownership
        if api_key.user_id != user_id:
            logger.warning(
                f"User {user_id} attempted to delete API key owned by {api_key.user_id}"
            )
            return False

        # Remove from prefix index
        if api_key.prefix in self._prefix_index:
            self._prefix_index[api_key.prefix].remove(key_id)
            if not self._prefix_index[api_key.prefix]:
                del self._prefix_index[api_key.prefix]

        # Delete key
        del self._api_keys[key_id]
        logger.info(f"Deleted API key: {api_key.name} (ID: {key_id})")
        return True


# Global API key manager instance (singleton)
_api_key_manager = None


def get_api_key_manager() -> APIKeyManager:
    """
    Get global API key manager instance.

    Returns:
        APIKeyManager: Singleton API key manager instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


async def get_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to authenticate using API key.

    This can be used alongside JWT authentication as an alternative method.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        Dict[str, Any]: User information if valid, None otherwise

    Raises:
        HTTPException: If API key is invalid

    Example:
        >>> @app.get("/protected")
        >>> async def protected_route(
        >>>     user: dict = Depends(get_user_from_api_key)
        >>> ):
        >>>     return {"user_id": user["sub"]}
    """
    if not api_key:
        return None

    # Validate API key
    manager = get_api_key_manager()
    user_id = manager.validate_api_key(api_key)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Get user information
    user_manager = get_user_manager()
    user = user_manager.get_user_by_id(user_id)

    if not user or not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Return user info in same format as JWT
    return {
        "sub": user.id,
        "email": user.email,
        "roles": user.roles,
        "auth_method": "api_key"
    }

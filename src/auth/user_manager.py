"""
User authentication and management.

This module provides user management functionality including:
- User registration
- User authentication (login)
- Password management
- User activation/deactivation

In production, this integrates with the database layer (Phase 2).
For Phase 1, we provide an in-memory implementation for testing.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from src.auth.jwt_handler import jwt_handler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class User:
    """
    User model (in-memory for Phase 1).

    In Phase 2, this will be replaced with SQLAlchemy model.
    """

    def __init__(
        self,
        id: str,
        email: str,
        hashed_password: str,
        roles: list = None,
        is_active: bool = True,
        created_at: datetime = None,
        last_login: datetime = None
    ):
        self.id = id
        self.email = email
        self.hashed_password = hashed_password
        self.roles = roles or ["user"]
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow()
        self.last_login = last_login

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary (without password)."""
        return {
            "id": self.id,
            "email": self.email,
            "roles": self.roles,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class UserManager:
    """
    Manages user authentication and authorization.

    This is an in-memory implementation for Phase 1.
    In Phase 2, this will be refactored to use PostgreSQL.
    """

    def __init__(self):
        """Initialize user manager with in-memory storage."""
        # In-memory user storage (will be replaced with database in Phase 2)
        self._users: Dict[str, User] = {}
        self._email_index: Dict[str, str] = {}  # email -> user_id mapping

        # Create default admin user for testing
        self._create_default_users()

    def _create_default_users(self):
        """Create default users for testing."""
        # Admin user
        admin = self.create_user(
            email="admin@blockchain-anomaly.com",
            password="admin123",  # Change in production!
            roles=["user", "admin"]
        )
        logger.info(f"Created default admin user: {admin.email}")

        # Regular user
        user = self.create_user(
            email="user@blockchain-anomaly.com",
            password="user123",  # Change in production!
            roles=["user"]
        )
        logger.info(f"Created default user: {user.email}")

    def create_user(
        self,
        email: str,
        password: str,
        roles: list = None
    ) -> User:
        """
        Create a new user.

        Args:
            email: User email (must be unique)
            password: Plain text password (will be hashed)
            roles: List of roles (default: ["user"])

        Returns:
            User: Created user object

        Raises:
            ValueError: If user already exists

        Example:
            >>> manager = UserManager()
            >>> user = manager.create_user(
            ...     email="user@example.com",
            ...     password="securepass123"
            ... )
        """
        # Check if user exists
        if email in self._email_index:
            logger.warning(f"Attempted to create duplicate user: {email}")
            raise ValueError(f"User with email {email} already exists")

        # Hash password
        hashed_password = jwt_handler.hash_password(password)

        # Create user
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            email=email,
            hashed_password=hashed_password,
            roles=roles or ["user"],
            is_active=True,
            created_at=datetime.utcnow()
        )

        # Store user
        self._users[user_id] = user
        self._email_index[email] = user_id

        logger.info(f"Created user: {email} (ID: {user_id})")
        return user

    def authenticate_user(
        self,
        email: str,
        password: str
    ) -> Optional[User]:
        """
        Authenticate user with email and password.

        Args:
            email: User email
            password: Plain text password

        Returns:
            User: Authenticated user object, or None if authentication fails

        Example:
            >>> manager = UserManager()
            >>> user = manager.authenticate_user(
            ...     email="user@example.com",
            ...     password="securepass123"
            ... )
            >>> if user:
            ...     print(f"Authenticated: {user.email}")
        """
        # Get user by email
        user_id = self._email_index.get(email)
        if not user_id:
            logger.warning(f"Authentication failed: User not found ({email})")
            return None

        user = self._users.get(user_id)
        if not user:
            logger.error(f"Data inconsistency: User ID {user_id} not in storage")
            return None

        # Check if user is active
        if not user.is_active:
            logger.warning(f"Authentication failed: User inactive ({email})")
            return None

        # Verify password
        if not jwt_handler.verify_password(password, user.hashed_password):
            logger.warning(f"Authentication failed: Invalid password ({email})")
            return None

        # Update last login
        user.last_login = datetime.utcnow()

        logger.info(f"User authenticated: {email}")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User: User object, or None if not found
        """
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.

        Args:
            email: User email

        Returns:
            User: User object, or None if not found
        """
        user_id = self._email_index.get(email)
        if user_id:
            return self._users.get(user_id)
        return None

    def update_user(
        self,
        user_id: str,
        **updates
    ) -> Optional[User]:
        """
        Update user attributes.

        Args:
            user_id: User ID
            **updates: Attributes to update

        Returns:
            User: Updated user object, or None if not found

        Example:
            >>> manager.update_user(
            ...     user_id="123",
            ...     is_active=False
            ... )
        """
        user = self._users.get(user_id)
        if not user:
            return None

        # Update allowed attributes
        allowed_updates = ['is_active', 'roles']
        for key, value in updates.items():
            if key in allowed_updates:
                setattr(user, key, value)
                logger.info(f"Updated user {user_id}: {key} = {value}")

        return user

    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password.

        Args:
            user_id: User ID
            old_password: Current password (for verification)
            new_password: New password

        Returns:
            bool: True if password changed successfully

        Example:
            >>> manager.change_password(
            ...     user_id="123",
            ...     old_password="oldpass",
            ...     new_password="newpass123"
            ... )
        """
        user = self._users.get(user_id)
        if not user:
            logger.warning(f"Password change failed: User not found ({user_id})")
            return False

        # Verify old password
        if not jwt_handler.verify_password(old_password, user.hashed_password):
            logger.warning(f"Password change failed: Invalid old password ({user_id})")
            return False

        # Hash and set new password
        user.hashed_password = jwt_handler.hash_password(new_password)
        logger.info(f"Password changed for user {user_id}")
        return True

    def delete_user(self, user_id: str) -> bool:
        """
        Delete user.

        Args:
            user_id: User ID

        Returns:
            bool: True if user deleted successfully
        """
        user = self._users.get(user_id)
        if not user:
            return False

        # Remove from indexes
        if user.email in self._email_index:
            del self._email_index[user.email]

        # Remove user
        del self._users[user_id]

        logger.info(f"Deleted user: {user.email} (ID: {user_id})")
        return True

    def list_users(self, skip: int = 0, limit: int = 100) -> list:
        """
        List all users.

        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return

        Returns:
            list: List of user dictionaries
        """
        users = list(self._users.values())
        return [user.to_dict() for user in users[skip:skip+limit]]

    def count_users(self) -> int:
        """Get total number of users."""
        return len(self._users)


# Global user manager instance (singleton)
_user_manager = None


def get_user_manager() -> UserManager:
    """
    Get global user manager instance.

    Returns:
        UserManager: Singleton user manager instance

    Example:
        >>> from src.auth.user_manager import get_user_manager
        >>> manager = get_user_manager()
        >>> user = manager.authenticate_user("email", "password")
    """
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager

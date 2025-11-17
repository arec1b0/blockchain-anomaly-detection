"""
Authentication security tests.

Tests for:
- JWT token creation and validation
- User registration
- User login/logout
- Token expiration
- Password hashing
- Token refresh
- Invalid credentials
"""

import pytest
from datetime import timedelta
from fastapi import HTTPException

from src.auth.jwt_handler import jwt_handler, get_current_user
from src.auth.user_manager import UserManager, get_user_manager
from tests.security.conftest import (
    decode_token_unsafe,
    create_user_payload,
    create_login_payload
)


class TestJWTTokens:
    """Test JWT token creation and validation."""

    def test_create_access_token(self, test_user):
        """Test access token creation."""
        token = jwt_handler.create_access_token(
            user_id=test_user.id,
            email=test_user.email,
            roles=test_user.roles
        )

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify payload
        payload = decode_token_unsafe(token)
        assert payload["sub"] == test_user.id
        assert payload["email"] == test_user.email
        assert payload["roles"] == test_user.roles
        assert payload["type"] == "access"
        assert "exp" in payload
        assert "iat" in payload

    def test_create_refresh_token(self, test_user):
        """Test refresh token creation."""
        token = jwt_handler.create_refresh_token(user_id=test_user.id)

        assert token is not None
        assert isinstance(token, str)

        # Decode and verify payload
        payload = decode_token_unsafe(token)
        assert payload["sub"] == test_user.id
        assert payload["type"] == "refresh"
        assert "exp" in payload
        assert "iat" in payload

    def test_decode_valid_token(self, test_access_token):
        """Test decoding a valid token."""
        payload = jwt_handler.decode_token(test_access_token)

        assert payload is not None
        assert "sub" in payload
        assert "email" in payload
        assert "roles" in payload
        assert payload["type"] == "access"

    def test_decode_expired_token(self, expired_token):
        """Test decoding an expired token raises exception."""
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.decode_token(expired_token)

        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_decode_invalid_token(self):
        """Test decoding an invalid token raises exception."""
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.decode_token("invalid.token.here")

        assert exc_info.value.status_code == 401

    def test_decode_malformed_token(self):
        """Test decoding a malformed token raises exception."""
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.decode_token("not-a-jwt-token")

        assert exc_info.value.status_code == 401

    def test_token_type_validation(self, test_access_token, test_refresh_token):
        """Test token type validation."""
        # Access token should validate as access
        access_payload = jwt_handler.decode_token(test_access_token)
        assert jwt_handler.validate_token_type(access_payload, "access")

        # Refresh token should fail access validation
        refresh_payload = jwt_handler.decode_token(test_refresh_token)
        with pytest.raises(HTTPException) as exc_info:
            jwt_handler.validate_token_type(refresh_payload, "access")

        assert exc_info.value.status_code == 401
        assert "Invalid token type" in exc_info.value.detail

    def test_custom_expiration(self, test_user):
        """Test token with custom expiration."""
        custom_expiry = timedelta(hours=1)
        token = jwt_handler.create_access_token(
            user_id=test_user.id,
            email=test_user.email,
            roles=test_user.roles,
            expires_delta=custom_expiry
        )

        payload = decode_token_unsafe(token)
        assert "exp" in payload

    def test_token_includes_all_roles(self, fresh_user_manager):
        """Test token includes all user roles."""
        user = fresh_user_manager.create_user(
            email="multirole@example.com",
            password="pass123",
            roles=["user", "admin", "moderator"]
        )

        token = jwt_handler.create_access_token(
            user_id=user.id,
            email=user.email,
            roles=user.roles
        )

        payload = decode_token_unsafe(token)
        assert set(payload["roles"]) == {"user", "admin", "moderator"}


class TestPasswordHashing:
    """Test password hashing and verification."""

    def test_hash_password(self):
        """Test password hashing."""
        password = "mySecurePassword123"
        hashed = jwt_handler.hash_password(password)

        assert hashed != password
        assert isinstance(hashed, str)
        assert len(hashed) > 20  # Bcrypt hashes are long

    def test_verify_correct_password(self):
        """Test verifying correct password."""
        password = "correctPassword123"
        hashed = jwt_handler.hash_password(password)

        assert jwt_handler.verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        """Test verifying incorrect password."""
        password = "correctPassword123"
        hashed = jwt_handler.hash_password(password)

        assert jwt_handler.verify_password("wrongPassword", hashed) is False

    def test_hash_produces_different_hashes(self):
        """Test same password produces different hashes (salt)."""
        password = "samePassword123"
        hash1 = jwt_handler.hash_password(password)
        hash2 = jwt_handler.hash_password(password)

        assert hash1 != hash2
        assert jwt_handler.verify_password(password, hash1)
        assert jwt_handler.verify_password(password, hash2)


class TestUserRegistration:
    """Test user registration."""

    def test_create_user_success(self, fresh_user_manager):
        """Test successful user creation."""
        user = fresh_user_manager.create_user(
            email="newuser@example.com",
            password="securepass123",
            roles=["user"]
        )

        assert user is not None
        assert user.email == "newuser@example.com"
        assert user.roles == ["user"]
        assert user.is_active is True
        assert user.hashed_password != "securepass123"  # Should be hashed

    def test_create_user_with_default_role(self, fresh_user_manager):
        """Test user creation with default role."""
        user = fresh_user_manager.create_user(
            email="defaultrole@example.com",
            password="pass123"
        )

        assert user.roles == ["user"]

    def test_create_duplicate_user_fails(self, fresh_user_manager):
        """Test creating duplicate user raises error."""
        fresh_user_manager.create_user(
            email="duplicate@example.com",
            password="pass123"
        )

        with pytest.raises(ValueError) as exc_info:
            fresh_user_manager.create_user(
                email="duplicate@example.com",
                password="pass123"
            )

        assert "already exists" in str(exc_info.value)

    def test_create_admin_user(self, fresh_user_manager):
        """Test creating admin user."""
        admin = fresh_user_manager.create_user(
            email="admin@example.com",
            password="adminpass123",
            roles=["user", "admin"]
        )

        assert "admin" in admin.roles
        assert "user" in admin.roles


class TestUserAuthentication:
    """Test user authentication (login)."""

    def test_authenticate_valid_credentials(self, fresh_user_manager):
        """Test authentication with valid credentials."""
        # Create user
        user = fresh_user_manager.create_user(
            email="auth@example.com",
            password="validpass123"
        )

        # Authenticate
        authenticated = fresh_user_manager.authenticate_user(
            email="auth@example.com",
            password="validpass123"
        )

        assert authenticated is not None
        assert authenticated.id == user.id
        assert authenticated.email == user.email
        assert authenticated.last_login is not None

    def test_authenticate_invalid_password(self, fresh_user_manager, test_user):
        """Test authentication with invalid password."""
        result = fresh_user_manager.authenticate_user(
            email=test_user.email,
            password="wrongpassword"
        )

        assert result is None

    def test_authenticate_nonexistent_user(self, fresh_user_manager):
        """Test authentication with nonexistent user."""
        result = fresh_user_manager.authenticate_user(
            email="nonexistent@example.com",
            password="anypassword"
        )

        assert result is None

    def test_authenticate_inactive_user(self, fresh_user_manager):
        """Test authentication with inactive user."""
        user = fresh_user_manager.create_user(
            email="inactive@example.com",
            password="pass123"
        )

        # Deactivate user
        fresh_user_manager.update_user(user.id, is_active=False)

        # Try to authenticate
        result = fresh_user_manager.authenticate_user(
            email="inactive@example.com",
            password="pass123"
        )

        assert result is None

    def test_last_login_updated(self, fresh_user_manager):
        """Test last login timestamp is updated."""
        user = fresh_user_manager.create_user(
            email="logintime@example.com",
            password="pass123"
        )

        initial_login = user.last_login
        assert initial_login is None

        # Authenticate
        authenticated = fresh_user_manager.authenticate_user(
            email="logintime@example.com",
            password="pass123"
        )

        assert authenticated.last_login is not None


class TestPasswordManagement:
    """Test password change functionality."""

    def test_change_password_success(self, fresh_user_manager, test_user):
        """Test successful password change."""
        result = fresh_user_manager.change_password(
            user_id=test_user.id,
            old_password="testpass123",
            new_password="newpass123"
        )

        assert result is True

        # Verify new password works
        authenticated = fresh_user_manager.authenticate_user(
            email=test_user.email,
            password="newpass123"
        )
        assert authenticated is not None

        # Verify old password doesn't work
        old_auth = fresh_user_manager.authenticate_user(
            email=test_user.email,
            password="testpass123"
        )
        assert old_auth is None

    def test_change_password_wrong_old_password(self, fresh_user_manager, test_user):
        """Test password change with wrong old password."""
        result = fresh_user_manager.change_password(
            user_id=test_user.id,
            old_password="wrongoldpass",
            new_password="newpass123"
        )

        assert result is False

    def test_change_password_nonexistent_user(self, fresh_user_manager):
        """Test password change for nonexistent user."""
        result = fresh_user_manager.change_password(
            user_id="nonexistent-id",
            old_password="old",
            new_password="new"
        )

        assert result is False


class TestUserRetrieval:
    """Test user retrieval methods."""

    def test_get_user_by_id(self, fresh_user_manager, test_user):
        """Test retrieving user by ID."""
        user = fresh_user_manager.get_user_by_id(test_user.id)

        assert user is not None
        assert user.id == test_user.id
        assert user.email == test_user.email

    def test_get_user_by_email(self, fresh_user_manager, test_user):
        """Test retrieving user by email."""
        user = fresh_user_manager.get_user_by_email(test_user.email)

        assert user is not None
        assert user.id == test_user.id
        assert user.email == test_user.email

    def test_get_nonexistent_user_by_id(self, fresh_user_manager):
        """Test retrieving nonexistent user by ID."""
        user = fresh_user_manager.get_user_by_id("nonexistent-id")

        assert user is None

    def test_get_nonexistent_user_by_email(self, fresh_user_manager):
        """Test retrieving nonexistent user by email."""
        user = fresh_user_manager.get_user_by_email("nonexistent@example.com")

        assert user is None


class TestUserManagement:
    """Test user management operations."""

    def test_update_user_is_active(self, fresh_user_manager, test_user):
        """Test updating user is_active status."""
        result = fresh_user_manager.update_user(
            user_id=test_user.id,
            is_active=False
        )

        assert result is not None
        assert result.is_active is False

    def test_update_user_roles(self, fresh_user_manager, test_user):
        """Test updating user roles."""
        result = fresh_user_manager.update_user(
            user_id=test_user.id,
            roles=["user", "admin"]
        )

        assert result is not None
        assert set(result.roles) == {"user", "admin"}

    def test_update_nonexistent_user(self, fresh_user_manager):
        """Test updating nonexistent user."""
        result = fresh_user_manager.update_user(
            user_id="nonexistent",
            is_active=False
        )

        assert result is None

    def test_delete_user(self, fresh_user_manager, test_user):
        """Test deleting user."""
        result = fresh_user_manager.delete_user(test_user.id)

        assert result is True

        # Verify user is gone
        user = fresh_user_manager.get_user_by_id(test_user.id)
        assert user is None

    def test_delete_nonexistent_user(self, fresh_user_manager):
        """Test deleting nonexistent user."""
        result = fresh_user_manager.delete_user("nonexistent")

        assert result is False

    def test_list_users(self, fresh_user_manager):
        """Test listing users."""
        # Create multiple users
        for i in range(5):
            fresh_user_manager.create_user(
                email=f"user{i}@example.com",
                password="pass123"
            )

        users = fresh_user_manager.list_users()

        assert len(users) == 5
        assert all("email" in user for user in users)
        assert all("id" in user for user in users)
        # Password should not be in dict
        assert all("password" not in user for user in users)
        assert all("hashed_password" not in user for user in users)

    def test_list_users_pagination(self, fresh_user_manager):
        """Test listing users with pagination."""
        # Create users
        for i in range(10):
            fresh_user_manager.create_user(
                email=f"user{i}@example.com",
                password="pass123"
            )

        # Get first page
        page1 = fresh_user_manager.list_users(skip=0, limit=5)
        assert len(page1) == 5

        # Get second page
        page2 = fresh_user_manager.list_users(skip=5, limit=5)
        assert len(page2) == 5

        # Verify different users
        page1_ids = {user["id"] for user in page1}
        page2_ids = {user["id"] for user in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_count_users(self, fresh_user_manager):
        """Test counting users."""
        # Create users
        for i in range(7):
            fresh_user_manager.create_user(
                email=f"user{i}@example.com",
                password="pass123"
            )

        count = fresh_user_manager.count_users()
        assert count == 7


class TestSecurityBestPractices:
    """Test security best practices are followed."""

    def test_password_not_in_user_dict(self, fresh_user_manager):
        """Test password is never exposed in user dict."""
        user = fresh_user_manager.create_user(
            email="security@example.com",
            password="securepass123"
        )

        user_dict = user.to_dict()

        assert "password" not in user_dict
        assert "hashed_password" not in user_dict

    def test_token_contains_no_sensitive_data(self, test_user):
        """Test JWT token doesn't contain sensitive data."""
        token = jwt_handler.create_access_token(
            user_id=test_user.id,
            email=test_user.email,
            roles=test_user.roles
        )

        payload = decode_token_unsafe(token)

        assert "password" not in payload
        assert "hashed_password" not in payload

    def test_case_sensitive_email(self, fresh_user_manager):
        """Test emails are case-insensitive for lookups."""
        user = fresh_user_manager.create_user(
            email="CaseSensitive@Example.com",
            password="pass123"
        )

        # Email should be stored in lowercase
        assert user.email == user.email.lower()

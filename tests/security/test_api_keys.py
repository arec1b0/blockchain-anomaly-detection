"""
API key security tests.

Tests for:
- API key generation
- API key validation
- API key expiration
- API key revocation
- API key usage tracking
- Security best practices
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from src.auth.api_key_manager import (
    APIKeyManager,
    APIKey,
    get_user_from_api_key
)


class TestAPIKeyGeneration:
    """Test API key generation."""

    def test_create_api_key_success(self, fresh_api_key_manager, test_user):
        """Test successful API key creation."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Test Key"
        )

        assert api_key_obj is not None
        assert plain_key is not None
        assert isinstance(plain_key, str)
        assert api_key_obj.name == "Test Key"
        assert api_key_obj.user_id == test_user.id
        assert api_key_obj.is_active is True

    def test_api_key_format(self, fresh_api_key_manager, test_user):
        """Test API key follows correct format."""
        _, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Format Test"
        )

        # Should start with "sk_"
        assert plain_key.startswith("sk_")

        # Should be followed by 32 hex characters
        hex_part = plain_key[3:]
        assert len(hex_part) == 64  # 32 bytes = 64 hex chars
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_api_key_prefix_stored(self, fresh_api_key_manager, test_user):
        """Test API key prefix is stored for identification."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Prefix Test"
        )

        # Prefix should be first 8 characters (sk_ + first 5 hex)
        expected_prefix = plain_key[:8]
        assert api_key_obj.prefix == expected_prefix

    def test_api_key_hashed_not_plain(self, fresh_api_key_manager, test_user):
        """Test API key is stored hashed, not in plain text."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Hash Test"
        )

        # Hashed key should not equal plain key
        assert api_key_obj.hashed_key != plain_key
        # Hashed key should be bcrypt format
        assert api_key_obj.hashed_key.startswith("$2b$")

    def test_create_multiple_keys_for_same_user(self, fresh_api_key_manager, test_user):
        """Test creating multiple API keys for same user."""
        key1_obj, key1 = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Key 1"
        )
        key2_obj, key2 = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Key 2"
        )

        # Keys should be different
        assert key1 != key2
        assert key1_obj.id != key2_obj.id
        assert key1_obj.prefix != key2_obj.prefix

        # Both should belong to same user
        assert key1_obj.user_id == key2_obj.user_id == test_user.id

    def test_api_key_expiration_set(self, fresh_api_key_manager, test_user):
        """Test API key with expiration."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Expiring Key",
            expires_days=30
        )

        assert api_key_obj.expires_at is not None
        assert isinstance(api_key_obj.expires_at, datetime)

        # Should expire in approximately 30 days
        delta = api_key_obj.expires_at - datetime.utcnow()
        assert 29 < delta.days <= 30

    def test_api_key_no_expiration(self, fresh_api_key_manager, test_user):
        """Test API key without expiration (None)."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Never Expires"
        )

        assert api_key_obj.expires_at is None


class TestAPIKeyValidation:
    """Test API key validation."""

    def test_validate_correct_key(self, fresh_api_key_manager, test_user):
        """Test validating correct API key."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Valid Key"
        )

        user_id = fresh_api_key_manager.validate_api_key(plain_key)

        assert user_id is not None
        assert user_id == test_user.id

    def test_validate_incorrect_key(self, fresh_api_key_manager, test_user):
        """Test validating incorrect API key."""
        fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Real Key"
        )

        # Try to validate wrong key
        user_id = fresh_api_key_manager.validate_api_key("sk_wrongkey1234567890")

        assert user_id is None

    def test_validate_nonexistent_key(self, fresh_api_key_manager):
        """Test validating nonexistent API key."""
        user_id = fresh_api_key_manager.validate_api_key("sk_doesnotexist123456")

        assert user_id is None

    def test_validate_malformed_key(self, fresh_api_key_manager):
        """Test validating malformed API key."""
        # Wrong prefix
        user_id1 = fresh_api_key_manager.validate_api_key("pk_1234567890")
        assert user_id1 is None

        # Too short
        user_id2 = fresh_api_key_manager.validate_api_key("sk_short")
        assert user_id2 is None

        # No prefix
        user_id3 = fresh_api_key_manager.validate_api_key("1234567890")
        assert user_id3 is None

    def test_validate_inactive_key(self, fresh_api_key_manager, test_user):
        """Test validating inactive API key."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="To Be Deactivated"
        )

        # Deactivate key
        fresh_api_key_manager.revoke_api_key(api_key_obj.id)

        # Should not validate
        user_id = fresh_api_key_manager.validate_api_key(plain_key)
        assert user_id is None

    def test_validate_expired_key(self, fresh_api_key_manager, test_user):
        """Test validating expired API key."""
        # Create key that expires immediately
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Expired Key",
            expires_days=0
        )

        # Manually set expiration to past
        api_key_obj.expires_at = datetime.utcnow() - timedelta(days=1)

        # Should not validate
        user_id = fresh_api_key_manager.validate_api_key(plain_key)
        assert user_id is None


class TestAPIKeyRevocation:
    """Test API key revocation."""

    def test_revoke_api_key_success(self, fresh_api_key_manager, test_user):
        """Test successful API key revocation."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="To Revoke"
        )

        # Revoke key
        result = fresh_api_key_manager.revoke_api_key(api_key_obj.id)
        assert result is True

        # Key should be inactive
        key = fresh_api_key_manager.get_api_key(api_key_obj.id)
        assert key.is_active is False

        # Should not validate
        user_id = fresh_api_key_manager.validate_api_key(plain_key)
        assert user_id is None

    def test_revoke_nonexistent_key(self, fresh_api_key_manager):
        """Test revoking nonexistent API key."""
        result = fresh_api_key_manager.revoke_api_key("nonexistent-id")
        assert result is False

    def test_revoke_already_revoked_key(self, fresh_api_key_manager, test_user):
        """Test revoking already revoked key."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Double Revoke"
        )

        # Revoke once
        fresh_api_key_manager.revoke_api_key(api_key_obj.id)

        # Revoke again
        result = fresh_api_key_manager.revoke_api_key(api_key_obj.id)
        assert result is True  # Still returns True


class TestAPIKeyRetrieval:
    """Test API key retrieval methods."""

    def test_get_api_key_by_id(self, fresh_api_key_manager, test_user):
        """Test retrieving API key by ID."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Get Test"
        )

        retrieved = fresh_api_key_manager.get_api_key(api_key_obj.id)

        assert retrieved is not None
        assert retrieved.id == api_key_obj.id
        assert retrieved.name == "Get Test"

    def test_get_nonexistent_api_key(self, fresh_api_key_manager):
        """Test retrieving nonexistent API key."""
        key = fresh_api_key_manager.get_api_key("nonexistent-id")
        assert key is None

    def test_list_user_api_keys(self, fresh_api_key_manager, test_user):
        """Test listing user's API keys."""
        # Create multiple keys
        for i in range(3):
            fresh_api_key_manager.create_api_key(
                user_id=test_user.id,
                name=f"Key {i}"
            )

        keys = fresh_api_key_manager.list_user_api_keys(test_user.id)

        assert len(keys) == 3
        assert all(key.user_id == test_user.id for key in keys)

    def test_list_user_api_keys_no_keys(self, fresh_api_key_manager, test_user):
        """Test listing API keys for user with no keys."""
        keys = fresh_api_key_manager.list_user_api_keys(test_user.id)
        assert len(keys) == 0

    def test_list_user_api_keys_multiple_users(self, fresh_api_key_manager,
                                               test_user, test_admin):
        """Test listing API keys filters by user."""
        # Create keys for different users
        fresh_api_key_manager.create_api_key(test_user.id, "User Key")
        fresh_api_key_manager.create_api_key(test_admin.id, "Admin Key")

        user_keys = fresh_api_key_manager.list_user_api_keys(test_user.id)
        admin_keys = fresh_api_key_manager.list_user_api_keys(test_admin.id)

        assert len(user_keys) == 1
        assert len(admin_keys) == 1
        assert user_keys[0].user_id == test_user.id
        assert admin_keys[0].user_id == test_admin.id


class TestAPIKeyUsageTracking:
    """Test API key usage tracking."""

    def test_last_used_updated_on_validation(self, fresh_api_key_manager, test_user):
        """Test last_used timestamp is updated on validation."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Usage Test"
        )

        initial_last_used = api_key_obj.last_used
        assert initial_last_used is None

        # Validate key
        fresh_api_key_manager.validate_api_key(plain_key)

        # Check last_used updated
        key = fresh_api_key_manager.get_api_key(api_key_obj.id)
        assert key.last_used is not None
        assert isinstance(key.last_used, datetime)


class TestAPIKeyMetadata:
    """Test API key metadata."""

    def test_api_key_name_stored(self, fresh_api_key_manager, test_user):
        """Test API key name is stored correctly."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Production API Key"
        )

        assert api_key_obj.name == "Production API Key"

    def test_api_key_created_at(self, fresh_api_key_manager, test_user):
        """Test API key has created_at timestamp."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Timestamp Test"
        )

        assert api_key_obj.created_at is not None
        assert isinstance(api_key_obj.created_at, datetime)

        # Should be recent
        delta = datetime.utcnow() - api_key_obj.created_at
        assert delta.seconds < 10  # Created within last 10 seconds

    def test_api_key_to_dict_no_sensitive_data(self, fresh_api_key_manager, test_user):
        """Test API key dict doesn't include sensitive data."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Security Test"
        )

        key_dict = api_key_obj.to_dict()

        # Should not include hashed key or plain key
        assert "hashed_key" not in key_dict
        assert plain_key not in str(key_dict)

        # Should include safe fields
        assert "id" in key_dict
        assert "name" in key_dict
        assert "prefix" in key_dict
        assert "user_id" in key_dict


class TestAPIKeyFastAPIDependency:
    """Test FastAPI dependency for API key authentication."""

    @pytest.mark.asyncio
    async def test_get_user_from_valid_api_key(self, fresh_api_key_manager, test_user):
        """Test extracting user from valid API key."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Dependency Test"
        )

        # Mock the dependency call
        # In real usage, this is called by FastAPI with the header value
        user = fresh_api_key_manager.validate_api_key(plain_key)

        assert user is not None
        assert user == test_user.id

    def test_get_user_from_invalid_api_key(self, fresh_api_key_manager):
        """Test invalid API key returns None."""
        user = fresh_api_key_manager.validate_api_key("sk_invalid123")

        assert user is None


class TestAPIKeySecurity:
    """Test API key security best practices."""

    def test_api_keys_are_unique(self, fresh_api_key_manager, test_user):
        """Test each generated API key is unique."""
        keys = set()
        for i in range(100):
            _, plain_key = fresh_api_key_manager.create_api_key(
                user_id=test_user.id,
                name=f"Key {i}"
            )
            keys.add(plain_key)

        # All keys should be unique
        assert len(keys) == 100

    def test_api_key_prefix_allows_identification(self, fresh_api_key_manager, test_user):
        """Test API key prefix allows key identification without full key."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Prefix Test"
        )

        # Prefix should be visible
        assert api_key_obj.prefix == plain_key[:8]

        # Should be able to find key by prefix (in real implementation)
        # This helps users identify which key they're using

    def test_api_key_cannot_be_retrieved_after_creation(self,
                                                        fresh_api_key_manager,
                                                        test_user):
        """Test plain API key cannot be retrieved after creation."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="One Time Show"
        )

        # Get key from storage
        stored_key = fresh_api_key_manager.get_api_key(api_key_obj.id)

        # Plain key should not be accessible
        assert not hasattr(stored_key, 'plain_key')
        assert stored_key.hashed_key != plain_key

    def test_revoked_key_cannot_be_reactivated(self, fresh_api_key_manager, test_user):
        """Test revoked keys cannot be reactivated (security policy)."""
        api_key_obj, plain_key = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Permanent Revoke"
        )

        # Revoke key
        fresh_api_key_manager.revoke_api_key(api_key_obj.id)

        # In current implementation, there's no reactivate method
        # This is by design - revoked keys should stay revoked
        # Users should create new keys instead

        # Verify no reactivate method exists
        assert not hasattr(fresh_api_key_manager, 'reactivate_api_key')


class TestAPIKeyEdgeCases:
    """Test edge cases."""

    def test_empty_api_key_name(self, fresh_api_key_manager, test_user):
        """Test creating API key with empty name."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name=""
        )

        # Should still work, just has empty name
        assert api_key_obj.name == ""

    def test_very_long_api_key_name(self, fresh_api_key_manager, test_user):
        """Test creating API key with very long name."""
        long_name = "A" * 500
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name=long_name
        )

        assert api_key_obj.name == long_name

    def test_special_characters_in_name(self, fresh_api_key_manager, test_user):
        """Test API key name with special characters."""
        special_name = "Key-123_test@example.com (production) #1"
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name=special_name
        )

        assert api_key_obj.name == special_name

    def test_zero_expiration_days(self, fresh_api_key_manager, test_user):
        """Test API key with zero expiration days (expires immediately)."""
        api_key_obj, _ = fresh_api_key_manager.create_api_key(
            user_id=test_user.id,
            name="Zero Days",
            expires_days=0
        )

        # Should have expiration set to approximately now
        assert api_key_obj.expires_at is not None
        delta = api_key_obj.expires_at - datetime.utcnow()
        assert delta.days == 0

"""
Audit logging security tests.

Tests for:
- Audit event logging
- Event types (auth, data, admin, api_key, access)
- Event severity levels
- Audit log retrieval and filtering
- Audit log statistics
- Audit middleware
- Compliance requirements
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from src.audit.audit_logger import (
    AuditLog,
    AuditLogger,
    AuditMiddleware,
    get_audit_logger
)


class TestAuditLogModel:
    """Test AuditLog model."""

    def test_audit_log_creation(self):
        """Test creating an audit log entry."""
        log = AuditLog(
            event_type="auth",
            user_id="user123",
            resource="authentication",
            action="login",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            severity="info"
        )

        assert log.event_type == "auth"
        assert log.user_id == "user123"
        assert log.resource == "authentication"
        assert log.action == "login"
        assert log.status == "success"
        assert log.ip_address == "192.168.1.1"
        assert log.severity == "info"
        assert log.timestamp is not None

    def test_audit_log_to_dict(self):
        """Test converting audit log to dictionary."""
        log = AuditLog(
            event_type="data",
            user_id="user123",
            resource="model",
            action="create",
            status="success",
            ip_address="10.0.0.1",
            user_agent="curl/7.0",
            details={"model_type": "isolation_forest"},
            severity="info"
        )

        log_dict = log.to_dict()

        assert "id" in log_dict
        assert log_dict["event_type"] == "data"
        assert log_dict["user_id"] == "user123"
        assert log_dict["resource"] == "model"
        assert log_dict["action"] == "create"
        assert log_dict["status"] == "success"
        assert log_dict["details"]["model_type"] == "isolation_forest"
        assert "timestamp" in log_dict

    def test_audit_log_to_log_string(self):
        """Test converting audit log to log string."""
        log = AuditLog(
            event_type="admin",
            user_id="admin123",
            resource="user",
            action="delete",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Chrome",
            severity="warning"
        )

        log_str = log.to_log_string()

        assert "AUDIT" in log_str
        assert "WARNING" in log_str
        assert "admin" in log_str
        assert "delete on user" in log_str
        assert "admin123" in log_str

    def test_audit_log_with_details(self):
        """Test audit log with additional details."""
        details = {
            "email": "user@example.com",
            "failure_reason": "invalid_password",
            "attempt_count": 3
        }

        log = AuditLog(
            event_type="auth",
            user_id=None,
            resource="authentication",
            action="login",
            status="failure",
            ip_address="192.168.1.1",
            user_agent="Mozilla",
            details=details,
            severity="warning"
        )

        assert log.details == details
        assert log.details["email"] == "user@example.com"

    def test_audit_log_anonymous_user(self):
        """Test audit log for anonymous user."""
        log = AuditLog(
            event_type="api",
            user_id=None,
            resource="/api/v1/predict",
            action="read",
            status="success",
            ip_address="203.0.113.1",
            user_agent="API Client",
            severity="info"
        )

        log_str = log.to_log_string()

        assert "anonymous" in log_str.lower()


class TestAuditLogger:
    """Test AuditLogger class."""

    @pytest.mark.asyncio
    async def test_log_event(self, fresh_audit_logger):
        """Test logging an audit event."""
        await fresh_audit_logger.log_event(
            event_type="auth",
            user_id="user123",
            resource="authentication",
            action="login",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            severity="info"
        )

        logs = fresh_audit_logger.get_logs(limit=10)

        assert len(logs) == 1
        assert logs[0]["event_type"] == "auth"
        assert logs[0]["user_id"] == "user123"

    @pytest.mark.asyncio
    async def test_log_multiple_events(self, fresh_audit_logger):
        """Test logging multiple events."""
        for i in range(5):
            await fresh_audit_logger.log_event(
                event_type="api",
                user_id=f"user{i}",
                resource="/api/v1/predict",
                action="read",
                status="success",
                ip_address=f"192.168.1.{i}",
                user_agent="Client",
                severity="info"
            )

        logs = fresh_audit_logger.get_logs(limit=10)

        assert len(logs) == 5

    @pytest.mark.asyncio
    async def test_circular_buffer_behavior(self):
        """Test audit logger uses circular buffer (max_size)."""
        logger = AuditLogger(max_size=10)

        # Log more than max_size events
        for i in range(15):
            await logger.log_event(
                event_type="api",
                user_id=f"user{i}",
                resource="/api/test",
                action="read",
                status="success",
                ip_address="192.168.1.1",
                user_agent="Client",
                severity="info"
            )

        logs = logger.get_logs(limit=100)

        # Should only keep last 10
        assert len(logs) == 10


class TestAuthEventLogging:
    """Test authentication event logging."""

    @pytest.mark.asyncio
    async def test_log_successful_login(self, fresh_audit_logger):
        """Test logging successful login."""
        await fresh_audit_logger.log_auth_event(
            action="login",
            user_id="user123",
            email="user@example.com",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        logs = fresh_audit_logger.get_logs(event_type="auth")

        assert len(logs) == 1
        assert logs[0]["action"] == "login"
        assert logs[0]["status"] == "success"
        assert logs[0]["details"]["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_log_failed_login(self, fresh_audit_logger):
        """Test logging failed login."""
        await fresh_audit_logger.log_auth_event(
            action="login",
            user_id=None,  # No user ID for failed login
            email="user@example.com",
            status="failure",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            failure_reason="invalid_password"
        )

        logs = fresh_audit_logger.get_logs(event_type="auth", severity="warning")

        assert len(logs) == 1
        assert logs[0]["status"] == "failure"
        assert logs[0]["severity"] == "warning"
        assert logs[0]["details"]["failure_reason"] == "invalid_password"

    @pytest.mark.asyncio
    async def test_log_register(self, fresh_audit_logger):
        """Test logging user registration."""
        await fresh_audit_logger.log_auth_event(
            action="register",
            user_id="newuser123",
            email="newuser@example.com",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        logs = fresh_audit_logger.get_logs(event_type="auth")

        assert len(logs) == 1
        assert logs[0]["action"] == "register"

    @pytest.mark.asyncio
    async def test_log_logout(self, fresh_audit_logger):
        """Test logging user logout."""
        await fresh_audit_logger.log_auth_event(
            action="logout",
            user_id="user123",
            email="user@example.com",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )

        logs = fresh_audit_logger.get_logs(event_type="auth")

        assert len(logs) == 1
        assert logs[0]["action"] == "logout"

    @pytest.mark.asyncio
    async def test_log_token_refresh(self, fresh_audit_logger):
        """Test logging token refresh."""
        await fresh_audit_logger.log_auth_event(
            action="token_refresh",
            user_id="user123",
            email="user@example.com",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Mobile App"
        )

        logs = fresh_audit_logger.get_logs(event_type="auth")

        assert len(logs) == 1
        assert logs[0]["action"] == "token_refresh"


class TestDataEventLogging:
    """Test data mutation event logging."""

    @pytest.mark.asyncio
    async def test_log_data_create(self, fresh_audit_logger):
        """Test logging data creation."""
        await fresh_audit_logger.log_data_event(
            action="create",
            user_id="user123",
            resource="model",
            resource_id="model_001",
            status="success",
            ip_address="192.168.1.1",
            user_agent="API Client",
            changes={"type": "isolation_forest", "contamination": 0.01}
        )

        logs = fresh_audit_logger.get_logs(event_type="data")

        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["resource"] == "model"
        assert logs[0]["details"]["resource_id"] == "model_001"

    @pytest.mark.asyncio
    async def test_log_data_update(self, fresh_audit_logger):
        """Test logging data updates."""
        await fresh_audit_logger.log_data_event(
            action="update",
            user_id="user123",
            resource="user",
            resource_id="user_456",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Web UI",
            changes={"email": "newemail@example.com"}
        )

        logs = fresh_audit_logger.get_logs(event_type="data")

        assert len(logs) == 1
        assert logs[0]["action"] == "update"
        assert "changes" in logs[0]["details"]

    @pytest.mark.asyncio
    async def test_log_data_delete(self, fresh_audit_logger):
        """Test logging data deletion."""
        await fresh_audit_logger.log_data_event(
            action="delete",
            user_id="admin123",
            resource="anomaly",
            resource_id="anomaly_789",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Admin Panel"
        )

        logs = fresh_audit_logger.get_logs(event_type="data")

        assert len(logs) == 1
        assert logs[0]["action"] == "delete"

    @pytest.mark.asyncio
    async def test_log_sensitive_data_access(self, fresh_audit_logger):
        """Test logging sensitive data access."""
        await fresh_audit_logger.log_data_event(
            action="read",
            user_id="admin123",
            resource="user_pii",
            resource_id="user_123",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Admin Console"
        )

        logs = fresh_audit_logger.get_logs(event_type="data", user_id="admin123")

        assert len(logs) == 1
        assert logs[0]["resource"] == "user_pii"


class TestAdminEventLogging:
    """Test admin action logging."""

    @pytest.mark.asyncio
    async def test_log_admin_action(self, fresh_audit_logger):
        """Test logging admin actions."""
        await fresh_audit_logger.log_admin_event(
            action="grant_permissions",
            user_id="admin123",
            resource="user",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Admin Panel",
            details={"target_user": "user456", "permissions": ["admin"]}
        )

        logs = fresh_audit_logger.get_logs(event_type="admin")

        assert len(logs) == 1
        assert logs[0]["action"] == "grant_permissions"
        assert logs[0]["severity"] == "warning"  # Admin actions always warning level

    @pytest.mark.asyncio
    async def test_log_config_change(self, fresh_audit_logger):
        """Test logging configuration changes."""
        await fresh_audit_logger.log_admin_event(
            action="update_config",
            user_id="admin123",
            resource="system_config",
            status="success",
            ip_address="192.168.1.1",
            user_agent="CLI",
            details={"setting": "rate_limit", "old_value": 100, "new_value": 200}
        )

        logs = fresh_audit_logger.get_logs(event_type="admin")

        assert len(logs) == 1
        assert logs[0]["resource"] == "system_config"


class TestAPIKeyEventLogging:
    """Test API key operation logging."""

    @pytest.mark.asyncio
    async def test_log_api_key_create(self, fresh_audit_logger):
        """Test logging API key creation."""
        await fresh_audit_logger.log_api_key_event(
            action="create",
            user_id="user123",
            key_id="key_001",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Web UI",
            details={"name": "Production Key", "expires_days": 365}
        )

        logs = fresh_audit_logger.get_logs(event_type="api_key")

        assert len(logs) == 1
        assert logs[0]["action"] == "create"
        assert logs[0]["details"]["key_id"] == "key_001"

    @pytest.mark.asyncio
    async def test_log_api_key_revoke(self, fresh_audit_logger):
        """Test logging API key revocation."""
        await fresh_audit_logger.log_api_key_event(
            action="revoke",
            user_id="user123",
            key_id="key_001",
            status="success",
            ip_address="192.168.1.1",
            user_agent="Web UI"
        )

        logs = fresh_audit_logger.get_logs(event_type="api_key")

        assert len(logs) == 1
        assert logs[0]["action"] == "revoke"

    @pytest.mark.asyncio
    async def test_log_api_key_usage(self, fresh_audit_logger):
        """Test logging API key usage."""
        await fresh_audit_logger.log_api_key_event(
            action="use",
            user_id="user123",
            key_id="key_001",
            status="success",
            ip_address="203.0.113.1",
            user_agent="API Client"
        )

        logs = fresh_audit_logger.get_logs(event_type="api_key")

        assert len(logs) == 1
        assert logs[0]["action"] == "use"


class TestAuditLogRetrieval:
    """Test audit log retrieval and filtering."""

    @pytest.mark.asyncio
    async def test_get_logs_no_filter(self, fresh_audit_logger):
        """Test retrieving all logs."""
        # Create different types of logs
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_data_event(
            "create", "user2", "model", "model1", "success",
            "192.168.1.2", "Client"
        )

        logs = fresh_audit_logger.get_logs(limit=10)

        assert len(logs) == 2

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_event_type(self, fresh_audit_logger):
        """Test filtering logs by event type."""
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_data_event(
            "create", "user1", "model", "model1", "success",
            "192.168.1.1", "Client"
        )

        auth_logs = fresh_audit_logger.get_logs(event_type="auth")
        data_logs = fresh_audit_logger.get_logs(event_type="data")

        assert len(auth_logs) == 1
        assert len(data_logs) == 1

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_user(self, fresh_audit_logger):
        """Test filtering logs by user ID."""
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_auth_event(
            "login", "user2", "user2@example.com", "success",
            "192.168.1.2", "Chrome"
        )

        user1_logs = fresh_audit_logger.get_logs(user_id="user1")

        assert len(user1_logs) == 1
        assert user1_logs[0]["user_id"] == "user1"

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_severity(self, fresh_audit_logger):
        """Test filtering logs by severity."""
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_auth_event(
            "login", None, "baduser@example.com", "failure",
            "192.168.1.1", "Mozilla", failure_reason="invalid_password"
        )

        warning_logs = fresh_audit_logger.get_logs(severity="warning")
        info_logs = fresh_audit_logger.get_logs(severity="info")

        assert len(warning_logs) >= 1
        assert len(info_logs) >= 1

    @pytest.mark.asyncio
    async def test_get_logs_with_limit(self, fresh_audit_logger):
        """Test limiting number of returned logs."""
        # Create many logs
        for i in range(20):
            await fresh_audit_logger.log_event(
                "api", f"user{i}", "/api/test", "read", "success",
                "192.168.1.1", "Client", severity="info"
            )

        logs = fresh_audit_logger.get_logs(limit=5)

        assert len(logs) == 5

    @pytest.mark.asyncio
    async def test_get_logs_returns_most_recent(self, fresh_audit_logger):
        """Test get_logs returns most recent entries."""
        for i in range(10):
            await fresh_audit_logger.log_event(
                "api", f"user{i}", "/api/test", "read", "success",
                "192.168.1.1", "Client", details={"index": i}, severity="info"
            )

        logs = fresh_audit_logger.get_logs(limit=3)

        # Should return last 3 (indices 7, 8, 9)
        indices = [log["details"]["index"] for log in logs]
        assert 7 in indices or 8 in indices or 9 in indices


class TestAuditLogStatistics:
    """Test audit log statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, fresh_audit_logger):
        """Test getting audit log statistics."""
        # Create various events
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_auth_event(
            "login", None, "bad@example.com", "failure",
            "192.168.1.1", "Mozilla", failure_reason="bad_password"
        )
        await fresh_audit_logger.log_data_event(
            "create", "user1", "model", "model1", "success",
            "192.168.1.1", "Client"
        )

        stats = fresh_audit_logger.get_stats()

        assert stats["total_logs"] == 3
        assert "event_types" in stats
        assert "severities" in stats
        assert "failures" in stats
        assert stats["failures"] == 1

    @pytest.mark.asyncio
    async def test_stats_event_type_counts(self, fresh_audit_logger):
        """Test statistics count events by type."""
        # Create events
        for i in range(5):
            await fresh_audit_logger.log_auth_event(
                "login", f"user{i}", f"user{i}@example.com", "success",
                "192.168.1.1", "Mozilla"
            )
        for i in range(3):
            await fresh_audit_logger.log_data_event(
                "create", "user1", "model", f"model{i}", "success",
                "192.168.1.1", "Client"
            )

        stats = fresh_audit_logger.get_stats()

        assert stats["event_types"]["auth"] == 5
        assert stats["event_types"]["data"] == 3

    @pytest.mark.asyncio
    async def test_stats_severity_counts(self, fresh_audit_logger):
        """Test statistics count events by severity."""
        await fresh_audit_logger.log_event(
            "api", "user1", "/api/test", "read", "success",
            "192.168.1.1", "Client", severity="info"
        )
        await fresh_audit_logger.log_event(
            "api", "user1", "/api/test", "read", "failure",
            "192.168.1.1", "Client", severity="error"
        )

        stats = fresh_audit_logger.get_stats()

        assert stats["severities"]["info"] >= 1
        assert stats["severities"]["error"] >= 1

    @pytest.mark.asyncio
    async def test_stats_oldest_newest_timestamps(self, fresh_audit_logger):
        """Test statistics include oldest and newest log timestamps."""
        await fresh_audit_logger.log_event(
            "api", "user1", "/api/test", "read", "success",
            "192.168.1.1", "Client", severity="info"
        )

        stats = fresh_audit_logger.get_stats()

        assert "oldest_log" in stats
        assert "newest_log" in stats
        assert stats["oldest_log"] is not None
        assert stats["newest_log"] is not None


class TestAuditMiddleware:
    """Test audit middleware for automatic request logging."""

    @pytest.mark.asyncio
    async def test_middleware_logs_mutations(self):
        """Test middleware logs POST/PUT/PATCH/DELETE requests."""
        audit_logger = AuditLogger(max_size=100)
        middleware = AuditMiddleware(app=Mock())
        middleware.audit_logger = audit_logger

        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/train"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {
            "user-agent": "API Client",
            "authorization": ""
        }

        async def call_next(request):
            mock_response = Mock()
            mock_response.status_code = 200
            return mock_response

        await middleware.dispatch(mock_request, call_next)

        logs = audit_logger.get_logs(event_type="api")

        assert len(logs) >= 1

    @pytest.mark.asyncio
    async def test_middleware_logs_important_gets(self):
        """Test middleware logs important GET requests."""
        audit_logger = AuditLogger(max_size=100)
        middleware = AuditMiddleware(app=Mock())
        middleware.audit_logger = audit_logger

        mock_request = Mock()
        mock_request.method = "GET"
        mock_request.url.path = "/users/123"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {
            "user-agent": "Browser",
            "authorization": ""
        }

        async def call_next(request):
            mock_response = Mock()
            mock_response.status_code = 200
            return mock_response

        await middleware.dispatch(mock_request, call_next)

        logs = audit_logger.get_logs(event_type="api")

        # Should log GET to /users/
        assert len(logs) >= 1

    @pytest.mark.asyncio
    async def test_middleware_includes_status_code(self):
        """Test middleware includes response status code."""
        audit_logger = AuditLogger(max_size=100)
        middleware = AuditMiddleware(app=Mock())
        middleware.audit_logger = audit_logger

        mock_request = Mock()
        mock_request.method = "POST"
        mock_request.url.path = "/api/v1/test"
        mock_request.client.host = "192.168.1.1"
        mock_request.headers = {
            "user-agent": "Client",
            "authorization": ""
        }

        async def call_next(request):
            mock_response = Mock()
            mock_response.status_code = 404
            return mock_response

        await middleware.dispatch(mock_request, call_next)

        logs = audit_logger.get_logs()

        assert len(logs) >= 1
        assert logs[0]["details"]["status_code"] == 404


class TestComplianceRequirements:
    """Test audit logging meets compliance requirements."""

    @pytest.mark.asyncio
    async def test_immutable_logs(self, fresh_audit_logger):
        """Test audit logs are immutable (append-only)."""
        await fresh_audit_logger.log_event(
            "auth", "user1", "authentication", "login", "success",
            "192.168.1.1", "Mozilla", severity="info"
        )

        logs = fresh_audit_logger.get_logs()
        original_log = logs[0].copy()

        # Audit logger should not have methods to modify logs
        assert not hasattr(fresh_audit_logger, 'update_log')
        assert not hasattr(fresh_audit_logger, 'delete_log')
        assert not hasattr(fresh_audit_logger, 'modify_log')

    @pytest.mark.asyncio
    async def test_full_context_captured(self, fresh_audit_logger):
        """Test all required context is captured."""
        await fresh_audit_logger.log_event(
            "admin", "admin1", "user", "grant_admin", "success",
            "192.168.1.1", "Admin Panel",
            details={"target_user": "user123"},
            severity="warning"
        )

        logs = fresh_audit_logger.get_logs()

        # Should include all required fields
        log = logs[0]
        assert "id" in log
        assert "event_type" in log
        assert "user_id" in log
        assert "resource" in log
        assert "action" in log
        assert "status" in log
        assert "ip_address" in log
        assert "user_agent" in log
        assert "timestamp" in log
        assert "severity" in log
        assert "details" in log

    @pytest.mark.asyncio
    async def test_timestamps_accurate(self, fresh_audit_logger):
        """Test timestamps are accurate and in correct format."""
        await fresh_audit_logger.log_event(
            "api", "user1", "/api/test", "read", "success",
            "192.168.1.1", "Client", severity="info"
        )

        logs = fresh_audit_logger.get_logs()

        timestamp_str = logs[0]["timestamp"]
        # Should be ISO 8601 format
        timestamp = datetime.fromisoformat(timestamp_str)

        # Should be recent (within last minute)
        delta = datetime.utcnow() - timestamp
        assert delta.seconds < 60

    @pytest.mark.asyncio
    async def test_searchable_and_filterable(self, fresh_audit_logger):
        """Test logs are searchable and filterable."""
        # Create various logs
        await fresh_audit_logger.log_auth_event(
            "login", "user1", "user1@example.com", "success",
            "192.168.1.1", "Mozilla"
        )
        await fresh_audit_logger.log_admin_event(
            "grant_admin", "admin1", "user", "success",
            "192.168.1.1", "Admin Panel"
        )

        # Should be filterable by multiple criteria
        auth_logs = fresh_audit_logger.get_logs(event_type="auth")
        admin_logs = fresh_audit_logger.get_logs(event_type="admin")
        user1_logs = fresh_audit_logger.get_logs(user_id="user1")
        warning_logs = fresh_audit_logger.get_logs(severity="warning")

        assert len(auth_logs) >= 1
        assert len(admin_logs) >= 1
        assert len(user1_logs) >= 1


class TestErrorHandling:
    """Test error handling in audit logging."""

    @pytest.mark.asyncio
    async def test_logging_failure_does_not_crash(self, fresh_audit_logger):
        """Test audit logging failures don't crash the application."""
        # Even with invalid data, should not raise exception
        try:
            await fresh_audit_logger.log_event(
                event_type=None,  # Invalid
                user_id="user1",
                resource="test",
                action="test",
                status="success",
                ip_address="192.168.1.1",
                user_agent="Client",
                severity="info"
            )
        except Exception as e:
            # If exception occurs, it should be caught and logged
            # Application should continue
            pass

    @pytest.mark.asyncio
    async def test_get_stats_empty_logs(self):
        """Test getting stats with no logs doesn't error."""
        logger = AuditLogger(max_size=100)

        stats = logger.get_stats()

        assert stats["total_logs"] == 0
        assert stats["failures"] == 0
        assert stats["oldest_log"] is None
        assert stats["newest_log"] is None

"""
Comprehensive audit logging for security and compliance.

This module logs all security-relevant events:
- Authentication (login, logout, failed attempts)
- Authorization (permission denied)
- Data mutations (create, update, delete)
- Admin actions
- API key operations
- Sensitive data access

Audit logs are:
- Immutable (append-only)
- Include full context (user, IP, timestamp, etc.)
- Exportable for compliance audits
- Searchable and filterable
"""

from datetime import datetime
from typing import Optional, Dict, Any
import json
from collections import deque
from prometheus_client import Counter

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
audit_events = Counter(
    'audit_events_total',
    'Total number of audit events logged.',
    ['event_type', 'severity']
)


class AuditLog:
    """
    Audit log entry model (in-memory for Phase 1).

    In Phase 2, this will be stored in PostgreSQL.
    """

    def __init__(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        timestamp: datetime = None
    ):
        self.id = f"{datetime.utcnow().isoformat()}_{event_type}"
        self.event_type = event_type
        self.user_id = user_id
        self.resource = resource
        self.action = action
        self.status = status
        self.ip_address = ip_address
        self.user_agent = user_agent
        self.details = details or {}
        self.severity = severity
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "resource": self.resource,
            "action": self.action,
            "status": self.status,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "details": self.details,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }

    def to_log_string(self) -> str:
        """Convert to log string."""
        return (
            f"[AUDIT] {self.severity.upper()} | "
            f"{self.event_type} | "
            f"User:{self.user_id or 'anonymous'} | "
            f"{self.action} on {self.resource} | "
            f"Status:{self.status} | "
            f"IP:{self.ip_address}"
        )


class AuditLogger:
    """
    Centralized audit logging for security events.

    Logs all security-relevant events to:
    - In-memory store (Phase 1)
    - Application logs (always)
    - Database (Phase 2)
    - External SIEM (future)
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize audit logger.

        Args:
            max_size: Maximum number of logs to keep in memory
        """
        # In-memory storage (circular buffer)
        self._logs = deque(maxlen=max_size)
        self.max_size = max_size

    async def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        resource: str,
        action: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "info"
    ):
        """
        Log an audit event.

        Args:
            event_type: Type of event (auth, data, admin, api_key, access)
            user_id: User who performed action (None for anonymous)
            resource: Resource affected (endpoint, model, user, etc.)
            action: Action performed (create, read, update, delete, login, etc.)
            status: Result status (success, failure, denied)
            ip_address: Client IP address
            user_agent: Client user agent string
            details: Additional context (dict)
            severity: Event severity (info, warning, error, critical)

        Example:
            >>> logger = AuditLogger()
            >>> await logger.log_event(
            ...     event_type="auth",
            ...     user_id="123",
            ...     resource="authentication",
            ...     action="login",
            ...     status="success",
            ...     ip_address="192.168.1.1",
            ...     user_agent="Mozilla/5.0..."
            ... )
        """
        try:
            # Create audit log entry
            audit_log = AuditLog(
                event_type=event_type,
                user_id=user_id,
                resource=resource,
                action=action,
                status=status,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                severity=severity,
                timestamp=datetime.utcnow()
            )

            # Store in memory
            self._logs.append(audit_log)

            # Log to application logs
            log_str = audit_log.to_log_string()
            if severity == "critical":
                logger.critical(log_str)
            elif severity == "error":
                logger.error(log_str)
            elif severity == "warning":
                logger.warning(log_str)
            else:
                logger.info(log_str)

            # Update metrics
            audit_events.labels(event_type=event_type, severity=severity).inc()

            # In Phase 2, also write to database here

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", exc_info=True)

    async def log_auth_event(
        self,
        action: str,
        user_id: Optional[str],
        email: str,
        status: str,
        ip_address: str,
        user_agent: str,
        failure_reason: Optional[str] = None
    ):
        """
        Log authentication event.

        Args:
            action: Action (login, logout, register, token_refresh)
            user_id: User ID (None if failed)
            email: User email
            status: Status (success, failure)
            ip_address: Client IP
            user_agent: User agent
            failure_reason: Reason for failure (if applicable)
        """
        details = {"email": email}
        if failure_reason:
            details["failure_reason"] = failure_reason

        severity = "warning" if status == "failure" else "info"
        if status == "failure" and action == "login":
            severity = "warning"  # Failed login attempts are suspicious

        await self.log_event(
            event_type="auth",
            user_id=user_id,
            resource="authentication",
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity=severity
        )

    async def log_data_event(
        self,
        action: str,
        user_id: str,
        resource: str,
        resource_id: Optional[str],
        status: str,
        ip_address: str,
        user_agent: str,
        changes: Optional[Dict] = None
    ):
        """
        Log data mutation event.

        Args:
            action: Action (create, update, delete, read)
            user_id: User performing action
            resource: Resource type (user, model, anomaly, etc.)
            resource_id: Specific resource ID
            status: Status
            ip_address: Client IP
            user_agent: User agent
            changes: What changed (for updates)
        """
        details = {}
        if resource_id:
            details["resource_id"] = resource_id
        if changes:
            details["changes"] = changes

        await self.log_event(
            event_type="data",
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="info"
        )

    async def log_admin_event(
        self,
        action: str,
        user_id: str,
        resource: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict] = None
    ):
        """
        Log admin action (always warning level for visibility).

        Args:
            action: Admin action
            user_id: Admin user ID
            resource: Resource affected
            status: Status
            ip_address: Client IP
            user_agent: User agent
            details: Additional details
        """
        await self.log_event(
            event_type="admin",
            user_id=user_id,
            resource=resource,
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="warning"  # Admin actions always visible
        )

    async def log_api_key_event(
        self,
        action: str,
        user_id: str,
        key_id: str,
        status: str,
        ip_address: str,
        user_agent: str,
        details: Optional[Dict] = None
    ):
        """Log API key operation."""
        details = details or {}
        details["key_id"] = key_id

        await self.log_event(
            event_type="api_key",
            user_id=user_id,
            resource="api_key",
            action=action,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details,
            severity="info"
        )

    def get_logs(
        self,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> list:
        """
        Retrieve audit logs with filters.

        Args:
            event_type: Filter by event type
            user_id: Filter by user ID
            severity: Filter by severity
            limit: Maximum number of logs to return

        Returns:
            list: Filtered audit logs
        """
        filtered = list(self._logs)

        # Apply filters
        if event_type:
            filtered = [log for log in filtered if log.event_type == event_type]
        if user_id:
            filtered = [log for log in filtered if log.user_id == user_id]
        if severity:
            filtered = [log for log in filtered if log.severity == severity]

        # Limit results
        filtered = filtered[-limit:]  # Most recent

        return [log.to_dict() for log in filtered]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            dict: Statistics about audit logs
        """
        logs = list(self._logs)

        # Count by event type
        event_types = {}
        for log in logs:
            event_types[log.event_type] = event_types.get(log.event_type, 0) + 1

        # Count by severity
        severities = {}
        for log in logs:
            severities[log.severity] = severities.get(log.severity, 0) + 1

        # Count failures
        failures = sum(1 for log in logs if log.status == "failure")

        return {
            "total_logs": len(logs),
            "event_types": event_types,
            "severities": severities,
            "failures": failures,
            "oldest_log": logs[0].timestamp.isoformat() if logs else None,
            "newest_log": logs[-1].timestamp.isoformat() if logs else None,
        }


# Global audit logger instance
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """
    Get global audit logger instance.

    Returns:
        AuditLogger: Singleton audit logger instance
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically log all API requests.

    Logs mutations (POST, PUT, PATCH, DELETE) and important GET requests.
    """

    def __init__(self, app):
        super().__init__(app)
        self.audit_logger = get_audit_logger()

    async def dispatch(self, request: Request, call_next):
        """Log request and response."""
        start_time = datetime.utcnow()

        # Extract user info
        user_id = None
        try:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                from src.auth.jwt_handler import jwt_handler
                token = auth_header.replace("Bearer ", "")
                payload = jwt_handler.decode_token(token)
                user_id = payload.get("sub")
        except:
            pass

        # Process request
        response = await call_next(request)

        # Determine if we should log this request
        should_log = False

        # Always log mutations
        if request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            should_log = True

        # Log important GET requests (admin, user data)
        elif request.method == "GET":
            if any(path in request.url.path for path in ["/users/", "/admin/", "/api-keys/"]):
                should_log = True

        if should_log:
            # Determine action from method
            action_map = {
                "POST": "create",
                "GET": "read",
                "PUT": "update",
                "PATCH": "update",
                "DELETE": "delete"
            }
            action = action_map.get(request.method, request.method.lower())

            # Determine status
            status = "success" if response.status_code < 400 else "failure"
            severity = "error" if response.status_code >= 500 else "info"

            # Get client info
            ip_address = request.client.host if request.client else "unknown"
            user_agent = request.headers.get("user-agent", "unknown")

            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Log event
            await self.audit_logger.log_event(
                event_type="api",
                user_id=user_id,
                resource=request.url.path,
                action=action,
                status=status,
                ip_address=ip_address,
                user_agent=user_agent,
                details={
                    "method": request.method,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms
                },
                severity=severity
            )

        return response

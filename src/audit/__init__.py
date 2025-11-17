"""
Audit logging module for security and compliance.

This module provides:
- Comprehensive audit logging for all security events
- Authentication event tracking
- Data mutation tracking
- Admin action logging
- Audit middleware for automatic logging
"""

from src.audit.audit_logger import AuditLogger, get_audit_logger, AuditMiddleware

__all__ = [
    'AuditLogger',
    'get_audit_logger',
    'AuditMiddleware',
]

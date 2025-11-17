"""
Repository for audit log data access.

This module provides methods for querying audit logs.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from src.database.models import AuditLog
from src.database.repositories.base_repository import BaseRepository


class AuditRepository(BaseRepository[AuditLog]):
    """Repository for audit logs."""

    def __init__(self, db: Session):
        super().__init__(AuditLog, db)

    def get_by_event_type(
        self,
        event_type: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by event type."""
        return self.db.query(AuditLog)\
            .filter(AuditLog.event_type == event_type)\
            .order_by(AuditLog.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_user(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs for a user."""
        return self.db.query(AuditLog)\
            .filter(AuditLog.user_id == user_id)\
            .order_by(AuditLog.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_severity(
        self,
        severity: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by severity."""
        return self.db.query(AuditLog)\
            .filter(AuditLog.severity == severity)\
            .order_by(AuditLog.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 1000
    ) -> List[AuditLog]:
        """Get audit logs in date range."""
        return self.db.query(AuditLog)\
            .filter(and_(
                AuditLog.timestamp >= start_date,
                AuditLog.timestamp <= end_date
            ))\
            .order_by(AuditLog.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()


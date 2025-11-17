"""
Repository for anomaly data access.

This module provides methods for querying and managing detected anomalies.
"""

from typing import List, Optional
from datetime import datetime, timedelta
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session, joinedload

from src.database.models import Anomaly, SeverityEnum, Transaction
from src.database.repositories.base_repository import BaseRepository


class AnomalyRepository(BaseRepository[Anomaly]):
    """Repository for detected anomalies."""

    def __init__(self, db: Session):
        super().__init__(Anomaly, db)

    def get_with_transaction(self, id: str) -> Optional[Anomaly]:
        """Get anomaly with related transaction data."""
        return self.db.query(Anomaly)\
            .options(joinedload(Anomaly.transaction))\
            .filter(Anomaly.id == id)\
            .first()

    def get_by_severity(
        self,
        severity: SeverityEnum,
        skip: int = 0,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies by severity level."""
        return self.db.query(Anomaly)\
            .filter(Anomaly.severity == severity)\
            .order_by(Anomaly.detected_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_unreviewed(
        self,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies awaiting review."""
        return self.db.query(Anomaly)\
            .filter(Anomaly.reviewed == False)\
            .order_by(
                Anomaly.severity.desc(),  # Critical first
                Anomaly.detected_at.asc()  # Oldest first
            )\
            .limit(limit)\
            .all()

    def get_recent(
        self,
        hours: int = 24,
        limit: int = 100
    ) -> List[Anomaly]:
        """Get anomalies from last N hours."""
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.db.query(Anomaly)\
            .filter(Anomaly.detected_at >= since)\
            .order_by(Anomaly.detected_at.desc())\
            .limit(limit)\
            .all()

    def mark_reviewed(
        self,
        id: str,
        reviewer_id: str,
        is_false_positive: bool
    ) -> Optional[Anomaly]:
        """Mark anomaly as reviewed."""
        anomaly = self.get(id)
        if anomaly:
            anomaly.reviewed = True
            anomaly.reviewed_by = reviewer_id
            anomaly.reviewed_at = datetime.utcnow()
            anomaly.false_positive = is_false_positive
            self.db.commit()
            self.db.refresh(anomaly)
        return anomaly

    def get_false_positive_rate(
        self,
        start_date: Optional[datetime] = None
    ) -> float:
        """Calculate false positive rate."""
        query = self.db.query(Anomaly).filter(Anomaly.reviewed == True)
        if start_date:
            query = query.filter(Anomaly.detected_at >= start_date)

        total = query.count()
        if total == 0:
            return 0.0

        false_positives = query.filter(Anomaly.false_positive == True).count()
        return false_positives / total


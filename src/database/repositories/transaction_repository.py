"""
Repository for transaction data access.

This module provides methods for querying and managing blockchain transactions.
"""

from typing import List, Optional
from datetime import datetime
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from src.database.models import Transaction, Anomaly
from src.database.repositories.base_repository import BaseRepository


class TransactionRepository(BaseRepository[Transaction]):
    """Repository for blockchain transactions."""

    def __init__(self, db: Session):
        super().__init__(Transaction, db)

    def get_by_hash(self, hash: str) -> Optional[Transaction]:
        """Get transaction by hash."""
        return self.db.query(Transaction).filter(Transaction.hash == hash).first()

    def get_by_address(
        self,
        address: str,
        is_sender: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[Transaction]:
        """Get transactions for an address (as sender or receiver)."""
        if is_sender:
            filter_col = Transaction.from_address
        else:
            filter_col = Transaction.to_address

        return self.db.query(Transaction)\
            .filter(filter_col == address)\
            .order_by(Transaction.timestamp.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        skip: int = 0,
        limit: int = 1000
    ) -> List[Transaction]:
        """Get transactions in date range."""
        return self.db.query(Transaction)\
            .filter(and_(
                Transaction.timestamp >= start_date,
                Transaction.timestamp <= end_date
            ))\
            .order_by(Transaction.timestamp.asc())\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_high_value_transactions(
        self,
        min_value: float,
        limit: int = 100
    ) -> List[Transaction]:
        """Get high-value transactions."""
        return self.db.query(Transaction)\
            .filter(Transaction.value >= min_value)\
            .order_by(Transaction.value.desc())\
            .limit(limit)\
            .all()

    def get_statistics(self, start_date: Optional[datetime] = None):
        """Get transaction statistics."""
        query = self.db.query(
            func.count(Transaction.id).label('total'),
            func.sum(Transaction.value).label('total_value'),
            func.avg(Transaction.value).label('avg_value'),
            func.max(Transaction.value).label('max_value'),
            func.min(Transaction.value).label('min_value'),
            func.avg(Transaction.gas_price).label('avg_gas_price')
        )
        if start_date:
            query = query.filter(Transaction.timestamp >= start_date)
        result = query.first()
        return {
            'total': result.total if result else 0,
            'total_value': float(result.total_value) if result and result.total_value else 0.0,
            'avg_value': float(result.avg_value) if result and result.avg_value else 0.0,
            'max_value': float(result.max_value) if result and result.max_value else 0.0,
            'min_value': float(result.min_value) if result and result.min_value else 0.0,
            'avg_gas_price': float(result.avg_gas_price) if result and result.avg_gas_price else 0.0
        }

    def bulk_insert(self, transactions: List[dict]) -> int:
        """
        Bulk insert transactions (efficient for large datasets).

        Returns:
            Number of records inserted
        """
        try:
            self.db.bulk_insert_mappings(Transaction, transactions)
            self.db.commit()
            return len(transactions)
        except Exception as e:
            self.db.rollback()
            raise e


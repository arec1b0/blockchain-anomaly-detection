"""
Repository pattern for database access.

This module provides repositories for accessing database entities
with a clean separation of concerns.
"""

from src.database.repositories.base_repository import BaseRepository
from src.database.repositories.transaction_repository import TransactionRepository
from src.database.repositories.anomaly_repository import AnomalyRepository
from src.database.repositories.model_repository import ModelRepository
from src.database.repositories.user_repository import UserRepository
from src.database.repositories.audit_repository import AuditRepository

__all__ = [
    "BaseRepository",
    "TransactionRepository",
    "AnomalyRepository",
    "ModelRepository",
    "UserRepository",
    "AuditRepository"
]


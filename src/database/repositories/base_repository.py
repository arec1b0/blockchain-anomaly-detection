"""
Base repository with common CRUD operations.

This module provides a base repository class with common database
operations that can be extended by specific repositories.
"""

from typing import TypeVar, Generic, Type, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.models import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository for common database operations."""

    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db

    def get(self, id: str) -> Optional[ModelType]:
        """Get single record by ID."""
        return self.db.query(self.model).filter(self.model.id == id).first()

    def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: str = "created_at"
    ) -> List[ModelType]:
        """Get all records with pagination."""
        query = self.db.query(self.model)
        if hasattr(self.model, order_by):
            query = query.order_by(desc(getattr(self.model, order_by)))
        return query.offset(skip).limit(limit).all()

    def create(self, obj: ModelType) -> ModelType:
        """Create new record."""
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def update(self, id: str, updates: dict) -> Optional[ModelType]:
        """Update record by ID."""
        obj = self.get(id)
        if obj:
            for key, value in updates.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            self.db.commit()
            self.db.refresh(obj)
        return obj

    def delete(self, id: str) -> bool:
        """Delete record by ID."""
        obj = self.get(id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
            return True
        return False

    def count(self) -> int:
        """Count total records."""
        return self.db.query(self.model).count()


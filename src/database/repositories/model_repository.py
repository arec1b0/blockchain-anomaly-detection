"""
Repository for model data access.

This module provides methods for querying and managing ML models and versions.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from src.database.models import Model, ModelVersion
from src.database.repositories.base_repository import BaseRepository


class ModelRepository(BaseRepository[Model]):
    """Repository for ML models."""

    def __init__(self, db: Session):
        super().__init__(Model, db)

    def get_active(self) -> List[Model]:
        """Get all active models."""
        return self.db.query(Model).filter(Model.is_active == True).all()

    def get_by_type(self, model_type: str) -> List[Model]:
        """Get models by type."""
        return self.db.query(Model).filter(Model.model_type == model_type).all()

    def get_by_name(self, name: str) -> Optional[Model]:
        """Get model by name."""
        return self.db.query(Model).filter(Model.name == name).first()


class ModelVersionRepository(BaseRepository[ModelVersion]):
    """Repository for model versions."""

    def __init__(self, db: Session):
        super().__init__(ModelVersion, db)

    def get_deployed(self, model_id: Optional[str] = None) -> List[ModelVersion]:
        """Get deployed model versions."""
        query = self.db.query(ModelVersion).filter(ModelVersion.is_deployed == True)
        if model_id:
            query = query.filter(ModelVersion.model_id == model_id)
        return query.all()

    def get_by_model(self, model_id: str) -> List[ModelVersion]:
        """Get all versions for a model."""
        return self.db.query(ModelVersion)\
            .filter(ModelVersion.model_id == model_id)\
            .order_by(ModelVersion.created_at.desc())\
            .all()

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get latest version for a model."""
        return self.db.query(ModelVersion)\
            .filter(ModelVersion.model_id == model_id)\
            .order_by(ModelVersion.created_at.desc())\
            .first()

    def get_by_model_id(self, model_id: str) -> List[ModelVersion]:
        """Get all versions for a model (alias for get_by_model)."""
        return self.get_by_model(model_id)


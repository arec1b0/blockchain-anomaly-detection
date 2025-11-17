"""
Model deployment manager for loading and serving ML models.

This module handles model loading, caching, and serving for predictions.
"""

import os
import pickle
import json
import tempfile
from typing import Dict, Any, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from src.database.repositories.model_repository import ModelVersionRepository
from src.database.models import ModelVersion
from src.ml.storage import ModelStorage
from src.ml.deployment.ab_tester import ABTester
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class ModelManager:
    """
    Manages model loading, caching, and serving.

    Features:
    - Model caching for fast inference
    - A/B testing integration
    - Automatic model version selection
    - Lazy loading to save memory

    Usage:
        manager = ModelManager(db_session)

        # Get model for prediction (handles A/B testing)
        model = manager.get_model_for_prediction(model_id="default", user_id="user123")

        # Make prediction
        result = model.predict(features)

        # Clear cache
        manager.clear_cache()
    """

    def __init__(self, db_session: Session):
        """
        Initialize ModelManager.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.model_version_repo = ModelVersionRepository(db_session)
        self.storage = ModelStorage()
        self.ab_tester = ABTester(db_session)

        # Model cache: {model_version_id: (model, metadata, load_time)}
        self._cache: Dict[str, tuple] = {}

        # Cache configuration
        self.cache_enabled = os.getenv("MODEL_CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl_hours = int(os.getenv("MODEL_CACHE_TTL_HOURS", "24"))

    def get_model_for_prediction(
        self,
        model_id: str,
        user_id: Optional[str] = None
    ) -> Any:
        """
        Get model for making predictions.

        Handles A/B testing automatically by selecting the appropriate model
        version based on traffic allocation.

        Args:
            model_id: ID of the model
            user_id: Optional user ID for consistent A/B testing routing

        Returns:
            Loaded model ready for prediction

        Raises:
            ValueError: If no model found or model loading fails
        """
        # Use A/B tester to get the active model version
        model_version = self.ab_tester.get_active_model(model_id, user_id)

        if not model_version:
            raise ValueError(f"No active model version found for model: {model_id}")

        # Load model (uses cache if available)
        return self.load_model(model_version.id)

    def load_model(
        self,
        model_version_id: str,
        force_reload: bool = False
    ) -> Any:
        """
        Load a specific model version.

        Args:
            model_version_id: ID of model version to load
            force_reload: Force reload even if cached

        Returns:
            Loaded model object

        Raises:
            ValueError: If model version not found
            RuntimeError: If model loading fails
        """
        # Check cache
        if not force_reload and self.cache_enabled and model_version_id in self._cache:
            model, metadata, load_time = self._cache[model_version_id]

            # Check if cache entry is still valid
            hours_since_load = (datetime.utcnow() - load_time).total_seconds() / 3600
            if hours_since_load < self.cache_ttl_hours:
                logger.debug(f"Returning cached model {model_version_id}")
                return model
            else:
                logger.info(f"Cache expired for model {model_version_id}, reloading")
                del self._cache[model_version_id]

        # Load model from storage
        logger.info(f"Loading model version {model_version_id}")

        # Get model version metadata
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {model_version_id}")

        # Download model from storage
        tmp_dir = tempfile.mkdtemp()
        try:
            local_path = self.storage.download_model(
                storage_path=model_version.storage_path,
                local_dir=tmp_dir
            )

            # Load model pickle
            model_file = os.path.join(local_path, 'model.pkl')
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Model file not found: {model_file}")

            with open(model_file, 'rb') as f:
                model = pickle.load(f)

            # Load metadata
            metadata_file = os.path.join(local_path, 'metadata.json')
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            logger.info(f"Model {model_version_id} loaded successfully")

            # Cache the model
            if self.cache_enabled:
                self._cache[model_version_id] = (model, metadata, datetime.utcnow())
                logger.debug(f"Model {model_version_id} cached")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_version_id}: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

        finally:
            # Cleanup temporary directory
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def get_model_metadata(self, model_version_id: str) -> Dict[str, Any]:
        """
        Get metadata for a model version.

        Args:
            model_version_id: ID of model version

        Returns:
            Metadata dictionary

        Raises:
            ValueError: If model version not found
        """
        # Check cache first
        if self.cache_enabled and model_version_id in self._cache:
            _, metadata, _ = self._cache[model_version_id]
            return metadata

        # Load from storage
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {model_version_id}")

        # Download metadata only
        tmp_dir = tempfile.mkdtemp()
        try:
            local_path = self.storage.download_model(
                storage_path=model_version.storage_path,
                local_dir=tmp_dir
            )

            metadata_file = os.path.join(local_path, 'metadata.json')
            if not os.path.exists(metadata_file):
                return {}

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            return metadata

        except Exception as e:
            logger.error(f"Failed to load metadata for model {model_version_id}: {e}")
            return {}

        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def preload_models(self, model_ids: Optional[list] = None):
        """
        Preload models into cache.

        Useful for warming up cache on application startup.

        Args:
            model_ids: List of model IDs to preload (None = preload all deployed)
        """
        if not self.cache_enabled:
            logger.warning("Model caching is disabled, skipping preload")
            return

        logger.info("Preloading models into cache")

        if model_ids is None:
            # Get all deployed model versions
            deployed_versions = self.model_version_repo.get_deployed()
        else:
            # Get specific models
            deployed_versions = []
            for model_id in model_ids:
                versions = self.model_version_repo.get_deployed(model_id)
                deployed_versions.extend(versions)

        for version in deployed_versions:
            try:
                self.load_model(version.id)
                logger.info(f"Preloaded model {version.id}")
            except Exception as e:
                logger.error(f"Failed to preload model {version.id}: {e}")

        logger.info(f"Preloaded {len(self._cache)} models")

    def clear_cache(self, model_version_id: Optional[str] = None):
        """
        Clear model cache.

        Args:
            model_version_id: Specific model to clear (None = clear all)
        """
        if model_version_id:
            if model_version_id in self._cache:
                del self._cache[model_version_id]
                logger.info(f"Cleared cache for model {model_version_id}")
        else:
            self._cache.clear()
            logger.info("Cleared all model cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.cache_enabled:
            return {
                "enabled": False,
                "cached_models": 0
            }

        cached_models = []
        for version_id, (_, _, load_time) in self._cache.items():
            hours_cached = (datetime.utcnow() - load_time).total_seconds() / 3600
            cached_models.append({
                "version_id": version_id,
                "loaded_at": load_time.isoformat(),
                "hours_cached": round(hours_cached, 2),
                "expires_in_hours": round(self.cache_ttl_hours - hours_cached, 2)
            })

        return {
            "enabled": True,
            "cached_models": len(self._cache),
            "ttl_hours": self.cache_ttl_hours,
            "models": cached_models
        }

    def warmup_cache(self):
        """
        Alias for preload_models() for convenience.
        """
        self.preload_models()


class ModelRegistry:
    """
    Model registry for managing model lifecycle.

    Provides high-level operations for model management:
    - List available models
    - Get model information
    - Deploy/undeploy models
    - Track model lineage
    """

    def __init__(self, db_session: Session):
        """
        Initialize ModelRegistry.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.model_version_repo = ModelVersionRepository(db_session)

    def list_models(
        self,
        model_id: Optional[str] = None,
        deployed_only: bool = False
    ) -> list:
        """
        List model versions.

        Args:
            model_id: Filter by model ID
            deployed_only: Only return deployed versions

        Returns:
            List of model version dictionaries
        """
        if deployed_only:
            versions = self.model_version_repo.get_deployed(model_id)
        elif model_id:
            versions = self.model_version_repo.get_by_model(model_id)
        else:
            versions = self.db.query(ModelVersion).all()

        return [
            {
                "id": v.id,
                "model_id": v.model_id,
                "version": v.version,
                "is_deployed": v.is_deployed,
                "traffic_percentage": v.traffic_percentage,
                "deployed_at": v.deployed_at.isoformat() if v.deployed_at else None,
                "created_at": v.created_at.isoformat(),
                "metrics": v.metrics,
                "hyperparameters": v.hyperparameters
            }
            for v in versions
        ]

    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get model lineage (version history).

        Args:
            model_id: ID of the model

        Returns:
            Dictionary with model lineage information
        """
        versions = self.model_version_repo.get_by_model(model_id)

        lineage = []
        for v in versions:
            lineage.append({
                "version": v.version,
                "version_id": v.id,
                "created_at": v.created_at.isoformat(),
                "is_deployed": v.is_deployed,
                "traffic_percentage": v.traffic_percentage,
                "training_duration_seconds": v.training_duration_seconds,
                "metrics": v.metrics
            })

        # Sort by creation time (newest first)
        lineage.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "model_id": model_id,
            "total_versions": len(lineage),
            "versions": lineage
        }

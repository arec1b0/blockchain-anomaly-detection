"""
A/B testing framework for ML model deployment.

This module provides functionality for gradual model rollout with A/B testing,
including shadow deployment, traffic splitting, and performance comparison.
"""

import random
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session

from src.database.repositories.model_repository import ModelVersionRepository
from src.database.repositories.anomaly_repository import AnomalyRepository
from src.database.models import ModelVersion
from src.utils.logger import get_logger
from src.utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class DeploymentStrategy(str, Enum):
    """Model deployment strategies."""
    SHADOW = "shadow"  # 0% traffic, parallel execution for comparison
    CANARY = "canary"  # Gradual rollout: 10% → 50% → 100%
    BLUE_GREEN = "blue_green"  # Instant switch with quick rollback capability
    FULL = "full"  # 100% traffic immediately


class ABTester:
    """
    A/B testing framework for model deployment.

    Supports multiple deployment strategies:
    - Shadow: New model runs alongside old, 0% traffic
    - Canary: Gradual rollout (10% → 50% → 100%)
    - Blue-Green: Instant switch with quick rollback
    - Full: Immediate 100% deployment

    Usage:
        ab_tester = ABTester(db_session)

        # Start shadow deployment
        ab_tester.deploy_model(model_version_id, strategy=DeploymentStrategy.SHADOW)

        # After validation, increase traffic
        ab_tester.update_traffic(model_version_id, traffic_percentage=10.0)

        # Gradually increase
        ab_tester.update_traffic(model_version_id, traffic_percentage=50.0)
        ab_tester.update_traffic(model_version_id, traffic_percentage=100.0)

        # Rollback if needed
        ab_tester.rollback_deployment(model_version_id)
    """

    def __init__(self, db_session: Session):
        """
        Initialize ABTester.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session
        self.model_version_repo = ModelVersionRepository(db_session)
        self.anomaly_repo = AnomalyRepository(db_session)

    def deploy_model(
        self,
        model_version_id: str,
        strategy: DeploymentStrategy = DeploymentStrategy.SHADOW,
        initial_traffic: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Deploy a model version with specified strategy.

        Args:
            model_version_id: ID of model version to deploy
            strategy: Deployment strategy
            initial_traffic: Initial traffic percentage (None = strategy default)

        Returns:
            Deployment status dictionary

        Raises:
            ValueError: If model version not found or invalid parameters
        """
        logger.info(f"Deploying model version {model_version_id} with strategy: {strategy}")

        # Get model version
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {model_version_id}")

        # Determine initial traffic based on strategy
        if initial_traffic is None:
            if strategy == DeploymentStrategy.SHADOW:
                initial_traffic = 0.0
            elif strategy == DeploymentStrategy.CANARY:
                initial_traffic = 10.0
            elif strategy == DeploymentStrategy.BLUE_GREEN:
                initial_traffic = 0.0  # Start at 0, switch instantly when ready
            elif strategy == DeploymentStrategy.FULL:
                initial_traffic = 100.0

        # Validate traffic percentage
        if not (0.0 <= initial_traffic <= 100.0):
            raise ValueError(f"Invalid traffic percentage: {initial_traffic}")

        # Check if there's a currently deployed model for this model
        currently_deployed = self.model_version_repo.get_deployed(model_version.model_id)

        # If deploying at 100%, undeploy others
        if initial_traffic == 100.0:
            for deployed in currently_deployed:
                if deployed.id != model_version_id:
                    logger.info(f"Undeploying model version {deployed.id}")
                    deployed.is_deployed = False
                    deployed.traffic_percentage = 0.0
                    self.model_version_repo.update(deployed)

        # Update model version
        model_version.is_deployed = True
        model_version.deployed_at = datetime.utcnow()
        model_version.traffic_percentage = initial_traffic
        self.model_version_repo.update(model_version)

        logger.info(
            f"Model version {model_version_id} deployed with {initial_traffic}% traffic"
        )

        return {
            "model_version_id": model_version_id,
            "strategy": strategy,
            "traffic_percentage": initial_traffic,
            "deployed_at": model_version.deployed_at.isoformat(),
            "status": "deployed"
        }

    def update_traffic(
        self,
        model_version_id: str,
        traffic_percentage: float
    ) -> Dict[str, Any]:
        """
        Update traffic percentage for a deployed model.

        Args:
            model_version_id: ID of model version
            traffic_percentage: New traffic percentage (0-100)

        Returns:
            Updated deployment status

        Raises:
            ValueError: If model not deployed or invalid percentage
        """
        logger.info(f"Updating traffic for model {model_version_id} to {traffic_percentage}%")

        # Validate percentage
        if not (0.0 <= traffic_percentage <= 100.0):
            raise ValueError(f"Invalid traffic percentage: {traffic_percentage}")

        # Get model version
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {model_version_id}")

        if not model_version.is_deployed:
            raise ValueError(f"Model version {model_version_id} is not deployed")

        old_percentage = model_version.traffic_percentage

        # If increasing to 100%, undeploy other versions
        if traffic_percentage == 100.0:
            deployed = self.model_version_repo.get_deployed(model_version.model_id)
            for other in deployed:
                if other.id != model_version_id:
                    logger.info(f"Undeploying model version {other.id}")
                    other.is_deployed = False
                    other.traffic_percentage = 0.0
                    self.model_version_repo.update(other)

        # Update traffic
        model_version.traffic_percentage = traffic_percentage
        self.model_version_repo.update(model_version)

        logger.info(
            f"Traffic updated for model {model_version_id}: "
            f"{old_percentage}% → {traffic_percentage}%"
        )

        return {
            "model_version_id": model_version_id,
            "old_traffic": old_percentage,
            "new_traffic": traffic_percentage,
            "status": "updated"
        }

    def rollback_deployment(
        self,
        model_version_id: str,
        restore_previous: bool = True
    ) -> Dict[str, Any]:
        """
        Rollback a model deployment.

        Args:
            model_version_id: ID of model version to rollback
            restore_previous: Whether to restore previous deployed version

        Returns:
            Rollback status dictionary
        """
        logger.warning(f"Rolling back model version {model_version_id}")

        # Get model version
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version:
            raise ValueError(f"Model version not found: {model_version_id}")

        # Undeploy current version
        old_traffic = model_version.traffic_percentage
        model_version.is_deployed = False
        model_version.traffic_percentage = 0.0
        self.model_version_repo.update(model_version)

        # Restore previous version if requested
        restored_version_id = None
        if restore_previous:
            # Get all versions for this model, ordered by deployment time
            all_versions = self.model_version_repo.get_by_model(model_version.model_id)

            # Find the most recent previously deployed version
            for version in all_versions:
                if version.id != model_version_id and version.deployed_at:
                    # Restore this version
                    version.is_deployed = True
                    version.traffic_percentage = 100.0
                    self.model_version_repo.update(version)
                    restored_version_id = version.id
                    logger.info(f"Restored previous model version {version.id}")
                    break

        logger.info(f"Model version {model_version_id} rolled back (was at {old_traffic}%)")

        return {
            "rolled_back_version": model_version_id,
            "old_traffic": old_traffic,
            "restored_version": restored_version_id,
            "status": "rolled_back"
        }

    def should_use_model(
        self,
        model_version_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Determine if a request should use this model version based on traffic percentage.

        Uses consistent hashing if user_id provided (same user always gets same model),
        otherwise uses random selection.

        Args:
            model_version_id: ID of model version
            user_id: Optional user ID for consistent routing

        Returns:
            True if this model should be used for the request
        """
        model_version = self.model_version_repo.get_by_id(model_version_id)
        if not model_version or not model_version.is_deployed:
            return False

        traffic_percentage = model_version.traffic_percentage

        if traffic_percentage == 0.0:
            return False
        elif traffic_percentage == 100.0:
            return True
        else:
            # Use consistent hashing if user_id provided
            if user_id:
                # Hash user_id to get consistent routing
                import hashlib
                hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                return (hash_value % 100) < traffic_percentage
            else:
                # Random selection
                return random.random() * 100 < traffic_percentage

    def get_active_model(
        self,
        model_id: str,
        user_id: Optional[str] = None
    ) -> Optional[ModelVersion]:
        """
        Get the active model version to use for a request.

        Handles A/B testing by selecting model based on traffic percentage.

        Args:
            model_id: ID of the model
            user_id: Optional user ID for consistent routing

        Returns:
            ModelVersion to use, or None if no model deployed
        """
        deployed_versions = self.model_version_repo.get_deployed(model_id)

        if not deployed_versions:
            return None

        # If only one deployed, use it
        if len(deployed_versions) == 1:
            version = deployed_versions[0]
            if self.should_use_model(version.id, user_id):
                return version
            return None

        # Multiple deployed versions (A/B test)
        # Check each version in order of traffic percentage (highest first)
        sorted_versions = sorted(
            deployed_versions,
            key=lambda v: v.traffic_percentage,
            reverse=True
        )

        for version in sorted_versions:
            if self.should_use_model(version.id, user_id):
                return version

        # Fallback to highest traffic version
        return sorted_versions[0] if sorted_versions else None

    def compare_models(
        self,
        model_version_id_a: str,
        model_version_id_b: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Compare performance metrics of two model versions.

        Args:
            model_version_id_a: ID of first model version
            model_version_id_b: ID of second model version
            time_window_hours: Time window for comparison (hours)

        Returns:
            Comparison metrics dictionary
        """
        logger.info(f"Comparing models {model_version_id_a} vs {model_version_id_b}")

        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        # Get anomalies detected by each model
        anomalies_a = self.anomaly_repo.get_by_model_version(
            model_version_id_a,
            start_date=cutoff_time
        )
        anomalies_b = self.anomaly_repo.get_by_model_version(
            model_version_id_b,
            start_date=cutoff_time
        )

        # Calculate metrics
        metrics_a = self._calculate_model_metrics(anomalies_a)
        metrics_b = self._calculate_model_metrics(anomalies_b)

        # Determine winner
        winner = self._determine_winner(metrics_a, metrics_b)

        return {
            "model_a": {
                "id": model_version_id_a,
                "metrics": metrics_a
            },
            "model_b": {
                "id": model_version_id_b,
                "metrics": metrics_b
            },
            "winner": winner,
            "time_window_hours": time_window_hours
        }

    def _calculate_model_metrics(self, anomalies: List) -> Dict[str, Any]:
        """Calculate metrics for a list of anomalies."""
        if not anomalies:
            return {
                "total_anomalies": 0,
                "avg_confidence": 0.0,
                "avg_anomaly_score": 0.0,
                "severity_distribution": {}
            }

        total = len(anomalies)
        avg_confidence = sum(a.confidence for a in anomalies) / total
        avg_score = sum(a.anomaly_score for a in anomalies) / total

        # Severity distribution
        severity_counts = {}
        for anomaly in anomalies:
            severity = anomaly.severity.value if hasattr(anomaly.severity, 'value') else str(anomaly.severity)
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_anomalies": total,
            "avg_confidence": round(avg_confidence, 3),
            "avg_anomaly_score": round(avg_score, 3),
            "severity_distribution": severity_counts
        }

    def _determine_winner(
        self,
        metrics_a: Dict[str, Any],
        metrics_b: Dict[str, Any]
    ) -> str:
        """
        Determine which model performs better.

        Criteria (in order of priority):
        1. Higher average confidence
        2. More high-severity anomalies detected
        3. Higher total anomalies (if both have similar confidence)
        """
        conf_a = metrics_a.get("avg_confidence", 0)
        conf_b = metrics_b.get("avg_confidence", 0)

        # If confidence difference > 5%, higher confidence wins
        if abs(conf_a - conf_b) > 0.05:
            return "model_a" if conf_a > conf_b else "model_b"

        # Check high-severity anomalies
        sev_a = metrics_a.get("severity_distribution", {})
        sev_b = metrics_b.get("severity_distribution", {})

        high_sev_a = sev_a.get("critical", 0) + sev_a.get("high", 0)
        high_sev_b = sev_b.get("critical", 0) + sev_b.get("high", 0)

        if high_sev_a != high_sev_b:
            return "model_a" if high_sev_a > high_sev_b else "model_b"

        # Fallback to total anomalies
        total_a = metrics_a.get("total_anomalies", 0)
        total_b = metrics_b.get("total_anomalies", 0)

        if total_a == total_b:
            return "tie"

        return "model_a" if total_a > total_b else "model_b"

    def get_deployment_status(self, model_id: str) -> Dict[str, Any]:
        """
        Get current deployment status for all versions of a model.

        Args:
            model_id: ID of the model

        Returns:
            Deployment status dictionary
        """
        deployed_versions = self.model_version_repo.get_deployed(model_id)

        versions_info = []
        for version in deployed_versions:
            versions_info.append({
                "version_id": version.id,
                "version": version.version,
                "traffic_percentage": version.traffic_percentage,
                "deployed_at": version.deployed_at.isoformat() if version.deployed_at else None,
                "metrics": version.metrics
            })

        # Sort by traffic percentage (highest first)
        versions_info.sort(key=lambda x: x["traffic_percentage"], reverse=True)

        return {
            "model_id": model_id,
            "deployed_versions": versions_info,
            "total_versions": len(versions_info),
            "traffic_allocated": sum(v["traffic_percentage"] for v in versions_info)
        }

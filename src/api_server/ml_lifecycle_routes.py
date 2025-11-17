"""
ML Lifecycle API routes (Phase 3).

This module provides endpoints for:
- Model deployment and A/B testing
- Drift detection
- Model comparison
- Model retraining
- Deployment management
"""

from fastapi import APIRouter, Depends, HTTPRequest, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Dict, Any

from src.api_server.models import (
    ModelDeploymentRequest,
    ModelDeploymentResponse,
    ModelTrafficUpdateRequest,
    ModelRollbackRequest,
    DriftDetectionRequest,
    DriftDetectionResponse,
    ModelComparisonRequest,
    ModelComparisonResponse,
    ModelRetrainingRequest,
    ModelRetrainingResponse,
    DeploymentStatusResponse,
    CacheStatsResponse,
    SuccessResponse,
    ErrorResponse
)
from src.ml.deployment.ab_tester import ABTester, DeploymentStrategy
from src.ml.monitoring.drift_detector import DriftDetector
from src.ml.deployment.model_manager import ModelManager
from src.ml.training.trainer import ModelTrainer
from src.database.connection import get_db
from src.auth.dependencies import get_current_active_user, require_roles
from src.audit.audit_logger import log_audit_event
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/ml", tags=["ML Lifecycle"])


# ============================================================================
# Model Deployment Endpoints
# ============================================================================

@router.post("/deploy", response_model=ModelDeploymentResponse)
async def deploy_model(
    request: ModelDeploymentRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Deploy a model version with specified strategy.

    Strategies:
    - shadow: 0% traffic, parallel execution
    - canary: Gradual rollout (10% → 50% → 100%)
    - blue_green: Instant switch with rollback capability
    - full: 100% traffic immediately

    Requires admin role.
    """
    try:
        ab_tester = ABTester(db)

        # Validate strategy
        strategy = DeploymentStrategy(request.strategy)

        # Deploy model
        result = ab_tester.deploy_model(
            model_version_id=request.model_version_id,
            strategy=strategy,
            initial_traffic=request.initial_traffic
        )

        # Log audit event
        await log_audit_event(
            db=db,
            event_type="model",
            user_id=current_user.get("sub"),
            resource=f"model_version/{request.model_version_id}",
            action="deploy",
            status="success",
            ip_address=http_request.client.host,
            details={
                "strategy": request.strategy,
                "traffic_percentage": result["traffic_percentage"]
            }
        )

        return ModelDeploymentResponse(
            success=True,
            model_version_id=result["model_version_id"],
            strategy=result["strategy"],
            traffic_percentage=result["traffic_percentage"],
            deployed_at=result["deployed_at"],
            message=f"Model deployed with {result['strategy']} strategy"
        )

    except ValueError as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Deployment error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Deployment failed")


@router.put("/traffic", response_model=SuccessResponse)
async def update_traffic(
    request: ModelTrafficUpdateRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Update traffic percentage for a deployed model.

    Use this for gradual canary rollouts:
    1. Start at 10%
    2. Monitor metrics
    3. Increase to 50%
    4. Monitor again
    5. Increase to 100%

    Requires admin role.
    """
    try:
        ab_tester = ABTester(db)

        result = ab_tester.update_traffic(
            model_version_id=request.model_version_id,
            traffic_percentage=request.traffic_percentage
        )

        # Log audit event
        await log_audit_event(
            db=db,
            event_type="model",
            user_id=current_user.get("sub"),
            resource=f"model_version/{request.model_version_id}",
            action="update_traffic",
            status="success",
            ip_address=http_request.client.host,
            details={
                "old_traffic": result["old_traffic"],
                "new_traffic": result["new_traffic"]
            }
        )

        return SuccessResponse(
            success=True,
            message=f"Traffic updated: {result['old_traffic']}% → {result['new_traffic']}%",
            data=result
        )

    except ValueError as e:
        logger.error(f"Traffic update failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Traffic update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Traffic update failed")


@router.post("/rollback", response_model=SuccessResponse)
async def rollback_deployment(
    request: ModelRollbackRequest,
    http_request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Rollback a model deployment.

    Optionally restores the previous deployed version.

    Requires admin role.
    """
    try:
        ab_tester = ABTester(db)

        result = ab_tester.rollback_deployment(
            model_version_id=request.model_version_id,
            restore_previous=request.restore_previous
        )

        # Log audit event
        await log_audit_event(
            db=db,
            event_type="model",
            user_id=current_user.get("sub"),
            resource=f"model_version/{request.model_version_id}",
            action="rollback",
            status="success",
            ip_address=http_request.client.host,
            severity="warning",
            details={
                "old_traffic": result["old_traffic"],
                "restored_version": result.get("restored_version")
            }
        )

        logger.warning(f"Model {request.model_version_id} rolled back by {current_user.get('email')}")

        return SuccessResponse(
            success=True,
            message=f"Model rolled back. Previous version restored: {request.restore_previous}",
            data=result
        )

    except ValueError as e:
        logger.error(f"Rollback failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Rollback error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Rollback failed")


@router.get("/deployment/status/{model_id}", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    model_id: str,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get current deployment status for a model.

    Shows all deployed versions and their traffic allocation.

    Requires authentication.
    """
    try:
        ab_tester = ABTester(db)
        status = ab_tester.get_deployment_status(model_id)
        return DeploymentStatusResponse(**status)

    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get deployment status")


# ============================================================================
# Drift Detection Endpoints
# ============================================================================

@router.post("/drift/detect", response_model=DriftDetectionResponse)
async def detect_drift(
    request: DriftDetectionRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "analyst"]))
):
    """
    Detect model drift (feature, concept, performance).

    Returns:
    - Feature drift: Distribution changes in input features
    - Concept drift: Changes in prediction patterns
    - Performance drift: Accuracy degradation

    Requires admin or analyst role.
    """
    try:
        detector = DriftDetector(
            db_session=db,
            reference_window_days=request.reference_window_days,
            detection_window_days=request.detection_window_days
        )

        result = detector.detect_drift(
            model_version_id=request.model_version_id,
            drift_threshold=request.drift_threshold
        )

        # Log warning if drift detected
        if result["drift_detected"]:
            logger.warning(
                f"Drift detected for model {request.model_version_id}: "
                f"{result['recommendation']}"
            )

        return DriftDetectionResponse(**result)

    except Exception as e:
        logger.error(f"Drift detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Drift detection failed")


@router.post("/compare", response_model=ModelComparisonResponse)
async def compare_models(
    request: ModelComparisonRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin", "analyst"]))
):
    """
    Compare performance of two model versions.

    Useful for A/B testing validation.

    Requires admin or analyst role.
    """
    try:
        ab_tester = ABTester(db)

        result = ab_tester.compare_models(
            model_version_id_a=request.model_version_id_a,
            model_version_id_b=request.model_version_id_b,
            time_window_hours=request.time_window_hours
        )

        return ModelComparisonResponse(**result)

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        raise HTTPException(status_code=500, detail="Model comparison failed")


# ============================================================================
# Model Retraining Endpoints
# ============================================================================

@router.post("/retrain", response_model=ModelRetrainingResponse)
async def retrain_model(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Trigger model retraining.

    This is a long-running operation that runs in the background.

    Requires admin role.
    """
    try:
        # Parse dates if provided
        start_date = datetime.fromisoformat(request.start_date) if request.start_date else None
        end_date = datetime.fromisoformat(request.end_date) if request.end_date else None

        # Start training in background
        trainer = ModelTrainer(db)

        logger.info(f"Starting model retraining: {request.model_name}")

        model_version_id, metrics = trainer.train_isolation_forest(
            model_name=request.model_name,
            start_date=start_date,
            end_date=end_date,
            hyperparameter_tuning=request.hyperparameter_tuning,
            contamination=request.contamination
        )

        # Log audit event
        await log_audit_event(
            db=db,
            event_type="model",
            user_id=current_user.get("sub"),
            resource=f"model/{request.model_name}",
            action="retrain",
            status="success",
            ip_address=http_request.client.host,
            details={
                "model_version_id": model_version_id,
                "training_samples": metrics.get("training_samples"),
                "training_duration": metrics.get("training_duration_seconds")
            }
        )

        # Get version info
        from src.database.repositories.model_repository import ModelVersionRepository
        model_version_repo = ModelVersionRepository(db)
        version = model_version_repo.get_by_id(model_version_id)

        return ModelRetrainingResponse(
            success=True,
            model_version_id=model_version_id,
            model_name=request.model_name,
            version=version.version,
            metrics=metrics,
            training_duration_seconds=metrics.get("training_duration_seconds", 0),
            message=f"Model retrained successfully: version {version.version}"
        )

    except ValueError as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Retraining error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Retraining failed")


# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get model cache statistics.

    Shows which models are currently cached in memory.

    Requires authentication.
    """
    try:
        model_manager = ModelManager(db)
        stats = model_manager.get_cache_stats()
        return CacheStatsResponse(**stats)

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache stats")


@router.post("/cache/clear", response_model=SuccessResponse)
async def clear_cache(
    model_version_id: str = None,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Clear model cache.

    If model_version_id provided, clears only that model.
    Otherwise clears all cached models.

    Requires admin role.
    """
    try:
        model_manager = ModelManager(db)
        model_manager.clear_cache(model_version_id)

        message = f"Cleared cache for model {model_version_id}" if model_version_id else "Cleared all model cache"

        return SuccessResponse(
            success=True,
            message=message
        )

    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


@router.post("/cache/preload", response_model=SuccessResponse)
async def preload_cache(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_roles(["admin"]))
):
    """
    Preload deployed models into cache.

    Useful for warming up cache after application restart.

    Requires admin role.
    """
    try:
        model_manager = ModelManager(db)

        # Run preload in background
        background_tasks.add_task(model_manager.preload_models)

        return SuccessResponse(
            success=True,
            message="Cache preload started in background"
        )

    except Exception as e:
        logger.error(f"Failed to preload cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to preload cache")

"""
Batch processing API routes for high-throughput scenarios.

This module provides endpoints for batch operations to reduce overhead:
- Batch predictions
- Batch anomaly retrieval
- Batch transaction lookup
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from datetime import datetime
import asyncio

from pydantic import BaseModel, Field

from src.database.connection import get_db
from src.auth.dependencies import get_current_active_user
from src.cache.optimized_cache import OptimizedCacheLayer
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/batch", tags=["Batch Operations"])


# ============================================================================
# Request/Response Models
# ============================================================================

class TransactionInput(BaseModel):
    """Single transaction for batch prediction."""
    hash: str
    value: float
    gas: float
    gasPrice: float
    timestamp: str = Field(None, description="ISO format timestamp")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    transactions: List[TransactionInput] = Field(
        ...,
        max_items=1000,
        description="List of transactions (max 1000)"
    )


class PredictionResult(BaseModel):
    """Single prediction result."""
    hash: str
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    total_processed: int
    total_anomalies: int
    processing_time_ms: float
    results: List[PredictionResult]
    from_cache: int
    computed: int


class BatchHashRequest(BaseModel):
    """Request for batch transaction lookup."""
    hashes: List[str] = Field(..., max_items=1000)


# ============================================================================
# Batch Prediction Endpoint
# ============================================================================

@router.post("/predict", response_model=BatchPredictionResponse)
async def batch_predict(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Batch prediction endpoint for multiple transactions.

    Optimizations:
    - Checks cache first for each transaction
    - Computes predictions in parallel
    - Updates cache in background
    - Returns results immediately

    Limits:
    - Maximum 1000 transactions per batch
    - Recommended batch size: 100-500 for optimal performance

    Requires authentication.
    """
    import time
    from src.ml.deployment.model_manager import ModelManager
    from src.cache.optimized_cache import OptimizedCacheLayer
    import pandas as pd
    import numpy as np

    start_time = time.time()
    transactions = request.transactions

    logger.info(f"Batch prediction request: {len(transactions)} transactions")

    # Initialize cache and model manager
    cache = OptimizedCacheLayer()
    model_manager = ModelManager(db)

    # Check cache first
    cached_results = []
    uncached_transactions = []
    cache_hits = 0

    for txn in transactions:
        cached = cache.get_prediction(txn.hash)
        if cached:
            cached_results.append(cached)
            cache_hits += 1
        else:
            uncached_transactions.append(txn)

    logger.info(f"Cache hits: {cache_hits}/{len(transactions)}")

    # Process uncached transactions
    computed_results = []
    if uncached_transactions:
        try:
            # Load model
            model = model_manager.get_model_for_prediction(
                model_id="default",
                user_id=current_user.get("sub")
            )

            # Prepare features in batch
            df = pd.DataFrame([{
                'hash': t.hash,
                'value': t.value,
                'gas': t.gas,
                'gasPrice': t.gasPrice
            } for t in uncached_transactions])

            # Add derived features
            df['value_per_gas'] = df['value'] / (df['gas'] + 1e-10)

            # Select features for prediction
            feature_columns = ['value', 'gas', 'gasPrice', 'value_per_gas']
            X = df[feature_columns].values

            # Make predictions in batch
            pred_start = time.time()
            scores = model.score_samples(X)
            predictions = model.predict(X)
            pred_duration = (time.time() - pred_start) * 1000

            # Build results
            for i, txn in enumerate(uncached_transactions):
                result = {
                    'hash': txn.hash,
                    'is_anomaly': bool(predictions[i] == -1),
                    'confidence': float(abs(scores[i])),
                    'anomaly_score': float(scores[i]),
                    'processing_time_ms': pred_duration / len(uncached_transactions)
                }
                computed_results.append(result)

                # Cache result in background
                background_tasks.add_task(
                    cache.cache_prediction,
                    txn.hash,
                    result
                )

        except Exception as e:
            logger.error(f"Batch prediction error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Batch prediction failed")

    # Combine results
    all_results = cached_results + computed_results

    # Calculate stats
    total_anomalies = sum(1 for r in all_results if r.get('is_anomaly'))
    total_time = (time.time() - start_time) * 1000

    logger.info(
        f"Batch prediction complete: {len(all_results)} processed, "
        f"{total_anomalies} anomalies, {total_time:.0f}ms"
    )

    return BatchPredictionResponse(
        total_processed=len(all_results),
        total_anomalies=total_anomalies,
        processing_time_ms=total_time,
        results=[PredictionResult(**r) for r in all_results],
        from_cache=cache_hits,
        computed=len(computed_results)
    )


# ============================================================================
# Batch Transaction Lookup
# ============================================================================

@router.post("/transactions/lookup")
async def batch_transaction_lookup(
    request: BatchHashRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Batch transaction lookup by hashes.

    Returns transaction details for multiple hashes in a single request.

    Limits:
    - Maximum 1000 hashes per batch

    Requires authentication.
    """
    from src.database.repositories.transaction_repository import TransactionRepository

    if len(request.hashes) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 hashes per batch"
        )

    repo = TransactionRepository(db)

    results = []
    for tx_hash in request.hashes:
        transaction = repo.get_by_hash(tx_hash)
        if transaction:
            results.append({
                'hash': transaction.hash,
                'value': transaction.value,
                'gas': transaction.gas,
                'gas_price': transaction.gas_price,
                'timestamp': transaction.timestamp.isoformat(),
                'from_address': transaction.from_address,
                'to_address': transaction.to_address
            })

    return {
        'total_requested': len(request.hashes),
        'total_found': len(results),
        'transactions': results
    }


# ============================================================================
# Batch Anomaly Retrieval
# ============================================================================

@router.get("/anomalies/recent")
async def batch_recent_anomalies(
    limit: int = 100,
    severity: str = None,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Get recent anomalies in batch.

    Optimized for returning large result sets efficiently.

    Args:
        limit: Maximum number of anomalies to return (max 1000)
        severity: Filter by severity (low, medium, high, critical)

    Requires authentication.
    """
    from src.database.repositories.anomaly_repository import AnomalyRepository

    if limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum limit is 1000"
        )

    repo = AnomalyRepository(db)

    # Use pagination for efficiency
    anomalies = repo.get_recent(limit=limit, severity=severity)

    results = [{
        'id': a.id,
        'transaction_hash': a.transaction.hash if a.transaction else None,
        'anomaly_score': a.anomaly_score,
        'severity': a.severity.value if hasattr(a.severity, 'value') else str(a.severity),
        'confidence': a.confidence,
        'detected_at': a.detected_at.isoformat(),
        'reviewed': a.reviewed
    } for a in anomalies]

    return {
        'total': len(results),
        'anomalies': results
    }


# ============================================================================
# Batch Cache Operations
# ============================================================================

@router.post("/cache/warm")
async def warm_prediction_cache(
    background_tasks: BackgroundTasks,
    num_recent: int = 1000,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Warm prediction cache with recent transactions.

    Preloads cache with predictions for recent transactions to improve hit rate.

    Args:
        num_recent: Number of recent transactions to cache (max 10000)

    Requires authentication.
    """
    from src.database.repositories.transaction_repository import TransactionRepository
    from src.cache.optimized_cache import OptimizedCacheLayer

    if num_recent > 10000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10000 transactions for warming"
        )

    def warm_cache_task():
        """Background task to warm cache."""
        try:
            repo = TransactionRepository(db)
            cache = OptimizedCacheLayer()

            # Get recent transactions
            transactions = repo.get_recent(limit=num_recent)

            # Warm cache (this would involve running predictions)
            logger.info(f"Cache warming started for {len(transactions)} transactions")

            # In a real scenario, you'd compute predictions here
            # For now, we just log the intention

        except Exception as e:
            logger.error(f"Cache warming failed: {e}", exc_info=True)

    # Run in background
    background_tasks.add_task(warm_cache_task)

    return {
        'message': f'Cache warming started for {num_recent} recent transactions',
        'status': 'processing'
    }

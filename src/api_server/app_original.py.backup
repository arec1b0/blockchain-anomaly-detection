"""
FastAPI application for blockchain anomaly detection.

This module defines a FastAPI application that provides a REST API for real-time
blockchain transaction anomaly detection. It includes endpoints for prediction,
model management, health checks, and streaming status.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from src.api_server.models import (
    TransactionData,
    BatchTransactionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ModelTrainingRequest,
    ModelTrainingResponse,
    ModelInfo,
    ModelListResponse,
    ModelUpdateRequest,
    AnomalyListResponse,
    AnomalyRecord,
    StreamStatusResponse,
    HealthCheckResponse,
    ErrorResponse,
    SuccessResponse
)
from src.api_server.monitoring import (
    health_checker,
    http_requests_total,
    http_request_duration_seconds,
    http_requests_in_progress
)
from src.streaming.stream_processor import StreamProcessor
from src.streaming.kafka_consumer import KafkaConsumerService
from src.cache import get_cache_layer, get_redis_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global state
stream_processor: Optional[StreamProcessor] = None
kafka_consumer: Optional[KafkaConsumerService] = None
cache_layer = None
app_state = {
    'models': {},
    'active_model_id': None,
    'streaming_enabled': False,
    'cache_enabled': False
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for handling startup and shutdown events.

    During startup, it initializes the stream processor and Kafka consumer.
    During shutdown, it disconnects the Kafka consumer if it's running.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    # Startup
    logger.info("Starting Blockchain Anomaly Detection API")

    # Initialize Redis cache if enabled
    global cache_layer
    if os.getenv('REDIS_ENABLED', 'false').lower() == 'true':
        try:
            redis_client = get_redis_client(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                password=os.getenv('REDIS_PASSWORD'),
                max_connections=int(os.getenv('REDIS_MAX_CONNECTIONS', 50))
            )
            cache_layer = get_cache_layer(redis_client=redis_client)
            app_state['cache_enabled'] = True
            logger.info("Redis cache layer initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Redis cache: {e}")
            cache_layer = None
            app_state['cache_enabled'] = False

    # Initialize stream processor
    global stream_processor
    model_path = os.getenv('MODEL_PATH', './models/default_model.pkl')
    stream_processor = StreamProcessor(
        model_path=model_path if os.path.exists(model_path) else None,
        batch_size=int(os.getenv('BATCH_SIZE', 100)),
        contamination=float(os.getenv('CONTAMINATION', 0.01)),
        anomaly_buffer_max_size=int(os.getenv('ANOMALY_BUFFER_MAX_SIZE', 10000)),
        anomaly_buffer_ttl_seconds=int(os.getenv('ANOMALY_BUFFER_TTL_SECONDS', 3600))
    )
    logger.info("Stream processor initialized")

    # Set health checker dependencies
    health_checker.set_stream_processor(stream_processor)
    if cache_layer:
        health_checker.set_cache_layer(cache_layer)

    # Initialize Kafka consumer if enabled
    if os.getenv('KAFKA_ENABLED', 'false').lower() == 'true':
        global kafka_consumer
        kafka_consumer = KafkaConsumerService(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
            topic=os.getenv('KAFKA_TOPIC', 'blockchain-transactions'),
            group_id=os.getenv('KAFKA_GROUP_ID', 'anomaly-detection-group')
        )
        try:
            kafka_consumer.connect()
            app_state['streaming_enabled'] = True
            logger.info("Kafka consumer connected")
        except Exception as e:
            logger.error(f"Failed to connect Kafka consumer: {e}")
            app_state['streaming_enabled'] = False

    yield

    # Shutdown
    logger.info("Shutting down Blockchain Anomaly Detection API")

    if kafka_consumer:
        kafka_consumer.disconnect()
        logger.info("Kafka consumer disconnected")

    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Blockchain Anomaly Detection API",
    description="REST API for real-time blockchain transaction anomaly detection",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/api/metrics")


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """
    Track HTTP requests with Prometheus metrics.

    This middleware records the total number of HTTP requests, their duration,
    and the number of requests in progress.

    Args:
        request (Request): The incoming HTTP request.
        call_next (Callable): The next middleware or endpoint in the chain.

    Returns:
        Response: The HTTP response.
    """
    method = request.method
    path = request.url.path

    http_requests_in_progress.labels(method=method, endpoint=path).inc()
    start_time = time.time()

    try:
        response = await call_next(request)

        # Record metrics
        duration = time.time() - start_time
        http_requests_total.labels(
            method=method,
            endpoint=path,
            status=response.status_code
        ).inc()
        http_request_duration_seconds.labels(
            method=method,
            endpoint=path
        ).observe(duration)

        return response
    finally:
        http_requests_in_progress.labels(method=method, endpoint=path).dec()


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns detailed health status including system metrics.

    Returns:
        HealthCheckResponse: The health status of the service.

    Raises:
        HTTPException: If the service is unhealthy.
    """
    health_status = health_checker.check_health()

    if health_status['status'] == 'unhealthy':
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )

    return health_status


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.

    Indicates if the service is ready to accept traffic.

    Returns:
        dict: A dictionary indicating the readiness of the service.

    Raises:
        HTTPException: If the service is not ready.
    """
    readiness = health_checker.check_readiness()

    if not readiness['ready']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness
        )

    return readiness


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness check endpoint.

    Indicates if the service is alive and responsive.

    Returns:
        dict: A dictionary indicating the liveness of the service.
    """
    return health_checker.check_liveness()


# Prediction endpoints
@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Prediction"]
)
async def predict_single(transaction: TransactionData):
    """
    Predict if a single transaction is anomalous.

    Args:
        transaction (TransactionData): Transaction data to analyze.

    Returns:
        PredictionResponse: Prediction result with anomaly score.

    Raises:
        HTTPException: If the prediction fails.
    """
    try:
        start_time = time.time()

        # Process transaction
        stream_processor.process_transaction(transaction.model_dump(by_alias=True))
        stream_processor.flush()

        # Get anomalies
        anomalies = stream_processor.get_anomalies()

        # Check if this transaction was flagged
        is_anomaly = any(a['hash'] == transaction.hash for a in anomalies)

        if is_anomaly:
            anomaly_data = next(a for a in anomalies if a['hash'] == transaction.hash)
            severity = anomaly_data.get('severity', 'low')
        else:
            severity = None

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            hash=transaction.hash,
            is_anomaly=is_anomaly,
            severity=severity,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"]
)
async def predict_batch(request: BatchTransactionRequest):
    """
    Predict anomalies for a batch of transactions.

    Args:
        request (BatchTransactionRequest): Batch of transactions to analyze.

    Returns:
        BatchPredictionResponse: Batch prediction results.

    Raises:
        HTTPException: If the batch prediction fails.
    """
    try:
        start_time = time.time()

        # Process all transactions
        for transaction in request.transactions:
            stream_processor.process_transaction(transaction.model_dump(by_alias=True))

        # Flush to process the batch
        stream_processor.flush()

        # Get anomalies
        anomalies = stream_processor.get_anomalies()
        anomaly_hashes = {a['hash']: a for a in anomalies}

        # Create predictions for all transactions
        predictions = []
        for transaction in request.transactions:
            is_anomaly = transaction.hash in anomaly_hashes
            severity = anomaly_hashes[transaction.hash].get('severity') if is_anomaly else None

            predictions.append(PredictionResponse(
                hash=transaction.hash,
                is_anomaly=is_anomaly,
                severity=severity,
                timestamp=datetime.utcnow().isoformat()
            ))

        processing_time = (time.time() - start_time) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_processed=len(request.transactions),
            anomalies_detected=len([p for p in predictions if p.is_anomaly]),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Error in batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Model management endpoints
@app.post(
    "/api/v1/models/train",
    response_model=ModelTrainingResponse,
    tags=["Models"]
)
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """
    Train a new anomaly detection model.

    Args:
        request (ModelTrainingRequest): Training configuration.
        background_tasks (BackgroundTasks): FastAPI background tasks.

    Returns:
        ModelTrainingResponse: Training status and model information.

    Raises:
        HTTPException: If model training fails.
    """
    try:
        start_time = time.time()

        # Generate model ID
        model_id = f"model_{int(time.time())}"

        # This is a simplified version - in production you'd load actual data
        logger.info(f"Training model {model_id} with contamination={request.contamination}")

        # Save model info
        app_state['models'][model_id] = {
            'model_id': model_id,
            'model_type': request.model_type,
            'created_at': datetime.utcnow().isoformat(),
            'contamination': request.contamination,
            'status': 'active'
        }

        training_time = (time.time() - start_time) * 1000

        return ModelTrainingResponse(
            success=True,
            model_id=model_id,
            training_samples=0,  # Would be actual count in production
            contamination=request.contamination,
            training_time_ms=training_time,
            message="Model training initiated successfully"
        )

    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}"
        )


@app.get(
    "/api/v1/models",
    response_model=ModelListResponse,
    tags=["Models"]
)
async def list_models():
    """
    List all available models.

    Returns:
        ModelListResponse: List of model information.
    """
    models = []
    for model_id, model_data in app_state['models'].items():
        models.append(ModelInfo(
            model_id=model_data['model_id'],
            model_type=model_data.get('model_type', 'isolation_forest'),
            created_at=model_data['created_at'],
            last_updated=model_data.get('last_updated', model_data['created_at']),
            version="1.0",
            contamination=model_data.get('contamination', 0.01),
            training_samples=model_data.get('training_samples', 0),
            is_active=model_data.get('status') == 'active'
        ))

    return ModelListResponse(
        models=models,
        total_models=len(models)
    )


@app.get(
    "/api/v1/models/{model_id}",
    response_model=ModelInfo,
    tags=["Models"]
)
async def get_model(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id (str): Model identifier.

    Returns:
        ModelInfo: Model information.

    Raises:
        HTTPException: If the model is not found.
    """
    if model_id not in app_state['models']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    model_data = app_state['models'][model_id]
    return ModelInfo(
        model_id=model_data['model_id'],
        model_type=model_data.get('model_type', 'isolation_forest'),
        created_at=model_data['created_at'],
        last_updated=model_data.get('last_updated', model_data['created_at']),
        version="1.0",
        contamination=model_data.get('contamination', 0.01),
        training_samples=model_data.get('training_samples', 0),
        is_active=model_data.get('status') == 'active'
    )


@app.delete(
    "/api/v1/models/{model_id}",
    response_model=SuccessResponse,
    tags=["Models"]
)
async def delete_model(model_id: str):
    """
    Delete a model.

    Args:
        model_id (str): Model identifier.

    Returns:
        SuccessResponse: Success message.

    Raises:
        HTTPException: If the model is not found.
    """
    if model_id not in app_state['models']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    del app_state['models'][model_id]

    return SuccessResponse(
        success=True,
        message=f"Model {model_id} deleted successfully"
    )


# Anomaly endpoints
@app.get(
    "/api/v1/anomalies",
    response_model=AnomalyListResponse,
    tags=["Anomalies"]
)
async def get_anomalies(limit: Optional[int] = 100, severity: Optional[str] = None):
    """
    Get detected anomalies.

    Args:
        limit (Optional[int]): Maximum number of anomalies to return. Defaults to 100.
        severity (Optional[str]): Filter by severity level.

    Returns:
        AnomalyListResponse: List of detected anomalies.
    """
    anomalies = stream_processor.get_anomalies(limit=limit)

    # Filter by severity if specified
    if severity:
        anomalies = [a for a in anomalies if a.get('severity') == severity]

    anomaly_records = [
        AnomalyRecord(
            hash=a['hash'],
            value=a['value'],
            gas=a['gas'],
            gasPrice=a['gasPrice'],
            from_address=a.get('from'),
            to_address=a.get('to'),
            timestamp=a.get('timestamp', ''),
            detected_at=a.get('detected_at', ''),
            severity=a.get('severity', 'low')
        )
        for a in anomalies
    ]

    return AnomalyListResponse(
        anomalies=anomaly_records,
        total_count=len(anomaly_records)
    )


@app.delete(
    "/api/v1/anomalies",
    response_model=SuccessResponse,
    tags=["Anomalies"]
)
async def clear_anomalies():
    """
    Clear the anomaly buffer.

    Returns:
        SuccessResponse: Success message.
    """
    stream_processor.clear_anomaly_buffer()

    return SuccessResponse(
        success=True,
        message="Anomaly buffer cleared successfully"
    )


# Streaming endpoints
@app.get(
    "/api/v1/stream/status",
    response_model=StreamStatusResponse,
    tags=["Streaming"]
)
async def get_stream_status():
    """
    Get streaming service status.

    Returns:
        StreamStatusResponse: Streaming service status.
    """
    stats = stream_processor.get_stats()

    return StreamStatusResponse(
        is_running=app_state['streaming_enabled'],
        consumer_connected=kafka_consumer is not None and kafka_consumer.consumer is not None,
        transactions_processed=0,  # Would track actual count
        anomalies_detected=stats['anomalies_detected'],
        buffer_size=stats['buffer_size'],
        uptime_seconds=health_checker.check_liveness()['uptime_seconds']
    )


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.

    Returns:
        dict: API information.
    """
    return {
        "name": "Blockchain Anomaly Detection API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
        "health": "/health"
    }


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns:
        Response: Prometheus-formatted metrics.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions.

    Args:
        request (Request): The HTTP request.
        exc (HTTPException): The HTTP exception.

    Returns:
        JSONResponse: The JSON response with error details.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions.

    Args:
        request (Request): The HTTP request.
        exc (Exception): The exception.

    Returns:
        JSONResponse: The JSON response with error details.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api_server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

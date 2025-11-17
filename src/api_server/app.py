"""
FastAPI application for blockchain anomaly detection with authentication.

This module extends the base FastAPI application with:
- JWT-based authentication
- API key authentication
- Role-based access control
- Rate limiting
- Comprehensive audit logging

All prediction and model management endpoints now require authentication.
Admin endpoints require admin role.
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Any
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, status
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator

from src.api_server.models import (
    # Existing models
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
    SuccessResponse,
    # Authentication models
    RegisterRequest,
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    ChangePasswordRequest,
    UserResponse,
    UserListResponse,
    # API Key models
    APIKeyCreateRequest,
    APIKeyResponse,
    APIKeyInfo,
    APIKeyListResponse,
    # Audit Log models
    AuditLogEntry,
    AuditLogListResponse,
    AuditLogStatsResponse
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
from src.database import get_db, check_db_connection
from src.database.repositories import (
    TransactionRepository,
    AnomalyRepository,
    ModelRepository,
    ModelVersionRepository
)
from src.database.models import (
    Transaction as TransactionModel,
    Anomaly as AnomalyModel,
    Prediction as PredictionModel,
    SeverityEnum
)

# Authentication imports
from src.auth.jwt_handler import (
    jwt_handler,
    get_current_user,
    get_current_active_user,
    require_role,
    require_all_roles
)
from src.auth.user_manager import get_user_manager
from src.auth.api_key_manager import get_api_key_manager, get_user_from_api_key

# Middleware imports
from src.middleware.rate_limiter import RateLimitMiddleware, RateLimiter
from src.audit.audit_logger import get_audit_logger, AuditMiddleware

logger = get_logger(__name__)
security = HTTPBearer()

# Global state
stream_processor: Optional[StreamProcessor] = None
kafka_consumer: Optional[KafkaConsumerService] = None
cache_layer = None
rate_limiter = None
audit_logger = None
user_manager = None
api_key_manager = None

app_state = {
    'models': {},
    'active_model_id': None,
    'streaming_enabled': False,
    'cache_enabled': False,
    'authentication_enabled': True
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for handling startup and shutdown events.

    Initializes:
    - Authentication (user manager, API key manager, audit logger)
    - Rate limiter
    - Redis cache
    - Stream processor
    - Kafka consumer
    """
    # Startup
    logger.info("Starting Blockchain Anomaly Detection API with Authentication")

    # Initialize authentication components
    global user_manager, api_key_manager, audit_logger
    user_manager = get_user_manager()
    api_key_manager = get_api_key_manager()
    audit_logger = get_audit_logger()
    logger.info("Authentication components initialized")

    # Initialize rate limiter
    global rate_limiter
    redis_enabled = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'
    rate_limiter = RateLimiter(use_redis=redis_enabled)
    logger.info(f"Rate limiter initialized (Redis: {redis_enabled})")

    # Initialize Redis cache if enabled
    global cache_layer
    if redis_enabled:
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
    
    # Check database connection
    if check_db_connection():
        logger.info("Database connection verified")
    else:
        logger.warning("Database connection check failed - some features may not work")

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
    description="REST API for real-time blockchain transaction anomaly detection with authentication",
    version="2.0.0",
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

# Add Rate Limit middleware
@app.on_event("startup")
async def add_rate_limit_middleware():
    """Add rate limit middleware after rate_limiter is initialized."""
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    logger.info("Rate limit middleware added")

# Add Audit middleware
@app.on_event("startup")
async def add_audit_middleware():
    """Add audit middleware after audit_logger is initialized."""
    app.add_middleware(AuditMiddleware)
    logger.info("Audit middleware added")

# Initialize Prometheus instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/api/metrics")


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track HTTP requests with Prometheus metrics."""
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


# ============================================================================
# Authentication Endpoints
# ============================================================================

@app.post("/api/v1/auth/register", response_model=SuccessResponse, tags=["Authentication"])
async def register(request: RegisterRequest, http_request: Request):
    """
    Register a new user account.

    Creates a new user with email and password.
    Default role is 'user'.
    """
    try:
        # Create user
        user = user_manager.create_user(
            email=request.email,
            password=request.password,
            roles=["user"]
        )

        # Log audit event
        await audit_logger.log_auth_event(
            action="register",
            user_id=user.id,
            email=user.email,
            status="success",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown")
        )

        logger.info(f"New user registered: {user.email}")

        return SuccessResponse(
            success=True,
            message="User registered successfully",
            data={"user_id": user.id, "email": user.email}
        )

    except ValueError as e:
        # Log failed registration
        await audit_logger.log_auth_event(
            action="register",
            user_id=None,
            email=request.email,
            status="failure",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown"),
            failure_reason=str(e)
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/api/v1/auth/login", response_model=LoginResponse, tags=["Authentication"])
async def login(request: LoginRequest, http_request: Request):
    """
    Login with email and password.

    Returns access token and refresh token on successful authentication.
    """
    # Authenticate user
    user = user_manager.authenticate_user(
        email=request.email,
        password=request.password
    )

    if not user:
        # Log failed login
        await audit_logger.log_auth_event(
            action="login",
            user_id=None,
            email=request.email,
            status="failure",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown"),
            failure_reason="invalid_credentials"
        )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    # Create tokens
    access_token = jwt_handler.create_access_token(
        user_id=user.id,
        email=user.email,
        roles=user.roles
    )
    refresh_token = jwt_handler.create_refresh_token(user_id=user.id)

    # Log successful login
    await audit_logger.log_auth_event(
        action="login",
        user_id=user.id,
        email=user.email,
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown")
    )

    logger.info(f"User logged in: {user.email}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=30 * 60,  # 30 minutes
        user=user.to_dict()
    )


@app.post("/api/v1/auth/refresh", response_model=LoginResponse, tags=["Authentication"])
async def refresh_token(request: RefreshTokenRequest, http_request: Request):
    """
    Refresh access token using refresh token.

    Returns new access token and refresh token.
    """
    try:
        # Decode refresh token
        payload = jwt_handler.decode_token(request.refresh_token)

        # Validate it's a refresh token
        jwt_handler.validate_token_type(payload, "refresh")

        # Get user
        user_id = payload["sub"]
        user = user_manager.get_user_by_id(user_id)

        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        # Create new tokens
        access_token = jwt_handler.create_access_token(
            user_id=user.id,
            email=user.email,
            roles=user.roles
        )
        new_refresh_token = jwt_handler.create_refresh_token(user_id=user.id)

        # Log token refresh
        await audit_logger.log_auth_event(
            action="token_refresh",
            user_id=user.id,
            email=user.email,
            status="success",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown")
        )

        return LoginResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=30 * 60,
            user=user.to_dict()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@app.post("/api/v1/auth/change-password", response_model=SuccessResponse, tags=["Authentication"])
async def change_password(
    request: ChangePasswordRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Change user password.

    Requires current password for verification.
    """
    user_id = current_user["sub"]

    success = user_manager.change_password(
        user_id=user_id,
        old_password=request.old_password,
        new_password=request.new_password
    )

    if not success:
        await audit_logger.log_auth_event(
            action="change_password",
            user_id=user_id,
            email=current_user["email"],
            status="failure",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown"),
            failure_reason="invalid_old_password"
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid old password"
        )

    # Log successful password change
    await audit_logger.log_auth_event(
        action="change_password",
        user_id=user_id,
        email=current_user["email"],
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown")
    )

    logger.info(f"Password changed for user: {current_user['email']}")

    return SuccessResponse(
        success=True,
        message="Password changed successfully"
    )


@app.get("/api/v1/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user information.

    Returns user profile based on JWT token.
    """
    user = user_manager.get_user_by_id(current_user["sub"])

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        roles=user.roles,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None
    )


# ============================================================================
# User Management Endpoints (Admin Only)
# ============================================================================

@app.get("/api/v1/users", response_model=UserListResponse, tags=["User Management"])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    List all users (Admin only).

    Returns paginated list of users.
    """
    users = user_manager.list_users(skip=skip, limit=limit)
    total = user_manager.count_users()

    user_responses = [
        UserResponse(
            id=user["id"],
            email=user["email"],
            roles=user["roles"],
            is_active=user["is_active"],
            created_at=user["created_at"],
            last_login=user["last_login"]
        )
        for user in users
    ]

    return UserListResponse(
        users=user_responses,
        total=total
    )


@app.get("/api/v1/users/{user_id}", response_model=UserResponse, tags=["User Management"])
async def get_user(
    user_id: str,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Get user by ID (Admin only).
    """
    user = user_manager.get_user_by_id(user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        roles=user.roles,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None
    )


@app.delete("/api/v1/users/{user_id}", response_model=SuccessResponse, tags=["User Management"])
async def delete_user(
    user_id: str,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Delete user (Admin only).

    Cannot delete your own account.
    """
    if user_id == current_user["sub"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )

    success = user_manager.delete_user(user_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found"
        )

    # Log admin action
    await audit_logger.log_admin_event(
        action="delete_user",
        user_id=current_user["sub"],
        resource="user",
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown"),
        details={"target_user_id": user_id}
    )

    return SuccessResponse(
        success=True,
        message=f"User {user_id} deleted successfully"
    )


# ============================================================================
# API Key Management Endpoints
# ============================================================================

@app.post("/api/v1/api-keys", response_model=APIKeyResponse, tags=["API Keys"])
async def create_api_key(
    request: APIKeyCreateRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new API key.

    Returns the API key - save it securely as it won't be shown again!
    """
    api_key_obj, plain_key = api_key_manager.create_api_key(
        user_id=current_user["sub"],
        name=request.name,
        expires_days=request.expires_days
    )

    # Log API key creation
    await audit_logger.log_api_key_event(
        action="create",
        user_id=current_user["sub"],
        key_id=api_key_obj.id,
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown"),
        details={"name": request.name, "expires_days": request.expires_days}
    )

    logger.info(f"API key created for user {current_user['email']}: {request.name}")

    return APIKeyResponse(
        id=api_key_obj.id,
        name=api_key_obj.name,
        key=plain_key,  # Only shown once!
        prefix=api_key_obj.prefix,
        created_at=api_key_obj.created_at.isoformat(),
        expires_at=api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None
    )


@app.get("/api/v1/api-keys", response_model=APIKeyListResponse, tags=["API Keys"])
async def list_api_keys(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    List user's API keys.

    Returns list of API keys (without the actual key values).
    """
    keys = api_key_manager.list_user_api_keys(current_user["sub"])

    key_infos = [
        APIKeyInfo(
            id=key.id,
            name=key.name,
            prefix=key.prefix,
            is_active=key.is_active,
            created_at=key.created_at.isoformat(),
            last_used=key.last_used.isoformat() if key.last_used else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None
        )
        for key in keys
    ]

    return APIKeyListResponse(
        api_keys=key_infos,
        total=len(key_infos)
    )


@app.delete("/api/v1/api-keys/{key_id}", response_model=SuccessResponse, tags=["API Keys"])
async def revoke_api_key(
    key_id: str,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Revoke (delete) an API key.

    Once revoked, the key cannot be used for authentication.
    """
    # Verify key belongs to user
    key = api_key_manager.get_api_key(key_id)

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    if key.user_id != current_user["sub"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot revoke another user's API key"
        )

    success = api_key_manager.revoke_api_key(key_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Log API key revocation
    await audit_logger.log_api_key_event(
        action="revoke",
        user_id=current_user["sub"],
        key_id=key_id,
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown")
    )

    logger.info(f"API key revoked by user {current_user['email']}: {key_id}")

    return SuccessResponse(
        success=True,
        message="API key revoked successfully"
    )


# ============================================================================
# Audit Log Endpoints (Admin Only)
# ============================================================================

@app.get("/api/v1/audit-logs", response_model=AuditLogListResponse, tags=["Audit Logs"])
async def get_audit_logs(
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Get audit logs (Admin only).

    Supports filtering by event_type, user_id, and severity.
    """
    logs = audit_logger.get_logs(
        event_type=event_type,
        user_id=user_id,
        severity=severity,
        limit=limit
    )

    log_entries = [
        AuditLogEntry(
            id=log["id"],
            event_type=log["event_type"],
            user_id=log["user_id"],
            resource=log["resource"],
            action=log["action"],
            status=log["status"],
            ip_address=log["ip_address"],
            timestamp=log["timestamp"],
            severity=log["severity"],
            details=log.get("details")
        )
        for log in logs
    ]

    return AuditLogListResponse(
        logs=log_entries,
        total=len(log_entries)
    )


@app.get("/api/v1/audit-logs/stats", response_model=AuditLogStatsResponse, tags=["Audit Logs"])
async def get_audit_stats(current_user: Dict[str, Any] = Depends(require_role(["admin"]))):
    """
    Get audit log statistics (Admin only).

    Returns aggregated statistics about audit logs.
    """
    stats = audit_logger.get_stats()

    return AuditLogStatsResponse(
        total_logs=stats["total_logs"],
        event_types=stats["event_types"],
        severities=stats["severities"],
        failures=stats["failures"],
        oldest_log=stats["oldest_log"],
        newest_log=stats["newest_log"]
    )


# ============================================================================
# Health Check Endpoints (No Authentication Required)
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.

    Returns detailed health status including system metrics.
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
    """Readiness check endpoint."""
    readiness = health_checker.check_readiness()

    if not readiness['ready']:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=readiness
        )

    return readiness


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """Liveness check endpoint."""
    return health_checker.check_liveness()


# ============================================================================
# Prediction Endpoints (Authentication Required)
# ============================================================================

async def get_user_from_auth(
    http_authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key_user: Optional[Dict[str, Any]] = Depends(get_user_from_api_key)
) -> Dict[str, Any]:
    """
    Get user from either JWT token or API key.

    Supports both authentication methods.
    """
    # Try API key first
    if api_key_user:
        return api_key_user

    # Fall back to JWT
    if http_authorization:
        token = http_authorization.credentials
        return jwt_handler.decode_token(token)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(
    transaction: TransactionData,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Predict if a single transaction is anomalous.

    Requires authentication via JWT token or API key.
    Stores transaction and prediction in database.
    """
    try:
        start_time = time.time()

        # Create repositories
        transaction_repo = TransactionRepository(db)
        anomaly_repo = AnomalyRepository(db)

        # Check if transaction already exists
        existing_tx = transaction_repo.get_by_hash(transaction.hash)
        
        if not existing_tx:
            # Store transaction
            tx_model = TransactionModel(
                hash=transaction.hash,
                block_number=transaction.blockNumber or 0,
                timestamp=datetime.fromisoformat(transaction.timestamp.replace('Z', '+00:00')) if transaction.timestamp else datetime.utcnow(),
                from_address=transaction.from_address or "0x0",
                to_address=transaction.to_address,
                value=transaction.value,
                gas=transaction.gas,
                gas_price=transaction.gasPrice,
                nonce=getattr(transaction, 'nonce', 0) or 0
            )
            existing_tx = transaction_repo.create(tx_model)

        # Process transaction through stream processor
        result = stream_processor.process_transaction(transaction.model_dump(by_alias=True))
        stream_processor.flush()

        # Get anomalies
        anomalies = stream_processor.get_anomalies()
        is_anomaly = any(a['hash'] == transaction.hash for a in anomalies)

        # Determine severity and confidence
        severity_str = None
        confidence = 0.0
        anomaly_score = None
        
        if is_anomaly:
            anomaly_data = next(a for a in anomalies if a['hash'] == transaction.hash)
            severity_str = anomaly_data.get('severity', 'low')
            confidence = anomaly_data.get('confidence', 0.0)
            anomaly_score = anomaly_data.get('anomaly_score', 0.0)
            
            # Map severity string to enum
            severity_map = {
                'low': SeverityEnum.low,
                'medium': SeverityEnum.medium,
                'high': SeverityEnum.high,
                'critical': SeverityEnum.critical
            }
            severity_enum = severity_map.get(severity_str, SeverityEnum.low)
            
            # Store anomaly in database
            # Get active model version ID (simplified - should use actual model version)
            model_version_id = app_state.get('active_model_id', 'default-model-id')
            
            anomaly_model = AnomalyModel(
                transaction_id=existing_tx.id,
                model_id=model_version_id,
                anomaly_score=anomaly_score,
                severity=severity_enum,
                confidence=confidence,
                features_used=anomaly_data.get('features', {}),
                detected_at=datetime.utcnow()
            )
            anomaly_repo.create(anomaly_model)

        # Store prediction
        model_version_id = app_state.get('active_model_id', 'default-model-id')
        prediction_model = PredictionModel(
            transaction_id=existing_tx.id,
            model_version_id=model_version_id,
            user_id=current_user.get("sub"),
            is_anomaly=is_anomaly,
            confidence=confidence,
            anomaly_score=anomaly_score,
            response_time_ms=(time.time() - start_time) * 1000,
            created_at=datetime.utcnow()
        )
        db.add(prediction_model)
        db.commit()

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            hash=transaction.hash,
            is_anomaly=is_anomaly,
            severity=severity_str,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchTransactionRequest,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """
    Predict anomalies for a batch of transactions.

    Requires authentication via JWT token or API key.
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


# ============================================================================
# Model Management Endpoints (Admin Only)
# ============================================================================

@app.post("/api/v1/models/train", response_model=ModelTrainingResponse, tags=["Models"])
async def train_model(
    request: ModelTrainingRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """
    Train a new anomaly detection model (Admin only).
    """
    try:
        start_time = time.time()

        # Generate model ID
        model_id = f"model_{int(time.time())}"

        logger.info(f"Training model {model_id} with contamination={request.contamination}")

        # Save model info
        app_state['models'][model_id] = {
            'model_id': model_id,
            'model_type': request.model_type,
            'created_at': datetime.utcnow().isoformat(),
            'contamination': request.contamination,
            'status': 'active'
        }

        # Log admin action
        await audit_logger.log_admin_event(
            action="train_model",
            user_id=current_user["sub"],
            resource="model",
            status="success",
            ip_address=http_request.client.host,
            user_agent=http_request.headers.get("user-agent", "unknown"),
            details={"model_id": model_id, "model_type": request.model_type}
        )

        training_time = (time.time() - start_time) * 1000

        return ModelTrainingResponse(
            success=True,
            model_id=model_id,
            training_samples=0,
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


@app.get("/api/v1/models", response_model=ModelListResponse, tags=["Models"])
async def list_models(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """List all available models (Authentication required)."""
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


@app.get("/api/v1/models/{model_id}", response_model=ModelInfo, tags=["Models"])
async def get_model(
    model_id: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get information about a specific model (Authentication required)."""
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


@app.delete("/api/v1/models/{model_id}", response_model=SuccessResponse, tags=["Models"])
async def delete_model(
    model_id: str,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """Delete a model (Admin only)."""
    if model_id not in app_state['models']:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )

    del app_state['models'][model_id]

    # Log admin action
    await audit_logger.log_admin_event(
        action="delete_model",
        user_id=current_user["sub"],
        resource="model",
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown"),
        details={"model_id": model_id}
    )

    return SuccessResponse(
        success=True,
        message=f"Model {model_id} deleted successfully"
    )


# ============================================================================
# Anomaly Endpoints (Authentication Required)
# ============================================================================

@app.get("/api/v1/anomalies", response_model=AnomalyListResponse, tags=["Anomalies"])
async def get_anomalies(
    limit: Optional[int] = 100,
    severity: Optional[str] = None,
    skip: int = 0,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get detected anomalies from database (Authentication required)."""
    anomaly_repo = AnomalyRepository(db)
    
    if severity:
        # Get by severity
        severity_enum = SeverityEnum[severity.lower()]
        anomalies = anomaly_repo.get_by_severity(severity_enum, skip=skip, limit=limit)
    else:
        # Get all anomalies
        anomalies = anomaly_repo.get_all(skip=skip, limit=limit)

    anomaly_records = [
        AnomalyRecord(
            hash=a.transaction.hash,
            value=a.transaction.value,
            gas=a.transaction.gas,
            gasPrice=a.transaction.gas_price,
            from_address=a.transaction.from_address,
            to_address=a.transaction.to_address,
            timestamp=a.transaction.timestamp.isoformat() if a.transaction.timestamp else '',
            detected_at=a.detected_at.isoformat() if a.detected_at else '',
            severity=a.severity.value if isinstance(a.severity, SeverityEnum) else str(a.severity)
        )
        for a in anomalies
    ]

    return AnomalyListResponse(
        anomalies=anomaly_records,
        total_count=anomaly_repo.count()
    )


@app.get("/api/v1/transactions/{hash}", response_model=TransactionData, tags=["Transactions"])
async def get_transaction(
    hash: str,
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get transaction details by hash."""
    transaction_repo = TransactionRepository(db)
    tx = transaction_repo.get_by_hash(hash)
    
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found"
        )
    
    return TransactionData(
        hash=tx.hash,
        value=tx.value,
        gas=tx.gas,
        gasPrice=tx.gas_price,
        from_address=tx.from_address,
        to_address=tx.to_address,
        blockNumber=tx.block_number,
        timestamp=tx.timestamp.isoformat() if tx.timestamp else None
    )


@app.delete("/api/v1/anomalies", response_model=SuccessResponse, tags=["Anomalies"])
async def clear_anomalies(
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_role(["admin"]))
):
    """Clear the anomaly buffer (Admin only)."""
    stream_processor.clear_anomaly_buffer()

    # Log admin action
    await audit_logger.log_admin_event(
        action="clear_anomalies",
        user_id=current_user["sub"],
        resource="anomaly_buffer",
        status="success",
        ip_address=http_request.client.host,
        user_agent=http_request.headers.get("user-agent", "unknown")
    )

    return SuccessResponse(
        success=True,
        message="Anomaly buffer cleared successfully"
    )


# ============================================================================
# Streaming Endpoints (Authentication Required)
# ============================================================================

@app.get("/api/v1/stream/status", response_model=StreamStatusResponse, tags=["Streaming"])
async def get_stream_status(current_user: Dict[str, Any] = Depends(get_current_active_user)):
    """Get streaming service status (Authentication required)."""
    stats = stream_processor.get_stats()

    return StreamStatusResponse(
        is_running=app_state['streaming_enabled'],
        consumer_connected=kafka_consumer is not None and kafka_consumer.consumer is not None,
        transactions_processed=0,
        anomalies_detected=stats['anomalies_detected'],
        buffer_size=stats['buffer_size'],
        uptime_seconds=health_checker.check_liveness()['uptime_seconds']
    )


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Blockchain Anomaly Detection API",
        "version": "2.0.0",
        "status": "running",
        "authentication": "enabled",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
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
        "src.api_server.app_authenticated:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

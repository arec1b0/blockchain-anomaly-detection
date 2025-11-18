"""
Pydantic models for API request/response validation.

This module defines the Pydantic models used for validating the data
in API requests and responses. These models ensure that the data
conforms to the expected structure and types.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TransactionData(BaseModel):
    """
    Represents a single blockchain transaction.

    Attributes:
        hash (str): The transaction hash.
        value (float): The value of the transaction.
        gas (float): The gas used by the transaction.
        gasPrice (float): The price of gas for the transaction.
        from_address (Optional[str]): The sender's address.
        to_address (Optional[str]): The recipient's address.
        blockNumber (Optional[int]): The block number of the transaction.
        timestamp (Optional[str]): The timestamp of the transaction.
    """
    hash: str = Field(..., description="Transaction hash")
    value: float = Field(..., ge=0, description="Transaction value")
    gas: float = Field(..., ge=0, description="Gas used")
    gasPrice: float = Field(..., ge=0, description="Gas price")
    from_address: Optional[str] = Field(None, alias="from", description="Sender address")
    to_address: Optional[str] = Field(None, alias="to", description="Recipient address")
    blockNumber: Optional[int] = Field(None, description="Block number")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")

    class Config:
        populate_by_name = True


class BatchTransactionRequest(BaseModel):
    """
    Represents a batch of transactions for prediction.

    Attributes:
        transactions (List[TransactionData]): A list of transactions to be processed.
    """
    transactions: List[TransactionData] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """
    Represents the anomaly prediction for a single transaction.

    Attributes:
        hash (str): The transaction hash.
        is_anomaly (bool): Whether the transaction is an anomaly.
        anomaly_score (Optional[float]): The anomaly score of the transaction.
        severity (Optional[str]): The severity of the anomaly.
        timestamp (str): The timestamp of the prediction.
    """
    hash: str
    is_anomaly: bool
    anomaly_score: Optional[float] = None
    severity: Optional[str] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """
    Represents the response for a batch prediction request.

    Attributes:
        predictions (List[PredictionResponse]): A list of prediction results.
        total_processed (int): The total number of transactions processed.
        anomalies_detected (int): The number of anomalies detected.
        processing_time_ms (float): The time taken to process the batch in milliseconds.
    """
    predictions: List[PredictionResponse]
    total_processed: int
    anomalies_detected: int
    processing_time_ms: float


class ModelTrainingRequest(BaseModel):
    """
    Represents a request to train a new model.

    Attributes:
        data_source (str): The path to the training data or data source.
        contamination (float): The expected contamination rate.
        model_type (str): The type of model to train.
        parameters (Optional[Dict[str, Any]]): Additional model parameters.
    """
    data_source: str = Field(..., description="Path to training data or data source")
    contamination: float = Field(0.01, ge=0.001, le=0.5, description="Expected contamination rate")
    model_type: str = Field("isolation_forest", description="Type of model to train")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")


class ModelTrainingResponse(BaseModel):
    """
    Represents the response for a model training request.

    Attributes:
        success (bool): Whether the training was successful.
        model_id (str): The ID of the trained model.
        training_samples (int): The number of samples used for training.
        contamination (float): The contamination rate used for training.
        training_time_ms (float): The time taken to train the model in milliseconds.
        metrics (Optional[Dict[str, Any]]): The metrics of the trained model.
        message (str): A message about the training status.
    """
    success: bool
    model_id: str
    training_samples: int
    contamination: float
    training_time_ms: float
    metrics: Optional[Dict[str, Any]] = None
    message: str


class ModelInfo(BaseModel):
    """
    Represents information about a model.

    Attributes:
        model_id (str): The ID of the model.
        model_type (str): The type of the model.
        created_at (str): The timestamp of when the model was created.
        last_updated (str): The timestamp of when the model was last updated.
        version (str): The version of the model.
        contamination (float): The contamination rate of the model.
        training_samples (int): The number of samples used to train the model.
        is_active (bool): Whether the model is active.
    """
    model_id: str
    model_type: str
    created_at: str
    last_updated: str
    version: str
    contamination: float
    training_samples: int
    is_active: bool


class ModelListResponse(BaseModel):
    """
    Represents a list of available models.

    Attributes:
        models (List[ModelInfo]): A list of model information.
        total_models (int): The total number of models.
    """
    models: List[ModelInfo]
    total_models: int


class ModelStatus(str, Enum):
    """
    Represents the status of a model.
    """
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"


class ModelUpdateRequest(BaseModel):
    """
    Represents a request to update a model.

    Attributes:
        status (Optional[ModelStatus]): The new status of the model.
        is_active (Optional[bool]): Whether the model should be active.
    """
    status: Optional[ModelStatus] = None
    is_active: Optional[bool] = None


class AnomalyRecord(BaseModel):
    """
    Represents a detected anomaly.

    Attributes:
        hash (str): The transaction hash.
        value (float): The value of the transaction.
        gas (float): The gas used by the transaction.
        gasPrice (float): The price of gas for the transaction.
        from_address (Optional[str]): The sender's address.
        to_address (Optional[str]): The recipient's address.
        timestamp (str): The timestamp of the transaction.
        detected_at (str): The timestamp of when the anomaly was detected.
        severity (str): The severity of the anomaly.
    """
    hash: str
    value: float
    gas: float
    gasPrice: float
    from_address: Optional[str] = Field(None, alias="from")
    to_address: Optional[str] = Field(None, alias="to")
    timestamp: str
    detected_at: str
    severity: str

    class Config:
        populate_by_name = True


class AnomalyListResponse(BaseModel):
    """
    Represents a list of detected anomalies.

    Attributes:
        anomalies (List[AnomalyRecord]): A list of anomaly records.
        total_count (int): The total number of anomalies.
        time_range (Optional[Dict[str, str]]): The time range of the anomalies.
    """
    anomalies: List[AnomalyRecord]
    total_count: int
    time_range: Optional[Dict[str, str]] = None


class StreamStatusResponse(BaseModel):
    """
    Represents the status of the streaming service.

    Attributes:
        is_running (bool): Whether the streaming service is running.
        consumer_connected (bool): Whether the Kafka consumer is connected.
        transactions_processed (int): The number of transactions processed.
        anomalies_detected (int): The number of anomalies detected.
        buffer_size (int): The size of the transaction buffer.
        uptime_seconds (float): The uptime of the service in seconds.
    """
    is_running: bool
    consumer_connected: bool
    transactions_processed: int
    anomalies_detected: int
    buffer_size: int
    uptime_seconds: float


class HealthCheckResponse(BaseModel):
    """
    Represents the response for a health check.

    Attributes:
        status (str): The status of the service.
        timestamp (str): The timestamp of the health check.
        uptime_seconds (float): The uptime of the service in seconds.
        checks (Dict[str, Any]): A dictionary of health check results.
    """
    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]


class ErrorResponse(BaseModel):
    """
    Represents an error response.

    Attributes:
        error (str): The error message.
        detail (Optional[str]): The details of the error.
        timestamp (str): The timestamp of the error.
    """
    error: str
    detail: Optional[str] = None
    timestamp: str


class SuccessResponse(BaseModel):
    """
    Represents a generic success response.

    Attributes:
        success (bool): Whether the request was successful.
        message (str): A message about the success status.
        data (Optional[Dict[str, Any]]): Additional data.
    """
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# Authentication Models (Phase 1)
# ============================================================================

class RegisterRequest(BaseModel):
    """
    Represents a user registration request.

    Attributes:
        email (str): User email address (must be valid email)
        password (str): User password (min 8 characters)
        confirm_password (str): Password confirmation
    """
    email: str = Field(..., description="User email address", min_length=3)
    password: str = Field(..., description="User password", min_length=8)
    confirm_password: str = Field(..., description="Password confirmation")

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        """Validate email format."""
        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v.lower()

    @field_validator('confirm_password')
    @classmethod
    def passwords_match(cls, v, info):
        """Validate that passwords match."""
        if 'password' in info.data and v != info.data['password']:
            raise ValueError('Passwords do not match')
        return v


class LoginRequest(BaseModel):
    """
    Represents a user login request.

    Attributes:
        email (str): User email address
        password (str): User password
    """
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class LoginResponse(BaseModel):
    """
    Represents the response for a successful login.

    Attributes:
        access_token (str): JWT access token
        refresh_token (str): JWT refresh token
        token_type (str): Token type (always "bearer")
        expires_in (int): Token expiration time in seconds
        user (Dict[str, Any]): User information
    """
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Optional[Dict[str, Any]] = None


class RefreshTokenRequest(BaseModel):
    """
    Represents a token refresh request.

    Attributes:
        refresh_token (str): The refresh token
    """
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """
    Represents a password change request.

    Attributes:
        old_password (str): Current password
        new_password (str): New password
        confirm_password (str): New password confirmation
    """
    old_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password", min_length=8)
    confirm_password: str = Field(..., description="New password confirmation")

    @field_validator('confirm_password')
    @classmethod
    def passwords_match(cls, v, info):
        """Validate that passwords match."""
        if 'new_password' in info.data and v != info.data['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UserResponse(BaseModel):
    """
    Represents user information (without sensitive data).

    Attributes:
        id (str): User ID
        email (str): User email
        roles (List[str]): User roles
        is_active (bool): Whether user is active
        created_at (str): When user was created
        last_login (Optional[str]): Last login timestamp
    """
    id: str
    email: str
    roles: List[str]
    is_active: bool
    created_at: str
    last_login: Optional[str] = None


class UserListResponse(BaseModel):
    """
    Represents a list of users.

    Attributes:
        users (List[UserResponse]): List of users
        total (int): Total number of users
    """
    users: List[UserResponse]
    total: int


# ============================================================================
# API Key Models (Phase 1)
# ============================================================================

class APIKeyCreateRequest(BaseModel):
    """
    Represents a request to create an API key.

    Attributes:
        name (str): Descriptive name for the API key
        expires_days (Optional[int]): Days until expiration (None = never)
    """
    name: str = Field(..., description="Descriptive name for the API key", min_length=3, max_length=100)
    expires_days: Optional[int] = Field(None, description="Days until expiration", ge=1, le=365)


class APIKeyResponse(BaseModel):
    """
    Represents an API key (returned only on creation).

    Attributes:
        id (str): API key ID
        name (str): API key name
        key (str): The actual API key (only shown once!)
        prefix (str): Key prefix for identification
        created_at (str): When key was created
        expires_at (Optional[str]): When key expires
    """
    id: str
    name: str
    key: str  # Only included on creation!
    prefix: str
    created_at: str
    expires_at: Optional[str] = None


class APIKeyInfo(BaseModel):
    """
    Represents API key information (without the actual key).

    Attributes:
        id (str): API key ID
        name (str): API key name
        prefix (str): Key prefix
        is_active (bool): Whether key is active
        created_at (str): When key was created
        last_used (Optional[str]): When key was last used
        expires_at (Optional[str]): When key expires
    """
    id: str
    name: str
    prefix: str
    is_active: bool
    created_at: str
    last_used: Optional[str] = None
    expires_at: Optional[str] = None


class APIKeyListResponse(BaseModel):
    """
    Represents a list of API keys.

    Attributes:
        api_keys (List[APIKeyInfo]): List of API keys
        total (int): Total number of keys
    """
    api_keys: List[APIKeyInfo]
    total: int


# ============================================================================
# Audit Log Models (Phase 1)
# ============================================================================

class AuditLogEntry(BaseModel):
    """
    Represents an audit log entry.

    Attributes:
        id (str): Log entry ID
        event_type (str): Type of event
        user_id (Optional[str]): User who performed action
        resource (str): Resource affected
        action (str): Action performed
        status (str): Result status
        ip_address (str): Client IP address
        timestamp (str): When event occurred
        severity (str): Event severity
        details (Optional[Dict[str, Any]]): Additional details
    """
    id: str
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    status: str
    ip_address: str
    timestamp: str
    severity: str
    details: Optional[Dict[str, Any]] = None


class AuditLogListResponse(BaseModel):
    """
    Represents a list of audit logs.

    Attributes:
        logs (List[AuditLogEntry]): List of audit log entries
        total (int): Total number of logs
    """
    logs: List[AuditLogEntry]
    total: int


class AuditLogStatsResponse(BaseModel):
    """
    Represents audit log statistics.

    Attributes:
        total_logs (int): Total number of logs
        event_types (Dict[str, int]): Count by event type
        severities (Dict[str, int]): Count by severity
        failures (int): Number of failures
        oldest_log (Optional[str]): Oldest log timestamp
        newest_log (Optional[str]): Newest log timestamp
    """
    total_logs: int
    event_types: Dict[str, int]
    severities: Dict[str, int]
    failures: int
    oldest_log: Optional[str] = None
    newest_log: Optional[str] = None


# ============================================================================
# ML Model Lifecycle Models (Phase 3)
# ============================================================================

class ModelDeploymentRequest(BaseModel):
    """
    Represents a request to deploy a model version.

    Attributes:
        model_version_id (str): ID of model version to deploy
        strategy (str): Deployment strategy (shadow, canary, blue_green, full)
        initial_traffic (Optional[float]): Initial traffic percentage (0-100)
    """
    model_version_id: str
    strategy: str = Field("shadow", description="Deployment strategy")
    initial_traffic: Optional[float] = Field(None, description="Initial traffic percentage", ge=0, le=100)


class ModelDeploymentResponse(BaseModel):
    """
    Represents the response for a model deployment.

    Attributes:
        success (bool): Whether deployment was successful
        model_version_id (str): ID of deployed model version
        strategy (str): Deployment strategy used
        traffic_percentage (float): Current traffic percentage
        deployed_at (str): Deployment timestamp
        message (str): Status message
    """
    success: bool
    model_version_id: str
    strategy: str
    traffic_percentage: float
    deployed_at: str
    message: str


class ModelTrafficUpdateRequest(BaseModel):
    """
    Represents a request to update model traffic percentage.

    Attributes:
        model_version_id (str): ID of model version
        traffic_percentage (float): New traffic percentage (0-100)
    """
    model_version_id: str
    traffic_percentage: float = Field(..., ge=0, le=100, description="Traffic percentage")


class ModelRollbackRequest(BaseModel):
    """
    Represents a request to rollback a model deployment.

    Attributes:
        model_version_id (str): ID of model version to rollback
        restore_previous (bool): Whether to restore previous version
    """
    model_version_id: str
    restore_previous: bool = Field(True, description="Restore previous deployed version")


class DriftDetectionRequest(BaseModel):
    """
    Represents a request for drift detection.

    Attributes:
        model_version_id (str): ID of model version to check
        drift_threshold (float): Threshold for drift detection (0.0-1.0)
        reference_window_days (Optional[int]): Days for reference period
        detection_window_days (Optional[int]): Days for detection period
    """
    model_version_id: str
    drift_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Drift threshold")
    reference_window_days: Optional[int] = Field(30, ge=1, le=90)
    detection_window_days: Optional[int] = Field(7, ge=1, le=30)


class DriftDetectionResponse(BaseModel):
    """
    Represents the response for drift detection.

    Attributes:
        model_version_id (str): ID of model version
        drift_detected (bool): Whether drift was detected
        drift_threshold (float): Threshold used
        feature_drift (Dict[str, Any]): Feature drift details
        concept_drift (Dict[str, Any]): Concept drift details
        performance_drift (Dict[str, Any]): Performance drift details
        recommendation (str): Recommended action
        reference_period (Dict[str, Any]): Reference period info
        detection_period (Dict[str, Any]): Detection period info
    """
    model_version_id: str
    drift_detected: bool
    drift_threshold: float
    feature_drift: Dict[str, Any]
    concept_drift: Dict[str, Any]
    performance_drift: Dict[str, Any]
    recommendation: str
    reference_period: Dict[str, Any]
    detection_period: Dict[str, Any]


class ModelComparisonRequest(BaseModel):
    """
    Represents a request to compare two model versions.

    Attributes:
        model_version_id_a (str): ID of first model version
        model_version_id_b (str): ID of second model version
        time_window_hours (int): Time window for comparison
    """
    model_version_id_a: str
    model_version_id_b: str
    time_window_hours: int = Field(24, ge=1, le=168, description="Time window in hours")


class ModelComparisonResponse(BaseModel):
    """
    Represents the response for model comparison.

    Attributes:
        model_a (Dict[str, Any]): Metrics for model A
        model_b (Dict[str, Any]): Metrics for model B
        winner (str): Which model performed better
        time_window_hours (int): Time window used
    """
    model_a: Dict[str, Any]
    model_b: Dict[str, Any]
    winner: str
    time_window_hours: int


class ModelRetrainingRequest(BaseModel):
    """
    Represents a request to retrain a model.

    Attributes:
        model_name (str): Name of model to retrain
        start_date (Optional[str]): Training data start date (ISO format)
        end_date (Optional[str]): Training data end date (ISO format)
        hyperparameter_tuning (bool): Whether to tune hyperparameters
        contamination (Optional[float]): Expected anomaly proportion
    """
    model_name: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    hyperparameter_tuning: bool = Field(True, description="Enable hyperparameter tuning")
    contamination: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModelRetrainingResponse(BaseModel):
    """
    Represents the response for model retraining.

    Attributes:
        success (bool): Whether retraining was successful
        model_version_id (str): ID of new model version
        model_name (str): Name of model
        version (str): Version string
        metrics (Dict[str, Any]): Training metrics
        training_duration_seconds (float): Training time
        message (str): Status message
    """
    success: bool
    model_version_id: str
    model_name: str
    version: str
    metrics: Dict[str, Any]
    training_duration_seconds: float
    message: str


class ModelVersionInfo(BaseModel):
    """
    Represents detailed information about a model version.

    Attributes:
        id (str): Model version ID
        model_id (str): Parent model ID
        version (str): Version string
        is_deployed (bool): Deployment status
        traffic_percentage (float): Current traffic percentage
        deployed_at (Optional[str]): Deployment timestamp
        created_at (str): Creation timestamp
        metrics (Dict[str, Any]): Model metrics
        hyperparameters (Dict[str, Any]): Hyperparameters used
        training_duration_seconds (Optional[float]): Training time
    """
    id: str
    model_id: str
    version: str
    is_deployed: bool
    traffic_percentage: float
    deployed_at: Optional[str]
    created_at: str
    metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    training_duration_seconds: Optional[float] = None


class DeploymentStatusResponse(BaseModel):
    """
    Represents current deployment status for a model.

    Attributes:
        model_id (str): Model ID
        deployed_versions (List[Dict[str, Any]]): List of deployed versions
        total_versions (int): Total number of deployed versions
        traffic_allocated (float): Total traffic percentage allocated
    """
    model_id: str
    deployed_versions: List[Dict[str, Any]]
    total_versions: int
    traffic_allocated: float


class CacheStatsResponse(BaseModel):
    """
    Represents model cache statistics.

    Attributes:
        enabled (bool): Whether caching is enabled
        cached_models (int): Number of models in cache
        ttl_hours (Optional[int]): Cache TTL in hours
        models (Optional[List[Dict[str, Any]]]): Cached model details
    """
    enabled: bool
    cached_models: int
    ttl_hours: Optional[int] = None
    models: Optional[List[Dict[str, Any]]] = None

"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TransactionData(BaseModel):
    """Transaction data model."""

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
    """Batch transaction prediction request."""

    transactions: List[TransactionData] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Anomaly prediction response."""

    hash: str
    is_anomaly: bool
    anomaly_score: Optional[float] = None
    severity: Optional[str] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    total_processed: int
    anomalies_detected: int
    processing_time_ms: float


class ModelTrainingRequest(BaseModel):
    """Model training request."""

    data_source: str = Field(..., description="Path to training data or data source")
    contamination: float = Field(0.01, ge=0.001, le=0.5, description="Expected contamination rate")
    model_type: str = Field("isolation_forest", description="Type of model to train")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")


class ModelTrainingResponse(BaseModel):
    """Model training response."""

    success: bool
    model_id: str
    training_samples: int
    contamination: float
    training_time_ms: float
    metrics: Optional[Dict[str, Any]] = None
    message: str


class ModelInfo(BaseModel):
    """Model information."""

    model_id: str
    model_type: str
    created_at: str
    last_updated: str
    version: str
    contamination: float
    training_samples: int
    is_active: bool


class ModelListResponse(BaseModel):
    """List of available models."""

    models: List[ModelInfo]
    total_models: int


class ModelStatus(str, Enum):
    """Model status enum."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"


class ModelUpdateRequest(BaseModel):
    """Model update request."""

    status: Optional[ModelStatus] = None
    is_active: Optional[bool] = None


class AnomalyRecord(BaseModel):
    """Anomaly record model."""

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
    """List of detected anomalies."""

    anomalies: List[AnomalyRecord]
    total_count: int
    time_range: Optional[Dict[str, str]] = None


class StreamStatusResponse(BaseModel):
    """Streaming service status."""

    is_running: bool
    consumer_connected: bool
    transactions_processed: int
    anomalies_detected: int
    buffer_size: int
    uptime_seconds: float


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: Optional[str] = None
    timestamp: str


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

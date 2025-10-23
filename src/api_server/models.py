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

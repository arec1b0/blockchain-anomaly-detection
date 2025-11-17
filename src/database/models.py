"""
SQLAlchemy database models.

This module defines all database models for the blockchain anomaly detection system.
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, Index, JSON, BigInteger, Enum as SQLEnum
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import enum

Base = declarative_base()


class User(Base):
    """User authentication and profile."""

    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    roles = Column(JSONB, default=["user"])  # ["user", "admin", "analyst"]
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user")

    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
    )


class APIKey(Base):
    """API keys for service-to-service authentication."""

    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    hashed_key = Column(String(255), nullable=False)
    prefix = Column(String(8), nullable=False, index=True)  # First 8 chars for lookup
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index('idx_apikey_prefix_active', 'prefix', 'is_active'),
    )


class Transaction(Base):
    """Blockchain transactions (historical data)."""

    __tablename__ = "transactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    hash = Column(String(66), unique=True, nullable=False, index=True)  # 0x + 64 hex chars
    block_number = Column(BigInteger, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    from_address = Column(String(42), nullable=False, index=True)
    to_address = Column(String(42), nullable=True, index=True)
    value = Column(Float, nullable=False)
    gas = Column(Float, nullable=False)
    gas_price = Column(Float, nullable=False)
    gas_used = Column(Float, nullable=True)
    nonce = Column(Integer, nullable=False)
    input_data = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    anomalies = relationship("Anomaly", back_populates="transaction")
    predictions = relationship("Prediction", back_populates="transaction")

    __table_args__ = (
        Index('idx_transaction_timestamp_value', 'timestamp', 'value'),
        Index('idx_transaction_from_to', 'from_address', 'to_address'),
    )


class SeverityEnum(enum.Enum):
    """Anomaly severity levels."""

    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Anomaly(Base):
    """Detected anomalies."""

    __tablename__ = "anomalies"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(BigInteger, ForeignKey("transactions.id"), nullable=False)
    model_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)

    # Anomaly details
    anomaly_score = Column(Float, nullable=False)
    severity = Column(SQLEnum(SeverityEnum), nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    # Explanation
    features_used = Column(JSONB, nullable=True)  # {"value": 1000, "gas": 50000}
    explanation = Column(Text, nullable=True)

    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow, index=True)
    reviewed = Column(Boolean, default=False)
    reviewed_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    false_positive = Column(Boolean, nullable=True)

    # Relationships
    transaction = relationship("Transaction", back_populates="anomalies")
    model_version = relationship("ModelVersion")

    __table_args__ = (
        Index('idx_anomaly_severity_detected', 'severity', 'detected_at'),
        Index('idx_anomaly_reviewed', 'reviewed', 'detected_at'),
    )


class Model(Base):
    """ML model metadata."""

    __tablename__ = "models"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    model_type = Column(String(50), nullable=False)  # "isolation_forest", "arima", etc.
    description = Column(Text, nullable=True)

    # Status
    is_active = Column(Boolean, default=True)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    versions = relationship("ModelVersion", back_populates="model", cascade="all, delete-orphan")


class ModelVersion(Base):
    """ML model versions (supports A/B testing)."""

    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey("models.id"), nullable=False)
    version = Column(String(50), nullable=False)  # "1.0.0", "1.0.1", etc.

    # Model artifacts
    storage_path = Column(String(500), nullable=False)  # S3/GCS path
    checksum = Column(String(64), nullable=False)  # SHA256

    # Training metadata
    training_dataset_size = Column(Integer, nullable=True)
    training_duration_seconds = Column(Float, nullable=True)
    hyperparameters = Column(JSONB, nullable=True)

    # Performance metrics
    metrics = Column(JSONB, nullable=True)  # {"accuracy": 0.95, "f1": 0.92, ...}

    # Deployment
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime, nullable=True)
    traffic_percentage = Column(Float, default=0.0)  # For A/B testing (0-100)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Relationships
    model = relationship("Model", back_populates="versions")

    __table_args__ = (
        Index('idx_model_version_deployed', 'model_id', 'is_deployed'),
    )


class Prediction(Base):
    """Prediction history for audit and analysis."""

    __tablename__ = "predictions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id = Column(BigInteger, ForeignKey("transactions.id"), nullable=True)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)

    # Prediction
    is_anomaly = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=True)

    # Request metadata
    request_id = Column(String(36), nullable=True, index=True)
    response_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    transaction = relationship("Transaction", back_populates="predictions")
    user = relationship("User", back_populates="predictions")

    __table_args__ = (
        Index('idx_prediction_created_at', 'created_at'),
        Index('idx_prediction_user_created', 'user_id', 'created_at'),
    )


class AuditLog(Base):
    """Security audit logs."""

    __tablename__ = "audit_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)

    # Event details
    event_type = Column(String(50), nullable=False, index=True)  # "auth", "data", "admin"
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    resource = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # "create", "read", "update", "delete"
    status = Column(String(20), nullable=False)  # "success", "failure", "denied"

    # Context
    ip_address = Column(String(45), nullable=False)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    details = Column(JSONB, nullable=True)

    # Severity
    severity = Column(String(20), nullable=False, default="info")  # "info", "warning", "error", "critical"

    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('idx_audit_event_timestamp', 'event_type', 'timestamp'),
        Index('idx_audit_severity_timestamp', 'severity', 'timestamp'),
    )


class SystemMetric(Base):
    """System performance metrics over time."""

    __tablename__ = "system_metrics"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    tags = Column(JSONB, nullable=True)  # {"pod": "api-1", "endpoint": "/predict"}
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index('idx_system_metric_name_timestamp', 'metric_name', 'timestamp'),
    )


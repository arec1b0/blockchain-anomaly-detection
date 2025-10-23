"""
Tests for FastAPI server endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import os
from datetime import datetime, timedelta

# Set environment variables before importing app
os.environ['KAFKA_ENABLED'] = 'false'

from src.api_server.app import app


@pytest.fixture
def client():
    """Create test client."""
    with patch('src.api_server.app.stream_processor', MagicMock()) as mock_stream_processor:
        yield TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_liveness_check(self, client):
        """Test liveness endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data['alive'] is True
        assert 'uptime_seconds' in data

    @patch('src.api_server.app.health_checker.check_readiness')
    def test_readiness_check(self, mock_check_readiness, client):
        """Test readiness endpoint."""
        mock_check_readiness.return_value = {'ready': True}
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert 'ready' in data

    def test_health_check(self, client):
        """Test comprehensive health check."""
        response = client.get("/health")
        assert response.status_code in [200, 503]  # May be unhealthy in test environment
        data = response.json()
        assert 'status' in data
        assert 'checks' in data


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data['name'] == "Blockchain Anomaly Detection API"
        assert data['version'] == "1.0.0"
        assert data['status'] == "running"


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_single_transaction(self, client):
        """Test single transaction prediction."""
        transaction = {
            "hash": "0x123",
            "value": 100.0,
            "gas": 21000.0,
            "gasPrice": 20.0,
            "from": "0xabc",
            "to": "0xdef"
        }

        response = client.post("/api/v1/predict", json=transaction)
        assert response.status_code == 200

        data = response.json()
        assert data['hash'] == "0x123"
        assert 'is_anomaly' in data
        assert isinstance(data['is_anomaly'], bool)

    def test_predict_single_invalid_data(self, client):
        """Test prediction with invalid data."""
        transaction = {
            "hash": "0x123"
            # Missing required fields
        }

        response = client.post("/api/v1/predict", json=transaction)
        assert response.status_code == 422  # Validation error

    def test_predict_batch(self, client):
        """Test batch prediction."""
        transactions = {
            "transactions": [
                {
                    "hash": f"0x{i}",
                    "value": 100.0,
                    "gas": 21000.0,
                    "gasPrice": 20.0
                }
                for i in range(5)
            ]
        }

        response = client.post("/api/v1/predict/batch", json=transactions)
        assert response.status_code == 200

        data = response.json()
        assert data['total_processed'] == 5
        assert len(data['predictions']) == 5
        assert 'anomalies_detected' in data

    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty list."""
        transactions = {
            "transactions": []
        }

        response = client.post("/api/v1/predict/batch", json=transactions)
        assert response.status_code == 422  # Validation error

    def test_predict_batch_too_large(self, client):
        """Test batch prediction with too many transactions."""
        transactions = {
            "transactions": [
                {
                    "hash": f"0x{i}",
                    "value": 100.0,
                    "gas": 21000.0,
                    "gasPrice": 20.0
                }
                for i in range(1001)  # Exceeds max of 1000
            ]
        }

        response = client.post("/api/v1/predict/batch", json=transactions)
        assert response.status_code == 422  # Validation error


class TestModelEndpoints:
    """Test model management endpoints."""

    def test_list_models_empty(self, client):
        """Test listing models when none exist."""
        response = client.get("/api/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert 'models' in data
        assert isinstance(data['models'], list)
        assert data['total_models'] >= 0

    def test_train_model(self, client):
        """Test model training."""
        training_request = {
            "data_source": "/path/to/data",
            "contamination": 0.01,
            "model_type": "isolation_forest"
        }

        response = client.post("/api/v1/models/train", json=training_request)
        assert response.status_code == 200

        data = response.json()
        assert data['success'] is True
        assert 'model_id' in data
        assert data['contamination'] == 0.01

    def test_train_model_invalid_contamination(self, client):
        """Test model training with invalid contamination."""
        training_request = {
            "data_source": "/path/to/data",
            "contamination": 1.5,  # Invalid, must be < 1
            "model_type": "isolation_forest"
        }

        response = client.post("/api/v1/models/train", json=training_request)
        assert response.status_code == 422  # Validation error

    def test_get_model_not_found(self, client):
        """Test getting a non-existent model."""
        response = client.get("/api/v1/models/nonexistent")
        assert response.status_code == 404

    def test_delete_model_not_found(self, client):
        """Test deleting a non-existent model."""
        response = client.delete("/api/v1/models/nonexistent")
        assert response.status_code == 404


class TestAnomalyEndpoints:
    """Test anomaly endpoints."""

    def test_get_anomalies(self, client):
        """Test getting anomalies."""
        response = client.get("/api/v1/anomalies")
        assert response.status_code == 200

        data = response.json()
        assert 'anomalies' in data
        assert isinstance(data['anomalies'], list)
        assert 'total_count' in data

    def test_get_anomalies_with_limit(self, client):
        """Test getting anomalies with limit."""
        response = client.get("/api/v1/anomalies?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert len(data['anomalies']) <= 10

    def test_get_anomalies_with_severity(self, client):
        """Test getting anomalies filtered by severity."""
        response = client.get("/api/v1/anomalies?severity=high")
        assert response.status_code == 200

        data = response.json()
        # Verify all returned anomalies have high severity
        for anomaly in data['anomalies']:
            assert anomaly['severity'] == 'high'

    def test_clear_anomalies(self, client):
        """Test clearing anomaly buffer."""
        response = client.delete("/api/v1/anomalies")
        assert response.status_code == 200

        data = response.json()
        assert data['success'] is True


class TestStreamEndpoints:
    """Test streaming endpoints."""

    def test_stream_status(self, client):
        """Test getting stream status."""
        response = client.get("/api/v1/stream/status")
        assert response.status_code == 200

        data = response.json()
        assert 'is_running' in data
        assert 'consumer_connected' in data
        assert 'anomalies_detected' in data


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert 'text/plain' in response.headers['content-type']


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_endpoint(self, client):
        """Test accessing invalid endpoint."""
        response = client.get("/api/v1/invalid")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test using wrong HTTP method."""
        response = client.post("/health/live")
        assert response.status_code == 405

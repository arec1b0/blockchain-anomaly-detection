# API Documentation

## Overview

The Blockchain Anomaly Detection API provides a comprehensive REST API for real-time blockchain transaction anomaly detection. Built with FastAPI, it offers high-performance endpoints for model management, predictions, and monitoring.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding authentication middleware.

## Endpoints

### Health Check Endpoints

#### GET /health
Comprehensive health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "uptime_seconds": 3600.5,
  "checks": {
    "system": {
      "status": "healthy",
      "cpu_percent": 45.2,
      "message": "CPU usage at 45.2%"
    },
    "memory": {
      "status": "healthy",
      "memory_percent": 60.5,
      "memory_available_mb": 4096.0,
      "message": "Memory usage at 60.5%"
    },
    "disk": {
      "status": "healthy",
      "disk_percent": 55.0,
      "disk_free_gb": 100.5,
      "message": "Disk usage at 55.0%"
    }
  }
}
```

#### GET /health/live
Liveness probe endpoint.

**Response:**
```json
{
  "alive": true,
  "timestamp": "2024-01-01T00:00:00",
  "uptime_seconds": 3600.5
}
```

#### GET /health/ready
Readiness probe endpoint.

**Response:**
```json
{
  "ready": true,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### Prediction Endpoints

#### POST /api/v1/predict
Predict if a single transaction is anomalous.

**Request Body:**
```json
{
  "hash": "0x123abc...",
  "value": 100.0,
  "gas": 21000.0,
  "gasPrice": 20.0,
  "from": "0xabc...",
  "to": "0xdef...",
  "blockNumber": 12345,
  "timestamp": "2024-01-01T00:00:00"
}
```

**Response:**
```json
{
  "hash": "0x123abc...",
  "is_anomaly": false,
  "anomaly_score": 0.12,
  "severity": null,
  "timestamp": "2024-01-01T00:00:00"
}
```

#### POST /api/v1/predict/batch
Predict anomalies for a batch of transactions.

**Request Body:**
```json
{
  "transactions": [
    {
      "hash": "0x123...",
      "value": 100.0,
      "gas": 21000.0,
      "gasPrice": 20.0
    },
    {
      "hash": "0x456...",
      "value": 1000000.0,
      "gas": 21000.0,
      "gasPrice": 200.0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "hash": "0x123...",
      "is_anomaly": false,
      "severity": null,
      "timestamp": "2024-01-01T00:00:00"
    },
    {
      "hash": "0x456...",
      "is_anomaly": true,
      "severity": "high",
      "timestamp": "2024-01-01T00:00:00"
    }
  ],
  "total_processed": 2,
  "anomalies_detected": 1,
  "processing_time_ms": 45.2
}
```

**Limits:**
- Minimum: 1 transaction
- Maximum: 1000 transactions per request

---

### Model Management Endpoints

#### POST /api/v1/models/train
Train a new anomaly detection model.

**Request Body:**
```json
{
  "data_source": "/path/to/training/data.csv",
  "contamination": 0.01,
  "model_type": "isolation_forest",
  "parameters": {
    "n_estimators": 100,
    "max_samples": "auto"
  }
}
```

**Response:**
```json
{
  "success": true,
  "model_id": "model_1234567890",
  "training_samples": 10000,
  "contamination": 0.01,
  "training_time_ms": 5000.0,
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.92,
    "recall": 0.89
  },
  "message": "Model training initiated successfully"
}
```

#### GET /api/v1/models
List all available models.

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_1234567890",
      "model_type": "isolation_forest",
      "created_at": "2024-01-01T00:00:00",
      "last_updated": "2024-01-01T00:00:00",
      "version": "1.0",
      "contamination": 0.01,
      "training_samples": 10000,
      "is_active": true
    }
  ],
  "total_models": 1
}
```

#### GET /api/v1/models/{model_id}
Get information about a specific model.

**Response:**
```json
{
  "model_id": "model_1234567890",
  "model_type": "isolation_forest",
  "created_at": "2024-01-01T00:00:00",
  "last_updated": "2024-01-01T00:00:00",
  "version": "1.0",
  "contamination": 0.01,
  "training_samples": 10000,
  "is_active": true
}
```

#### DELETE /api/v1/models/{model_id}
Delete a model.

**Response:**
```json
{
  "success": true,
  "message": "Model model_1234567890 deleted successfully"
}
```

---

### Anomaly Endpoints

#### GET /api/v1/anomalies
Get detected anomalies.

**Query Parameters:**
- `limit` (optional): Maximum number of anomalies to return (default: 100)
- `severity` (optional): Filter by severity level (low, medium, high, critical)

**Response:**
```json
{
  "anomalies": [
    {
      "hash": "0x123...",
      "value": 1000000.0,
      "gas": 21000.0,
      "gasPrice": 200.0,
      "from": "0xabc...",
      "to": "0xdef...",
      "timestamp": "2024-01-01T00:00:00",
      "detected_at": "2024-01-01T00:05:00",
      "severity": "high"
    }
  ],
  "total_count": 1
}
```

#### DELETE /api/v1/anomalies
Clear the anomaly buffer.

**Response:**
```json
{
  "success": true,
  "message": "Anomaly buffer cleared successfully"
}
```

---

### Streaming Endpoints

#### GET /api/v1/stream/status
Get streaming service status.

**Response:**
```json
{
  "is_running": true,
  "consumer_connected": true,
  "transactions_processed": 15234,
  "anomalies_detected": 152,
  "buffer_size": 45,
  "uptime_seconds": 3600.5
}
```

---

### Monitoring Endpoints

#### GET /metrics
Prometheus metrics endpoint.

**Response:**
Plain text Prometheus metrics format.

#### GET /api/metrics
Alternative metrics endpoint with FastAPI instrumentation.

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-01-01T00:00:00"
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service unhealthy

---

## Rate Limiting

Currently, no rate limiting is implemented. For production use, consider adding rate limiting middleware.

---

## Examples

### Using cURL

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hash": "0x123",
    "value": 100.0,
    "gas": 21000.0,
    "gasPrice": 20.0
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/api/v1/predict",
    json={
        "hash": "0x123",
        "value": 100.0,
        "gas": 21000.0,
        "gasPrice": 20.0
    }
)
print(response.json())

# Get anomalies
response = requests.get("http://localhost:8000/api/v1/anomalies?limit=10")
print(response.json())
```

---

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to test all endpoints directly from your browser.

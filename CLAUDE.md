# CLAUDE.md - AI Assistant Guide for Blockchain Anomaly Detection

**Version:** 2.0
**Last Updated:** 2025-11-17
**Project Owner:** Daniil Krizhanonovskyi

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Codebase Architecture](#codebase-architecture)
3. [Directory Structure](#directory-structure)
4. [Key Technologies](#key-technologies)
5. [Development Setup](#development-setup)
6. [Code Conventions](#code-conventions)
7. [Testing Strategy](#testing-strategy)
8. [Common Workflows](#common-workflows)
9. [Configuration Management](#configuration-management)
10. [Deployment](#deployment)
11. [Important Files Reference](#important-files-reference)
12. [Best Practices for AI Assistants](#best-practices-for-ai-assistants)

---

## Project Overview

### Purpose
The **Blockchain Anomaly Detection** project is an enterprise-grade system for detecting anomalies in blockchain transaction data using machine learning. It provides both batch processing and real-time streaming capabilities for fraud detection and suspicious activity identification.

### Key Features
- **ML-based Anomaly Detection**: Isolation Forest algorithm for anomaly detection
- **Time Series Forecasting**: ARIMA model for trend prediction
- **Real-time Processing**: Kafka-based streaming for live transaction analysis
- **REST API**: FastAPI-powered endpoints for predictions and model management
- **Distributed Caching**: Redis-based caching for performance optimization
- **Scalability**: Kubernetes-ready with horizontal auto-scaling
- **Monitoring**: Comprehensive Prometheus metrics and Grafana dashboards
- **CI/CD**: Automated testing and deployment pipeline

### Architecture Patterns
- **Single Responsibility Principle**: Each module has one clear purpose
- **Configuration Management**: Environment-based configuration with `.env` files
- **Dependency Injection**: Centralized configuration and dependency management
- **Error Handling**: Comprehensive error handling with Sentry integration
- **Observability**: Structured logging and metrics collection

---

## Codebase Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────┐
│                   NGINX Ingress / Load Balancer         │
└────────────────────────┬────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│   FastAPI       │◄────────────►│  Redis Cache    │
│   API Server    │              │  (Distributed)  │
│  (3-10 pods)    │              │                 │
│                 │              │  - Predictions  │
│ - Health checks │              │  - Features     │
│ - Predictions   │              │  - Query cache  │
│ - Model mgmt    │              └─────────────────┘
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Kafka Consumer Pods           │
│   (2-8 pods, 4 threads each)    │
│                                 │
│   - Stream Processor            │
│   - Anomaly Detection           │
│   - Bounded Buffer (TTL)        │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Kafka Cluster (External)      │
│   Topic: blockchain-transactions│
└─────────────────────────────────┘
```

### Component Responsibilities

1. **API Server** (`src/api_server/`): REST API for predictions, model management, and health checks
2. **Stream Processor** (`src/streaming/`): Real-time transaction processing and anomaly detection
3. **ML Models** (`src/anomaly_detection/`): Isolation Forest and ARIMA implementations
4. **Data Processing** (`src/data_processing/`): Data cleaning, transformation, and feature engineering
5. **Caching Layer** (`src/cache/`): Redis-based distributed caching
6. **API Client** (`src/api/`): Etherscan API integration for blockchain data
7. **Utilities** (`src/utils/`): Configuration, logging, and Sentry integration
8. **Visualization** (`src/visualization/`): Data visualization and plotting

---

## Directory Structure

```
blockchain-anomaly-detection/
│
├── src/                              # Source code
│   ├── main.py                       # Batch processing entry point
│   ├── anomaly_detection/            # ML models
│   │   ├── isolation_forest.py       # Anomaly detection model
│   │   └── arima_model.py            # Time series forecasting
│   ├── api/                          # External API integrations
│   │   ├── etherscan_api.py          # Etherscan API client
│   │   └── api_utils.py              # API utilities
│   ├── api_server/                   # FastAPI application
│   │   ├── app.py                    # Main FastAPI app
│   │   ├── models.py                 # Pydantic request/response models
│   │   └── monitoring.py             # Health checks and metrics
│   ├── streaming/                    # Kafka streaming
│   │   ├── kafka_consumer.py         # Kafka consumer service
│   │   ├── stream_processor.py       # Real-time anomaly detection
│   │   └── bounded_buffer.py         # Memory-safe anomaly buffer
│   ├── data_processing/              # Data pipeline
│   │   ├── data_cleaning.py          # Pandas-based cleaning
│   │   ├── data_cleaning_dask.py     # Dask-based cleaning (large datasets)
│   │   └── data_transformation.py    # Feature engineering
│   ├── cache/                        # Caching layer
│   │   ├── redis_client.py           # Redis connection management
│   │   └── cache_layer.py            # Application caching logic
│   ├── utils/                        # Utilities
│   │   ├── config.py                 # Configuration management
│   │   ├── logger.py                 # Structured logging
│   │   └── sentry.py                 # Error tracking
│   └── visualization/                # Data visualization
│       └── visualization.py          # Plotting utilities
│
├── tests/                            # Test suite (pytest)
│   ├── test_api_server.py            # API endpoint tests
│   ├── test_kafka_consumer.py        # Streaming tests
│   ├── test_stream_processor.py      # Processor tests
│   ├── test_data_cleaning.py         # Data processing tests
│   ├── test_monitoring.py            # Health check tests
│   └── test_integration.py           # End-to-end tests
│
├── k8s/                              # Kubernetes manifests
│   ├── api-deployment.yaml           # API deployment (3-10 pods)
│   ├── consumer-deployment.yaml      # Consumer deployment (2-8 pods)
│   ├── redis-statefulset.yaml        # Redis cache
│   ├── hpa.yaml                      # Auto-scaling rules
│   ├── ingress.yaml                  # Load balancer config
│   └── configmap.yaml                # Configuration
│
├── monitoring/                       # Observability
│   ├── prometheus.yml                # Prometheus config
│   ├── grafana-dashboard.json        # Grafana dashboard
│   └── grafana-datasources.yml       # Data sources
│
├── docker/                           # Docker configuration
│   └── Dockerfile                    # Production image
│
├── docs/                             # Documentation
│   ├── API.md                        # API documentation
│   ├── OPTIMIZATION_PLAN.md          # Performance optimization roadmap
│   ├── OPTIMIZATION_QUICK_START.md   # Quick optimization guide
│   ├── architecture.md               # Architecture details
│   └── troubleshooting.md            # Common issues
│
├── .github/workflows/                # CI/CD pipeline
│   └── ci-cd.yml                     # GitHub Actions workflow
│
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Build configuration
├── setup.py                          # Package setup
├── Makefile                          # Common commands
├── .env.example                      # Environment variables template
├── .gitignore                        # Git ignore patterns
├── README.md                         # Project documentation
├── CONTRIBUTING.md                   # Contribution guidelines
├── CHANGELOG.md                      # Version history
├── SECURITY.md                       # Security policy
└── SCALABILITY_ENHANCEMENTS.md       # Scalability documentation
```

---

## Key Technologies

### Core Stack
- **Python 3.9-3.11**: Primary language
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Kafka**: Streaming platform
- **Redis**: Distributed caching

### Data Processing
- **Pandas 2.0.3**: Data manipulation
- **NumPy 1.24.3**: Numerical computing
- **Dask 2023.5.0**: Parallel computing for large datasets

### Machine Learning
- **scikit-learn 1.2.2**: ML algorithms (Isolation Forest)
- **statsmodels 0.13.5**: Time series analysis (ARIMA)

### Visualization
- **Matplotlib 3.7.1**: Plotting
- **Seaborn 0.12.2**: Statistical visualization

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Sentry**: Error tracking
- **prometheus-client**: Python metrics
- **prometheus-fastapi-instrumentator**: FastAPI metrics

### Testing
- **pytest 7.3.2**: Testing framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **httpx**: Async HTTP client for testing

### Infrastructure
- **Docker**: Containerization
- **Kubernetes**: Orchestration
- **NGINX**: Load balancing
- **GitHub Actions**: CI/CD

---

## Development Setup

### Prerequisites
- Python 3.9+ (3.10 recommended)
- pip or poetry for dependency management
- Docker and Docker Compose (optional, for local services)
- Git for version control

### Initial Setup

```bash
# 1. Clone the repository
git clone https://github.com/arec1b0/blockchain-anomaly-detection.git
cd blockchain-anomaly-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment template
cp .env.example .env

# 5. Configure environment variables (see Configuration Management)
# Edit .env with your values
nano .env
```

### Environment Configuration

**Required Variables:**
```bash
# Etherscan API
ETHERSCAN_API_KEY=your_api_key_here
ETHERSCAN_ADDRESS=ethereum_address_here

# Application
LOG_LEVEL=INFO
ENVIRONMENT=development
```

**Optional Features:**
```bash
# Kafka Streaming
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=blockchain-transactions

# Redis Caching
REDIS_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379

# Sentry Error Tracking
SENTRY_ENABLED=true
SENTRY_DSN=your_sentry_dsn
```

### Running the Application

**Batch Processing Mode:**
```bash
# Run traditional batch pipeline
python src/main.py
```

**API Server Mode:**
```bash
# Start FastAPI server
uvicorn src.api_server.app:app --reload --host 0.0.0.0 --port 8000
```

**With Docker Compose (Recommended for Development):**
```bash
# Start all services (Kafka, Redis, API, Prometheus, Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Useful Make Commands

```bash
make install      # Install dependencies
make run          # Run batch processing
make test         # Run test suite
make lint         # Check code style
make clean        # Clean cache files
make docker-build # Build Docker image
```

---

## Code Conventions

### Python Style Guide
- **PEP 8 Compliance**: Follow PEP 8 style guidelines
- **Line Length**: Maximum 100 characters (configured in pyproject.toml)
- **Code Formatter**: Black (line-length=100)
- **Import Sorting**: isort
- **Linting**: flake8, pylint

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `data_cleaning.py`)
- **Classes**: `PascalCase` (e.g., `DataCleaner`, `AnomalyDetector`)
- **Functions/Methods**: `snake_case` (e.g., `clean_data()`, `detect_anomalies()`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `API_KEY`, `MAX_RETRIES`)
- **Private Members**: `_leading_underscore` (e.g., `_config`, `_validate()`)

### Documentation Standards
- **Docstrings**: Required for all public classes, methods, and functions
- **Format**: Google-style docstrings
- **Type Hints**: Use type hints for all function parameters and returns

**Example:**
```python
def detect_anomalies(data: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    """
    Detects anomalies in the provided transaction data.

    Args:
        data (pd.DataFrame): Transaction data with required features.
        contamination (float): Expected proportion of anomalies (default: 0.01).

    Returns:
        pd.DataFrame: Data with 'anomaly' column (-1 for anomalies, 1 for normal).

    Raises:
        ValueError: If required columns are missing from data.
    """
    # Implementation
```

### Error Handling
- **Use Specific Exceptions**: Catch specific exceptions, not bare `except:`
- **Log Errors**: Always log exceptions with context
- **Sentry Integration**: Capture exceptions in production
- **Validation**: Validate inputs at module boundaries

**Example:**
```python
try:
    transactions = api.get_transactions(address)
except requests.RequestException as e:
    logger.error(f"Failed to fetch transactions: {e}", exc_info=True)
    capture_exception(e, context={"address": address})
    raise
```

### Configuration Access
- **Use Config Class**: Access all config via `get_config()`
- **No Hardcoded Values**: All configurable values should be in environment variables
- **Validation**: Validate configuration at startup

```python
from src.utils.config import get_config

config = get_config()
api_key = config.API_KEY  # Not: os.getenv("API_KEY")
```

### Logging Standards
- **Use Logger**: Always use the logger from `src.utils.logger`
- **Log Levels**: INFO for normal flow, WARNING for recoverable issues, ERROR for failures
- **Structured Logging**: Include relevant context in log messages

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info(f"Processing {len(transactions)} transactions")
logger.warning(f"Cache miss for key {cache_key}")
logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
```

---

## Testing Strategy

### Test Organization
- **Location**: All tests in `tests/` directory
- **Naming**: `test_*.py` for files, `test_*` for functions
- **Coverage Target**: 80%+ (configured in pyproject.toml)

### Test Types

1. **Unit Tests**: Test individual functions/classes in isolation
   - `test_data_cleaning.py`
   - `test_anomaly_detection.py`
   - `test_config.py`

2. **Integration Tests**: Test component interactions
   - `test_api_server.py`
   - `test_kafka_consumer.py`
   - `test_integration.py`

3. **End-to-End Tests**: Test complete workflows
   - `test_integration.py`

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test file
pytest tests/test_api_server.py -v

# Run tests matching pattern
pytest -k "test_anomaly" -v

# Run with verbose output
pytest -vv

# Run in parallel (requires pytest-xdist)
pytest -n auto
```

### Writing Tests

**Example Unit Test:**
```python
import pytest
from src.data_processing.data_cleaning import DataCleaner

def test_remove_duplicates():
    """Test that duplicates are removed correctly."""
    data = pd.DataFrame({
        'hash': ['0x1', '0x1', '0x2'],
        'value': [100, 100, 200]
    })
    cleaner = DataCleaner(data)
    cleaned = cleaner.remove_duplicates()
    assert len(cleaned) == 2
    assert '0x1' in cleaned['hash'].values
```

**Example Integration Test:**
```python
@pytest.mark.asyncio
async def test_predict_endpoint(client):
    """Test the /api/v1/predict endpoint."""
    response = await client.post(
        "/api/v1/predict",
        json={
            "hash": "0x123",
            "value": 100.0,
            "gas": 21000.0,
            "gasPrice": 20.0
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "is_anomaly" in data
    assert "confidence" in data
```

### Test Fixtures
- Use pytest fixtures for reusable test setup
- Define fixtures in `conftest.py` for sharing across test files

---

## Common Workflows

### Adding a New API Endpoint

1. **Define Request/Response Models** in `src/api_server/models.py`:
   ```python
   class NewFeatureRequest(BaseModel):
       param1: str
       param2: int

   class NewFeatureResponse(BaseModel):
       result: bool
       message: str
   ```

2. **Implement Endpoint** in `src/api_server/app.py`:
   ```python
   @app.post("/api/v1/new-feature", response_model=NewFeatureResponse)
   async def new_feature(request: NewFeatureRequest):
       # Implementation
       return NewFeatureResponse(result=True, message="Success")
   ```

3. **Add Tests** in `tests/test_api_server.py`:
   ```python
   async def test_new_feature_endpoint(client):
       response = await client.post("/api/v1/new-feature", json={...})
       assert response.status_code == 200
   ```

4. **Update API Documentation** in `docs/API.md`

### Adding a New ML Model

1. **Create Model Class** in `src/anomaly_detection/new_model.py`:
   ```python
   class NewModel:
       def __init__(self, params):
           self.model = ...

       def train(self, data):
           # Training logic

       def predict(self, data):
           # Prediction logic
   ```

2. **Add Tests** in `tests/test_new_model.py`

3. **Integrate into Pipeline** in `src/main.py` or `src/streaming/stream_processor.py`

4. **Update Documentation**

### Adding Environment Variables

1. **Update `.env.example`** with new variable and description
2. **Add to Config Class** in `src/utils/config.py`:
   ```python
   self.NEW_VARIABLE = os.getenv("NEW_VARIABLE", "default_value")
   ```
3. **Update Validation** in `Config.validate()` if required
4. **Update Kubernetes ConfigMap** in `k8s/configmap.yaml`
5. **Document in CLAUDE.md** (this file)

### Making Database Schema Changes

⚠️ **Note**: This project currently doesn't use a traditional database. Anomalies are stored in-memory with Redis caching. If adding persistent storage:

1. Create migration scripts
2. Update models/schemas
3. Add tests for new schema
4. Update Kubernetes PVC if needed

### Debugging Issues

**Check Logs:**
```bash
# Application logs
tail -f logs/app.log

# Docker logs
docker-compose logs -f api

# Kubernetes logs
kubectl logs -f deployment/api -n blockchain-anomaly-detection
```

**Check Health:**
```bash
# Health check endpoint
curl http://localhost:8000/health

# Liveness probe
curl http://localhost:8000/health/live

# Readiness probe
curl http://localhost:8000/health/ready
```

**Check Metrics:**
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Access Prometheus UI
open http://localhost:9090

# Access Grafana
open http://localhost:3000  # admin/admin
```

---

## Configuration Management

### Environment-Based Configuration

Configuration is managed through environment variables and the `Config` class in `src/utils/config.py`.

**Priority Order:**
1. Environment variables (highest priority)
2. `.env` file
3. Default values in `Config` class (lowest priority)

### Configuration Categories

**API Configuration:**
- `ETHERSCAN_API_KEY`: API key for Etherscan
- `ETHERSCAN_ADDRESS`: Ethereum address to monitor
- `ETHERSCAN_BASE_URL`: API base URL
- `REQUEST_TIMEOUT`: HTTP request timeout (seconds)
- `MAX_RETRIES`: Max retry attempts
- `RETRY_BACKOFF`: Backoff factor for retries

**Application Settings:**
- `ENVIRONMENT`: Environment (development/production)
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `USE_DASK`: Enable Dask for large datasets

**Kafka Configuration:**
- `KAFKA_ENABLED`: Enable Kafka streaming
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses
- `KAFKA_TOPIC`: Topic name
- `KAFKA_GROUP_ID`: Consumer group ID
- `KAFKA_NUM_WORKER_THREADS`: Worker threads per consumer (default: 4)
- `KAFKA_MAX_QUEUE_SIZE`: Max processing queue size

**Redis Configuration:**
- `REDIS_ENABLED`: Enable Redis caching
- `REDIS_HOST`: Redis server host
- `REDIS_PORT`: Redis server port
- `REDIS_DB`: Redis database number
- `REDIS_PASSWORD`: Redis password (optional)
- `REDIS_MAX_CONNECTIONS`: Connection pool size

**Model Configuration:**
- `MODEL_PATH`: Path to saved model
- `BATCH_SIZE`: Processing batch size
- `CONTAMINATION`: Expected anomaly proportion (0.01 = 1%)

**Buffer Configuration:**
- `ANOMALY_BUFFER_MAX_SIZE`: Max anomalies in buffer (default: 10000)
- `ANOMALY_BUFFER_TTL_SECONDS`: TTL for anomalies (default: 3600)

**Monitoring:**
- `SENTRY_ENABLED`: Enable Sentry error tracking
- `SENTRY_DSN`: Sentry DSN
- `SENTRY_ENVIRONMENT`: Sentry environment tag
- `SENTRY_TRACES_SAMPLE_RATE`: Sample rate for traces

### Validation

Configuration is validated at startup via `Config.validate()`. Missing required variables will raise `ValueError`.

---

## Deployment

### Docker Deployment

**Build Image:**
```bash
docker build -f docker/Dockerfile -t blockchain-anomaly-detection:latest .
```

**Run Container:**
```bash
docker run -d -p 8000:8000 \
  -e KAFKA_ENABLED=false \
  -e REDIS_ENABLED=false \
  blockchain-anomaly-detection:latest
```

### Kubernetes Deployment

**Prerequisites:**
- Kubernetes cluster (v1.24+)
- kubectl configured
- NGINX Ingress Controller
- Metrics Server (for HPA)

**Deploy:**
```bash
# 1. Create namespace
kubectl create namespace blockchain-anomaly-detection

# 2. Create secrets
kubectl create secret generic redis-secret \
  --from-literal=password='your-password' \
  -n blockchain-anomaly-detection

kubectl create secret generic etherscan-secret \
  --from-literal=api_key='your-api-key' \
  -n blockchain-anomaly-detection

# 3. Apply manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/redis-statefulset.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/consumer-deployment.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml

# 4. Verify deployment
kubectl get pods -n blockchain-anomaly-detection
kubectl get hpa -n blockchain-anomaly-detection
```

**Scaling:**
```bash
# Manual scaling
kubectl scale deployment api --replicas=5 -n blockchain-anomaly-detection

# HPA automatically scales based on CPU/memory (configured in hpa.yaml)
```

See `k8s/README.md` and `SCALABILITY_ENHANCEMENTS.md` for detailed deployment instructions.

### CI/CD Pipeline

The project uses GitHub Actions for CI/CD (`.github/workflows/ci-cd.yml`):

**Stages:**
1. **Lint**: Code quality checks (Black, isort, flake8)
2. **Test**: Run tests on Python 3.9, 3.10, 3.11
3. **Security**: Vulnerability scanning (Safety, Bandit)
4. **Build**: Build and test Docker image
5. **Integration Test**: End-to-end tests with Docker Compose
6. **Deploy**: Deploy to production (main branch only)

**Triggers:**
- Push to `main`, `develop`, or `claude/**` branches
- Pull requests to `main` or `develop`
- Manual workflow dispatch

---

## Important Files Reference

### Core Application Files

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `src/main.py` | Batch processing entry point | `main()` - orchestrates pipeline |
| `src/api_server/app.py` | FastAPI application | Endpoints, startup/shutdown logic |
| `src/api_server/models.py` | Request/response models | Pydantic models for API |
| `src/api_server/monitoring.py` | Health checks & metrics | `health_checker`, Prometheus metrics |
| `src/streaming/stream_processor.py` | Real-time anomaly detection | `StreamProcessor` class |
| `src/streaming/kafka_consumer.py` | Kafka consumer service | `KafkaConsumerService` |
| `src/streaming/bounded_buffer.py` | Memory-safe anomaly buffer | `BoundedBuffer` with TTL |
| `src/cache/redis_client.py` | Redis connection management | `get_redis_client()` |
| `src/cache/cache_layer.py` | Application caching logic | `CacheLayer`, `@cached` decorator |
| `src/utils/config.py` | Configuration management | `Config` class, `get_config()` |
| `src/utils/logger.py` | Logging setup | `get_logger()` |
| `src/utils/sentry.py` | Error tracking | `init_sentry()`, `capture_exception()` |

### ML Model Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/anomaly_detection/isolation_forest.py` | Anomaly detection | `AnomalyDetectorIsolationForest` |
| `src/anomaly_detection/arima_model.py` | Time series forecasting | `ARIMAModel` |

### Data Processing Files

| File | Purpose | Key Classes |
|------|---------|-------------|
| `src/data_processing/data_cleaning.py` | Pandas data cleaning | `DataCleaner` |
| `src/data_processing/data_cleaning_dask.py` | Dask data cleaning (large data) | `DataCleanerDask` |
| `src/data_processing/data_transformation.py` | Feature engineering | `DataTransformer` |

### Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Environment variables template |
| `pyproject.toml` | Build config, tool settings (Black, pytest, mypy) |
| `requirements.txt` | Python dependencies |
| `setup.py` | Package setup and metadata |
| `Makefile` | Common development commands |
| `.gitignore` | Git ignore patterns |

### Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and setup guide |
| `CLAUDE.md` | This file - AI assistant guide |
| `CONTRIBUTING.md` | Contribution guidelines |
| `SECURITY.md` | Security policy |
| `CHANGELOG.md` | Version history |
| `SCALABILITY_ENHANCEMENTS.md` | Scalability features |
| `docs/API.md` | API endpoint documentation |
| `docs/OPTIMIZATION_PLAN.md` | Performance optimization roadmap |
| `docs/architecture.md` | Architecture deep dive |
| `docs/troubleshooting.md` | Common issues and solutions |

### Deployment Files

| File | Purpose |
|------|---------|
| `docker/Dockerfile` | Production Docker image |
| `k8s/*.yaml` | Kubernetes manifests |
| `.github/workflows/ci-cd.yml` | CI/CD pipeline |
| `monitoring/*.yml` | Prometheus/Grafana config |

---

## Best Practices for AI Assistants

### When Reading Code

1. **Start with High-Level Context**: Read README.md and this CLAUDE.md first
2. **Check Configuration**: Review `.env.example` and `src/utils/config.py` for available settings
3. **Understand Architecture**: Review the architecture diagrams in this file and `docs/architecture.md`
4. **Follow Imports**: Trace import statements to understand dependencies
5. **Read Docstrings**: Classes and functions have comprehensive docstrings
6. **Check Tests**: Test files often reveal usage patterns and edge cases

### When Writing Code

1. **Follow Existing Patterns**: Match the coding style of existing modules
2. **Use Type Hints**: Add type hints to all function signatures
3. **Write Docstrings**: Document all public classes, methods, and functions
4. **Handle Errors**: Add proper error handling and logging
5. **Add Tests**: Write tests for new functionality (aim for 80%+ coverage)
6. **Update Documentation**: Update relevant docs when adding features
7. **Validate Configuration**: Add validation for new config variables
8. **Use Config Class**: Access configuration via `get_config()`, not directly from `os.getenv()`

### When Debugging

1. **Check Logs**: Look at logs first (`logs/app.log` or container logs)
2. **Verify Configuration**: Ensure required environment variables are set
3. **Test Health Endpoints**: Use `/health`, `/health/live`, `/health/ready`
4. **Check Metrics**: Review Prometheus metrics at `/metrics`
5. **Isolate Components**: Test components individually before integration
6. **Review Recent Changes**: Check git history for recent modifications

### When Refactoring

1. **Run Tests First**: Ensure all tests pass before refactoring
2. **Refactor Small**: Make small, incremental changes
3. **Keep Tests Green**: Run tests after each change
4. **Update Documentation**: Keep docs synchronized with code
5. **Preserve API Contracts**: Don't break existing API endpoints
6. **Consider Backward Compatibility**: Maintain compatibility when possible

### When Reviewing Code

1. **Check Style**: Verify PEP 8 compliance and Black formatting
2. **Review Error Handling**: Ensure exceptions are properly caught and logged
3. **Verify Tests**: Confirm tests exist and provide good coverage
4. **Check Documentation**: Ensure docstrings and comments are clear
5. **Assess Performance**: Consider performance implications of changes
6. **Security Review**: Look for potential security issues (injection, XSS, etc.)

### Common Pitfalls to Avoid

1. **Don't Hardcode Values**: Use environment variables and configuration
2. **Don't Ignore Errors**: Always handle exceptions appropriately
3. **Don't Skip Tests**: Write tests for all new functionality
4. **Don't Modify Core Logic Without Tests**: Especially for ML models
5. **Don't Break Backward Compatibility**: Coordinate breaking changes
6. **Don't Commit Secrets**: Never commit API keys or passwords
7. **Don't Bypass Configuration**: Always use `get_config()`
8. **Don't Use Bare Except**: Catch specific exceptions

### Working with Git Branches

**Branch Naming:**
- Feature: `feature/description` or `claude/description-sessionid`
- Bug fix: `fix/description`
- Hotfix: `hotfix/description`

**Commit Message Format:**
```
type(scope): short description

Longer explanation if needed

- Bullet points for details
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(api): add batch prediction endpoint

Added /api/v1/predict/batch endpoint for processing multiple
transactions in a single request.

- Supports up to 100 transactions per batch
- Returns anomaly scores for all transactions
- Includes rate limiting
```

### Performance Considerations

1. **Use Caching**: Leverage Redis cache for expensive operations
2. **Batch Processing**: Process transactions in batches when possible
3. **Async Operations**: Use async/await for I/O-bound operations
4. **Memory Management**: Be mindful of memory usage with large datasets
5. **Monitor Metrics**: Use Prometheus metrics to identify bottlenecks
6. **Scale Horizontally**: Add more pods rather than increasing resources

### Security Considerations

1. **Validate Input**: Always validate user input
2. **Sanitize Data**: Sanitize data before processing
3. **Use Secrets**: Store sensitive data in Kubernetes secrets
4. **Follow OWASP Top 10**: Be aware of common vulnerabilities
5. **Update Dependencies**: Keep dependencies up to date
6. **Run Security Scans**: CI/CD includes Safety and Bandit scans
7. **Rate Limiting**: API includes rate limiting (100 RPS per IP)

---

## Quick Reference Commands

### Development
```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Run
python src/main.py                    # Batch processing
uvicorn src.api_server.app:app --reload  # API server

# Test
pytest                                # All tests
pytest --cov=src                      # With coverage
pytest tests/test_api_server.py -v    # Specific file

# Lint
black src tests                       # Format code
isort src tests                       # Sort imports
flake8 src tests                      # Lint check

# Clean
make clean                            # Remove cache
```

### Docker
```bash
# Build
docker build -f docker/Dockerfile -t blockchain-anomaly-detection .

# Run
docker run -p 8000:8000 -e KAFKA_ENABLED=false blockchain-anomaly-detection

# Compose
docker-compose up -d                  # Start all services
docker-compose logs -f api            # View logs
docker-compose down                   # Stop services
```

### Kubernetes
```bash
# Deploy
kubectl apply -f k8s/

# Monitor
kubectl get pods -n blockchain-anomaly-detection
kubectl logs -f deployment/api -n blockchain-anomaly-detection
kubectl get hpa -n blockchain-anomaly-detection

# Scale
kubectl scale deployment api --replicas=5 -n blockchain-anomaly-detection

# Debug
kubectl exec -it pod-name -n blockchain-anomaly-detection -- /bin/bash
```

### Git
```bash
# Create feature branch
git checkout -b claude/feature-name-sessionid

# Commit
git add .
git commit -m "feat(api): description"

# Push
git push -u origin claude/feature-name-sessionid

# Create PR via GitHub web interface
```

---

## Additional Resources

- **Project Repository**: https://github.com/arec1b0/blockchain-anomaly-detection
- **API Documentation**: See `docs/API.md` or http://localhost:8000/docs (when running)
- **Architecture Details**: See `docs/architecture.md`
- **Optimization Guide**: See `docs/OPTIMIZATION_PLAN.md`
- **Troubleshooting**: See `docs/troubleshooting.md`
- **Contributing**: See `CONTRIBUTING.md`
- **Security Policy**: See `SECURITY.md`
- **Scalability**: See `SCALABILITY_ENHANCEMENTS.md`

---

## Version History

- **v2.0** (2025-11-17): Added comprehensive AI assistant guide
- **v1.0** (2024): Initial project release with core features

---

## Contact

**Project Owner**: Daniil Krizhanonovskyi
**Email**: daniill.krizhanovskyi@hotmail.com
**Repository**: https://github.com/arec1b0/blockchain-anomaly-detection

---

**Note for AI Assistants**: This document is regularly updated to reflect the current state of the codebase. When in doubt, verify information by checking the actual source code or documentation files. Always run tests after making changes and ensure the CI/CD pipeline passes before merging.

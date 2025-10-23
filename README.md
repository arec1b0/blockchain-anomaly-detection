# Blockchain Anomaly Detection

Welcome to the **Blockchain Anomaly Detection** project! My name is Daniil Krizhanonovskyi, and I created this open-source tool to provide an effective way to detect anomalies in blockchain transaction data using machine learning techniques. This project offers a comprehensive solution for cleaning, processing, analyzing, and visualizing blockchain data with the aim of identifying unusual patterns that could represent fraudulent or suspicious activity.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Architecture](#architecture)
- [Environment Variables](#environment-variables)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Contribution](#contribution)
- [License](#license)

## Project Overview

The **Blockchain Anomaly Detection** project integrates multiple machine learning models to analyze transaction data, including:
- **Isolation Forest**: A powerful tool for detecting anomalies in high-dimensional data.
- **ARIMA**: A time-series forecasting model used to predict future trends in transaction activity.

The tool also provides functionality for fetching blockchain transaction data directly from the **Etherscan API**, cleaning and transforming it for analysis, and producing visualizations of the results. The primary goal is to provide a customizable framework for blockchain analytics and fraud detection that can be used by developers, researchers, and security analysts.

## Features

### Core Features
- **Data Cleaning**: Automated data cleaning and preprocessing using Pandas and Dask, removing duplicates and handling missing values.
- **Anomaly Detection**: Detection of anomalies in transaction data using the Isolation Forest algorithm.
- **Time Series Forecasting**: Future prediction of transaction trends using the ARIMA model.
- **API Integration**: Direct integration with the Etherscan API to fetch blockchain transaction data.
- **Data Visualization**: Visualization of transaction data, including anomalies, using Matplotlib and Seaborn.
- **Scalability**: Support for large datasets using Dask for parallelized data processing.

### New Features (v2.0)
- **Real-time Stream Processing**: Kafka-based streaming for real-time anomaly detection
- **REST API**: FastAPI-powered REST API for model management and predictions
- **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
- **Health Checks**: Kubernetes-ready health check endpoints (liveness, readiness, health)
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Docker Compose**: Complete local development environment with all services
- **Enhanced Testing**: Comprehensive test coverage for all components

## Project Structure

Here is the structure of the project to give you a clear idea of where everything is located:

```
blockchain-anomaly-detection/
│
├── README.md                        # Project overview and setup instructions
├── requirements.txt                 # Project dependencies
├── docker-compose.yml               # Docker Compose configuration
│
├── .github/                         # GitHub configuration
│   └── workflows/                   # GitHub Actions workflows
│       └── ci-cd.yml                # CI/CD pipeline
│
├── data/                            # Data directory
│   ├── processed/                   # Processed data
│   └── raw/                         # Raw data
│
├── docker/                          # Docker configuration
│   └── Dockerfile                   # Docker image setup
│
├── docs/                            # Documentation
│   └── API.md                       # API documentation
│
├── monitoring/                      # Monitoring configuration
│   ├── prometheus.yml               # Prometheus configuration
│   ├── grafana-dashboard.json       # Grafana dashboard
│   ├── grafana-datasources.yml      # Grafana data sources
│   └── grafana-dashboards.yml       # Grafana dashboard provisioning
│
├── logs/                            # Logs directory
│   └── app.log                      # Application logs
│
├── src/                             # Source code
│   ├── main.py                      # Main script to launch the project
│   ├── anomaly_detection/           # Anomaly detection models
│   │   ├── arima_model.py           # ARIMA model for time series forecasting
│   │   └── isolation_forest.py      # Isolation Forest for anomaly detection
│   ├── api/                         # API modules
│   │   ├── api_utils.py             # Utility functions for API requests
│   │   └── etherscan_api.py         # Etherscan API interaction
│   ├── api_server/                  # FastAPI REST API server
│   │   ├── app.py                   # Main FastAPI application
│   │   ├── models.py                # Pydantic models for request/response
│   │   └── monitoring.py            # Health checks and monitoring
│   ├── streaming/                   # Kafka streaming components
│   │   ├── kafka_consumer.py        # Kafka consumer service
│   │   └── stream_processor.py      # Stream processor for anomaly detection
│   ├── data_processing/             # Data cleaning and transformation
│   │   ├── data_cleaning.py         # Data cleaning using Pandas
│   │   ├── data_cleaning_dask.py    # Data cleaning using Dask for large datasets
│   │   └── data_transformation.py   # Data transformation and normalization
│   ├── utils/                       # Utility functions
│   │   ├── config.py                # Configuration settings (API keys, etc.)
│   │   ├── logger.py                # Logging utility
│   │   └── sentry.py                # Sentry integration
│   └── visualization/               # Data visualization
│       └── visualization.py         # Visualization module
│
└── tests/                           # Unit tests
    ├── test_anomaly_detection.py    # Tests for anomaly detection models
    ├── test_api.py                  # Tests for API interaction
    ├── test_api_server.py           # Tests for FastAPI endpoints
    ├── test_arima_model.py          # Tests for ARIMA model
    ├── test_data_cleaning.py        # Tests for data cleaning
    ├── test_integration.py          # Integration tests
    ├── test_kafka_consumer.py       # Tests for Kafka consumer
    ├── test_stream_processor.py     # Tests for stream processor
    ├── test_monitoring.py           # Tests for monitoring
    └── test_visualization.py        # Tests for data visualization
```

## Installation

To get started with the **Blockchain Anomaly Detection** project, you need to install the required dependencies and configure the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/dkrizhanovskyi/blockchain-anomaly-detection.git
   ```

2. Navigate into the project directory:
   ```bash
   cd blockchain-anomaly-detection
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```bash
   ETHERSCAN_API_KEY=<your_etherscan_api_key>
   ETHERSCAN_ADDRESS=<ethereum_wallet_address>
   ```

## Usage

### Quick Start with Docker Compose

The easiest way to get started is using Docker Compose, which sets up all services:

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

This will start:
- **Kafka** (port 9092): Message broker for streaming
- **Zookeeper** (port 2181): Kafka coordination
- **API Server** (port 8000): REST API for predictions and model management
- **Prometheus** (port 9090): Metrics collection
- **Grafana** (port 3000): Monitoring dashboards
- **Kafka UI** (port 8080): Kafka management interface

### Using the REST API

Once the services are running, you can interact with the API:

```bash
# Check API health
curl http://localhost:8000/health

# Make a prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hash": "0x123",
    "value": 100.0,
    "gas": 21000.0,
    "gasPrice": 20.0
  }'

# Get detected anomalies
curl http://localhost:8000/api/v1/anomalies?limit=10
```

### API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Full API Docs**: [docs/API.md](docs/API.md)

### Monitoring

Access monitoring dashboards:
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kafka UI**: http://localhost:8080

### Traditional Usage

You can also run the traditional batch processing pipeline:

```bash
python src/main.py
```

This will fetch transaction data from the Etherscan API, clean and transform the data, perform anomaly detection, and generate visualizations.

## Testing

Comprehensive unit tests are provided to ensure that the project works as expected. To run the tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test files
pytest tests/test_api_server.py -v
pytest tests/test_kafka_consumer.py -v
pytest tests/test_stream_processor.py -v

# Run integration tests
pytest tests/test_integration.py -v
```

The test suite covers:
- Anomaly detection models
- Data processing and cleaning
- API endpoints and responses
- Kafka consumer and stream processor
- Health checks and monitoring
- Integration tests

## Architecture

### System Components

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Kafka     │─────>│    Stream    │─────>│   Anomaly       │
│   Broker    │      │   Processor  │      │   Detection     │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            v
                     ┌──────────────┐
                     │   FastAPI    │
                     │   Server     │
                     └──────────────┘
                            │
                    ┌───────┴───────┐
                    v               v
            ┌──────────────┐  ┌──────────────┐
            │  Prometheus  │  │   Grafana    │
            │   Metrics    │  │  Dashboard   │
            └──────────────┘  └──────────────┘
```

### Data Flow

1. **Ingestion**: Transactions arrive via Kafka topics
2. **Processing**: Stream processor batches and analyzes transactions
3. **Detection**: ML models identify anomalies in real-time
4. **Storage**: Anomalies are buffered and accessible via API
5. **Monitoring**: Metrics are collected and visualized

## Environment Variables

Configure the application using these environment variables:

```bash
# Kafka Configuration
KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=blockchain-transactions
KAFKA_GROUP_ID=anomaly-detection-group

# Model Configuration
MODEL_PATH=./models/default_model.pkl
BATCH_SIZE=100
CONTAMINATION=0.01

# Etherscan API (for batch processing)
ETHERSCAN_API_KEY=your_api_key_here
ETHERSCAN_ADDRESS=ethereum_address_here

# Logging
LOG_LEVEL=INFO

# Sentry (optional)
SENTRY_DSN=your_sentry_dsn_here
```

## Docker

### Docker Compose (Recommended)

The recommended way to run the application is using Docker Compose:

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild services after code changes
docker-compose up -d --build
```

### Standalone Docker

You can also build and run the API service standalone:

```bash
# Build the Docker image
docker build -f docker/Dockerfile -t blockchain-anomaly-detection .

# Run the container
docker run -d -p 8000:8000 \
  -e KAFKA_ENABLED=false \
  blockchain-anomaly-detection
```

### Production Deployment

For production deployments, consider:
- Setting up Kafka cluster with multiple brokers
- Configuring persistent volumes for data storage
- Setting up load balancing for the API
- Implementing authentication and authorization
- Configuring SSL/TLS for secure communication

## CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline using GitHub Actions:

- **Linting**: Code quality checks with Black, isort, and flake8
- **Testing**: Automated tests on multiple Python versions
- **Security**: Vulnerability scanning with Safety and Bandit
- **Docker**: Container build and testing
- **Integration Tests**: End-to-end testing with Docker Compose

The pipeline runs on every push and pull request. See [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) for details.

## Contribution

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix: `git checkout -b feature-branch-name`.
3. Commit your changes: `git commit -m 'Add a new feature'`.
4. Push the branch: `git push origin feature-branch-name`.
5. Open a Pull Request to the `main` branch.

Please ensure that your code follows the project’s coding standards and passes all tests before submitting a PR.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute this software under the terms of the license.

---

Thank you for exploring the **Blockchain Anomaly Detection** project! I hope this tool will help you in your efforts to analyze blockchain data and detect potential fraudulent activities. If you have any questions or suggestions, feel free to reach out.

— Daniil Krizhanonovskyi

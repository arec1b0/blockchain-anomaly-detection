# Blockchain Anomaly Detection

Welcome to the **Blockchain Anomaly Detection** project! My name is Daniil Krizhanonovskyi, and I created this open-source tool to provide an effective way to detect anomalies in blockchain transaction data using machine learning techniques. This project offers a comprehensive solution for cleaning, processing, analyzing, and visualizing blockchain data with the aim of identifying unusual patterns that could represent fraudulent or suspicious activity.

## Project Overview

The **Blockchain Anomaly Detection** project integrates multiple machine learning models to analyze transaction data, including:
- **Isolation Forest**: A powerful tool for detecting anomalies in high-dimensional data.
- **ARIMA**: A time-series forecasting model used to predict future trends in transaction activity.

The tool also provides functionality for fetching blockchain transaction data directly from the **Etherscan API**, cleaning and transforming it for analysis, and producing visualizations of the results. The primary goal is to provide a customizable framework for blockchain analytics and fraud detection that can be used by developers, researchers, and security analysts.

## Features

- **Data Cleaning**: Automated data cleaning and preprocessing using Pandas and Dask, removing duplicates and handling missing values.
- **Anomaly Detection**: Detection of anomalies in transaction data using the Isolation Forest algorithm.
- **Time Series Forecasting**: Future prediction of transaction trends using the ARIMA model.
- **API Integration**: Direct integration with the Etherscan API to fetch blockchain transaction data.
- **Data Visualization**: Visualization of transaction data, including anomalies, using Matplotlib and Seaborn.
- **Scalability**: Support for large datasets using Dask for parallelized data processing.

## Project Structure

Here is the structure of the project to give you a clear idea of where everything is located:

```
blockchain-anomaly-detection/
│
├── README.md                        # Project overview and setup instructions
├── requirements.txt                 # Project dependencies
│
├── data/                            # Data directory
│   ├── processed/                   # Processed data
│   └── raw/                         # Raw data
│
├── docker/                          # Docker configuration
│   └── Dockerfile                   # Docker image setup
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
│   ├── data_processing/             # Data cleaning and transformation
│   │   ├── data_cleaning.py         # Data cleaning using Pandas
│   │   ├── data_cleaning_dask.py    # Data cleaning using Dask for large datasets
│   │   └── data_transformation.py   # Data transformation and normalization
│   ├── utils/                       # Utility functions
│   │   ├── config.py                # Configuration settings (API keys, etc.)
│   │   ├── logger.py                # Logging utility
│   └── visualization/               # Data visualization
│       └── visualization.py         # Visualization module
│
└── tests/                           # Unit tests
    ├── test_anomaly_detection.py    # Tests for anomaly detection models
    ├── test_api.py                  # Tests for API interaction
    ├── test_arima_model.py          # Tests for ARIMA model
    ├── test_data_cleaning.py        # Tests for data cleaning
    ├── test_integration.py          # Integration tests
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

Once the installation is complete, you can run the anomaly detection pipeline:

1. Fetch and analyze blockchain data:
   ```bash
   python src/main.py
   ```

This will fetch transaction data from the Etherscan API, clean and transform the data, perform anomaly detection, and generate visualizations.

### Example Workflow:

- **Step 1: Fetch Data**  
   The script will automatically fetch blockchain transaction data from the Etherscan API using the provided API key.

- **Step 2: Data Cleaning**  
   Data is cleaned by removing duplicates, handling missing values, and filtering out invalid transactions.

- **Step 3: Anomaly Detection**  
   The Isolation Forest algorithm is applied to detect anomalous transactions.

- **Step 4: Visualization**  
   Visualizations are generated for transaction trends and identified anomalies.

## Testing

Unit tests are provided to ensure that the project works as expected. To run the tests, use the following command:

```bash
pytest
```

This will execute the test suite located in the `tests/` directory, covering anomaly detection, data processing, API interaction, and visualization.

## Docker

To simplify deployment, a `Dockerfile` is provided. You can build and run the project inside a Docker container:

1. Build the Docker image:
   ```bash
   docker build -t blockchain-anomaly-detection .
   ```

2. Run the Docker container:
   ```bash
   docker run -d -p 80:80 blockchain-anomaly-detection
   ```

## Kubernetes

If you prefer deploying the project on Kubernetes, a sample `deployment.yaml` file is included in the `kubernetes/` directory. To deploy the application:

```bash
kubectl apply -f kubernetes/deployment.yaml
```

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


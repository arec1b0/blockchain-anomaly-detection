# Project Architecture

## Overview

The Blockchain Anomaly Detection project is designed to analyze blockchain transaction data using machine learning techniques such as anomaly detection and time-series forecasting.

The project follows a modular architecture, where each component has a clear responsibility:

1. **API Module**: Interacts with the Etherscan API to fetch blockchain transaction data.
2. **Data Processing Module**: Handles data cleaning, transformation, and normalization.
3. **Anomaly Detection Module**: Detects anomalies in transaction data using models like Isolation Forest.
4. **Visualization Module**: Provides tools for visualizing transaction data and anomalies.
5. **Main Application**: Coordinates the workflow from data fetching to anomaly detection and visualization.

## Components

- **Etherscan API**: Fetches real-time blockchain transaction data.
- **Isolation Forest Model**: Detects outliers in transaction data.
- **ARIMA Model**: Forecasts future trends in transaction values.

## Workflow

1. Fetch data from the Etherscan API.
2. Clean and transform the data.
3. Apply anomaly detection using Isolation Forest.
4. Forecast future trends using ARIMA.
5. Visualize the results.


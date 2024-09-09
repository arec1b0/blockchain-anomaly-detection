"""
example_usage.py

This script demonstrates the usage of the Blockchain Anomaly Detection project.
It covers the full pipeline from fetching data from the Etherscan API to detecting
anomalies and visualizing the results.

Make sure to set the environment variables for the Etherscan API key before running the script:
    export ETHERSCAN_API_KEY=your_etherscan_api_key
    export ETHERSCAN_ADDRESS=your_ethereum_address
"""

import os
import pandas as pd
from src.api.etherscan_api import EtherscanAPI
from src.data_processing.data_cleaning import DataCleaner
from src.data_processing.data_transformation import DataTransformer
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from src.visualization.visualization import DataVisualizer

# Fetch API key and Ethereum address from environment variables
API_KEY = os.getenv("ETHERSCAN_API_KEY")
ADDRESS = os.getenv("ETHERSCAN_ADDRESS")

# Step 1: Fetch transaction data from Etherscan API
api = EtherscanAPI(api_key=API_KEY)
print(f"Fetching transactions for address: {ADDRESS}")
transactions = api.get_transactions(ADDRESS)

if not transactions:
    print("Failed to fetch transactions.")
else:
    # Convert transactions to DataFrame
    df = pd.DataFrame(transactions)

    # Step 2: Clean the data
    cleaner = DataCleaner(df)
    cleaned_data = cleaner.clean_data()

    # Step 3: Transform the data
    transformer = DataTransformer(cleaned_data)
    transformed_data = transformer.transform_data()

    # Step 4: Detect anomalies
    detector = AnomalyDetectorIsolationForest(transformed_data)
    detector.train_model()
    result_df = detector.detect_anomalies()

    # Step 5: Visualize results
    visualizer = DataVisualizer(result_df)
    visualizer.plot_time_series()
    visualizer.plot_anomalies()
    visualizer.plot_distribution('value')

    print("Anomaly detection and visualization completed.")

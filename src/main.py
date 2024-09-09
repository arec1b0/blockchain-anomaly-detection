"""
main.py

This is the main entry point of the application. It orchestrates the entire workflow from fetching transaction data
via the Etherscan API, performing data cleaning and transformation, detecting anomalies, and visualizing results.

Adheres to the Single Responsibility Principle (SRP) by coordinating all system operations in a structured manner.
"""

from src.api.etherscan_api import EtherscanAPI
from src.data_processing.data_cleaning import DataCleaner
from src.data_processing.data_cleaning_dask import DataCleanerDask
from src.data_processing.data_transformation import DataTransformer
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from src.anomaly_detection.arima_model import ARIMAModel
from src.visualization.visualization import DataVisualizer
from src.utils.logger import get_logger
import pandas as pd
import os

# Initialize logger
logger = get_logger(__name__)


def main():
    """
    Main function orchestrating the entire process of fetching, cleaning, transforming, analyzing, 
    and visualizing transaction data.
    """
    try:
        # Step 1: Fetch transaction data using Etherscan API
        api_key = os.getenv("ETHERSCAN_API_KEY")
        if not api_key:
            logger.error("Etherscan API key is missing. Set 'ETHERSCAN_API_KEY' environment variable.")
            return

        address = os.getenv("ETHERSCAN_ADDRESS")
        if not address:
            logger.error("Ethereum address is missing. Set 'ETHERSCAN_ADDRESS' environment variable.")
            return

        logger.info(f"Fetching transactions for address: {address}")

        api = EtherscanAPI(api_key=api_key)
        transactions = api.get_transactions(address)

        if not transactions:
            logger.error("Failed to fetch transactions from Etherscan API.")
            return

        # Convert the transactions data into a DataFrame
        df = pd.DataFrame(transactions)
        logger.info(f"Fetched {len(df)} transactions for address {address}.")

        # Step 2: Data Cleaning
        use_dask = os.getenv("USE_DASK", "false").lower() == "true"
        if use_dask:
            logger.info("Using Dask for data cleaning.")
            cleaner = DataCleanerDask(df)
        else:
            logger.info("Using Pandas for data cleaning.")
            cleaner = DataCleaner(df)

        cleaned_data = cleaner.clean_data()

        # Step 3: Data Transformation
        logger.info("Starting data transformation...")
        transformer = DataTransformer(cleaned_data)
        transformed_data = transformer.transform_data()

        # Step 4: Anomaly Detection using Isolation Forest
        logger.info("Starting anomaly detection...")
        detector = AnomalyDetectorIsolationForest(transformed_data)
        detector.train_model()
        result_df = detector.detect_anomalies()

        # Step 5: Time Series Analysis using ARIMA
        logger.info("Starting ARIMA time series analysis...")
        arima_model = ARIMAModel(transformed_data, order=(5, 1, 0))
        arima_model.fit_model()
        forecast = arima_model.forecast(steps=10)
        logger.info(f"Forecasted values for the next 10 days: {forecast}")

        # Step 6: Visualization
        logger.info("Starting data visualization...")
        visualizer = DataVisualizer(result_df)
        visualizer.plot_time_series()
        visualizer.plot_anomalies()
        visualizer.plot_distribution('value')

    except Exception as e:
        logger.error(f"An error occurred during the execution of the pipeline: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()

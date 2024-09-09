import pytest
import os
import pandas as pd
from src.api.etherscan_api import EtherscanAPI
from src.data_processing.data_cleaning import DataCleaner
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest


def test_integration():
    api_key = os.getenv("ETHERSCAN_API_KEY")
    assert api_key, "API key is missing. Set 'ETHERSCAN_API_KEY' environment variable."

    address = os.getenv("ETHERSCAN_ADDRESS")
    assert address, "Ethereum address is missing. Set 'ETHERSCAN_ADDRESS' environment variable."

    api = EtherscanAPI(api_key=api_key)
    transactions = api.get_transactions(address)

    assert transactions is not None, "Failed to fetch transactions."

    df = pd.DataFrame(transactions)
    cleaner = DataCleaner(df)
    cleaned_data = cleaner.clean_data()

    assert not cleaned_data.empty, "Data cleaning failed."

    detector = AnomalyDetectorIsolationForest(cleaned_data)
    detector.train_model()
    result_df = detector.detect_anomalies()

    assert 'anomaly' in result_df.columns, "Anomaly detection failed."

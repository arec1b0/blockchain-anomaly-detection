import pytest
import os
import pandas as pd
from src.api.etherscan_api import EtherscanAPI
from src.data_processing.data_cleaning import DataCleaner
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest
from unittest.mock import patch


@patch('src.api.etherscan_api.EtherscanAPI.get_transactions')
def test_integration(mock_get_transactions):
    # Mock the API response
    mock_get_transactions.return_value = [
        {'value': '100', 'gas': '21000', 'gasPrice': '20'},
        {'value': '200', 'gas': '22000', 'gasPrice': '25'},
        {'value': '150', 'gas': '21500', 'gasPrice': '22'},
        {'value': '5000', 'gas': '25000', 'gasPrice': '30'}
    ]

    api_key = "dummy_key"
    address = "dummy_address"

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

import pytest
import pandas as pd
from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest


@pytest.fixture
def sample_data():
    data = {
        'value': [100, 200, 150, 0, 500, 600, 700],
        'gas': [21000, 21000, 21000, 21000, 21000, 21000, 21000],
        'gasPrice': [50, 50, 50, 50, 50, 50, 50]
    }
    return pd.DataFrame(data)


def test_train_model(sample_data):
    detector = AnomalyDetectorIsolationForest(sample_data)
    model = detector.train_model()

    assert model is not None, "Model training failed."


def test_detect_anomalies(sample_data):
    detector = AnomalyDetectorIsolationForest(sample_data)
    detector.train_model()
    result_df = detector.detect_anomalies()

    assert 'anomaly' in result_df.columns, "Anomaly detection failed."
    assert result_df['anomaly'].isin(['normal', 'anomaly']).all(), "Invalid anomaly labels detected."

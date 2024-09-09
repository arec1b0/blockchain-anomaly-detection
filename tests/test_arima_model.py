import pytest
import pandas as pd
from src.anomaly_detection.arima_model import ARIMAModel


@pytest.fixture
def sample_time_series_data():
    data = {
        'timeStamp': pd.date_range(start='2023-01-01', periods=7, freq='D'),
        'value': [100, 200, 150, 300, 500, 600, 700]
    }
    return pd.DataFrame(data)


def test_prepare_data(sample_time_series_data):
    arima = ARIMAModel(sample_time_series_data)
    prepared_data = arima.prepare_data()

    assert prepared_data is not None, "Time series preparation failed."
    assert prepared_data.index.is_monotonic_increasing, "Time series index is not sorted."
    assert prepared_data.isna().sum() == 0, "There are missing values in the prepared time series."


def test_fit_model(sample_time_series_data):
    arima = ARIMAModel(sample_time_series_data, order=(5, 1, 0))
    model = arima.fit_model()

    assert model is not None, "ARIMA model fitting failed."


def test_forecast(sample_time_series_data):
    arima = ARIMAModel(sample_time_series_data, order=(5, 1, 0))
    arima.fit_model()
    forecast = arima.forecast(steps=3)

    assert len(forecast) == 3, "Forecasting failed - incorrect number of steps."
    assert forecast.isna().sum() == 0, "Forecast contains NaN values."

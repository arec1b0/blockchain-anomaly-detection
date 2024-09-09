"""
arima_model.py

This module implements time series forecasting using the ARIMA (AutoRegressive Integrated Moving Average) model.
ARIMA is effective for modeling and forecasting transaction trends over time.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on time series analysis.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class ARIMAModel:
    """
    ARIMAModel uses the ARIMA algorithm to forecast future values in time series transaction data.
    """

    def __init__(self, df: pd.DataFrame, order: tuple = (5, 1, 0)):
        """
        Initializes the ARIMAModel with the provided DataFrame and ARIMA order.

        :param df: DataFrame containing transaction data.
        :param order: Tuple representing ARIMA order (p, d, q). Default is (5, 1, 0).
        """
        self.df = df
        self.order = order
        self.model = None

    def prepare_data(self):
        """
        Prepares the time series data for ARIMA modeling by resampling and filling missing values.

        :return: Prepared time series data.
        """
        if 'timeStamp' in self.df.columns and 'value' in self.df.columns:
            time_series = self.df.resample('D', on='timeStamp').sum()['value']
            time_series = time_series.replace(0, np.nan).ffill()
            logger.info("Time series data prepared for ARIMA modeling.")
            return time_series
        else:
            logger.error("Data must contain 'timeStamp' and 'value' columns.")
            raise ValueError("Invalid data structure for time series.")

    def fit_model(self):
        """
        Fits the ARIMA model to the prepared time series data.

        :return: Fitted ARIMA model.
        """
        time_series = self.prepare_data()
        logger.info(f"Fitting ARIMA model with order {self.order}...")
        self.model = ARIMA(time_series, order=self.order)
        self.model = self.model.fit()
        logger.info("ARIMA model fitting completed.")
        return self.model

    def forecast(self, steps: int = 10):
        """
        Forecasts future values based on the fitted ARIMA model.

        :param steps: Number of future steps (days) to forecast.
        :return: Forecasted values.
        """
        if self.model is None:
            logger.error("Model must be fitted before forecasting.")
            raise ValueError("Model is not fitted.")

        logger.info(f"Forecasting the next {steps} steps using ARIMA model.")
        forecast = self.model.forecast(steps=steps)
        logger.info("Forecasting completed.")
        return forecast

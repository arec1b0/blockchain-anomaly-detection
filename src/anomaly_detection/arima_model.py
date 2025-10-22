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
        :raises ValueError: If df is None, empty, or missing required columns.
        :raises TypeError: If df is not a pandas DataFrame or order is not a tuple.
        """
        # Validate DataFrame
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")

        # Validate required columns
        required_columns = ['timeStamp', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")

        # Validate order parameter
        if not isinstance(order, tuple):
            raise TypeError(f"order must be a tuple, got {type(order).__name__}")
        if len(order) != 3:
            raise ValueError(f"order must be a tuple of 3 integers (p, d, q), got length {len(order)}")
        if not all(isinstance(x, int) for x in order):
            raise TypeError("All elements in order tuple must be integers")
        if not all(x >= 0 for x in order):
            raise ValueError(f"All elements in order tuple must be non-negative, got {order}")

        self.df = df.copy()  # Create a copy to avoid modifying the original
        self.order = order
        self.model = None
        logger.info(f"ARIMAModel initialized with {len(self.df)} rows, order={order}")

    def prepare_data(self):
        """
        Prepares the time series data for ARIMA modeling by resampling and filling missing values.

        :return: Prepared time series data.
        :raises ValueError: If required columns are missing or data preparation fails.
        :raises RuntimeError: If resampling or data preparation fails.
        """
        try:
            # Validate required columns (should already be validated in __init__, but double-check)
            if 'timeStamp' not in self.df.columns or 'value' not in self.df.columns:
                logger.error("Data must contain 'timeStamp' and 'value' columns.")
                raise ValueError("DataFrame must contain 'timeStamp' and 'value' columns")

            # Ensure timeStamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df['timeStamp']):
                logger.warning("timeStamp column is not datetime type, attempting conversion")
                self.df['timeStamp'] = pd.to_datetime(self.df['timeStamp'])

            time_series = self.df.resample('D', on='timeStamp').sum()['value']
            time_series = time_series.replace(0, np.nan).ffill()

            # Check if we have enough data points
            if len(time_series.dropna()) < sum(self.order):
                raise ValueError(f"Insufficient data points ({len(time_series.dropna())}) for ARIMA order {self.order}")

            logger.info(f"Time series data prepared for ARIMA modeling. {len(time_series)} data points.")
            return time_series
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error preparing time series data: {e}")
            raise RuntimeError(f"Failed to prepare time series data: {e}") from e

    def fit_model(self):
        """
        Fits the ARIMA model to the prepared time series data.

        :return: Fitted ARIMA model.
        :raises RuntimeError: If model fitting fails.
        """
        try:
            time_series = self.prepare_data()
            logger.info(f"Fitting ARIMA model with order {self.order}...")
            self.model = ARIMA(time_series, order=self.order)
            self.model = self.model.fit()
            logger.info("ARIMA model fitting completed.")
            return self.model
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise RuntimeError(f"Failed to fit ARIMA model: {e}") from e

    def forecast(self, steps: int = 10):
        """
        Forecasts future values based on the fitted ARIMA model.

        :param steps: Number of future steps (days) to forecast.
        :return: Forecasted values.
        :raises ValueError: If model hasn't been fitted or steps is invalid.
        :raises TypeError: If steps is not an integer.
        :raises RuntimeError: If forecasting fails.
        """
        try:
            # Validate steps parameter
            if not isinstance(steps, int):
                raise TypeError(f"steps must be an integer, got {type(steps).__name__}")
            if steps <= 0:
                raise ValueError(f"steps must be positive, got {steps}")

            # Check if model has been fitted
            if self.model is None:
                logger.error("Model must be fitted before forecasting.")
                raise ValueError("Model has not been fitted. Call fit_model() first.")

            logger.info(f"Forecasting the next {steps} steps using ARIMA model.")
            forecast = self.model.forecast(steps=steps)
            logger.info("Forecasting completed.")
            return forecast
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error forecasting with ARIMA model: {e}")
            raise RuntimeError(f"Failed to forecast: {e}") from e

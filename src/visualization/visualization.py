"""
visualization.py

This module provides visualization functionalities for transaction data, including plotting time series,
highlighting anomalies, and visualizing data distributions.

Adheres to the Single Responsibility Principle (SRP) by focusing solely on data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class DataVisualizer:
    """
    DataVisualizer class provides methods to visualize transaction data, including time series and anomalies.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initializes the DataVisualizer with the provided DataFrame.

        :param df: Pandas DataFrame containing transaction data.
        :raises ValueError: If df is None or empty.
        :raises TypeError: If df is not a pandas DataFrame.
        """
        # Validate DataFrame
        if df is None:
            raise ValueError("DataFrame cannot be None")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("DataFrame cannot be empty")

        self.df = df.copy()  # Create a copy to avoid modifying the original
        logger.info(f"DataVisualizer initialized with {len(self.df)} rows")

    def plot_time_series(self):
        """
        Plots the transaction values over time as a time series.

        :return: Displays a line plot of transaction values over time.
        :raises ValueError: If required columns are missing.
        :raises RuntimeError: If plotting fails.
        """
        try:
            # Validate required columns
            required_columns = ['timeStamp', 'value']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")

            logger.info("Plotting time series of transaction values.")
            plt.figure(figsize=(10, 6))
            plt.plot(self.df['timeStamp'], self.df['value'], label='Transaction Value', color='blue')
            plt.title('Transaction Values Over Time')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.show()
            logger.info("Time series plot displayed successfully.")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error plotting time series: {e}")
            raise RuntimeError(f"Failed to plot time series: {e}") from e

    def plot_anomalies(self):
        """
        Plots the detected anomalies in the time series data.

        :return: Displays a scatter plot with anomalies highlighted.
        :raises ValueError: If required columns are missing.
        :raises RuntimeError: If plotting fails.
        """
        try:
            # Validate required columns
            required_columns = ['timeStamp', 'value', 'anomaly']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")

            logger.info("Plotting anomalies in the transaction data.")
            normal = self.df[self.df['anomaly'] == 'normal']
            anomalies = self.df[self.df['anomaly'] == 'anomaly']

            plt.figure(figsize=(10, 6))
            plt.plot(normal['timeStamp'], normal['value'], label='Normal', color='blue')
            plt.scatter(anomalies['timeStamp'], anomalies['value'], color='red', label='Anomaly', marker='x')
            plt.title('Transaction Values with Anomalies')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.show()
            logger.info("Anomaly plot displayed successfully.")
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error plotting anomalies: {e}")
            raise RuntimeError(f"Failed to plot anomalies: {e}") from e

    def plot_distribution(self, column_name: str):
        """
        Plots the distribution of a given column (e.g., transaction values or gas prices).

        :param column_name: The name of the column to plot the distribution for.
        :return: Displays a histogram or kernel density estimate plot.
        :raises ValueError: If column_name is invalid or column doesn't exist.
        :raises TypeError: If column_name is not a string.
        :raises RuntimeError: If plotting fails.
        """
        try:
            # Validate column_name parameter
            if column_name is None:
                raise ValueError("column_name cannot be None")
            if not isinstance(column_name, str):
                raise TypeError(f"column_name must be a string, got {type(column_name).__name__}")
            if not column_name or column_name.strip() == "":
                raise ValueError("column_name cannot be empty")

            # Check if column exists
            if column_name not in self.df.columns:
                logger.error(f"Column '{column_name}' is missing from the data.")
                raise ValueError(f"Column '{column_name}' not found in DataFrame")

            logger.info(f"Plotting distribution of {column_name}.")
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column_name], kde=True)
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
            logger.info(f"Distribution plot for '{column_name}' displayed successfully.")
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error plotting distribution for '{column_name}': {e}")
            raise RuntimeError(f"Failed to plot distribution: {e}") from e

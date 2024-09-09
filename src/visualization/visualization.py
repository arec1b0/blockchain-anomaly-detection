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
        """
        self.df = df

    def plot_time_series(self):
        """
        Plots the transaction values over time as a time series.

        :return: Displays a line plot of transaction values over time.
        """
        if 'timeStamp' in self.df.columns and 'value' in self.df.columns:
            logger.info("Plotting time series of transaction values.")
            plt.figure(figsize=(10, 6))
            plt.plot(self.df['timeStamp'], self.df['value'], label='Transaction Value', color='blue')
            plt.title('Transaction Values Over Time')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            logger.error("Columns 'timeStamp' and 'value' are required for time series plotting.")
            raise ValueError("Invalid data structure for time series plotting.")

    def plot_anomalies(self):
        """
        Plots the detected anomalies in the time series data.

        :return: Displays a scatter plot with anomalies highlighted.
        """
        if 'timeStamp' in self.df.columns and 'value' in self.df.columns and 'anomaly' in self.df.columns:
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
        else:
            logger.error("Columns 'timeStamp', 'value', and 'anomaly' are required for anomaly plotting.")
            raise ValueError("Invalid data structure for anomaly plotting.")

    def plot_distribution(self, column_name: str):
        """
        Plots the distribution of a given column (e.g., transaction values or gas prices).

        :param column_name: The name of the column to plot the distribution for.
        :return: Displays a histogram or kernel density estimate plot.
        """
        if column_name in self.df.columns:
            logger.info(f"Plotting distribution of {column_name}.")
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[column_name], kde=True)
            plt.title(f'Distribution of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
        else:
            logger.error(f"Column '{column_name}' is missing from the data.")
            raise ValueError(f"Column '{column_name}' not found.")

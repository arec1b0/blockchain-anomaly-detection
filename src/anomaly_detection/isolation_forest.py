"""
isolation_forest.py

This module provides functionality for anomaly detection in transaction data using the Isolation Forest algorithm.
Isolation Forest is particularly well-suited for detecting outliers in high-dimensional datasets.

Adheres to the Single Responsibility Principle (SRP) by focusing only on anomaly detection.
"""

from sklearn.ensemble import IsolationForest
import pandas as pd
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


class AnomalyDetectorIsolationForest:
    """
    AnomalyDetectorIsolationForest uses the Isolation Forest algorithm to detect anomalies in transaction data.
    """

    def __init__(self, df: pd.DataFrame, contamination: float = 0.01, random_state: int = 42):
        """
        Initializes the anomaly detection model with the provided data.

        :param df: DataFrame containing the transaction data.
        :param contamination: The proportion of outliers in the data set (default is 1%).
        :param random_state: Seed for the random number generator to ensure reproducibility.
        """
        self.df = df
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(contamination=self.contamination,
                                     random_state=self.random_state)
        self.features = df[['value', 'gas', 'gasPrice']]  # Select features for analysis

    def train_model(self):
        """
        Trains the Isolation Forest model on the selected features of the dataset.

        :return: Trained Isolation Forest model.
        """
        logger.info("Training Isolation Forest model...")
        self.model.fit(self.features)
        logger.info("Model training completed.")
        return self.model

    def detect_anomalies(self):
        """
        Detects anomalies in the dataset using the trained Isolation Forest model.

        :return: DataFrame with an additional 'anomaly' column indicating normal or anomalous transactions.
        """
        logger.info("Detecting anomalies using Isolation Forest model...")
        self.df['anomaly'] = self.model.predict(self.features)
        self.df['anomaly'] = self.df['anomaly'].map({1: 'normal', -1: 'anomaly'})
        num_anomalies = len(self.df[self.df['anomaly'] == 'anomaly'])
        logger.info(f"Detected {num_anomalies} anomalous transactions.")
        return self.df

    def get_anomalies(self):
        """
        Returns the subset of the dataset that contains only the anomalous transactions.

        :return: DataFrame containing only anomalous transactions.
        """
        anomalies = self.df[self.df['anomaly'] == 'anomaly']
        logger.info(f"Returning {len(anomalies)} anomalous transactions.")
        return anomalies

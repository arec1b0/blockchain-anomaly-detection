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

    def __init__(self, df: pd.DataFrame = None, contamination: float = 0.01, random_state: int = 42):
        """
        Initializes the anomaly detection model with the provided data.

        :param df: DataFrame containing the transaction data (optional, can be provided later).
        :param contamination: The proportion of outliers in the data set (default is 1%).
        :param random_state: Seed for the random number generator to ensure reproducibility.
        :raises ValueError: If df is provided but None, empty, or missing required columns.
        :raises TypeError: If df is not a pandas DataFrame or parameters are of incorrect type.
        """
        # Validate contamination parameter
        if not isinstance(contamination, (int, float)):
            raise TypeError(f"contamination must be numeric, got {type(contamination).__name__}")
        if not 0 < contamination < 1:
            raise ValueError(f"contamination must be between 0 and 1 (exclusive), got {contamination}")

        # Validate random_state parameter
        if not isinstance(random_state, int):
            raise TypeError(f"random_state must be an integer, got {type(random_state).__name__}")

        self.contamination = contamination
        self.random_state = random_state

        # Initialize model
        try:
            self.model = IsolationForest(contamination=self.contamination,
                                         random_state=self.random_state)
        except Exception as e:
            logger.error(f"Error initializing Isolation Forest model: {e}")
            raise RuntimeError(f"Failed to initialize Isolation Forest model: {e}") from e

        # If DataFrame provided, validate and set it up
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
            if len(df) == 0:
                raise ValueError("DataFrame cannot be empty")

            # Validate required columns
            required_columns = ['value', 'gas', 'gasPrice']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")

            self.df = df.copy()  # Create a copy to avoid modifying the original
            self.features = self.df[['value', 'gas', 'gasPrice']]  # Select features for analysis
            logger.info(f"AnomalyDetectorIsolationForest initialized with {len(self.df)} rows, contamination={contamination}")
        else:
            self.df = None
            self.features = None
            logger.info(f"AnomalyDetectorIsolationForest initialized without DataFrame, contamination={contamination}")

    def train_model(self, df: pd.DataFrame = None):
        """
        Trains the Isolation Forest model on the selected features of the dataset.

        :param df: Optional DataFrame to train on. If provided, updates self.df and self.features.
        :return: Trained Isolation Forest model.
        :raises RuntimeError: If model training fails.
        :raises ValueError: If no DataFrame is available for training.
        """
        try:
            # If DataFrame provided, use it
            if df is not None:
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
                if len(df) == 0:
                    raise ValueError("DataFrame cannot be empty")

                # Validate required columns
                required_columns = ['value', 'gas', 'gasPrice']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"DataFrame must contain the following columns: {missing_columns}")

                self.df = df.copy()
                self.features = self.df[['value', 'gas', 'gasPrice']]

            # Check if features are available
            if self.features is None:
                raise ValueError("No DataFrame available for training. Provide df parameter or initialize with DataFrame.")

            logger.info(f"Training Isolation Forest model on {len(self.features)} samples...")
            self.model.fit(self.features)
            logger.info("Model training completed.")
            return self.model
        except (ValueError, TypeError):
            raise
        except Exception as e:
            logger.error(f"Error training Isolation Forest model: {e}")
            raise RuntimeError(f"Failed to train Isolation Forest model: {e}") from e

    def detect_anomalies(self):
        """
        Detects anomalies in the dataset using the trained Isolation Forest model.

        :return: DataFrame with an additional 'anomaly' column indicating normal or anomalous transactions.
        :raises RuntimeError: If anomaly detection fails.
        :raises ValueError: If model hasn't been trained yet.
        """
        try:
            # Check if model has been trained
            if not hasattr(self.model, 'estimators_'):
                raise ValueError("Model must be trained before detecting anomalies. Call train_model() first.")

            logger.info("Detecting anomalies using Isolation Forest model...")
            self.df['anomaly'] = self.model.predict(self.features)
            self.df['anomaly'] = self.df['anomaly'].map({1: 'normal', -1: 'anomaly'})
            num_anomalies = len(self.df[self.df['anomaly'] == 'anomaly'])
            logger.info(f"Detected {num_anomalies} anomalous transactions.")
            return self.df
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise RuntimeError(f"Failed to detect anomalies: {e}") from e

    def get_anomalies(self):
        """
        Returns the subset of the dataset that contains only the anomalous transactions.

        :return: DataFrame containing only anomalous transactions.
        :raises ValueError: If anomaly detection hasn't been performed yet.
        :raises RuntimeError: If retrieval fails.
        """
        try:
            if 'anomaly' not in self.df.columns:
                raise ValueError("Anomaly column not found. Run detect_anomalies() first.")

            anomalies = self.df[self.df['anomaly'] == 'anomaly']
            logger.info(f"Returning {len(anomalies)} anomalous transactions.")
            return anomalies
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving anomalies: {e}")
            raise RuntimeError(f"Failed to retrieve anomalies: {e}") from e

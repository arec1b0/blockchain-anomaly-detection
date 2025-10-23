"""
Stream processor for real-time blockchain anomaly detection.

This module defines the `StreamProcessor` class, which is responsible for
processing streaming blockchain transactions for real-time anomaly detection.
It receives transaction data, applies data transformations, and uses trained
machine learning models to detect anomalies.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import pickle
import os
from prometheus_client import Counter, Histogram, Gauge

from src.anomaly_detection.isolation_forest import AnomalyDetectorIsolationForest

logger = logging.getLogger(__name__)

# Prometheus metrics
transactions_processed = Counter(
    'transactions_processed_total',
    'Total number of transactions processed.',
    ['status']
)

anomalies_detected = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected.',
    ['severity']
)

processing_duration = Histogram(
    'stream_processing_duration_seconds',
    'Time spent processing transactions in seconds.'
)

active_transactions = Gauge(
    'active_transactions_current',
    'Current number of transactions being processed.'
)

model_score = Gauge(
    'anomaly_model_score',
    'Latest anomaly score from the model.'
)


class StreamProcessor:
    """
    Processes streaming blockchain transactions for real-time anomaly detection.

    This processor receives transaction data, applies data transformations,
    and uses trained machine learning models to detect anomalies in real-time.

    Attributes:
        model_path (Optional[str]): The path to the pre-trained model file.
        batch_size (int): The number of transactions to batch before processing.
        contamination (float): The expected proportion of anomalies in the data.
        model (Optional[AnomalyDetectorIsolationForest]): The anomaly detection model.
        transaction_buffer (List[Dict[str, Any]]): A buffer for incoming transactions.
        anomaly_buffer (List[Dict[str, Any]]): A buffer for detected anomalies.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 100,
        contamination: float = 0.01
    ):
        """
        Initializes the StreamProcessor.

        Args:
            model_path (Optional[str]): The path to the pre-trained model file in pickle format.
                Defaults to None.
            batch_size (int): The number of transactions to batch before processing.
                Defaults to 100.
            contamination (float): The expected proportion of anomalies in the data.
                Defaults to 0.01.
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.contamination = contamination
        self.model: Optional[AnomalyDetectorIsolationForest] = None
        self.transaction_buffer: List[Dict[str, Any]] = []
        self.anomaly_buffer: List[Dict[str, Any]] = []

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded pre-trained model from {model_path}")
        else:
            logger.info("No pre-trained model provided. Will train on first batch.")

        logger.info(
            f"Stream processor initialized with batch_size={batch_size}, "
            f"contamination={contamination}"
        )

    def load_model(self, model_path: str) -> None:
        """
        Loads a pre-trained model from disk.

        Args:
            model_path (str): The path to the model file.

        Raises:
            FileNotFoundError: If the model file does not exist.
            Exception: If the model loading fails.
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_path}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def save_model(self, model_path: str) -> None:
        """
        Saves the current model to disk.

        Args:
            model_path (str): The path where the model should be saved.

        Raises:
            Exception: If the model saving fails.
        """
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved successfully to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def process_transaction(self, transaction: Dict[str, Any]) -> None:
        """
        Processes a single transaction from the stream.

        Args:
            transaction (Dict[str, Any]): The transaction data from Kafka.
        """
        with processing_duration.time():
            active_transactions.inc()
            try:
                # Validate and transform transaction data
                processed_tx = self._transform_transaction(transaction)

                # Add to buffer
                self.transaction_buffer.append(processed_tx)

                # Process batch if buffer is full
                if len(self.transaction_buffer) >= self.batch_size:
                    self._process_batch()

                transactions_processed.labels(status='success').inc()

            except Exception as e:
                logger.error(f"Error processing transaction: {e}", exc_info=True)
                transactions_processed.labels(status='error').inc()
            finally:
                active_transactions.dec()

    def _transform_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms and validates transaction data.

        Args:
            transaction (Dict[str, Any]): The raw transaction data.

        Returns:
            Dict[str, Any]: The transformed transaction data.

        Raises:
            ValueError: If required fields are missing or have invalid values.
        """
        # Required fields
        required_fields = ['hash', 'value', 'gas', 'gasPrice']

        # Validate required fields
        for field in required_fields:
            if field not in transaction:
                raise ValueError(f"Missing required field: {field}")

        # Convert string values to numeric
        try:
            value = float(transaction.get('value', 0))
            gas = float(transaction.get('gas', 0))
            gas_price = float(transaction.get('gasPrice', 0))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric value in transaction: {e}")

        # Create transformed transaction
        transformed = {
            'hash': transaction['hash'],
            'value': value,
            'gas': gas,
            'gasPrice': gas_price,
            'from': transaction.get('from', ''),
            'to': transaction.get('to', ''),
            'timestamp': transaction.get('timestamp', datetime.utcnow().isoformat()),
            'blockNumber': transaction.get('blockNumber', 0)
        }

        return transformed

    def _process_batch(self) -> None:
        """
        Processes a batch of transactions for anomaly detection.
        """
        if not self.transaction_buffer:
            return

        logger.info(f"Processing batch of {len(self.transaction_buffer)} transactions")

        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.transaction_buffer)

            # Ensure we have required columns
            if not all(col in df.columns for col in ['value', 'gas', 'gasPrice']):
                logger.error("Missing required columns in transaction batch")
                self.transaction_buffer.clear()
                return

            # Initialize or update model
            if self.model is None:
                logger.info("Training new model on first batch")
                self.model = AnomalyDetectorIsolationForest(
                    df=df,
                    contamination=self.contamination
                )
                self.model.train_model()
            else:
                # For pre-trained model, just use it for prediction
                pass

            # Detect anomalies
            result_df = self.model.detect_anomalies()

            # Process anomalies
            anomalies = result_df[result_df['anomaly'] == 'anomaly']

            if len(anomalies) > 0:
                logger.warning(f"Detected {len(anomalies)} anomalies in batch")

                for _, anomaly in anomalies.iterrows():
                    anomaly_record = {
                        'hash': anomaly.get('hash', 'unknown'),
                        'value': float(anomaly['value']),
                        'gas': float(anomaly['gas']),
                        'gasPrice': float(anomaly['gasPrice']),
                        'from': anomaly.get('from', ''),
                        'to': anomaly.get('to', ''),
                        'timestamp': anomaly.get('timestamp', datetime.utcnow().isoformat()),
                        'detected_at': datetime.utcnow().isoformat(),
                        'severity': self._calculate_severity(anomaly)
                    }

                    self.anomaly_buffer.append(anomaly_record)

                    # Update metrics
                    anomalies_detected.labels(
                        severity=anomaly_record['severity']
                    ).inc()

                    # Log high-severity anomalies
                    if anomaly_record['severity'] in ['high', 'critical']:
                        logger.warning(
                            f"High-severity anomaly detected: {anomaly_record['hash']}"
                        )

            # Clear buffer
            self.transaction_buffer.clear()

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
            self.transaction_buffer.clear()

    def _calculate_severity(self, anomaly: pd.Series) -> str:
        """
        Calculates the severity level of an anomaly.

        Args:
            anomaly (pd.Series): The anomaly transaction data.

        Returns:
            str: The severity level, which can be 'low', 'medium', 'high', or 'critical'.
        """
        try:
            # Calculate deviation from mean
            value = float(anomaly['value'])
            gas_price = float(anomaly['gasPrice'])

            # Simple heuristic based on transaction value and gas price
            if value > 1000000 or gas_price > 100000:
                return 'critical'
            elif value > 100000 or gas_price > 50000:
                return 'high'
            elif value > 10000 or gas_price > 10000:
                return 'medium'
            else:
                return 'low'
        except Exception as e:
            logger.error(f"Error calculating severity: {e}")
            return 'low'

    def get_anomalies(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Gets the detected anomalies from the buffer.

        Args:
            limit (Optional[int]): The maximum number of anomalies to return.
                If None, all anomalies are returned. Defaults to None.

        Returns:
            List[Dict[str, Any]]: A list of anomaly records.
        """
        if limit:
            return self.anomaly_buffer[-limit:]
        return self.anomaly_buffer.copy()

    def clear_anomaly_buffer(self) -> None:
        """
        Clears the anomaly buffer.
        """
        count = len(self.anomaly_buffer)
        self.anomaly_buffer.clear()
        logger.info(f"Cleared {count} anomalies from buffer")

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets the processing statistics.

        Returns:
            Dict[str, Any]: A dictionary containing processing statistics.
        """
        return {
            'buffer_size': len(self.transaction_buffer),
            'anomalies_detected': len(self.anomaly_buffer),
            'model_loaded': self.model is not None,
            'batch_size': self.batch_size,
            'contamination': self.contamination
        }

    def flush(self) -> None:
        """
        Forces the processing of the remaining transactions in the buffer.
        """
        if self.transaction_buffer:
            logger.info(f"Flushing {len(self.transaction_buffer)} transactions")
            self._process_batch()

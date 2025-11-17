"""
Model evaluation utilities.

This module provides utilities for evaluating ML model performance,
including metrics calculation and comparison.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Utility class for evaluating ML model performance.
    """

    @staticmethod
    def evaluate_classification(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate classification model performance.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities (optional)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # ROC AUC if scores provided
        if y_scores is not None:
            try:
                # Convert to binary if needed
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")

        return metrics

    @staticmethod
    def evaluate_anomaly_detection(
        predictions: np.ndarray,
        scores: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection model performance.

        Args:
            predictions: Model predictions (-1 for anomaly, 1 for normal)
            scores: Anomaly scores
            y_true: True labels (optional, for supervised evaluation)

        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'num_samples': len(predictions),
            'num_anomalies_detected': int(np.sum(predictions == -1)),
            'num_normal_detected': int(np.sum(predictions == 1)),
            'anomaly_rate': float(np.sum(predictions == -1) / len(predictions)),
            'mean_anomaly_score': float(np.mean(scores)),
            'std_anomaly_score': float(np.std(scores)),
            'min_anomaly_score': float(np.min(scores)),
            'max_anomaly_score': float(np.max(scores)),
            'score_range': float(np.max(scores) - np.min(scores))
        }

        # If true labels available, calculate supervised metrics
        if y_true is not None:
            # Convert predictions to binary (0 = normal, 1 = anomaly)
            y_pred_binary = (predictions == -1).astype(int)
            y_true_binary = (y_true == -1).astype(int) if isinstance(y_true[0], (int, np.integer)) else y_true

            try:
                metrics['precision'] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['recall'] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
                metrics['f1_score'] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

                # Confusion matrix
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                metrics['confusion_matrix'] = cm.tolist()
                metrics['true_positives'] = int(cm[1, 1])
                metrics['false_positives'] = int(cm[0, 1])
                metrics['true_negatives'] = int(cm[0, 0])
                metrics['false_negatives'] = int(cm[1, 0])
            except Exception as e:
                logger.warning(f"Could not calculate supervised metrics: {e}")

        return metrics

    @staticmethod
    def compare_models(
        metrics1: Dict[str, Any],
        metrics2: Dict[str, Any],
        metric_names: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            metrics1: Metrics from first model
            metrics2: Metrics from second model
            metric_names: List of metric names to compare (None = all)

        Returns:
            Comparison dictionary with differences
        """
        if metric_names is None:
            # Get common numeric metrics
            metric_names = set(metrics1.keys()) & set(metrics2.keys())
            metric_names = [m for m in metric_names if isinstance(metrics1[m], (int, float))]

        comparison = {}
        for metric in metric_names:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric]
                val2 = metrics2[metric]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else 0
                    comparison[metric] = {
                        'model1': val1,
                        'model2': val2,
                        'difference': diff,
                        'percent_change': pct_change
                    }

        return comparison


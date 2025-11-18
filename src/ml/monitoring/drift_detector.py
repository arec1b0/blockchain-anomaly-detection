"""
Model drift detection for monitoring model performance degradation.

This module provides functionality for detecting:
- Feature drift: Changes in input data distribution
- Concept drift: Changes in the relationship between features and predictions
- Performance drift: Degradation in model accuracy over time
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sqlalchemy.orm import Session

from src.database.repositories.transaction_repository import TransactionRepository
from src.database.repositories.anomaly_repository import AnomalyRepository
from src.database.repositories.model_repository import ModelVersionRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftType:
    """Types of drift that can be detected."""
    FEATURE_DRIFT = "feature_drift"
    CONCEPT_DRIFT = "concept_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class DriftDetector:
    """
    Detects various types of model drift.

    Monitors:
    1. Feature drift - Distribution changes in input features
    2. Concept drift - Changes in prediction patterns
    3. Performance drift - Accuracy degradation over time

    Uses statistical tests:
    - Kolmogorov-Smirnov test for continuous features
    - Chi-square test for categorical features
    - Population Stability Index (PSI) for distribution changes
    """

    def __init__(
        self,
        db_session: Session,
        reference_window_days: int = 30,
        detection_window_days: int = 7
    ):
        """
        Initialize DriftDetector.

        Args:
            db_session: SQLAlchemy database session
            reference_window_days: Days for reference baseline period
            detection_window_days: Days for current detection window
        """
        self.db = db_session
        self.transaction_repo = TransactionRepository(db_session)
        self.anomaly_repo = AnomalyRepository(db_session)
        self.model_version_repo = ModelVersionRepository(db_session)
        self.reference_window_days = reference_window_days
        self.detection_window_days = detection_window_days

    def detect_drift(
        self,
        model_version_id: str,
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect all types of drift for a model version.

        Args:
            model_version_id: ID of model version to check
            drift_threshold: Threshold for drift detection (0.1 = 10% change)

        Returns:
            Dictionary with drift detection results
        """
        logger.info(f"Detecting drift for model version {model_version_id}")

        # Get reference and detection periods
        now = datetime.utcnow()
        detection_start = now - timedelta(days=self.detection_window_days)
        reference_end = detection_start
        reference_start = reference_end - timedelta(days=self.reference_window_days)

        # Get transactions for both periods
        reference_transactions = self.transaction_repo.get_by_date_range(
            reference_start,
            reference_end,
            limit=10000
        )
        detection_transactions = self.transaction_repo.get_by_date_range(
            detection_start,
            now,
            limit=10000
        )

        if not reference_transactions or not detection_transactions:
            logger.warning("Insufficient data for drift detection")
            return {
                "drift_detected": False,
                "message": "Insufficient data",
                "reference_count": len(reference_transactions),
                "detection_count": len(detection_transactions)
            }

        # Convert to DataFrames
        ref_df = self._transactions_to_df(reference_transactions)
        det_df = self._transactions_to_df(detection_transactions)

        # Detect feature drift
        feature_drift = self._detect_feature_drift(ref_df, det_df, drift_threshold)

        # Detect concept drift (anomaly patterns)
        concept_drift = self._detect_concept_drift(
            model_version_id,
            reference_start,
            reference_end,
            detection_start,
            now,
            drift_threshold
        )

        # Detect performance drift (if we have labeled data)
        performance_drift = self._detect_performance_drift(
            model_version_id,
            reference_start,
            reference_end,
            detection_start,
            now,
            drift_threshold
        )

        # Aggregate results
        drift_detected = (
            feature_drift["drift_detected"] or
            concept_drift["drift_detected"] or
            performance_drift["drift_detected"]
        )

        return {
            "model_version_id": model_version_id,
            "drift_detected": drift_detected,
            "drift_threshold": drift_threshold,
            "reference_period": {
                "start": reference_start.isoformat(),
                "end": reference_end.isoformat(),
                "samples": len(reference_transactions)
            },
            "detection_period": {
                "start": detection_start.isoformat(),
                "end": now.isoformat(),
                "samples": len(detection_transactions)
            },
            "feature_drift": feature_drift,
            "concept_drift": concept_drift,
            "performance_drift": performance_drift,
            "recommendation": self._get_recommendation(
                feature_drift,
                concept_drift,
                performance_drift
            )
        }

    def _transactions_to_df(self, transactions: List) -> pd.DataFrame:
        """Convert transaction list to DataFrame."""
        return pd.DataFrame([{
            'value': t.value,
            'gas': t.gas,
            'gas_price': t.gas_price,
            'timestamp': t.timestamp
        } for t in transactions])

    def _detect_feature_drift(
        self,
        ref_df: pd.DataFrame,
        det_df: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions using statistical tests.

        Uses:
        - Kolmogorov-Smirnov test for continuous features
        - Population Stability Index (PSI) for distribution changes
        """
        drifted_features = []
        feature_metrics = {}

        # Features to check
        features = ['value', 'gas', 'gas_price']

        for feature in features:
            if feature not in ref_df.columns or feature not in det_df.columns:
                continue

            ref_data = ref_df[feature].dropna()
            det_data = det_df[feature].dropna()

            if len(ref_data) == 0 or len(det_data) == 0:
                continue

            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_data, det_data)

            # Population Stability Index (PSI)
            psi = self._calculate_psi(ref_data, det_data)

            # Distribution statistics
            ref_mean = float(ref_data.mean())
            det_mean = float(det_data.mean())
            mean_change = abs(det_mean - ref_mean) / (ref_mean + 1e-10)

            # Detect drift
            drift = (
                ks_pvalue < 0.05 or  # Significant distribution change
                psi > 0.2 or  # High PSI indicates drift
                mean_change > threshold  # Mean shifted significantly
            )

            feature_metrics[feature] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pvalue), 4),
                "psi": round(float(psi), 4),
                "reference_mean": round(ref_mean, 4),
                "detection_mean": round(det_mean, 4),
                "mean_change_pct": round(mean_change * 100, 2),
                "drift_detected": drift
            }

            if drift:
                drifted_features.append(feature)

        return {
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "feature_metrics": feature_metrics,
            "drift_severity": self._calculate_drift_severity(feature_metrics)
        }

    def _calculate_psi(
        self,
        reference: pd.Series,
        detection: pd.Series,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures the change in distribution between two samples.
        PSI < 0.1: No significant change
        0.1 < PSI < 0.2: Moderate change
        PSI > 0.2: Significant change (drift detected)
        """
        # Create bins based on reference data
        _, bin_edges = np.histogram(reference, bins=bins)

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        det_hist, _ = np.histogram(detection, bins=bin_edges)

        # Normalize to get percentages
        ref_pct = ref_hist / len(reference) + 1e-10  # Add small value to avoid log(0)
        det_pct = det_hist / len(detection) + 1e-10

        # Calculate PSI
        psi = np.sum((det_pct - ref_pct) * np.log(det_pct / ref_pct))

        return abs(float(psi))

    def _detect_concept_drift(
        self,
        model_version_id: str,
        ref_start: datetime,
        ref_end: datetime,
        det_start: datetime,
        det_end: datetime,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Detect concept drift by comparing anomaly detection patterns.

        Checks if the model's behavior has changed (detecting more/fewer anomalies,
        different severity distribution, etc.)
        """
        # Get anomalies for both periods
        ref_anomalies = self.anomaly_repo.get_by_model_version(
            model_version_id,
            start_date=ref_start,
            end_date=ref_end
        )
        det_anomalies = self.anomaly_repo.get_by_model_version(
            model_version_id,
            start_date=det_start,
            end_date=det_end
        )

        if not ref_anomalies:
            return {
                "drift_detected": False,
                "message": "No reference anomalies for comparison"
            }

        # Calculate metrics
        ref_rate = len(ref_anomalies) / max(self.reference_window_days, 1)
        det_rate = len(det_anomalies) / max(self.detection_window_days, 1)

        rate_change = abs(det_rate - ref_rate) / (ref_rate + 1e-10)

        # Average confidence comparison
        ref_confidence = np.mean([a.confidence for a in ref_anomalies]) if ref_anomalies else 0
        det_confidence = np.mean([a.confidence for a in det_anomalies]) if det_anomalies else 0
        confidence_change = abs(det_confidence - ref_confidence)

        # Severity distribution comparison
        ref_severity = self._get_severity_distribution(ref_anomalies)
        det_severity = self._get_severity_distribution(det_anomalies)

        # Detect drift
        drift = (
            rate_change > threshold or
            confidence_change > threshold
        )

        return {
            "drift_detected": drift,
            "anomaly_rate_change_pct": round(rate_change * 100, 2),
            "reference_anomaly_rate": round(ref_rate, 4),
            "detection_anomaly_rate": round(det_rate, 4),
            "confidence_change": round(confidence_change, 4),
            "reference_confidence": round(ref_confidence, 4),
            "detection_confidence": round(det_confidence, 4),
            "reference_severity": ref_severity,
            "detection_severity": det_severity
        }

    def _detect_performance_drift(
        self,
        model_version_id: str,
        ref_start: datetime,
        ref_end: datetime,
        det_start: datetime,
        det_end: datetime,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Detect performance drift using reviewed anomalies (labeled data).

        If we have user-reviewed anomalies, we can calculate precision/recall
        and detect if model performance has degraded.
        """
        # Get reviewed anomalies for both periods
        ref_anomalies = self.anomaly_repo.get_by_model_version(
            model_version_id,
            start_date=ref_start,
            end_date=ref_end
        )
        det_anomalies = self.anomaly_repo.get_by_model_version(
            model_version_id,
            start_date=det_start,
            end_date=det_end
        )

        # Filter for reviewed anomalies
        ref_reviewed = [a for a in ref_anomalies if a.reviewed]
        det_reviewed = [a for a in det_anomalies if a.reviewed]

        if not ref_reviewed:
            return {
                "drift_detected": False,
                "message": "No reviewed anomalies for performance comparison"
            }

        # Calculate precision (true anomalies / all detected)
        ref_precision = self._calculate_precision(ref_reviewed)
        det_precision = self._calculate_precision(det_reviewed) if det_reviewed else 0

        precision_change = abs(det_precision - ref_precision)

        # Detect drift
        drift = precision_change > threshold

        return {
            "drift_detected": drift,
            "reference_precision": round(ref_precision, 4),
            "detection_precision": round(det_precision, 4),
            "precision_change": round(precision_change, 4),
            "reference_reviewed_count": len(ref_reviewed),
            "detection_reviewed_count": len(det_reviewed)
        }

    def _get_severity_distribution(self, anomalies: List) -> Dict[str, int]:
        """Get distribution of anomaly severities."""
        distribution = {}
        for anomaly in anomalies:
            severity = anomaly.severity.value if hasattr(anomaly.severity, 'value') else str(anomaly.severity)
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution

    def _calculate_precision(self, reviewed_anomalies: List) -> float:
        """
        Calculate precision from reviewed anomalies.

        Precision = True Positives / (True Positives + False Positives)
        """
        if not reviewed_anomalies:
            return 0.0

        true_positives = sum(1 for a in reviewed_anomalies if not a.false_positive)
        total = len(reviewed_anomalies)

        return true_positives / total if total > 0 else 0.0

    def _calculate_drift_severity(self, feature_metrics: Dict[str, Any]) -> str:
        """
        Calculate overall drift severity.

        Returns: "none", "low", "moderate", "high", or "critical"
        """
        if not feature_metrics:
            return "none"

        drifted_count = sum(1 for m in feature_metrics.values() if m.get("drift_detected", False))
        total_features = len(feature_metrics)

        if drifted_count == 0:
            return "none"

        drift_ratio = drifted_count / total_features

        # Check max PSI across features
        max_psi = max(m.get("psi", 0) for m in feature_metrics.values())

        if drift_ratio >= 0.75 or max_psi > 0.5:
            return "critical"
        elif drift_ratio >= 0.5 or max_psi > 0.3:
            return "high"
        elif drift_ratio >= 0.25 or max_psi > 0.2:
            return "moderate"
        else:
            return "low"

    def _get_recommendation(
        self,
        feature_drift: Dict,
        concept_drift: Dict,
        performance_drift: Dict
    ) -> str:
        """Generate recommendation based on drift detection results."""
        feature_detected = feature_drift.get("drift_detected", False)
        concept_detected = concept_drift.get("drift_detected", False)
        performance_detected = performance_drift.get("drift_detected", False)

        severity = feature_drift.get("drift_severity", "none")

        if performance_detected:
            return "CRITICAL: Performance degradation detected. Retrain model immediately with recent data."

        if severity == "critical":
            return "CRITICAL: Severe feature drift detected. Retrain model with recent data."

        if feature_detected and concept_detected:
            return "HIGH: Both feature and concept drift detected. Consider retraining model."

        if severity == "high":
            return "HIGH: Significant feature drift. Monitor closely and schedule retraining."

        if concept_detected:
            return "MODERATE: Concept drift detected. Model behavior has changed."

        if severity == "moderate":
            return "MODERATE: Moderate feature drift. Continue monitoring."

        if feature_detected:
            return "LOW: Minor feature drift detected. No immediate action needed."

        return "NONE: No significant drift detected. Model is performing normally."

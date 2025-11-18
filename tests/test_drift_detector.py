"""
Tests for drift detection.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml.monitoring.drift_detector import DriftDetector, DriftType


@pytest.fixture
def db_session():
    """Mock database session."""
    return Mock()


@pytest.fixture
def drift_detector(db_session):
    """Create DriftDetector instance."""
    return DriftDetector(
        db_session,
        reference_window_days=30,
        detection_window_days=7
    )


@pytest.fixture
def sample_transactions():
    """Create sample transactions."""
    class Transaction:
        def __init__(self, value, gas, gas_price, timestamp):
            self.value = value
            self.gas = gas
            self.gas_price = gas_price
            self.timestamp = timestamp

    # Generate 100 sample transactions
    transactions = []
    base_time = datetime.utcnow()
    for i in range(100):
        transactions.append(Transaction(
            value=100.0 + np.random.randn() * 10,
            gas=21000.0 + np.random.randn() * 1000,
            gas_price=20.0 + np.random.randn() * 2,
            timestamp=base_time - timedelta(hours=i)
        ))

    return transactions


class TestCalculatePSI:
    """Tests for PSI calculation."""

    def test_identical_distributions(self, drift_detector):
        """Test PSI for identical distributions should be ~0."""
        # Setup
        data = pd.Series(np.random.randn(1000))

        # Execute
        psi = drift_detector._calculate_psi(data, data)

        # Assert - PSI should be very small for identical data
        assert psi < 0.01

    def test_different_distributions(self, drift_detector):
        """Test PSI for different distributions should be > 0.2."""
        # Setup
        reference = pd.Series(np.random.randn(1000))
        detection = pd.Series(np.random.randn(1000) + 5)  # Shifted distribution

        # Execute
        psi = drift_detector._calculate_psi(reference, detection)

        # Assert - PSI should be high for different distributions
        assert psi > 0.2


class TestDetectFeatureDrift:
    """Tests for feature drift detection."""

    def test_no_drift(self, drift_detector):
        """Test feature drift detection with no drift."""
        # Setup - same distribution
        ref_df = pd.DataFrame({
            'value': np.random.randn(1000) * 10 + 100,
            'gas': np.random.randn(1000) * 1000 + 21000,
            'gas_price': np.random.randn(1000) * 2 + 20
        })

        det_df = pd.DataFrame({
            'value': np.random.randn(1000) * 10 + 100,
            'gas': np.random.randn(1000) * 1000 + 21000,
            'gas_price': np.random.randn(1000) * 2 + 20
        })

        # Execute
        result = drift_detector._detect_feature_drift(ref_df, det_df, threshold=0.1)

        # Assert
        assert result["drift_detected"] is False or len(result["drifted_features"]) == 0

    def test_significant_drift(self, drift_detector):
        """Test feature drift detection with significant drift."""
        # Setup - shifted distribution
        ref_df = pd.DataFrame({
            'value': np.random.randn(1000) * 10 + 100,
            'gas': np.random.randn(1000) * 1000 + 21000,
            'gas_price': np.random.randn(1000) * 2 + 20
        })

        det_df = pd.DataFrame({
            'value': np.random.randn(1000) * 10 + 200,  # Shifted significantly
            'gas': np.random.randn(1000) * 1000 + 21000,
            'gas_price': np.random.randn(1000) * 2 + 20
        })

        # Execute
        result = drift_detector._detect_feature_drift(ref_df, det_df, threshold=0.1)

        # Assert
        assert result["drift_detected"] is True
        assert 'value' in result["drifted_features"]
        assert result["drift_severity"] in ["moderate", "high", "critical"]


class TestDetectDrift:
    """Tests for overall drift detection."""

    @patch.object(DriftDetector, '_detect_feature_drift')
    @patch.object(DriftDetector, '_detect_concept_drift')
    @patch.object(DriftDetector, '_detect_performance_drift')
    def test_detect_drift_with_mocks(
        self,
        mock_performance,
        mock_concept,
        mock_feature,
        drift_detector,
        sample_transactions
    ):
        """Test drift detection with mocked sub-detectors."""
        # Setup
        mock_feature.return_value = {
            "drift_detected": True,
            "drifted_features": ["value"],
            "drift_severity": "high"
        }
        mock_concept.return_value = {
            "drift_detected": False
        }
        mock_performance.return_value = {
            "drift_detected": False
        }

        drift_detector.transaction_repo.get_by_date_range = Mock(
            return_value=sample_transactions
        )

        # Execute
        result = drift_detector.detect_drift(
            model_version_id="test-version",
            drift_threshold=0.1
        )

        # Assert
        assert result["drift_detected"] is True
        assert result["model_version_id"] == "test-version"
        assert "feature_drift" in result
        assert "concept_drift" in result
        assert "performance_drift" in result
        assert "recommendation" in result


class TestGetRecommendation:
    """Tests for recommendation generation."""

    def test_critical_performance_drift(self, drift_detector):
        """Test recommendation for critical performance drift."""
        # Setup
        feature_drift = {"drift_detected": False, "drift_severity": "none"}
        concept_drift = {"drift_detected": False}
        performance_drift = {"drift_detected": True}

        # Execute
        recommendation = drift_detector._get_recommendation(
            feature_drift,
            concept_drift,
            performance_drift
        )

        # Assert
        assert "CRITICAL" in recommendation
        assert "retrain" in recommendation.lower()

    def test_no_drift(self, drift_detector):
        """Test recommendation when no drift detected."""
        # Setup
        feature_drift = {"drift_detected": False, "drift_severity": "none"}
        concept_drift = {"drift_detected": False}
        performance_drift = {"drift_detected": False}

        # Execute
        recommendation = drift_detector._get_recommendation(
            feature_drift,
            concept_drift,
            performance_drift
        )

        # Assert
        assert "NONE" in recommendation or "No" in recommendation


class TestCalculateDriftSeverity:
    """Tests for drift severity calculation."""

    def test_no_drift(self, drift_detector):
        """Test severity calculation with no drift."""
        # Setup
        feature_metrics = {
            "value": {"drift_detected": False, "psi": 0.05},
            "gas": {"drift_detected": False, "psi": 0.03}
        }

        # Execute
        severity = drift_detector._calculate_drift_severity(feature_metrics)

        # Assert
        assert severity == "none"

    def test_critical_drift(self, drift_detector):
        """Test severity calculation with critical drift."""
        # Setup
        feature_metrics = {
            "value": {"drift_detected": True, "psi": 0.6},
            "gas": {"drift_detected": True, "psi": 0.4},
            "gas_price": {"drift_detected": True, "psi": 0.7}
        }

        # Execute
        severity = drift_detector._calculate_drift_severity(feature_metrics)

        # Assert
        assert severity == "critical"

    def test_moderate_drift(self, drift_detector):
        """Test severity calculation with moderate drift."""
        # Setup
        feature_metrics = {
            "value": {"drift_detected": True, "psi": 0.25},
            "gas": {"drift_detected": False, "psi": 0.1}
        }

        # Execute
        severity = drift_detector._calculate_drift_severity(feature_metrics)

        # Assert
        assert severity in ["moderate", "low"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

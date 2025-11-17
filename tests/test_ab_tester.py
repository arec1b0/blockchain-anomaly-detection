"""
Tests for A/B testing framework.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.ml.deployment.ab_tester import ABTester, DeploymentStrategy
from src.database.models import ModelVersion, Model


@pytest.fixture
def db_session():
    """Mock database session."""
    return Mock()


@pytest.fixture
def ab_tester(db_session):
    """Create ABTester instance."""
    return ABTester(db_session)


@pytest.fixture
def sample_model_version():
    """Create sample model version."""
    return ModelVersion(
        id="test-version-1",
        model_id="test-model",
        version="1.0.0",
        is_deployed=False,
        traffic_percentage=0.0,
        storage_path="/models/test/1.0.0",
        checksum="abc123",
        metrics={"accuracy": 0.95}
    )


class TestDeployModel:
    """Tests for deploy_model method."""

    def test_deploy_shadow_strategy(self, ab_tester, sample_model_version, db_session):
        """Test shadow deployment (0% traffic)."""
        # Setup
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.deploy_model(
            model_version_id="test-version-1",
            strategy=DeploymentStrategy.SHADOW
        )

        # Assert
        assert result["model_version_id"] == "test-version-1"
        assert result["strategy"] == DeploymentStrategy.SHADOW
        assert result["traffic_percentage"] == 0.0
        assert result["status"] == "deployed"

    def test_deploy_canary_strategy(self, ab_tester, sample_model_version):
        """Test canary deployment (starts at 10%)."""
        # Setup
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.deploy_model(
            model_version_id="test-version-1",
            strategy=DeploymentStrategy.CANARY
        )

        # Assert
        assert result["traffic_percentage"] == 10.0

    def test_deploy_full_strategy_undeploys_others(self, ab_tester, sample_model_version):
        """Test full deployment undeploys other versions."""
        # Setup
        existing_version = ModelVersion(
            id="existing-version",
            model_id="test-model",
            version="0.9.0",
            is_deployed=True,
            traffic_percentage=100.0
        )

        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[existing_version])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.deploy_model(
            model_version_id="test-version-1",
            strategy=DeploymentStrategy.FULL
        )

        # Assert
        assert result["traffic_percentage"] == 100.0
        # Check that existing version was updated
        assert ab_tester.model_version_repo.update.call_count == 2  # Once for existing, once for new

    def test_deploy_invalid_model_version(self, ab_tester):
        """Test deploying non-existent model version raises error."""
        # Setup
        ab_tester.model_version_repo.get_by_id = Mock(return_value=None)

        # Execute & Assert
        with pytest.raises(ValueError, match="Model version not found"):
            ab_tester.deploy_model(
                model_version_id="non-existent",
                strategy=DeploymentStrategy.SHADOW
            )


class TestUpdateTraffic:
    """Tests for update_traffic method."""

    def test_increase_traffic(self, ab_tester, sample_model_version):
        """Test increasing traffic percentage."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 10.0

        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[sample_model_version])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.update_traffic("test-version-1", 50.0)

        # Assert
        assert result["old_traffic"] == 10.0
        assert result["new_traffic"] == 50.0
        assert result["status"] == "updated"

    def test_increase_to_100_undeploys_others(self, ab_tester, sample_model_version):
        """Test increasing to 100% undeploys other versions."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 50.0

        other_version = ModelVersion(
            id="other-version",
            model_id="test-model",
            version="0.9.0",
            is_deployed=True,
            traffic_percentage=50.0
        )

        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[sample_model_version, other_version])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.update_traffic("test-version-1", 100.0)

        # Assert
        assert result["new_traffic"] == 100.0
        # Check that other version was undeployed
        assert ab_tester.model_version_repo.update.call_count >= 2

    def test_update_traffic_not_deployed(self, ab_tester, sample_model_version):
        """Test updating traffic for non-deployed model raises error."""
        # Setup
        sample_model_version.is_deployed = False
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute & Assert
        with pytest.raises(ValueError, match="not deployed"):
            ab_tester.update_traffic("test-version-1", 50.0)

    def test_update_invalid_percentage(self, ab_tester, sample_model_version):
        """Test updating with invalid percentage raises error."""
        # Setup
        sample_model_version.is_deployed = True
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute & Assert
        with pytest.raises(ValueError, match="Invalid traffic percentage"):
            ab_tester.update_traffic("test-version-1", 150.0)


class TestRollback:
    """Tests for rollback_deployment method."""

    def test_rollback_with_restore(self, ab_tester, sample_model_version):
        """Test rollback with previous version restore."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 100.0
        sample_model_version.deployed_at = datetime.utcnow()

        previous_version = ModelVersion(
            id="previous-version",
            model_id="test-model",
            version="0.9.0",
            is_deployed=False,
            traffic_percentage=0.0,
            deployed_at=datetime.utcnow() - timedelta(days=1)
        )

        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_by_model = Mock(return_value=[sample_model_version, previous_version])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.rollback_deployment("test-version-1", restore_previous=True)

        # Assert
        assert result["rolled_back_version"] == "test-version-1"
        assert result["restored_version"] == "previous-version"
        assert result["status"] == "rolled_back"

    def test_rollback_without_restore(self, ab_tester, sample_model_version):
        """Test rollback without restoring previous version."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 100.0

        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)
        ab_tester.model_version_repo.get_by_model = Mock(return_value=[sample_model_version])
        ab_tester.model_version_repo.update = Mock()

        # Execute
        result = ab_tester.rollback_deployment("test-version-1", restore_previous=False)

        # Assert
        assert result["rolled_back_version"] == "test-version-1"
        assert result["restored_version"] is None


class TestShouldUseModel:
    """Tests for should_use_model method."""

    def test_0_percent_traffic_returns_false(self, ab_tester, sample_model_version):
        """Test that 0% traffic always returns False."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 0.0
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute & Assert
        assert ab_tester.should_use_model("test-version-1") is False

    def test_100_percent_traffic_returns_true(self, ab_tester, sample_model_version):
        """Test that 100% traffic always returns True."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 100.0
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute & Assert
        assert ab_tester.should_use_model("test-version-1") is True

    def test_consistent_hashing_with_user_id(self, ab_tester, sample_model_version):
        """Test that same user_id always gets same result."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 50.0
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute
        result1 = ab_tester.should_use_model("test-version-1", user_id="user123")
        result2 = ab_tester.should_use_model("test-version-1", user_id="user123")
        result3 = ab_tester.should_use_model("test-version-1", user_id="user123")

        # Assert - same user should always get same result
        assert result1 == result2 == result3


class TestGetActiveModel:
    """Tests for get_active_model method."""

    def test_single_deployed_model(self, ab_tester, sample_model_version):
        """Test getting active model with single deployment."""
        # Setup
        sample_model_version.is_deployed = True
        sample_model_version.traffic_percentage = 100.0
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[sample_model_version])
        ab_tester.model_version_repo.get_by_id = Mock(return_value=sample_model_version)

        # Execute
        result = ab_tester.get_active_model("test-model")

        # Assert
        assert result == sample_model_version

    def test_no_deployed_models(self, ab_tester):
        """Test getting active model with no deployments."""
        # Setup
        ab_tester.model_version_repo.get_deployed = Mock(return_value=[])

        # Execute
        result = ab_tester.get_active_model("test-model")

        # Assert
        assert result is None


class TestCompareModels:
    """Tests for compare_models method."""

    @patch.object(ABTester, '_calculate_model_metrics')
    @patch.object(ABTester, '_determine_winner')
    def test_compare_two_models(self, mock_determine_winner, mock_calculate_metrics, ab_tester):
        """Test comparing two model versions."""
        # Setup
        mock_calculate_metrics.return_value = {
            "total_anomalies": 10,
            "avg_confidence": 0.9,
            "avg_anomaly_score": -0.5
        }
        mock_determine_winner.return_value = "model_a"

        ab_tester.anomaly_repo.get_by_model_version = Mock(return_value=[])

        # Execute
        result = ab_tester.compare_models(
            model_version_id_a="version-a",
            model_version_id_b="version-b",
            time_window_hours=24
        )

        # Assert
        assert result["winner"] == "model_a"
        assert result["time_window_hours"] == 24
        assert "model_a" in result
        assert "model_b" in result

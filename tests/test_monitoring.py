"""
Tests for monitoring and health check functionality.
"""

import pytest
from unittest.mock import Mock, patch
from src.api_server.monitoring import HealthChecker
from datetime import datetime, timedelta


class TestHealthChecker:
    """Test cases for HealthChecker."""

    def test_initialization(self):
        """Test health checker initialization."""
        checker = HealthChecker()
        assert checker.start_time is not None
        assert isinstance(checker.checks, dict)

    def test_check_liveness(self):
        """Test liveness check."""
        checker = HealthChecker()
        result = checker.check_liveness()

        assert result['alive'] is True
        assert 'timestamp' in result
        assert 'uptime_seconds' in result
        assert result['uptime_seconds'] >= 0

    @patch('src.api_server.monitoring.psutil.virtual_memory')
    def test_check_readiness_success(self, mock_memory):
        """Test successful readiness check."""
        mock_memory.return_value.percent = 50

        checker = HealthChecker()
        # Mock the start_time to be 6 seconds in the past
        checker.start_time = datetime.utcnow() - timedelta(seconds=6)
        result = checker.check_readiness()

        assert result['ready'] is True

    @patch('src.api_server.monitoring.psutil.virtual_memory')
    def test_check_readiness_insufficient_memory(self, mock_memory):
        """Test readiness check with insufficient memory."""
        mock_memory.return_value.percent = 96

        checker = HealthChecker()
        result = checker.check_readiness()

        assert result['ready'] is False
        assert 'reason' in result

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    @patch('src.api_server.monitoring.psutil.virtual_memory')
    @patch('src.api_server.monitoring.psutil.disk_usage')
    def test_check_health(self, mock_disk, mock_memory, mock_cpu):
        """Test comprehensive health check."""
        mock_cpu.return_value = 50
        mock_memory.return_value.percent = 60
        mock_memory.return_value.used = 8000000000
        mock_memory.return_value.available = 4000000000
        mock_disk.return_value.percent = 70
        mock_disk.return_value.free = 50000000000

        checker = HealthChecker()
        result = checker.check_health()

        assert 'status' in result
        assert result['status'] in ['healthy', 'degraded', 'unhealthy']
        assert 'timestamp' in result
        assert 'uptime_seconds' in result
        assert 'checks' in result
        assert 'system' in result['checks']
        assert 'memory' in result['checks']
        assert 'disk' in result['checks']

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    def test_check_system_health_healthy(self, mock_cpu):
        """Test system health check when healthy."""
        mock_cpu.return_value = 50

        checker = HealthChecker()
        result = checker._check_system_health()

        assert result['status'] == 'healthy'
        assert result['cpu_percent'] == 50

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    def test_check_system_health_degraded(self, mock_cpu):
        """Test system health check when degraded."""
        mock_cpu.return_value = 80

        checker = HealthChecker()
        result = checker._check_system_health()

        assert result['status'] == 'degraded'

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    def test_check_system_health_unhealthy(self, mock_cpu):
        """Test system health check when unhealthy."""
        mock_cpu.return_value = 95

        checker = HealthChecker()
        result = checker._check_system_health()

        assert result['status'] == 'unhealthy'

    @patch('src.api_server.monitoring.psutil.virtual_memory')
    def test_check_memory_health_healthy(self, mock_memory):
        """Test memory health check when healthy."""
        mock_memory.return_value.percent = 50
        mock_memory.return_value.available = 8000000000

        checker = HealthChecker()
        result = checker._check_memory_health()

        assert result['status'] == 'healthy'
        assert 'memory_percent' in result
        assert 'memory_available_mb' in result

    @patch('src.api_server.monitoring.psutil.virtual_memory')
    def test_check_memory_health_degraded(self, mock_memory):
        """Test memory health check when degraded."""
        mock_memory.return_value.percent = 80
        mock_memory.return_value.available = 2000000000

        checker = HealthChecker()
        result = checker._check_memory_health()

        assert result['status'] == 'degraded'

    @patch('src.api_server.monitoring.psutil.virtual_memory')
    def test_check_memory_health_unhealthy(self, mock_memory):
        """Test memory health check when unhealthy."""
        mock_memory.return_value.percent = 95
        mock_memory.return_value.available = 500000000

        checker = HealthChecker()
        result = checker._check_memory_health()

        assert result['status'] == 'unhealthy'

    @patch('src.api_server.monitoring.psutil.disk_usage')
    def test_check_disk_health_healthy(self, mock_disk):
        """Test disk health check when healthy."""
        mock_disk.return_value.percent = 50
        mock_disk.return_value.free = 100000000000

        checker = HealthChecker()
        result = checker._check_disk_health()

        assert result['status'] == 'healthy'
        assert 'disk_percent' in result
        assert 'disk_free_gb' in result

    @patch('src.api_server.monitoring.psutil.disk_usage')
    def test_check_disk_health_degraded(self, mock_disk):
        """Test disk health check when degraded."""
        mock_disk.return_value.percent = 85
        mock_disk.return_value.free = 20000000000

        checker = HealthChecker()
        result = checker._check_disk_health()

        assert result['status'] == 'degraded'

    @patch('src.api_server.monitoring.psutil.disk_usage')
    def test_check_disk_health_unhealthy(self, mock_disk):
        """Test disk health check when unhealthy."""
        mock_disk.return_value.percent = 95
        mock_disk.return_value.free = 5000000000

        checker = HealthChecker()
        result = checker._check_disk_health()

        assert result['status'] == 'unhealthy'

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    @patch('src.api_server.monitoring.psutil.virtual_memory')
    @patch('src.api_server.monitoring.psutil.disk_usage')
    def test_update_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics update."""
        mock_cpu.return_value = 50
        mock_memory.return_value.used = 8000000000
        mock_memory.return_value.available = 4000000000
        mock_disk.return_value.percent = 70

        checker = HealthChecker()
        # Should not raise any exceptions
        checker._update_system_metrics()

    @patch('src.api_server.monitoring.psutil.cpu_percent')
    def test_health_check_with_error(self, mock_cpu):
        """Test health check handling errors gracefully."""
        mock_cpu.side_effect = Exception("System error")

        checker = HealthChecker()
        result = checker._check_system_health()

        assert result['status'] == 'unknown'
        assert 'message' in result

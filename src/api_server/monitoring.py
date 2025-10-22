"""
Monitoring and health check utilities for the API server.
"""

import logging
import psutil
import time
from typing import Dict, Any
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, Info
import platform

logger = logging.getLogger(__name__)

# Application info
app_info = Info('app', 'Application information')
app_info.info({
    'name': 'blockchain_anomaly_detection',
    'version': '1.0.0',
    'python_version': platform.python_version()
})

# HTTP metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests currently being processed',
    ['method', 'endpoint']
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_memory_available = Gauge(
    'system_memory_available_bytes',
    'System available memory in bytes'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage'
)

# Application metrics
app_uptime_seconds = Gauge(
    'app_uptime_seconds',
    'Application uptime in seconds'
)

app_start_time = time.time()


class HealthChecker:
    """
    Health check manager for monitoring service health.
    """

    def __init__(self):
        """Initialize health checker."""
        self.start_time = datetime.utcnow()
        self.checks = {}
        logger.info("Health checker initialized")

    def check_health(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dictionary with health status
        """
        # Update system metrics
        self._update_system_metrics()

        # Calculate uptime
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        app_uptime_seconds.set(uptime)

        # Gather all health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': uptime,
            'checks': {
                'system': self._check_system_health(),
                'memory': self._check_memory_health(),
                'disk': self._check_disk_health()
            }
        }

        # Determine overall status
        if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
            health_status['status'] = 'unhealthy'
        elif any(check['status'] == 'degraded' for check in health_status['checks'].values()):
            health_status['status'] = 'degraded'

        return health_status

    def check_readiness(self) -> Dict[str, Any]:
        """
        Check if service is ready to accept traffic.

        Returns:
            Dictionary with readiness status
        """
        readiness_status = {
            'ready': True,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }

        # Check if minimum uptime is met (5 seconds)
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        if uptime < 5:
            readiness_status['ready'] = False
            readiness_status['reason'] = 'Minimum uptime not met'

        # Check memory availability
        memory = psutil.virtual_memory()
        if memory.percent > 95:
            readiness_status['ready'] = False
            readiness_status['reason'] = 'Insufficient memory available'

        return readiness_status

    def check_liveness(self) -> Dict[str, Any]:
        """
        Check if service is alive and responsive.

        Returns:
            Dictionary with liveness status
        """
        return {
            'alive': True,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
        }

    def _update_system_metrics(self) -> None:
        """Update Prometheus system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage.set(memory.used)
            system_memory_available.set(memory.available)

            # Disk usage
            disk = psutil.disk_usage('/')
            system_disk_usage.set(disk.percent)

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")

    def _check_system_health(self) -> Dict[str, Any]:
        """
        Check system resource health.

        Returns:
            System health status
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            status = 'healthy'
            if cpu_percent > 90:
                status = 'unhealthy'
            elif cpu_percent > 75:
                status = 'degraded'

            return {
                'status': status,
                'cpu_percent': cpu_percent,
                'message': f'CPU usage at {cpu_percent:.1f}%'
            }
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking system health: {str(e)}'
            }

    def _check_memory_health(self) -> Dict[str, Any]:
        """
        Check memory health.

        Returns:
            Memory health status
        """
        try:
            memory = psutil.virtual_memory()

            status = 'healthy'
            if memory.percent > 90:
                status = 'unhealthy'
            elif memory.percent > 75:
                status = 'degraded'

            return {
                'status': status,
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'message': f'Memory usage at {memory.percent:.1f}%'
            }
        except Exception as e:
            logger.error(f"Error checking memory health: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking memory health: {str(e)}'
            }

    def _check_disk_health(self) -> Dict[str, Any]:
        """
        Check disk health.

        Returns:
            Disk health status
        """
        try:
            disk = psutil.disk_usage('/')

            status = 'healthy'
            if disk.percent > 90:
                status = 'unhealthy'
            elif disk.percent > 80:
                status = 'degraded'

            return {
                'status': status,
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024 * 1024 * 1024),
                'message': f'Disk usage at {disk.percent:.1f}%'
            }
        except Exception as e:
            logger.error(f"Error checking disk health: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking disk health: {str(e)}'
            }


# Global health checker instance
health_checker = HealthChecker()

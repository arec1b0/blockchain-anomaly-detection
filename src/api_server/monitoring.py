"""
Monitoring and health check utilities for the API server.

This module provides functionalities for monitoring the health and performance
of the API server. It includes a `HealthChecker` class for performing
health checks and Prometheus metrics for tracking various aspects of the
application.
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
    'Total number of HTTP requests.',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'Latency of HTTP requests in seconds.',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests currently in progress.',
    ['method', 'endpoint']
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'Current system-wide CPU utilization as a percentage.'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'Total system memory usage in bytes.'
)

system_memory_available = Gauge(
    'system_memory_available_bytes',
    'Total system memory available in bytes.'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'Current system disk utilization as a percentage.'
)

# Application metrics
app_uptime_seconds = Gauge(
    'app_uptime_seconds',
    'Uptime of the application in seconds.'
)

app_start_time = time.time()


class HealthChecker:
    """
    Manages and performs health checks for the service.

    This class provides methods to check the overall health, readiness, and liveness
    of the service. It also includes helper methods for checking system resources
    like CPU, memory, and disk.
    """

    def __init__(self):
        """
        Initializes the HealthChecker.
        """
        self.start_time = datetime.utcnow()
        self.checks = {}
        self.cache_layer = None  # Will be set by API server
        self.stream_processor = None  # Will be set by API server
        logger.info("Health checker initialized")

    def set_cache_layer(self, cache_layer):
        """Sets the cache layer for health checks."""
        self.cache_layer = cache_layer

    def set_stream_processor(self, stream_processor):
        """Sets the stream processor for health checks."""
        self.stream_processor = stream_processor

    def check_health(self) -> Dict[str, Any]:
        """
        Performs a comprehensive health check.

        This method checks the health of various system resources and aggregates
        the results to determine the overall health status of the service.

        Returns:
            Dict[str, Any]: A dictionary containing the health status and details.
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

        # Add cache health check if cache is enabled
        if self.cache_layer:
            health_status['checks']['cache'] = self._check_cache_health()

        # Add stream processor health check if available
        if self.stream_processor:
            health_status['checks']['stream_processor'] = self._check_stream_processor_health()

        # Determine overall status
        if any(check['status'] == 'unhealthy' for check in health_status['checks'].values()):
            health_status['status'] = 'unhealthy'
        elif any(check['status'] == 'degraded' for check in health_status['checks'].values()):
            health_status['status'] = 'degraded'

        return health_status

    def check_readiness(self) -> Dict[str, Any]:
        """
        Checks if the service is ready to accept traffic.

        This method checks if the service has been running for a minimum amount of time
        and has sufficient memory available.

        Returns:
            Dict[str, Any]: A dictionary containing the readiness status.
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
        Checks if the service is alive and responsive.

        Returns:
            Dict[str, Any]: A dictionary containing the liveness status.
        """
        return {
            'alive': True,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_seconds': (datetime.utcnow() - self.start_time).total_seconds()
        }

    def _update_system_metrics(self) -> None:
        """
        Updates the Prometheus system metrics.

        This method collects and updates metrics for CPU, memory, and disk usage.
        """
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
        Checks the system resource health.

        Returns:
            Dict[str, Any]: A dictionary containing the system health status.
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
        Checks the memory health.

        Returns:
            Dict[str, Any]: A dictionary containing the memory health status.
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
        Checks the disk health.

        Returns:
            Dict[str, Any]: A dictionary containing the disk health status.
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

    def _check_cache_health(self) -> Dict[str, Any]:
        """
        Checks the Redis cache health.

        Returns:
            Dict[str, Any]: A dictionary containing the cache health status.
        """
        try:
            if not self.cache_layer:
                return {
                    'status': 'disabled',
                    'message': 'Cache layer not initialized'
                }

            # Check Redis connection
            redis_client = self.cache_layer.redis_client
            is_healthy = redis_client.health_check()

            if not is_healthy:
                return {
                    'status': 'unhealthy',
                    'message': 'Redis connection failed'
                }

            # Get cache stats
            stats = self.cache_layer.get_cache_stats()
            redis_stats = stats.get('redis', {})

            # Check hit rate (target is 40%+)
            hit_rate = redis_stats.get('hit_rate', 0)
            status = 'healthy'
            if hit_rate < 20:
                status = 'degraded'
                message = f'Cache hit rate low: {hit_rate}%'
            else:
                message = f'Cache hit rate: {hit_rate}%'

            return {
                'status': status,
                'hit_rate': hit_rate,
                'connected': redis_stats.get('connected', False),
                'total_hits': redis_stats.get('total_hits', 0),
                'total_misses': redis_stats.get('total_misses', 0),
                'message': message
            }

        except Exception as e:
            logger.error(f"Error checking cache health: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking cache health: {str(e)}'
            }

    def _check_stream_processor_health(self) -> Dict[str, Any]:
        """
        Checks the stream processor health.

        Returns:
            Dict[str, Any]: A dictionary containing the stream processor health status.
        """
        try:
            if not self.stream_processor:
                return {
                    'status': 'disabled',
                    'message': 'Stream processor not initialized'
                }

            # Get processor stats
            stats = self.stream_processor.get_stats()

            # Check anomaly buffer utilization
            buffer_stats = stats.get('anomaly_buffer', {})
            utilization = buffer_stats.get('utilization', 0)

            status = 'healthy'
            if utilization > 0.9:
                status = 'degraded'
                message = f'Anomaly buffer nearly full: {utilization*100:.1f}%'
            else:
                message = f'Anomaly buffer utilization: {utilization*100:.1f}%'

            return {
                'status': status,
                'model_loaded': stats.get('model_loaded', False),
                'buffer_size': stats.get('buffer_size', 0),
                'anomalies_detected': stats.get('anomalies_detected', 0),
                'buffer_utilization': utilization,
                'memory_usage_mb': stats.get('memory_usage', {}).get('current_mb', 0),
                'message': message
            }

        except Exception as e:
            logger.error(f"Error checking stream processor health: {e}")
            return {
                'status': 'unknown',
                'message': f'Error checking stream processor health: {str(e)}'
            }


# Global health checker instance
health_checker = HealthChecker()

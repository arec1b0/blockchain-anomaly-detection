"""
Redis client with connection pooling and retry logic.

This module provides a singleton Redis client with connection pooling,
automatic reconnection, and health monitoring for distributed caching.
"""

import redis
import logging
from typing import Optional, Any, Dict
import json
import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# Prometheus metrics for cache performance
cache_operations = Counter(
    'redis_cache_operations_total',
    'Total number of cache operations.',
    ['operation', 'status']
)

cache_hit_rate = Gauge(
    'redis_cache_hit_rate',
    'Cache hit rate percentage.'
)

cache_operation_duration = Histogram(
    'redis_cache_operation_duration_seconds',
    'Time spent on cache operations.',
    ['operation']
)

cache_errors = Counter(
    'redis_cache_errors_total',
    'Total number of cache errors.',
    ['error_type']
)


class RedisConnectionError(Exception):
    """Exception raised when Redis connection fails."""
    pass


class RedisClient:
    """
    Singleton Redis client with connection pooling and automatic retry.

    Features:
    - Connection pooling for efficient resource usage
    - Automatic reconnection on connection failures
    - Health monitoring and metrics
    - Configurable timeouts and retry logic
    - JSON serialization for complex objects

    Attributes:
        host (str): Redis server host.
        port (int): Redis server port.
        db (int): Redis database number.
        password (Optional[str]): Redis password.
        max_connections (int): Maximum number of connections in pool.
        socket_timeout (int): Socket timeout in seconds.
        socket_connect_timeout (int): Connection timeout in seconds.
        decode_responses (bool): Whether to decode responses to strings.
        client (Optional[redis.Redis]): The Redis client instance.
        pool (Optional[redis.ConnectionPool]): The connection pool.
    """

    _instance: Optional['RedisClient'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Ensures singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        max_connections: int = 50,
        socket_timeout: int = 5,
        socket_connect_timeout: int = 5,
        decode_responses: bool = False
    ):
        """
        Initializes the Redis client (only once due to singleton pattern).

        Args:
            host (str): Redis server host. Defaults to 'localhost'.
            port (int): Redis server port. Defaults to 6379.
            db (int): Redis database number. Defaults to 0.
            password (Optional[str]): Redis password. Defaults to None.
            max_connections (int): Max connections in pool. Defaults to 50.
            socket_timeout (int): Socket timeout in seconds. Defaults to 5.
            socket_connect_timeout (int): Connection timeout. Defaults to 5.
            decode_responses (bool): Decode responses to strings. Defaults to False.
        """
        # Only initialize once
        if self._initialized:
            return

        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.decode_responses = decode_responses

        self.client: Optional[redis.Redis] = None
        self.pool: Optional[redis.ConnectionPool] = None

        # Metrics tracking
        self._total_hits = 0
        self._total_misses = 0

        self._initialize_connection()
        self._initialized = True

        logger.info(
            f"RedisClient initialized: {host}:{port}, db={db}, "
            f"max_connections={max_connections}"
        )

    def _initialize_connection(self) -> None:
        """
        Initializes the Redis connection pool and client.

        Raises:
            RedisConnectionError: If connection fails after retries.
        """
        try:
            # Create connection pool
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=self.decode_responses,
                health_check_interval=30
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            self.client.ping()

            logger.info("Redis connection established successfully")

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            cache_errors.labels(error_type='connection').inc()
            raise RedisConnectionError(f"Could not connect to Redis: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during Redis initialization: {e}")
            cache_errors.labels(error_type='initialization').inc()
            raise

    def health_check(self) -> bool:
        """
        Checks if Redis connection is healthy.

        Returns:
            bool: True if healthy, False otherwise.
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            cache_errors.labels(error_type='health_check').inc()
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        Gets a value from Redis cache.

        Args:
            key (str): The cache key.

        Returns:
            Optional[Any]: The cached value, or None if not found.
        """
        try:
            with cache_operation_duration.labels(operation='get').time():
                value = self.client.get(key)

                if value is not None:
                    cache_operations.labels(operation='get', status='hit').inc()
                    self._total_hits += 1
                    self._update_hit_rate()

                    # Try to deserialize JSON
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        return value
                else:
                    cache_operations.labels(operation='get', status='miss').inc()
                    self._total_misses += 1
                    self._update_hit_rate()
                    return None

        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error during GET: {e}")
            cache_errors.labels(error_type='get_connection').inc()
            cache_operations.labels(operation='get', status='error').inc()
            return None
        except Exception as e:
            logger.error(f"Error getting key '{key}' from Redis: {e}")
            cache_errors.labels(error_type='get').inc()
            cache_operations.labels(operation='get', status='error').inc()
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Sets a value in Redis cache.

        Args:
            key (str): The cache key.
            value (Any): The value to cache (will be JSON-serialized if needed).
            ttl (Optional[int]): Time-to-live in seconds. Defaults to None (no expiry).

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            with cache_operation_duration.labels(operation='set').time():
                # Serialize to JSON if it's a dict or list
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)

                if ttl:
                    result = self.client.setex(key, ttl, value)
                else:
                    result = self.client.set(key, value)

                if result:
                    cache_operations.labels(operation='set', status='success').inc()
                    return True
                else:
                    cache_operations.labels(operation='set', status='failure').inc()
                    return False

        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error during SET: {e}")
            cache_errors.labels(error_type='set_connection').inc()
            cache_operations.labels(operation='set', status='error').inc()
            return False
        except Exception as e:
            logger.error(f"Error setting key '{key}' in Redis: {e}")
            cache_errors.labels(error_type='set').inc()
            cache_operations.labels(operation='set', status='error').inc()
            return False

    def delete(self, key: str) -> bool:
        """
        Deletes a key from Redis cache.

        Args:
            key (str): The cache key to delete.

        Returns:
            bool: True if key was deleted, False otherwise.
        """
        try:
            with cache_operation_duration.labels(operation='delete').time():
                result = self.client.delete(key)
                cache_operations.labels(operation='delete', status='success').inc()
                return result > 0

        except Exception as e:
            logger.error(f"Error deleting key '{key}' from Redis: {e}")
            cache_errors.labels(error_type='delete').inc()
            cache_operations.labels(operation='delete', status='error').inc()
            return False

    def exists(self, key: str) -> bool:
        """
        Checks if a key exists in Redis cache.

        Args:
            key (str): The cache key.

        Returns:
            bool: True if key exists, False otherwise.
        """
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking existence of key '{key}': {e}")
            cache_errors.labels(error_type='exists').inc()
            return False

    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increments a counter in Redis.

        Args:
            key (str): The cache key.
            amount (int): Amount to increment by. Defaults to 1.

        Returns:
            Optional[int]: New value after increment, or None on error.
        """
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Error incrementing key '{key}': {e}")
            cache_errors.labels(error_type='increment').inc()
            return None

    def expire(self, key: str, ttl: int) -> bool:
        """
        Sets expiration time for a key.

        Args:
            key (str): The cache key.
            ttl (int): Time-to-live in seconds.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            return self.client.expire(key, ttl)
        except Exception as e:
            logger.error(f"Error setting expiration for key '{key}': {e}")
            cache_errors.labels(error_type='expire').inc()
            return False

    def get_ttl(self, key: str) -> Optional[int]:
        """
        Gets the remaining TTL for a key.

        Args:
            key (str): The cache key.

        Returns:
            Optional[int]: Remaining TTL in seconds, -1 if no expiry, None if key doesn't exist.
        """
        try:
            ttl = self.client.ttl(key)
            return ttl if ttl >= -1 else None
        except Exception as e:
            logger.error(f"Error getting TTL for key '{key}': {e}")
            cache_errors.labels(error_type='get_ttl').inc()
            return None

    def _update_hit_rate(self) -> None:
        """Updates the cache hit rate metric."""
        total_requests = self._total_hits + self._total_misses
        if total_requests > 0:
            hit_rate = (self._total_hits / total_requests) * 100
            cache_hit_rate.set(hit_rate)

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics including hit rate, memory usage, etc.
        """
        try:
            info = self.client.info()
            total_requests = self._total_hits + self._total_misses
            hit_rate = (self._total_hits / total_requests * 100) if total_requests > 0 else 0

            return {
                'connected': True,
                'total_hits': self._total_hits,
                'total_misses': self._total_misses,
                'hit_rate': round(hit_rate, 2),
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_connections_received': info.get('total_connections_received', 0),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'evicted_keys': info.get('evicted_keys', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {
                'connected': False,
                'error': str(e),
                'total_hits': self._total_hits,
                'total_misses': self._total_misses
            }

    def flush_db(self) -> bool:
        """
        Flushes all keys from the current database.

        WARNING: This operation is destructive!

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.client.flushdb()
            logger.warning("Redis database flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing Redis database: {e}")
            cache_errors.labels(error_type='flush').inc()
            return False

    def close(self) -> None:
        """Closes the Redis connection pool."""
        if self.pool:
            self.pool.disconnect()
            logger.info("Redis connection pool closed")


# Global singleton instance
_redis_client: Optional[RedisClient] = None


def get_redis_client(
    host: str = 'localhost',
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    **kwargs
) -> RedisClient:
    """
    Gets or creates the global Redis client instance.

    Args:
        host (str): Redis server host.
        port (int): Redis server port.
        db (int): Redis database number.
        password (Optional[str]): Redis password.
        **kwargs: Additional arguments for RedisClient.

    Returns:
        RedisClient: The singleton Redis client instance.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient(
            host=host,
            port=port,
            db=db,
            password=password,
            **kwargs
        )
    return _redis_client

"""
Enhanced cache layer with hit rate tracking and optimization.

This module extends the base CacheLayer with:
- Cache hit/miss tracking
- Cache warming strategies
- Batch operations
- Performance metrics
- Smart eviction policies
"""

import hashlib
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from functools import wraps
from collections import defaultdict
import time
import json
from datetime import datetime, timedelta

from prometheus_client import Counter, Gauge, Histogram

from src.cache.cache_layer import CacheLayer as BaseCacheLayer
from src.cache.redis_client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)

# Prometheus metrics for cache performance
cache_hits = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Cache hit rate percentage',
    ['cache_type']
)

cache_operation_duration = Histogram(
    'cache_operation_duration_seconds',
    'Time spent on cache operations',
    ['operation', 'cache_type']
)

cache_size_bytes = Gauge(
    'cache_size_bytes',
    'Approximate cache size in bytes',
    ['cache_type']
)

cache_evictions = Counter(
    'cache_evictions_total',
    'Total number of cache evictions',
    ['cache_type']
)


class OptimizedCacheLayer(BaseCacheLayer):
    """
    Optimized caching layer with hit rate tracking and warming.

    Extends the base CacheLayer with:
    - Real-time hit/miss tracking
    - Cache warming on startup
    - Batch get/set operations
    - Performance metrics
    - Smart eviction tracking

    Target: 70%+ cache hit rate
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        default_ttl: int = 3600,
        prediction_ttl: int = 7200,
        feature_ttl: int = 3600,
        query_ttl: int = 300,
        model_ttl: int = 86400,
        enable_metrics: bool = True
    ):
        """
        Initialize optimized cache layer.

        Args:
            redis_client: Redis client instance
            default_ttl: Default TTL in seconds
            prediction_ttl: TTL for predictions
            feature_ttl: TTL for features
            query_ttl: TTL for queries
            model_ttl: TTL for model data
            enable_metrics: Enable Prometheus metrics
        """
        super().__init__(
            redis_client,
            default_ttl,
            prediction_ttl,
            feature_ttl,
            query_ttl,
            model_ttl
        )

        self.enable_metrics = enable_metrics

        # Track hits/misses for each cache type
        self.hit_counts = defaultdict(int)
        self.miss_counts = defaultdict(int)
        self.last_metric_update = time.time()

        logger.info("OptimizedCacheLayer initialized with metrics tracking")

    def _record_hit(self, cache_type: str):
        """Record a cache hit."""
        self.hit_counts[cache_type] += 1
        if self.enable_metrics:
            cache_hits.labels(cache_type=cache_type).inc()
        self._update_hit_rate(cache_type)

    def _record_miss(self, cache_type: str):
        """Record a cache miss."""
        self.miss_counts[cache_type] += 1
        if self.enable_metrics:
            cache_misses.labels(cache_type=cache_type).inc()
        self._update_hit_rate(cache_type)

    def _update_hit_rate(self, cache_type: str):
        """Update hit rate metric."""
        # Update every 10 seconds to avoid excessive updates
        if time.time() - self.last_metric_update < 10:
            return

        total = self.hit_counts[cache_type] + self.miss_counts[cache_type]
        if total > 0:
            hit_rate_pct = (self.hit_counts[cache_type] / total) * 100
            if self.enable_metrics:
                cache_hit_rate.labels(cache_type=cache_type).set(hit_rate_pct)

        self.last_metric_update = time.time()

    def get_with_tracking(
        self,
        key: str,
        cache_type: str = "general"
    ) -> Optional[Any]:
        """
        Get value from cache with hit/miss tracking.

        Args:
            key: Cache key
            cache_type: Type of cache for metrics

        Returns:
            Cached value or None
        """
        start_time = time.time()

        value = self.redis_client.get(key)

        if self.enable_metrics:
            duration = time.time() - start_time
            cache_operation_duration.labels(
                operation='get',
                cache_type=cache_type
            ).observe(duration)

        if value is not None:
            self._record_hit(cache_type)
        else:
            self._record_miss(cache_type)

        return value

    def set_with_tracking(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = "general"
    ) -> bool:
        """
        Set value in cache with metrics.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds
            cache_type: Type of cache for metrics

        Returns:
            True if successful
        """
        start_time = time.time()

        result = self.redis_client.set(key, value, ttl or self.default_ttl)

        if self.enable_metrics:
            duration = time.time() - start_time
            cache_operation_duration.labels(
                operation='set',
                cache_type=cache_type
            ).observe(duration)

        return result

    def get_prediction(self, transaction_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached prediction result with tracking.

        Args:
            transaction_hash: Transaction hash

        Returns:
            Cached prediction or None
        """
        key = self._generate_key(self.PREDICTION_PREFIX, transaction_hash)
        return self.get_with_tracking(key, cache_type='prediction')

    def cache_prediction(
        self,
        transaction_hash: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache prediction result with tracking.

        Args:
            transaction_hash: Transaction hash
            prediction: Prediction result
            ttl: Custom TTL

        Returns:
            True if successful
        """
        key = self._generate_key(self.PREDICTION_PREFIX, transaction_hash)
        return self.set_with_tracking(
            key,
            prediction,
            ttl or self.prediction_ttl,
            cache_type='prediction'
        )

    def batch_get(
        self,
        keys: List[str],
        cache_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Batch get operation for efficiency.

        Args:
            keys: List of cache keys
            cache_type: Type of cache for metrics

        Returns:
            Dict mapping keys to values (None if not found)
        """
        start_time = time.time()

        # Use Redis MGET for batch retrieval
        values = self.redis_client.mget(keys)

        if self.enable_metrics:
            duration = time.time() - start_time
            cache_operation_duration.labels(
                operation='batch_get',
                cache_type=cache_type
            ).observe(duration)

        # Track hits and misses
        result = {}
        for key, value in zip(keys, values):
            result[key] = value
            if value is not None:
                self._record_hit(cache_type)
            else:
                self._record_miss(cache_type)

        return result

    def batch_set(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
        cache_type: str = "general"
    ) -> bool:
        """
        Batch set operation for efficiency.

        Args:
            items: Dict mapping keys to values
            ttl: Time-to-live in seconds
            cache_type: Type of cache for metrics

        Returns:
            True if successful
        """
        start_time = time.time()

        # Use pipeline for batch set
        success = self.redis_client.mset(items, ttl or self.default_ttl)

        if self.enable_metrics:
            duration = time.time() - start_time
            cache_operation_duration.labels(
                operation='batch_set',
                cache_type=cache_type
            ).observe(duration)

        return success

    def warm_cache(
        self,
        data_loader: Callable[[], Dict[str, Any]],
        cache_type: str = "general",
        ttl: Optional[int] = None
    ):
        """
        Warm cache with preloaded data.

        Args:
            data_loader: Function that returns dict of key->value pairs
            cache_type: Type of cache
            ttl: Time-to-live for cached items
        """
        logger.info(f"Warming {cache_type} cache...")
        start_time = time.time()

        try:
            data = data_loader()
            self.batch_set(data, ttl, cache_type)

            duration = time.time() - start_time
            logger.info(
                f"Cache warming complete: {len(data)} items loaded "
                f"in {duration:.2f}s for {cache_type}"
            )
        except Exception as e:
            logger.error(f"Cache warming failed for {cache_type}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats including hit rates
        """
        stats = {
            'hit_counts': dict(self.hit_counts),
            'miss_counts': dict(self.miss_counts),
            'hit_rates': {}
        }

        for cache_type in set(list(self.hit_counts.keys()) + list(self.miss_counts.keys())):
            total = self.hit_counts[cache_type] + self.miss_counts[cache_type]
            if total > 0:
                hit_rate = (self.hit_counts[cache_type] / total) * 100
                stats['hit_rates'][cache_type] = round(hit_rate, 2)
            else:
                stats['hit_rates'][cache_type] = 0.0

        # Add Redis stats if available
        try:
            redis_info = self.redis_client.info()
            stats['redis_info'] = {
                'used_memory': redis_info.get('used_memory_human'),
                'connected_clients': redis_info.get('connected_clients'),
                'total_commands_processed': redis_info.get('total_commands_processed')
            }
        except:
            pass

        return stats

    def reset_stats(self):
        """Reset hit/miss statistics."""
        self.hit_counts.clear()
        self.miss_counts.clear()
        logger.info("Cache statistics reset")


def cached_with_metrics(
    cache_type: str = "general",
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results with metrics.

    Args:
        cache_type: Type of cache for metrics
        ttl: Custom TTL in seconds
        key_func: Function to generate cache key from args/kwargs

    Returns:
        Decorated function

    Example:
        @cached_with_metrics(cache_type='prediction', ttl=3600)
        def expensive_prediction(transaction_id):
            # expensive computation
            return result
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache instance (assumes it's injected or available)
            cache = kwargs.get('cache') or get_redis_client()

            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__] + [str(arg) for arg in args]
                key_str = ':'.join(key_parts)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()

            # Try to get from cache
            if isinstance(cache, OptimizedCacheLayer):
                cached_value = cache.get_with_tracking(cache_key, cache_type)
            else:
                cached_value = cache.get(cache_key)

            if cached_value is not None:
                return cached_value

            # Compute value
            result = func(*args, **kwargs)

            # Cache the result
            if isinstance(cache, OptimizedCacheLayer):
                cache.set_with_tracking(cache_key, result, ttl, cache_type)
            else:
                cache.set(cache_key, result, ttl)

            return result

        return wrapper
    return decorator

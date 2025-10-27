"""
Cache module for distributed caching with Redis.

This module provides Redis-based caching functionality for the
blockchain anomaly detection system.
"""

from src.cache.redis_client import RedisClient, get_redis_client
from src.cache.cache_layer import CacheLayer, get_cache_layer

__all__ = ['RedisClient', 'get_redis_client', 'CacheLayer', 'get_cache_layer']

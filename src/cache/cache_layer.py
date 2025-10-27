"""
Distributed caching layer with smart caching strategies.

This module provides application-specific caching strategies for anomaly detection,
including prediction caching, model result caching, and query result caching.
Designed to achieve 40%+ cache hit rate.
"""

import hashlib
import logging
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import time
import json

from src.cache.redis_client import RedisClient, get_redis_client

logger = logging.getLogger(__name__)


class CacheLayer:
    """
    Application-specific caching layer with smart strategies.

    This layer provides high-level caching operations optimized for
    blockchain anomaly detection, including:
    - Prediction result caching (transaction hash -> prediction)
    - Feature computation caching (transaction -> features)
    - Anomaly query result caching (filters -> results)
    - Model metadata caching

    Features:
    - Automatic key generation with namespacing
    - Configurable TTLs for different data types
    - Cache warming strategies
    - Batch operations for efficiency

    Attributes:
        redis_client (RedisClient): The underlying Redis client.
        default_ttl (int): Default TTL in seconds.
        prediction_ttl (int): TTL for prediction results.
        feature_ttl (int): TTL for feature computations.
        query_ttl (int): TTL for query results.
        model_ttl (int): TTL for model metadata.
    """

    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        default_ttl: int = 3600,
        prediction_ttl: int = 7200,  # 2 hours
        feature_ttl: int = 3600,      # 1 hour
        query_ttl: int = 300,         # 5 minutes
        model_ttl: int = 86400        # 24 hours
    ):
        """
        Initializes the CacheLayer.

        Args:
            redis_client (Optional[RedisClient]): Redis client instance.
                If None, creates a new one. Defaults to None.
            default_ttl (int): Default TTL in seconds. Defaults to 3600.
            prediction_ttl (int): TTL for predictions. Defaults to 7200.
            feature_ttl (int): TTL for features. Defaults to 3600.
            query_ttl (int): TTL for queries. Defaults to 300.
            model_ttl (int): TTL for model data. Defaults to 86400.
        """
        self.redis_client = redis_client or get_redis_client()
        self.default_ttl = default_ttl
        self.prediction_ttl = prediction_ttl
        self.feature_ttl = feature_ttl
        self.query_ttl = query_ttl
        self.model_ttl = model_ttl

        # Namespace prefixes for different data types
        self.PREDICTION_PREFIX = "pred:"
        self.FEATURE_PREFIX = "feat:"
        self.QUERY_PREFIX = "query:"
        self.MODEL_PREFIX = "model:"
        self.ANOMALY_PREFIX = "anomaly:"

        logger.info(
            f"CacheLayer initialized with TTLs: "
            f"prediction={prediction_ttl}s, feature={feature_ttl}s, "
            f"query={query_ttl}s, model={model_ttl}s"
        )

    def _generate_key(self, prefix: str, identifier: str) -> str:
        """
        Generates a cache key with namespace prefix.

        Args:
            prefix (str): The namespace prefix.
            identifier (str): The unique identifier.

        Returns:
            str: The formatted cache key.
        """
        return f"{prefix}{identifier}"

    def _hash_object(self, obj: Any) -> str:
        """
        Generates a hash for a complex object.

        Args:
            obj (Any): The object to hash.

        Returns:
            str: The SHA256 hash of the object.
        """
        obj_str = json.dumps(obj, sort_keys=True)
        return hashlib.sha256(obj_str.encode()).hexdigest()

    # ==================== Prediction Caching ====================

    def cache_prediction(
        self,
        transaction_hash: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Caches a prediction result for a transaction.

        Args:
            transaction_hash (str): The transaction hash.
            prediction (Dict[str, Any]): The prediction result.
            ttl (Optional[int]): Custom TTL. Uses prediction_ttl if None.

        Returns:
            bool: True if cached successfully, False otherwise.
        """
        key = self._generate_key(self.PREDICTION_PREFIX, transaction_hash)
        ttl = ttl or self.prediction_ttl
        return self.redis_client.set(key, prediction, ttl=ttl)

    def get_prediction(self, transaction_hash: str) -> Optional[Dict[str, Any]]:
        """
        Gets a cached prediction result.

        Args:
            transaction_hash (str): The transaction hash.

        Returns:
            Optional[Dict[str, Any]]: The cached prediction, or None if not found.
        """
        key = self._generate_key(self.PREDICTION_PREFIX, transaction_hash)
        return self.redis_client.get(key)

    def cache_batch_predictions(
        self,
        predictions: Dict[str, Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> int:
        """
        Caches multiple predictions at once.

        Args:
            predictions (Dict[str, Dict[str, Any]]): Map of hash -> prediction.
            ttl (Optional[int]): Custom TTL. Uses prediction_ttl if None.

        Returns:
            int: Number of predictions successfully cached.
        """
        ttl = ttl or self.prediction_ttl
        success_count = 0

        for tx_hash, prediction in predictions.items():
            if self.cache_prediction(tx_hash, prediction, ttl=ttl):
                success_count += 1

        logger.info(f"Cached {success_count}/{len(predictions)} predictions")
        return success_count

    # ==================== Feature Caching ====================

    def cache_features(
        self,
        transaction_hash: str,
        features: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Caches computed features for a transaction.

        Args:
            transaction_hash (str): The transaction hash.
            features (Dict[str, Any]): The computed features.
            ttl (Optional[int]): Custom TTL. Uses feature_ttl if None.

        Returns:
            bool: True if cached successfully, False otherwise.
        """
        key = self._generate_key(self.FEATURE_PREFIX, transaction_hash)
        ttl = ttl or self.feature_ttl
        return self.redis_client.set(key, features, ttl=ttl)

    def get_features(self, transaction_hash: str) -> Optional[Dict[str, Any]]:
        """
        Gets cached features for a transaction.

        Args:
            transaction_hash (str): The transaction hash.

        Returns:
            Optional[Dict[str, Any]]: The cached features, or None if not found.
        """
        key = self._generate_key(self.FEATURE_PREFIX, transaction_hash)
        return self.redis_client.get(key)

    # ==================== Query Result Caching ====================

    def cache_query_result(
        self,
        query_params: Dict[str, Any],
        result: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Caches query results (e.g., anomaly listings with filters).

        Args:
            query_params (Dict[str, Any]): The query parameters.
            result (List[Dict[str, Any]]): The query result.
            ttl (Optional[int]): Custom TTL. Uses query_ttl if None.

        Returns:
            bool: True if cached successfully, False otherwise.
        """
        query_hash = self._hash_object(query_params)
        key = self._generate_key(self.QUERY_PREFIX, query_hash)
        ttl = ttl or self.query_ttl
        return self.redis_client.set(key, result, ttl=ttl)

    def get_query_result(
        self,
        query_params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Gets cached query results.

        Args:
            query_params (Dict[str, Any]): The query parameters.

        Returns:
            Optional[List[Dict[str, Any]]]: The cached result, or None if not found.
        """
        query_hash = self._hash_object(query_params)
        key = self._generate_key(self.QUERY_PREFIX, query_hash)
        return self.redis_client.get(key)

    def invalidate_query_cache(self) -> bool:
        """
        Invalidates all query result caches.

        This should be called when new anomalies are detected.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # In production, use SCAN to avoid blocking
            # For simplicity, this is a basic implementation
            pattern = f"{self.QUERY_PREFIX}*"
            keys = list(self.redis_client.client.scan_iter(pattern))

            if keys:
                self.redis_client.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} query cache entries")

            return True
        except Exception as e:
            logger.error(f"Error invalidating query cache: {e}")
            return False

    # ==================== Model Metadata Caching ====================

    def cache_model_metadata(
        self,
        model_id: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Caches model metadata (e.g., version, performance metrics).

        Args:
            model_id (str): The model identifier.
            metadata (Dict[str, Any]): The model metadata.
            ttl (Optional[int]): Custom TTL. Uses model_ttl if None.

        Returns:
            bool: True if cached successfully, False otherwise.
        """
        key = self._generate_key(self.MODEL_PREFIX, model_id)
        ttl = ttl or self.model_ttl
        return self.redis_client.set(key, metadata, ttl=ttl)

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets cached model metadata.

        Args:
            model_id (str): The model identifier.

        Returns:
            Optional[Dict[str, Any]]: The cached metadata, or None if not found.
        """
        key = self._generate_key(self.MODEL_PREFIX, model_id)
        return self.redis_client.get(key)

    # ==================== Anomaly Data Caching ====================

    def cache_anomaly(
        self,
        anomaly_id: str,
        anomaly_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Caches individual anomaly data.

        Args:
            anomaly_id (str): The anomaly identifier (e.g., transaction hash).
            anomaly_data (Dict[str, Any]): The anomaly data.
            ttl (Optional[int]): Custom TTL. Uses prediction_ttl if None.

        Returns:
            bool: True if cached successfully, False otherwise.
        """
        key = self._generate_key(self.ANOMALY_PREFIX, anomaly_id)
        ttl = ttl or self.prediction_ttl
        return self.redis_client.set(key, anomaly_data, ttl=ttl)

    def get_anomaly(self, anomaly_id: str) -> Optional[Dict[str, Any]]:
        """
        Gets cached anomaly data.

        Args:
            anomaly_id (str): The anomaly identifier.

        Returns:
            Optional[Dict[str, Any]]: The cached anomaly, or None if not found.
        """
        key = self._generate_key(self.ANOMALY_PREFIX, anomaly_id)
        return self.redis_client.get(key)

    # ==================== Decorator for Automatic Caching ====================

    def cached(
        self,
        prefix: str = "custom:",
        ttl: Optional[int] = None,
        key_func: Optional[Callable] = None
    ):
        """
        Decorator for automatic caching of function results.

        Args:
            prefix (str): Cache key prefix. Defaults to "custom:".
            ttl (Optional[int]): Custom TTL. Uses default_ttl if None.
            key_func (Optional[Callable]): Function to generate cache key from args.
                If None, uses function name and args hash.

        Returns:
            Callable: The decorated function.

        Example:
            @cache_layer.cached(prefix="stats:", ttl=300)
            def get_statistics(start_time, end_time):
                # expensive computation
                return stats
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    args_hash = self._hash_object({'args': args, 'kwargs': kwargs})
                    cache_key = f"{func.__name__}:{args_hash}"

                full_key = self._generate_key(prefix, cache_key)

                # Try to get from cache
                cached_result = self.redis_client.get(full_key)
                if cached_result is not None:
                    logger.debug(f"Cache hit for {full_key}")
                    return cached_result

                # Execute function
                logger.debug(f"Cache miss for {full_key}, executing function")
                result = func(*args, **kwargs)

                # Cache result
                cache_ttl = ttl or self.default_ttl
                self.redis_client.set(full_key, result, ttl=cache_ttl)

                return result

            return wrapper
        return decorator

    # ==================== Cache Statistics and Management ====================

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Gets cache statistics and health information.

        Returns:
            Dict[str, Any]: Cache statistics including hit rate and memory usage.
        """
        redis_stats = self.redis_client.get_stats()

        return {
            'redis': redis_stats,
            'ttl_config': {
                'default': self.default_ttl,
                'prediction': self.prediction_ttl,
                'feature': self.feature_ttl,
                'query': self.query_ttl,
                'model': self.model_ttl
            }
        }

    def warm_cache(
        self,
        recent_transactions: List[Dict[str, Any]],
        predictor: Callable
    ) -> int:
        """
        Warms the cache with predictions for recent transactions.

        This is useful on startup to achieve high hit rates immediately.

        Args:
            recent_transactions (List[Dict[str, Any]]): Recent transactions.
            predictor (Callable): Function to generate predictions.

        Returns:
            int: Number of cache entries warmed.
        """
        logger.info(f"Warming cache with {len(recent_transactions)} transactions")
        warmed_count = 0

        for transaction in recent_transactions:
            tx_hash = transaction.get('hash')
            if not tx_hash:
                continue

            try:
                # Generate prediction
                prediction = predictor(transaction)

                # Cache it
                if self.cache_prediction(tx_hash, prediction):
                    warmed_count += 1

            except Exception as e:
                logger.error(f"Error warming cache for {tx_hash}: {e}")

        logger.info(f"Cache warmed with {warmed_count} entries")
        return warmed_count

    def clear_all(self) -> bool:
        """
        Clears all cached data (USE WITH CAUTION).

        Returns:
            bool: True if successful, False otherwise.
        """
        logger.warning("Clearing all cache data")
        return self.redis_client.flush_db()


# Global singleton instance
_cache_layer: Optional[CacheLayer] = None


def get_cache_layer(
    redis_client: Optional[RedisClient] = None,
    **kwargs
) -> CacheLayer:
    """
    Gets or creates the global CacheLayer instance.

    Args:
        redis_client (Optional[RedisClient]): Redis client to use.
        **kwargs: Additional arguments for CacheLayer.

    Returns:
        CacheLayer: The singleton CacheLayer instance.
    """
    global _cache_layer
    if _cache_layer is None:
        _cache_layer = CacheLayer(redis_client=redis_client, **kwargs)
    return _cache_layer

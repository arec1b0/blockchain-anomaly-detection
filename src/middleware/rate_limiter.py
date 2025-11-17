"""
Rate limiting middleware for API protection.

This module implements token bucket rate limiting with multiple tiers:
- Global rate limit (all requests)
- Per-IP rate limit
- Per-user rate limit (authenticated users)
- Per-endpoint rate limit

Features:
- Configurable limits per tier
- Redis backend support (when available)
- In-memory fallback
- Prometheus metrics
- Retry-After headers
"""

import time
from typing import Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import asyncio

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
rate_limit_hits = Counter(
    'rate_limit_hits_total',
    'Total number of rate limit checks.',
    ['endpoint', 'limit_type']
)

rate_limit_exceeded = Counter(
    'rate_limit_exceeded_total',
    'Total number of requests that exceeded rate limits.',
    ['endpoint', 'client_ip', 'limit_type']
)

rate_limit_check_duration = Histogram(
    'rate_limit_check_duration_seconds',
    'Time spent checking rate limits.',
    ['limit_type']
)


class TokenBucket:
    """
    Token bucket algorithm for rate limiting.

    Allows bursts while maintaining average rate.
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Initialize token bucket.

        Args:
            capacity: Maximum number of tokens (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            bool: True if tokens were consumed, False if insufficient tokens
        """
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now

        # Try to consume tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def get_wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time until enough tokens available.

        Args:
            tokens: Number of tokens needed

        Returns:
            float: Seconds to wait
        """
        if self.tokens >= tokens:
            return 0.0

        needed = tokens - self.tokens
        return needed / self.refill_rate


class InMemoryRateLimiter:
    """
    In-memory rate limiter using token buckets.

    This is used when Redis is not available.
    Note: In distributed environments, use Redis-based rate limiter.
    """

    def __init__(self):
        """Initialize in-memory rate limiter."""
        self.buckets: Dict[str, TokenBucket] = {}
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()

    def _get_bucket_key(self, identifier: str, limit_type: str) -> str:
        """Generate bucket key."""
        return f"{limit_type}:{identifier}"

    def _cleanup_old_buckets(self):
        """Remove old unused buckets to prevent memory leak."""
        now = time.time()
        if now - self.last_cleanup < self.cleanup_interval:
            return

        # Remove buckets that haven't been accessed recently
        cutoff = now - 600  # 10 minutes
        to_remove = [
            key for key, bucket in self.buckets.items()
            if bucket.last_refill < cutoff
        ]

        for key in to_remove:
            del self.buckets[key]

        self.last_cleanup = now
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old rate limit buckets")

    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.

        Args:
            identifier: Unique identifier (IP, user_id, etc.)
            limit_type: Type of limit ("global", "per_ip", etc.)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, info_dict)
        """
        # Periodic cleanup
        self._cleanup_old_buckets()

        # Get or create bucket
        bucket_key = self._get_bucket_key(identifier, limit_type)
        if bucket_key not in self.buckets:
            # refill_rate = capacity / window
            refill_rate = max_requests / window_seconds
            self.buckets[bucket_key] = TokenBucket(max_requests, refill_rate)

        bucket = self.buckets[bucket_key]

        # Try to consume token
        is_allowed = bucket.consume(1)

        # Calculate info
        remaining = int(bucket.tokens)
        reset_time = int(time.time() + window_seconds)
        retry_after = int(bucket.get_wait_time(1)) if not is_allowed else 0

        info = {
            "limit": max_requests,
            "remaining": max(0, remaining),
            "reset": reset_time,
            "retry_after": retry_after
        }

        return is_allowed, info


class RateLimiter:
    """
    Rate limiter with configurable limits and backends.

    Supports multiple rate limit tiers and can use Redis when available.
    """

    def __init__(self, use_redis: bool = False):
        """
        Initialize rate limiter.

        Args:
            use_redis: If True, try to use Redis backend
        """
        self.use_redis = use_redis

        # Rate limit configurations (requests per window)
        self.limits = {
            "global": {"requests": 10000, "window": 60},        # 10K per minute globally
            "per_ip": {"requests": 100, "window": 60},          # 100 per minute per IP
            "per_user": {"requests": 200, "window": 60},        # 200 per minute per user
            "predict": {"requests": 500, "window": 60},         # 500 predictions/min
            "batch_predict": {"requests": 50, "window": 60},    # 50 batch/min
            "train": {"requests": 5, "window": 3600},           # 5 training jobs/hour
            "auth": {"requests": 10, "window": 300},            # 10 login attempts per 5 min
        }

        # Initialize backend
        if use_redis:
            try:
                # Try to import and use Redis
                from src.cache.redis_client import get_redis_client
                self.redis_client = get_redis_client()
                if self.redis_client and self.redis_client.ping():
                    logger.info("Rate limiter using Redis backend")
                    self.backend = "redis"
                else:
                    raise Exception("Redis not available")
            except Exception as e:
                logger.warning(f"Redis not available for rate limiting: {e}")
                logger.info("Falling back to in-memory rate limiter")
                self.backend = "memory"
                self.memory_limiter = InMemoryRateLimiter()
        else:
            self.backend = "memory"
            self.memory_limiter = InMemoryRateLimiter()

    async def check_rate_limit(
        self,
        key: str,
        limit_type: str,
        endpoint: str
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.

        Args:
            key: Unique identifier (IP, user_id, etc.)
            limit_type: Type of limit ("global", "per_ip", etc.)
            endpoint: API endpoint path

        Returns:
            Tuple of (is_allowed, info_dict)
        """
        start_time = time.time()

        config = self.limits.get(limit_type, self.limits["per_ip"])
        max_requests = config["requests"]
        window_seconds = config["window"]

        # Use appropriate backend
        if self.backend == "redis":
            is_allowed, info = await self._check_redis(
                key, limit_type, max_requests, window_seconds
            )
        else:
            is_allowed, info = await self.memory_limiter.check_rate_limit(
                key, limit_type, max_requests, window_seconds
            )

        # Update metrics
        rate_limit_hits.labels(endpoint=endpoint, limit_type=limit_type).inc()

        if not is_allowed:
            rate_limit_exceeded.labels(
                endpoint=endpoint,
                client_ip=key,
                limit_type=limit_type
            ).inc()
            logger.warning(
                f"Rate limit exceeded: {limit_type} for {key} on {endpoint}. "
                f"Retry after {info['retry_after']}s"
            )

        # Record duration
        duration = time.time() - start_time
        rate_limit_check_duration.labels(limit_type=limit_type).observe(duration)

        return is_allowed, info

    async def _check_redis(
        self,
        key: str,
        limit_type: str,
        max_requests: int,
        window_seconds: int
    ) -> Tuple[bool, Dict]:
        """
        Check rate limit using Redis backend.

        Uses sliding window algorithm with Redis.
        """
        now = int(time.time())
        window_key = f"rate_limit:{limit_type}:{key}:{now // window_seconds}"

        try:
            # Increment counter
            current = await self.redis_client.incr(window_key)

            # Set expiration on first request
            if current == 1:
                await self.redis_client.expire(window_key, window_seconds)

            # Calculate remaining and reset time
            remaining = max(0, max_requests - current)
            reset_time = ((now // window_seconds) + 1) * window_seconds

            # Check if limit exceeded
            is_allowed = current <= max_requests

            info = {
                "limit": max_requests,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": reset_time - now if not is_allowed else 0
            }

            return is_allowed, info

        except Exception as e:
            logger.error(f"Redis rate limit check error: {e}")
            # Fail open on errors (allow request)
            return True, {
                "limit": max_requests,
                "remaining": max_requests,
                "reset": now + window_seconds,
                "retry_after": 0
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Applies multiple rate limit tiers to protect the API.
    """

    def __init__(self, app, use_redis: bool = False):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            use_redis: If True, use Redis backend
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(use_redis=use_redis)

    async def dispatch(self, request: Request, call_next):
        """Process request and check rate limits."""

        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/health/ready", "/health/live", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Extract client identifier
        client_ip = request.client.host if request.client else "unknown"
        user_id = None

        # Get user_id from auth if available
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                from src.auth.jwt_handler import jwt_handler
                token = auth_header.replace("Bearer ", "")
                payload = jwt_handler.decode_token(token)
                user_id = payload.get("sub")
            except:
                pass

        # Determine endpoint type for specific limits
        path = request.url.path
        endpoint_type = None

        if "/predict/batch" in path:
            endpoint_type = "batch_predict"
        elif "/predict" in path:
            endpoint_type = "predict"
        elif "/train" in path:
            endpoint_type = "train"
        elif "/auth" in path:
            endpoint_type = "auth"

        # Check limits in order of specificity
        checks = [
            ("global", "global", path),
            ("per_ip", client_ip, path),
        ]

        # Add user-specific limit if authenticated
        if user_id:
            checks.append(("per_user", user_id, path))

        # Add endpoint-specific limit if applicable
        if endpoint_type:
            checks.append((endpoint_type, user_id or client_ip, path))

        # Perform rate limit checks
        for limit_type, identifier, endpoint in checks:
            is_allowed, info = await self.rate_limiter.check_rate_limit(
                identifier, limit_type, endpoint
            )

            if not is_allowed:
                # Rate limit exceeded
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit_type": limit_type,
                        "limit": info["limit"],
                        "retry_after": info["retry_after"],
                        "reset": info["reset"]
                    },
                    headers={
                        "X-RateLimit-Limit": str(info["limit"]),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(info["reset"]),
                        "Retry-After": str(info["retry_after"])
                    }
                )

        # All rate limits passed, process request
        response = await call_next(request)

        # Add rate limit headers to response (from most restrictive check)
        # Use the last check's info (usually most specific)
        if checks:
            _, last_identifier, _ = checks[-1]
            last_limit_type = checks[-1][0]
            _, info = await self.rate_limiter.check_rate_limit(
                last_identifier, last_limit_type, path
            )

            response.headers["X-RateLimit-Limit"] = str(info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(info["reset"])

        return response

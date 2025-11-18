"""
Performance monitoring middleware for FastAPI.

This module provides middleware for tracking API performance metrics including:
- Request/response latency
- Request rate
- Error rates
- Response sizes
- Slow request tracking
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
from prometheus_client import Counter, Histogram, Gauge
import asyncio

logger = logging.getLogger(__name__)

# Prometheus metrics
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

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

http_errors_total = Counter(
    'http_errors_total',
    'Total HTTP errors',
    ['method', 'endpoint', 'status']
)

slow_requests_total = Counter(
    'slow_requests_total',
    'Total slow requests (>1s)',
    ['method', 'endpoint']
)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for tracking API performance metrics.

    Tracks:
    - Request latency (p50, p95, p99)
    - Request rate
    - Error rates
    - Request/response sizes
    - Slow requests (>1s)
    - Concurrent requests

    Features:
    - Automatic Prometheus metrics
    - Slow request logging
    - Request ID tracking
    - Error tracking
    """

    def __init__(
        self,
        app,
        slow_request_threshold: float = 1.0,
        enable_request_logging: bool = True
    ):
        """
        Initialize performance monitoring middleware.

        Args:
            app: FastAPI application instance
            slow_request_threshold: Threshold in seconds for slow requests
            enable_request_logging: Enable detailed request logging
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.enable_request_logging = enable_request_logging
        logger.info(
            f"PerformanceMonitoringMiddleware initialized "
            f"(slow_threshold={slow_request_threshold}s)"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and track performance metrics.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/endpoint handler

        Returns:
            HTTP response
        """
        # Extract path and method
        path = request.url.path
        method = request.method

        # Simplify endpoint name (remove IDs)
        endpoint = self._normalize_path(path)

        # Track concurrent requests
        http_requests_in_progress.labels(
            method=method,
            endpoint=endpoint
        ).inc()

        # Start timing
        start_time = time.time()

        # Track request size
        request_size = int(request.headers.get('content-length', 0))
        if request_size > 0:
            http_request_size_bytes.labels(
                method=method,
                endpoint=endpoint
            ).observe(request_size)

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Track response metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status=response.status_code
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            # Track response size
            response_size = int(response.headers.get('content-length', 0))
            if response_size > 0:
                http_response_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)

            # Track errors
            if response.status_code >= 400:
                http_errors_total.labels(
                    method=method,
                    endpoint=endpoint,
                    status=response.status_code
                ).inc()

            # Track slow requests
            if duration > self.slow_request_threshold:
                slow_requests_total.labels(
                    method=method,
                    endpoint=endpoint
                ).inc()

                logger.warning(
                    f"Slow request detected: {method} {path} "
                    f"took {duration:.2f}s (threshold: {self.slow_request_threshold}s)"
                )

            # Log request if enabled
            if self.enable_request_logging:
                logger.info(
                    f"{method} {path} {response.status_code} {duration*1000:.0f}ms"
                )

            return response

        except Exception as e:
            # Track errors
            http_errors_total.labels(
                method=method,
                endpoint=endpoint,
                status=500
            ).inc()

            logger.error(f"Error processing request {method} {path}: {e}", exc_info=True)
            raise

        finally:
            # Decrement concurrent requests
            http_requests_in_progress.labels(
                method=method,
                endpoint=endpoint
            ).dec()

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for metrics (remove variable parts like IDs).

        Args:
            path: Request path

        Returns:
            Normalized path

        Examples:
            /api/v1/models/abc123 -> /api/v1/models/{id}
            /api/v1/transactions/0x123 -> /api/v1/transactions/{hash}
        """
        parts = path.split('/')

        # Replace UUIDs and hashes with placeholders
        normalized_parts = []
        for part in parts:
            if not part:
                continue

            # UUID pattern
            if len(part) == 36 and part.count('-') == 4:
                normalized_parts.append('{id}')
            # Hash pattern (starts with 0x)
            elif part.startswith('0x'):
                normalized_parts.append('{hash}')
            # Numeric ID
            elif part.isdigit():
                normalized_parts.append('{id}')
            else:
                normalized_parts.append(part)

        return '/' + '/'.join(normalized_parts)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to limit request body size.

    Prevents memory exhaustion from large requests.
    """

    def __init__(self, app, max_request_size: int = 10 * 1024 * 1024):  # 10MB default
        """
        Initialize request size limit middleware.

        Args:
            app: FastAPI application
            max_request_size: Maximum request size in bytes
        """
        super().__init__(app)
        self.max_request_size = max_request_size
        logger.info(f"RequestSizeLimitMiddleware initialized (max={max_request_size} bytes)")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check request size and reject if too large.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response
        """
        content_length = request.headers.get('content-length')

        if content_length:
            content_length = int(content_length)
            if content_length > self.max_request_size:
                logger.warning(
                    f"Request rejected: size {content_length} exceeds limit {self.max_request_size}"
                )
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request entity too large",
                        "max_size": self.max_request_size,
                        "actual_size": content_length
                    }
                )

        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.

    Note: For production, use Redis-based rate limiting for distributed rate limits.
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 100,
        burst_size: int = 20
    ):
        """
        Initialize rate limiting middleware.

        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute per IP
            burst_size: Max burst size
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

        # Simple in-memory tracking (use Redis in production)
        from collections import defaultdict, deque
        self.request_times = defaultdict(deque)

        logger.info(
            f"RateLimitMiddleware initialized "
            f"({requests_per_minute} req/min, burst={burst_size})"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limit and reject if exceeded.

        Args:
            request: Incoming request
            call_next: Next handler

        Returns:
            Response
        """
        # Get client IP
        client_ip = request.client.host

        # Current time
        now = time.time()

        # Clean old entries (older than 1 minute)
        cutoff = now - 60
        times = self.request_times[client_ip]

        while times and times[0] < cutoff:
            times.popleft()

        # Check rate limit
        if len(times) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")

            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "limit": self.requests_per_minute,
                    "window": "60s",
                    "retry_after": int(60 - (now - times[0]))
                },
                headers={
                    "Retry-After": str(int(60 - (now - times[0]))),
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(times[0] + 60))
                }
            )

        # Add current request
        times.append(now)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(times)
        )
        if times:
            response.headers["X-RateLimit-Reset"] = str(int(times[0] + 60))

        return response


def setup_performance_middleware(app, config: dict = None):
    """
    Setup all performance middleware for the application.

    Args:
        app: FastAPI application instance
        config: Configuration dict with middleware settings

    Example:
        from fastapi import FastAPI
        from src.api_server.performance_middleware import setup_performance_middleware

        app = FastAPI()
        setup_performance_middleware(app)
    """
    config = config or {}

    # Add GZip compression
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,  # Only compress responses > 1KB
        compresslevel=6     # Balance between speed and compression
    )
    logger.info("GZip compression middleware added")

    # Add request size limit
    max_size = config.get('max_request_size', 10 * 1024 * 1024)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_request_size=max_size
    )

    # Add rate limiting (if enabled)
    if config.get('enable_rate_limiting', True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=config.get('rate_limit_rpm', 100),
            burst_size=config.get('rate_limit_burst', 20)
        )

    # Add performance monitoring
    app.add_middleware(
        PerformanceMonitoringMiddleware,
        slow_request_threshold=config.get('slow_request_threshold', 1.0),
        enable_request_logging=config.get('enable_request_logging', True)
    )

    logger.info("All performance middleware configured")

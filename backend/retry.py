"""
Retry Utilities with Exponential Backoff

Provides decorators and utilities for resilient network operations.
Handles transient failures in HTTP requests, WebSocket connections,
and CLOB client operations.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type, Union
import aiohttp

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[tuple] = None,
        retryable_status_codes: Optional[set] = None,
    ):
        """
        Args:
            max_retries: Maximum number of retry attempts (0 = no retries)
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            exponential_base: Base for exponential backoff (2.0 = 1s, 2s, 4s, 8s...)
            jitter: Add random jitter to prevent thundering herd
            retryable_exceptions: Tuple of exception types to retry on
            retryable_status_codes: Set of HTTP status codes to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            ConnectionError,
            OSError,
        )
        self.retryable_status_codes = retryable_status_codes or {
            408,  # Request Timeout
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        }


# Default configs for different operation types
HTTP_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
)

WS_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=60.0,
)

CLOB_RETRY_CONFIG = RetryConfig(
    max_retries=2,  # Be conservative with trading operations
    base_delay=0.5,
    max_delay=5.0,
)


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate delay for retry attempt with exponential backoff and jitter."""
    delay = config.base_delay * (config.exponential_base ** attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add up to 25% jitter
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.1, delay)


def async_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for async functions with automatic retry on failure.

    Args:
        config: RetryConfig instance (defaults to HTTP_RETRY_CONFIG)
        on_retry: Optional callback(attempt, exception) called before each retry

    Example:
        @async_retry(config=HTTP_RETRY_CONFIG)
        async def fetch_data():
            async with session.get(url) as resp:
                return await resp.json()
    """
    if config is None:
        config = HTTP_RETRY_CONFIG

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"[Retry] {func.__name__} failed (attempt {attempt + 1}/{config.max_retries + 1}): "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(attempt, e)

                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"[Retry] {func.__name__} failed after {config.max_retries + 1} attempts: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                except Exception:
                    # Non-retryable exception, re-raise immediately
                    raise

            # Should not reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def sync_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception], None]] = None,
):
    """
    Decorator for synchronous functions with automatic retry on failure.

    Same as async_retry but for sync functions.
    """
    if config is None:
        config = HTTP_RETRY_CONFIG

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    if attempt < config.max_retries:
                        delay = calculate_delay(attempt, config)
                        logger.warning(
                            f"[Retry] {func.__name__} failed (attempt {attempt + 1}/{config.max_retries + 1}): "
                            f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[Retry] {func.__name__} failed after {config.max_retries + 1} attempts: "
                            f"{type(e).__name__}: {e}"
                        )
                        raise

                except Exception:
                    raise

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


async def retry_http_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> aiohttp.ClientResponse:
    """
    Execute HTTP request with automatic retry.

    Args:
        session: aiohttp ClientSession
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        config: RetryConfig (defaults to HTTP_RETRY_CONFIG)
        **kwargs: Additional arguments passed to session.request()

    Returns:
        aiohttp.ClientResponse

    Example:
        async with aiohttp.ClientSession() as session:
            resp = await retry_http_request(session, "GET", url, timeout=10)
            data = await resp.json()
    """
    if config is None:
        config = HTTP_RETRY_CONFIG

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            resp = await session.request(method, url, **kwargs)

            # Check if status code should be retried
            if resp.status in config.retryable_status_codes:
                if attempt < config.max_retries:
                    delay = calculate_delay(attempt, config)
                    logger.warning(
                        f"[Retry] HTTP {method} {url} returned {resp.status} "
                        f"(attempt {attempt + 1}/{config.max_retries + 1}). Retrying in {delay:.1f}s..."
                    )
                    await resp.release()
                    await asyncio.sleep(delay)
                    continue

            return resp

        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_retries:
                delay = calculate_delay(attempt, config)
                logger.warning(
                    f"[Retry] HTTP {method} {url} failed (attempt {attempt + 1}/{config.max_retries + 1}): "
                    f"{type(e).__name__}: {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"[Retry] HTTP {method} {url} failed after {config.max_retries + 1} attempts: "
                    f"{type(e).__name__}: {e}"
                )
                raise

    if last_exception:
        raise last_exception


class ConnectionHealthMonitor:
    """
    Monitors connection health and triggers reconnection when needed.

    Tracks last successful operation time and connection state for
    multiple named connections (e.g., "binance_ws", "polymarket_http").
    """

    def __init__(self, stale_threshold_sec: float = 60.0):
        """
        Args:
            stale_threshold_sec: Consider connection stale after this many seconds
                                 without successful operations
        """
        self.stale_threshold = stale_threshold_sec
        self._last_success: dict[str, float] = {}
        self._connected: dict[str, bool] = {}
        self._error_counts: dict[str, int] = {}
        self._callbacks: dict[str, Callable] = {}

    def register_connection(
        self,
        name: str,
        on_stale: Optional[Callable[[], None]] = None,
    ):
        """Register a connection to monitor."""
        self._last_success[name] = time.time()
        self._connected[name] = True
        self._error_counts[name] = 0
        if on_stale:
            self._callbacks[name] = on_stale

    def mark_success(self, name: str):
        """Mark successful operation for a connection."""
        self._last_success[name] = time.time()
        self._connected[name] = True
        self._error_counts[name] = 0

    def mark_error(self, name: str):
        """Mark failed operation for a connection."""
        self._error_counts[name] = self._error_counts.get(name, 0) + 1

    def mark_disconnected(self, name: str):
        """Mark connection as disconnected."""
        self._connected[name] = False

    def is_healthy(self, name: str) -> bool:
        """Check if a connection is healthy."""
        if name not in self._last_success:
            return False

        if not self._connected.get(name, False):
            return False

        age = time.time() - self._last_success[name]
        return age < self.stale_threshold

    def get_status(self) -> dict:
        """Get status of all monitored connections."""
        now = time.time()
        status = {}

        for name in self._last_success:
            age = now - self._last_success[name]
            status[name] = {
                "connected": self._connected.get(name, False),
                "last_success_age_sec": round(age, 1),
                "is_healthy": self.is_healthy(name),
                "error_count": self._error_counts.get(name, 0),
            }

        return status

    async def check_and_reconnect(self):
        """Check all connections and trigger reconnect for stale ones."""
        for name, callback in self._callbacks.items():
            if not self.is_healthy(name):
                logger.warning(f"[HealthMonitor] Connection '{name}' is stale, triggering reconnect")
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.error(f"[HealthMonitor] Reconnect callback for '{name}' failed: {e}")


# Global health monitor instance
connection_monitor = ConnectionHealthMonitor(stale_threshold_sec=120.0)

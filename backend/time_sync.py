"""
Time Synchronization Module

Ensures system clock is synchronized with authoritative time sources.
Critical for trading on 15-minute markets where timing matters.

Uses NTP servers and optionally checks against Polymarket server time.
"""

import asyncio
import logging
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional, Tuple

logger = logging.getLogger("time_sync")


# ============================================================================
# NTP TIME SYNC
# ============================================================================

# NTP servers to query (in priority order)
NTP_SERVERS = [
    "time.google.com",
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.windows.com",
]

# How often to resync (in seconds)
SYNC_INTERVAL_SEC = 300  # 5 minutes

# Maximum acceptable clock drift before warning
MAX_DRIFT_WARN_MS = 500  # 500ms
MAX_DRIFT_ERROR_MS = 2000  # 2 seconds


@dataclass
class TimeSync:
    """
    Stores time synchronization state.

    Attributes:
        offset_ms: Milliseconds to add to local time to get accurate time
        last_sync: Unix timestamp of last successful sync
        source: The source that provided the offset (NTP server or Polymarket)
        status: Current sync status
    """
    offset_ms: float = 0.0
    last_sync: float = 0.0
    source: str = "local"
    status: str = "not_synced"
    drift_ms: float = 0.0  # Detected clock drift


# Global state
_sync_state = TimeSync()
_sync_lock = asyncio.Lock()


def get_synced_time() -> float:
    """
    Get the current time adjusted for clock drift.

    Returns:
        Unix timestamp in seconds (float for sub-second precision)
    """
    local_time = time.time()
    offset_sec = _sync_state.offset_ms / 1000.0
    return local_time + offset_sec


def get_synced_timestamp() -> int:
    """
    Get the current time as Unix timestamp (seconds).

    Returns:
        Unix timestamp in seconds (integer)
    """
    return int(get_synced_time())


def get_synced_datetime() -> datetime:
    """
    Get the current time as a datetime object.

    Returns:
        timezone-aware datetime in UTC
    """
    return datetime.fromtimestamp(get_synced_time(), tz=timezone.utc)


def get_sync_status() -> dict:
    """
    Get current synchronization status.

    Returns:
        Dict with offset_ms, last_sync, source, status, drift_ms
    """
    return {
        "offset_ms": _sync_state.offset_ms,
        "last_sync": _sync_state.last_sync,
        "source": _sync_state.source,
        "status": _sync_state.status,
        "drift_ms": _sync_state.drift_ms,
        "local_time": time.time(),
        "synced_time": get_synced_time(),
    }


async def sync_with_ntp() -> Tuple[bool, str]:
    """
    Synchronize with NTP servers.

    Uses SNTP (Simple NTP) via HTTP time services as a fallback
    since raw NTP requires special handling.

    Returns:
        Tuple of (success, message)
    """
    global _sync_state

    async with _sync_lock:
        # Try each NTP server
        for server in NTP_SERVERS:
            try:
                # Use Google's time service API (returns accurate time)
                if "google" in server:
                    offset, source = await _sync_google_time()
                    if offset is not None:
                        _sync_state.offset_ms = offset
                        _sync_state.last_sync = time.time()
                        _sync_state.source = source
                        _sync_state.status = "synced"
                        _sync_state.drift_ms = abs(offset)

                        _log_sync_result(offset)
                        return True, f"Synced with {source}"

                # Try HTTP-based time sync for other servers
                offset, source = await _sync_http_time(server)
                if offset is not None:
                    _sync_state.offset_ms = offset
                    _sync_state.last_sync = time.time()
                    _sync_state.source = source
                    _sync_state.status = "synced"
                    _sync_state.drift_ms = abs(offset)

                    _log_sync_result(offset)
                    return True, f"Synced with {source}"

            except Exception as e:
                logger.debug(f"Failed to sync with {server}: {e}")
                continue

        # All servers failed
        _sync_state.status = "sync_failed"
        return False, "Failed to sync with any NTP server"


def _sync_url(url: str, timeout: float = 5.0) -> Tuple[Optional[str], float]:
    """
    Synchronous HTTP request that returns Date header.

    Returns:
        Tuple of (date_header, rtt_ms) or (None, 0) on failure
    """
    try:
        before = time.time() * 1000
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            after = time.time() * 1000
            date_header = resp.headers.get("Date")
            return date_header, after - before
    except Exception:
        return None, 0


async def sync_with_polymarket() -> Tuple[bool, str]:
    """
    Synchronize with Polymarket's server time.

    Queries Polymarket API and calculates offset from response headers
    or server time fields.

    Returns:
        Tuple of (success, message)
    """
    global _sync_state

    async with _sync_lock:
        try:
            before = time.time() * 1000
            date_header, rtt = await asyncio.to_thread(
                _sync_url, "https://gamma-api.polymarket.com/", 5.0
            )
            after = time.time() * 1000

            # Use the actual measured RTT if available
            if rtt > 0:
                actual_rtt = rtt
            else:
                actual_rtt = after - before

            # Get server time from Date header
            if date_header:
                server_time = parsedate_to_datetime(date_header)
                server_ms = server_time.timestamp() * 1000

                # Estimate when server generated response (midpoint of RTT)
                local_at_response = before + (actual_rtt / 2)

                # Calculate offset
                offset = server_ms - local_at_response

                _sync_state.offset_ms = offset
                _sync_state.last_sync = time.time()
                _sync_state.source = "polymarket"
                _sync_state.status = "synced"
                _sync_state.drift_ms = abs(offset)

                _log_sync_result(offset, "Polymarket")
                return True, f"Synced with Polymarket (offset: {offset:.0f}ms)"

            return False, "No Date header in Polymarket response"

        except asyncio.TimeoutError:
            logger.warning("Polymarket time sync timed out")
            return False, "Polymarket sync timed out"
        except Exception as e:
            logger.warning(f"Polymarket time sync failed: {e}")
            return False, f"Polymarket sync failed: {e}"


async def _sync_google_time() -> Tuple[Optional[float], str]:
    """
    Sync with Google's time service via HTTP.

    Returns:
        Tuple of (offset_ms, source) or (None, "") on failure
    """
    try:
        before = time.time() * 1000
        date_header, rtt = await asyncio.to_thread(
            _sync_url, "https://www.google.com", 3.0
        )
        after = time.time() * 1000

        if date_header:
            server_time = parsedate_to_datetime(date_header)
            server_ms = server_time.timestamp() * 1000

            actual_rtt = rtt if rtt > 0 else (after - before)
            local_at_response = before + (actual_rtt / 2)
            offset = server_ms - local_at_response

            return offset, "time.google.com"

    except Exception as e:
        logger.debug(f"Google time sync failed: {e}")

    return None, ""


def _sync_worldtime() -> Tuple[Optional[str], float, int]:
    """
    Synchronous HTTP request to worldtimeapi.

    Returns:
        Tuple of (date_header, rtt_ms, unixtime) or (None, 0, 0) on failure
    """
    import json
    try:
        before = time.time() * 1000
        req = urllib.request.Request("https://worldtimeapi.org/api/timezone/Etc/UTC")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            after = time.time() * 1000
            if resp.status == 200:
                data = json.loads(resp.read().decode())
                return resp.headers.get("Date"), after - before, data.get("unixtime", 0)
    except Exception:
        pass
    return None, 0, 0


async def _sync_http_time(server: str) -> Tuple[Optional[float], str]:
    """
    Sync with a server via HTTP Date header.

    Returns:
        Tuple of (offset_ms, source) or (None, "") on failure
    """
    try:
        # For pool.ntp.org and others, we can't use HTTP directly
        # Use worldtimeapi.org as a fallback
        before = time.time() * 1000
        _, rtt, unixtime = await asyncio.to_thread(_sync_worldtime)
        after = time.time() * 1000

        if unixtime > 0:
            server_ms = unixtime * 1000
            actual_rtt = rtt if rtt > 0 else (after - before)
            local_at_response = before + (actual_rtt / 2)
            offset = server_ms - local_at_response

            return offset, "worldtimeapi.org"

    except Exception as e:
        logger.debug(f"HTTP time sync failed for {server}: {e}")

    return None, ""


def _log_sync_result(offset_ms: float, source: str = "NTP"):
    """Log the sync result with appropriate level based on drift."""
    abs_offset = abs(offset_ms)

    if abs_offset > MAX_DRIFT_ERROR_MS:
        logger.error(
            f"[TimeSync] CRITICAL: Clock drift of {offset_ms:.0f}ms detected! "
            f"This may cause trading timing issues. Source: {source}"
        )
    elif abs_offset > MAX_DRIFT_WARN_MS:
        logger.warning(
            f"[TimeSync] Clock drift of {offset_ms:.0f}ms detected. Source: {source}"
        )
    else:
        logger.info(
            f"[TimeSync] Synced with {source}. Offset: {offset_ms:+.0f}ms"
        )


async def start_sync_loop():
    """
    Background task that periodically syncs time.

    Should be started on app startup.
    """
    while True:
        try:
            # Try Polymarket first (most relevant for trading)
            success, msg = await sync_with_polymarket()

            if not success:
                # Fall back to NTP/HTTP time
                success, msg = await sync_with_ntp()

            if success:
                logger.debug(f"Time sync successful: {msg}")
            else:
                logger.warning(f"Time sync failed: {msg}")

        except Exception as e:
            logger.error(f"Time sync loop error: {e}")

        # Wait for next sync interval
        await asyncio.sleep(SYNC_INTERVAL_SEC)


# ============================================================================
# MARKET TIMING HELPERS
# ============================================================================

def get_current_15min_window() -> Tuple[int, int]:
    """
    Get the current 15-minute market window boundaries.

    Returns:
        Tuple of (start_timestamp, end_timestamp) in Unix seconds
    """
    now = get_synced_timestamp()

    # Round down to nearest 15 minutes
    window_start = (now // 900) * 900
    window_end = window_start + 900

    return window_start, window_end


def get_next_15min_window() -> Tuple[int, int]:
    """
    Get the next 15-minute market window boundaries.

    Returns:
        Tuple of (start_timestamp, end_timestamp) in Unix seconds
    """
    window_start, _ = get_current_15min_window()
    next_start = window_start + 900
    next_end = next_start + 900

    return next_start, next_end


def get_time_until_window_end() -> int:
    """
    Get seconds remaining until current window ends.

    Returns:
        Seconds until window end
    """
    now = get_synced_timestamp()
    _, window_end = get_current_15min_window()
    return max(0, window_end - now)


def get_time_until_next_window() -> int:
    """
    Get seconds until next window starts.

    Returns:
        Seconds until next window (0 if we're between windows)
    """
    now = get_synced_timestamp()
    next_start, _ = get_next_15min_window()
    return max(0, next_start - now)


def is_market_open() -> bool:
    """
    Check if market is currently open.

    15-minute markets are open from :00 to :15, :15 to :30, etc.
    Returns True if we're within a window.
    """
    time_remaining = get_time_until_window_end()
    return time_remaining > 0


def format_synced_time() -> str:
    """
    Get formatted synced time for display.

    Returns:
        ISO format timestamp string
    """
    return get_synced_datetime().isoformat()

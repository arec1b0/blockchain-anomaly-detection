"""
Bounded buffer with TTL (Time-To-Live) for anomaly storage.

This module provides a thread-safe, bounded buffer with automatic expiration
of old entries to prevent memory leaks in long-running streaming applications.
"""

import threading
import time
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """
    A buffer entry with timestamp for TTL management.

    Attributes:
        data (Dict[str, Any]): The actual data being stored.
        inserted_at (float): Unix timestamp when entry was inserted.
    """
    data: Dict[str, Any]
    inserted_at: float = field(default_factory=time.time)


class BoundedTTLBuffer:
    """
    Thread-safe bounded buffer with TTL (Time-To-Live) eviction.

    This buffer maintains a maximum size and automatically removes entries
    that exceed the TTL threshold. It uses a deque for O(1) insertions and
    removals, and provides thread-safe operations using locks.

    Features:
    - Bounded size: automatically removes oldest entries when full
    - TTL-based eviction: removes entries older than specified TTL
    - Thread-safe: uses locks for concurrent access
    - Memory-efficient: uses deque for O(1) operations

    Attributes:
        max_size (int): Maximum number of entries to store.
        ttl_seconds (int): Time-to-live in seconds for each entry.
        buffer (deque): The underlying deque storing BufferEntry objects.
        lock (threading.Lock): Lock for thread-safe operations.
        total_evicted (int): Total number of entries evicted (for metrics).
        total_inserted (int): Total number of entries inserted (for metrics).
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initializes the BoundedTTLBuffer.

        Args:
            max_size (int): Maximum number of entries. Defaults to 10,000.
            ttl_seconds (int): Time-to-live in seconds. Defaults to 3600 (1 hour).
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.buffer: deque[BufferEntry] = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.total_evicted = 0
        self.total_inserted = 0

        logger.info(
            f"BoundedTTLBuffer initialized with max_size={max_size}, "
            f"ttl_seconds={ttl_seconds}"
        )

    def append(self, data: Dict[str, Any]) -> None:
        """
        Appends a new entry to the buffer.

        Automatically evicts expired entries and enforces size limit.

        Args:
            data (Dict[str, Any]): The data to append.
        """
        with self.lock:
            # Evict expired entries before adding new one
            self._evict_expired()

            # Check if we're at capacity (deque handles this automatically)
            if len(self.buffer) == self.max_size:
                self.total_evicted += 1

            # Add new entry
            entry = BufferEntry(data=data, inserted_at=time.time())
            self.buffer.append(entry)
            self.total_inserted += 1

    def get_all(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Gets all non-expired entries from the buffer.

        Args:
            limit (Optional[int]): Maximum number of entries to return.
                If None, returns all entries. Defaults to None.

        Returns:
            List[Dict[str, Any]]: List of data entries (most recent first).
        """
        with self.lock:
            # Evict expired entries
            self._evict_expired()

            # Get entries (most recent first)
            entries = list(reversed(self.buffer))

            if limit:
                entries = entries[:limit]

            return [entry.data for entry in entries]

    def get_by_severity(
        self,
        severity: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Gets entries filtered by severity level.

        Args:
            severity (str): The severity level to filter by.
            limit (Optional[int]): Maximum number of entries to return.

        Returns:
            List[Dict[str, Any]]: List of matching data entries.
        """
        with self.lock:
            # Evict expired entries
            self._evict_expired()

            # Filter by severity
            filtered = [
                entry.data for entry in reversed(self.buffer)
                if entry.data.get('severity') == severity
            ]

            if limit:
                filtered = filtered[:limit]

            return filtered

    def clear(self) -> int:
        """
        Clears all entries from the buffer.

        Returns:
            int: Number of entries cleared.
        """
        with self.lock:
            count = len(self.buffer)
            self.buffer.clear()
            logger.info(f"Cleared {count} entries from buffer")
            return count

    def _evict_expired(self) -> None:
        """
        Evicts entries that have exceeded their TTL.

        This method should only be called while holding the lock.
        """
        if not self.buffer:
            return

        current_time = time.time()
        expiration_threshold = current_time - self.ttl_seconds
        evicted_count = 0

        # Remove from left (oldest entries) while they're expired
        while self.buffer and self.buffer[0].inserted_at < expiration_threshold:
            self.buffer.popleft()
            evicted_count += 1
            self.total_evicted += 1

        if evicted_count > 0:
            logger.debug(f"Evicted {evicted_count} expired entries")

    def size(self) -> int:
        """
        Gets the current number of entries in the buffer.

        Returns:
            int: Current buffer size.
        """
        with self.lock:
            return len(self.buffer)

    def get_stats(self) -> Dict[str, Any]:
        """
        Gets buffer statistics.

        Returns:
            Dict[str, Any]: Statistics including size, capacity, eviction counts, etc.
        """
        with self.lock:
            # Calculate age statistics
            if self.buffer:
                current_time = time.time()
                oldest_age = current_time - self.buffer[0].inserted_at
                newest_age = current_time - self.buffer[-1].inserted_at
            else:
                oldest_age = 0
                newest_age = 0

            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'total_inserted': self.total_inserted,
                'total_evicted': self.total_evicted,
                'utilization': len(self.buffer) / self.max_size if self.max_size > 0 else 0,
                'oldest_entry_age_seconds': oldest_age,
                'newest_entry_age_seconds': newest_age
            }

    def get_memory_estimate(self) -> Dict[str, Any]:
        """
        Estimates the memory usage of the buffer.

        Returns:
            Dict[str, Any]: Memory usage estimates in bytes and MB.
        """
        with self.lock:
            # Rough estimate: 1KB per anomaly record on average
            bytes_per_entry = 1024
            estimated_bytes = len(self.buffer) * bytes_per_entry
            estimated_mb = estimated_bytes / (1024 * 1024)
            max_bytes = self.max_size * bytes_per_entry
            max_mb = max_bytes / (1024 * 1024)

            return {
                'current_bytes': estimated_bytes,
                'current_mb': round(estimated_mb, 2),
                'max_bytes': max_bytes,
                'max_mb': round(max_mb, 2)
            }

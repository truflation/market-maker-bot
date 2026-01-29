"""
Ring buffer implementation for indicator calculations.

Provides efficient fixed-size buffer for streaming data with O(1) operations.
"""

from collections import deque
from typing import Optional
import math


class RingBuffer:
    """
    Fixed-size ring buffer for streaming numeric data.

    Supports efficient append, retrieval, and statistical operations
    needed for volatility and other indicator calculations.
    """

    def __init__(self, capacity: int):
        """
        Initialize ring buffer with fixed capacity.

        Args:
            capacity: Maximum number of elements to store
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: deque[float] = deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum buffer size."""
        return self._capacity

    def __len__(self) -> int:
        """Current number of elements."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        """True if buffer has reached capacity."""
        return len(self._buffer) == self._capacity

    def append(self, value: float) -> None:
        """
        Add value to buffer, evicting oldest if at capacity.

        Args:
            value: Numeric value to append
        """
        self._buffer.append(value)

    def get_last_value(self) -> Optional[float]:
        """Get most recent value, or None if empty."""
        return self._buffer[-1] if self._buffer else None

    def get_first_value(self) -> Optional[float]:
        """Get oldest value, or None if empty."""
        return self._buffer[0] if self._buffer else None

    def get_as_list(self) -> list[float]:
        """Return buffer contents as a list (oldest to newest)."""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all elements."""
        self._buffer.clear()

    def sum(self) -> float:
        """Sum of all elements."""
        return sum(self._buffer)

    def mean(self) -> Optional[float]:
        """Arithmetic mean, or None if empty."""
        if not self._buffer:
            return None
        return sum(self._buffer) / len(self._buffer)

    def variance(self, ddof: int = 1) -> Optional[float]:
        """
        Sample variance with degrees of freedom adjustment.

        Args:
            ddof: Delta degrees of freedom (default 1 for sample variance)

        Returns:
            Variance or None if insufficient data
        """
        n = len(self._buffer)
        if n <= ddof:
            return None
        mean_val = sum(self._buffer) / n
        return sum((x - mean_val) ** 2 for x in self._buffer) / (n - ddof)

    def std(self, ddof: int = 1) -> Optional[float]:
        """
        Standard deviation with degrees of freedom adjustment.

        Args:
            ddof: Delta degrees of freedom (default 1 for sample std)

        Returns:
            Standard deviation or None if insufficient data
        """
        var = self.variance(ddof)
        return math.sqrt(var) if var is not None else None

    def rms_diff(self) -> Optional[float]:
        """
        Root mean square of consecutive differences.

        Used for instant volatility calculation:
        σ = sqrt(sum((x[i+1] - x[i])^2) / n)

        Returns:
            RMS of differences, or None if < 2 elements
        """
        if len(self._buffer) < 2:
            return None

        squared_diffs = 0.0
        prev = self._buffer[0]
        for curr in list(self._buffer)[1:]:
            diff = curr - prev
            squared_diffs += diff * diff
            prev = curr

        return math.sqrt(squared_diffs / len(self._buffer))

    def min(self) -> Optional[float]:
        """Minimum value, or None if empty."""
        return min(self._buffer) if self._buffer else None

    def max(self) -> Optional[float]:
        """Maximum value, or None if empty."""
        return max(self._buffer) if self._buffer else None

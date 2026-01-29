"""
Order book mid-price volatility indicator.

Implements instant volatility estimation for the Avellaneda-Stoikov
market making strategy using RMS of consecutive price differences.

This approach provides more stable readings for trending markets than
standard deviation around a mean.
"""

import logging
from typing import Optional

from ..utils.ring_buffer import RingBuffer
from ..models import VolatilityEstimate

logger = logging.getLogger(__name__)


class InstantVolatilityIndicator:
    """
    Instant volatility indicator based on consecutive price differences.

    Calculates volatility as RMS of tick-to-tick price changes:
    σ = sqrt(Σ(price[i+1] - price[i])² / N)

    This method is preferred over standard deviation around a mean because:
    - If the asset is trending, standard deviation would show artificially
      higher volatility as ticks drift from the mean
    - RMS of differences captures actual price movement volatility
    - The result is independent of trend direction

    For binary options, returns volatility in absolute cents (not percentage).
    """

    def __init__(
        self,
        buffer_size: int = 60,
        min_samples: int = 10,
        default_value: float = 5.0,
    ):
        """
        Initialize volatility indicator.

        Args:
            buffer_size: Number of samples to maintain
            min_samples: Minimum samples before returning calculated value
            default_value: Default volatility when insufficient samples
        """
        self._buffer = RingBuffer(buffer_size)
        self._min_samples = min_samples
        self._default_value = default_value

    @property
    def buffer_size(self) -> int:
        """Maximum buffer capacity."""
        return self._buffer.capacity

    @property
    def sample_count(self) -> int:
        """Current number of samples."""
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True if we have enough samples for reliable calculation."""
        return len(self._buffer) >= self._min_samples

    def add_sample(self, mid_price: float) -> None:
        """
        Add a new mid-price sample.

        Args:
            mid_price: Current mid price in cents
        """
        self._buffer.append(mid_price)

    def get_volatility(self) -> VolatilityEstimate:
        """
        Calculate current volatility.

        Returns:
            VolatilityEstimate with value, source, and sample count
        """
        if not self.is_ready:
            return VolatilityEstimate(
                value=self._default_value,
                source="default",
                samples=len(self._buffer),
            )

        vol = self._buffer.rms_diff()

        if vol is None or vol <= 0:
            return VolatilityEstimate(
                value=self._default_value,
                source="default",
                samples=len(self._buffer),
            )

        return VolatilityEstimate(
            value=vol,
            source="order_book",
            samples=len(self._buffer),
        )

    def get_value(self) -> float:
        """Convenience method to get just the volatility value."""
        return self.get_volatility().value

    def reset(self) -> None:
        """Clear all samples."""
        self._buffer.clear()

    def get_last_price(self) -> Optional[float]:
        """Get most recent price sample."""
        return self._buffer.get_last_value()


class VolatilityTracker:
    """
    Manages volatility tracking for multiple market outcomes.

    Each market outcome (YES/NO) has its own volatility indicator
    since they may have different trading patterns.
    """

    def __init__(
        self,
        buffer_size: int = 60,
        min_samples: int = 10,
        default_value: float = 5.0,
    ):
        """
        Initialize tracker.

        Args:
            buffer_size: Buffer size for each indicator
            min_samples: Minimum samples for each indicator
            default_value: Default volatility value
        """
        self._buffer_size = buffer_size
        self._min_samples = min_samples
        self._default_value = default_value
        self._indicators: dict[tuple[int, bool], InstantVolatilityIndicator] = {}

    def _get_key(self, query_id: int, outcome: bool) -> tuple[int, bool]:
        """Create key for indicator lookup."""
        return (query_id, outcome)

    def get_indicator(
        self, query_id: int, outcome: bool
    ) -> InstantVolatilityIndicator:
        """
        Get or create indicator for a market outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO

        Returns:
            InstantVolatilityIndicator for the market outcome
        """
        key = self._get_key(query_id, outcome)
        if key not in self._indicators:
            self._indicators[key] = InstantVolatilityIndicator(
                buffer_size=self._buffer_size,
                min_samples=self._min_samples,
                default_value=self._default_value,
            )
        return self._indicators[key]

    def add_sample(
        self, query_id: int, outcome: bool, mid_price: float
    ) -> None:
        """
        Add a mid-price sample for a market outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            mid_price: Current mid price in cents
        """
        indicator = self.get_indicator(query_id, outcome)
        indicator.add_sample(mid_price)

    def get_volatility(
        self, query_id: int, outcome: bool
    ) -> VolatilityEstimate:
        """
        Get volatility estimate for a market outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO

        Returns:
            VolatilityEstimate
        """
        indicator = self.get_indicator(query_id, outcome)
        return indicator.get_volatility()

    def reset(self, query_id: int, outcome: bool) -> None:
        """Reset indicator for a specific market outcome."""
        key = self._get_key(query_id, outcome)
        if key in self._indicators:
            self._indicators[key].reset()

    def reset_all(self) -> None:
        """Reset all indicators."""
        for indicator in self._indicators.values():
            indicator.reset()

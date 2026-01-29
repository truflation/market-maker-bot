"""Tests for volatility indicators."""

import pytest
import time
from datetime import datetime, timezone, timedelta

from market_maker_bot.utils.ring_buffer import RingBuffer
from market_maker_bot.indicators.volatility import (
    InstantVolatilityIndicator,
    VolatilityTracker,
)
from market_maker_bot.indicators.stream_volatility import (
    infer_stream_frequency,
    calculate_stream_volatility,
    calc_yang_zhang_volatility,
    calc_close_to_close_volatility,
    get_current_spot_value,
    StreamFrequency,
)


class TestRingBuffer:
    """Tests for RingBuffer utility."""

    def test_basic_append_and_retrieve(self):
        """Test basic append and get operations."""
        buf = RingBuffer(5)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)

        assert len(buf) == 3
        assert buf.get_last_value() == 3.0
        assert buf.get_first_value() == 1.0

    def test_capacity_overflow(self):
        """Test that buffer evicts oldest when at capacity."""
        buf = RingBuffer(3)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        buf.append(4.0)  # Should evict 1.0

        assert len(buf) == 3
        assert buf.get_first_value() == 2.0
        assert buf.get_last_value() == 4.0

    def test_is_full(self):
        """Test is_full property."""
        buf = RingBuffer(3)
        assert not buf.is_full

        buf.append(1.0)
        buf.append(2.0)
        assert not buf.is_full

        buf.append(3.0)
        assert buf.is_full

    def test_mean(self):
        """Test mean calculation."""
        buf = RingBuffer(5)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        buf.append(4.0)
        buf.append(5.0)

        assert buf.mean() == 3.0

    def test_variance(self):
        """Test variance calculation."""
        buf = RingBuffer(5)
        for v in [2.0, 4.0, 4.0, 4.0, 6.0]:
            buf.append(v)

        # Variance of [2, 4, 4, 4, 6] = 2.0
        assert abs(buf.variance() - 2.0) < 0.01

    def test_rms_diff(self):
        """Test RMS of consecutive differences."""
        buf = RingBuffer(5)
        # Constant differences of 1
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)
        buf.append(4.0)
        buf.append(5.0)

        # RMS = sqrt(sum((1)^2 * 4) / 5) = sqrt(4/5) ≈ 0.894
        rms = buf.rms_diff()
        assert rms is not None
        assert abs(rms - (4 / 5) ** 0.5) < 0.01

    def test_rms_diff_insufficient_data(self):
        """RMS diff should return None with < 2 elements."""
        buf = RingBuffer(5)
        assert buf.rms_diff() is None

        buf.append(1.0)
        assert buf.rms_diff() is None

    def test_empty_buffer_operations(self):
        """Test operations on empty buffer."""
        buf = RingBuffer(5)

        assert len(buf) == 0
        assert buf.get_last_value() is None
        assert buf.get_first_value() is None
        assert buf.mean() is None
        assert buf.variance() is None
        assert buf.std() is None

    def test_get_as_list(self):
        """Test conversion to list."""
        buf = RingBuffer(5)
        buf.append(1.0)
        buf.append(2.0)
        buf.append(3.0)

        assert buf.get_as_list() == [1.0, 2.0, 3.0]

    def test_clear(self):
        """Test clearing the buffer."""
        buf = RingBuffer(5)
        buf.append(1.0)
        buf.append(2.0)
        buf.clear()

        assert len(buf) == 0
        assert buf.get_last_value() is None


class TestInstantVolatilityIndicator:
    """Tests for InstantVolatilityIndicator."""

    def test_default_value_when_not_ready(self):
        """Returns default value when insufficient samples."""
        ind = InstantVolatilityIndicator(
            buffer_size=10,
            min_samples=5,
            default_value=5.0,
        )

        ind.add_sample(50.0)
        ind.add_sample(51.0)

        estimate = ind.get_volatility()
        assert estimate.value == 5.0
        assert estimate.source == "default"

    def test_calculated_value_when_ready(self):
        """Returns calculated value when enough samples."""
        ind = InstantVolatilityIndicator(
            buffer_size=20,
            min_samples=5,
            default_value=5.0,
        )

        # Add enough samples with known volatility pattern
        for i in range(10):
            ind.add_sample(50.0 + (i % 2))  # Oscillate between 50 and 51

        estimate = ind.get_volatility()
        assert estimate.source == "order_book"
        assert estimate.value > 0

    def test_is_ready(self):
        """Test is_ready property."""
        ind = InstantVolatilityIndicator(
            buffer_size=10,
            min_samples=5,
            default_value=5.0,
        )

        for i in range(4):
            ind.add_sample(50.0)
            assert not ind.is_ready

        ind.add_sample(50.0)
        assert ind.is_ready

    def test_reset(self):
        """Test reset functionality."""
        ind = InstantVolatilityIndicator(
            buffer_size=10,
            min_samples=5,
            default_value=5.0,
        )

        for i in range(10):
            ind.add_sample(50.0)

        assert ind.is_ready
        ind.reset()
        assert not ind.is_ready
        assert ind.sample_count == 0


class TestVolatilityTracker:
    """Tests for VolatilityTracker."""

    def test_separate_tracking_per_market(self):
        """Each market/outcome should have separate tracking."""
        tracker = VolatilityTracker(
            buffer_size=10,
            min_samples=3,
            default_value=5.0,
        )

        # Add samples to market 1 YES
        for i in range(5):
            tracker.add_sample(1, True, 50.0 + i)

        # Add samples to market 1 NO (with varying prices to get non-zero volatility)
        for i in range(3):
            tracker.add_sample(1, False, 30.0 + i)

        # Add samples to market 2 YES
        tracker.add_sample(2, True, 70.0)

        # Check separate tracking
        vol_1_yes = tracker.get_volatility(1, True)
        vol_1_no = tracker.get_volatility(1, False)
        vol_2_yes = tracker.get_volatility(2, True)

        assert vol_1_yes.source == "order_book"  # Has 5 samples
        assert vol_1_no.source == "order_book"   # Has 3 samples
        assert vol_2_yes.source == "default"     # Only 1 sample

    def test_reset_specific_market(self):
        """Test resetting a specific market outcome."""
        tracker = VolatilityTracker(buffer_size=10, min_samples=3)

        for i in range(5):
            tracker.add_sample(1, True, 50.0)
            tracker.add_sample(1, False, 50.0)

        tracker.reset(1, True)

        ind_yes = tracker.get_indicator(1, True)
        ind_no = tracker.get_indicator(1, False)

        assert ind_yes.sample_count == 0
        assert ind_no.sample_count == 5


class TestStreamVolatility:
    """Tests for stream volatility calculations."""

    def _make_records(self, values: list, hours_apart: float = 1.0):
        """Create test records with specified values."""
        now = datetime.now(timezone.utc)
        records = []
        for i, value in enumerate(values):
            event_time = int((now - timedelta(hours=i * hours_apart)).timestamp())
            records.append({"EventTime": event_time, "Value": value})
        return list(reversed(records))  # Oldest first

    def test_infer_frequency_hourly(self):
        """Test frequency inference for hourly stream."""
        now = datetime.now(timezone.utc)
        records = []
        # Create 20 events in last 48 hours
        for i in range(20):
            event_time = int((now - timedelta(hours=i * 2)).timestamp())
            records.append({"EventTime": event_time, "Value": 100})

        freq, count = infer_stream_frequency(records)
        assert freq == StreamFrequency.HOURLY
        assert count == 20

    def test_infer_frequency_daily(self):
        """Test frequency inference for daily stream."""
        now = datetime.now(timezone.utc)
        records = []
        # Create 10 events, but only 2 in last 48 hours
        for i in range(10):
            event_time = int((now - timedelta(days=i)).timestamp())
            records.append({"EventTime": event_time, "Value": 100})

        freq, count = infer_stream_frequency(records)
        # Should be DAILY (2 events in 48h)
        assert freq == StreamFrequency.DAILY

    def test_infer_frequency_monthly(self):
        """Test frequency inference for monthly stream."""
        now = datetime.now(timezone.utc)
        records = []
        # Create events with large gaps
        for i in range(5):
            event_time = int((now - timedelta(days=i * 30)).timestamp())
            records.append({"EventTime": event_time, "Value": 100})

        freq, count = infer_stream_frequency(records)
        assert freq == StreamFrequency.MONTHLY

    def test_close_to_close_basic(self):
        """Test basic Close-to-Close volatility calculation."""
        # Create records with known returns
        values = [100, 101, 102, 101, 100, 99, 100, 101, 100]
        records = self._make_records(values, hours_apart=24)

        result = calc_close_to_close_volatility(
            records, lookback_days=30, frequency=StreamFrequency.DAILY
        )

        assert result.data_points > 0
        assert result.daily_volatility > 0
        assert result.annual_volatility > result.daily_volatility

    def test_calculate_stream_volatility_applies_floor(self):
        """Test that minimum volatility floor is applied."""
        # Create records with very low volatility
        values = [100.0, 100.001, 100.002, 100.001, 100.0]
        records = self._make_records(values, hours_apart=24)

        result = calculate_stream_volatility(
            records, min_volatility=0.10  # 10% floor
        )

        assert result.annual_volatility >= 0.10

    def test_get_current_spot_value(self):
        """Test getting most recent value from records."""
        now = datetime.now(timezone.utc)
        records = [
            {"EventTime": int((now - timedelta(hours=10)).timestamp()), "Value": 100},
            {"EventTime": int((now - timedelta(hours=5)).timestamp()), "Value": 110},
            {"EventTime": int((now - timedelta(hours=1)).timestamp()), "Value": 105},
        ]

        spot = get_current_spot_value(records)
        assert spot == 105  # Most recent

    def test_get_current_spot_empty_records(self):
        """Test getting spot value from empty records."""
        spot = get_current_spot_value([])
        assert spot == 0.0

    def test_insufficient_data_returns_zero(self):
        """Test that insufficient data returns zero volatility."""
        records = self._make_records([100], hours_apart=24)

        result = calc_close_to_close_volatility(records, lookback_days=30)

        assert result.data_points <= 1
        assert "insufficient" in result.method_used.lower()

"""Tests for execution state management."""

import pytest
from datetime import datetime, time

from market_maker_bot.execution_state import (
    ExecutionTimeframeMode,
    ExecutionTimeframeConfig,
    ExecutionState,
    RunAlwaysExecutionState,
    RunInTimeExecutionState,
    create_execution_state,
)


class TestRunAlwaysExecutionState:
    """Tests for infinite/always-on execution mode."""

    def test_always_executes(self):
        """RunAlwaysExecutionState should always return True."""
        state = RunAlwaysExecutionState()

        # Test at various timestamps
        assert state.should_execute(0.0) is True
        assert state.should_execute(1000000000.0) is True
        assert state.should_execute(datetime.now().timestamp()) is True

    def test_no_time_limits(self):
        """RunAlwaysExecutionState should have no time limits."""
        state = RunAlwaysExecutionState()
        state.should_execute(datetime.now().timestamp())

        assert state.closing_time is None
        assert state.time_left is None

    def test_str_representation(self):
        """Test string representation."""
        state = RunAlwaysExecutionState()
        assert str(state) == "run continuously"


class TestRunInTimeExecutionState:
    """Tests for time-bounded execution modes."""

    def test_from_date_to_date_within_range(self):
        """Test execution within date range."""
        start = datetime(2024, 1, 1, 9, 0, 0)
        end = datetime(2024, 12, 31, 17, 0, 0)
        state = RunInTimeExecutionState(start, end)

        # Timestamp within range
        mid_timestamp = datetime(2024, 6, 15, 12, 0, 0).timestamp()
        assert state.should_execute(mid_timestamp) is True
        assert state.time_left > 0

    def test_from_date_to_date_outside_range(self):
        """Test execution outside date range."""
        start = datetime(2024, 1, 1, 9, 0, 0)
        end = datetime(2024, 12, 31, 17, 0, 0)
        state = RunInTimeExecutionState(start, end)

        # Timestamp outside range (before start)
        before_timestamp = datetime(2023, 12, 31, 12, 0, 0).timestamp()
        assert state.should_execute(before_timestamp) is False

    def test_from_date_only(self):
        """Test execution from date onwards (no end)."""
        start = datetime(2024, 1, 1, 9, 0, 0)
        state = RunInTimeExecutionState(start, end_timestamp=None)

        # Should execute after start
        after_timestamp = datetime(2024, 6, 15, 12, 0, 0).timestamp()
        assert state.should_execute(after_timestamp) is True

        # Should not execute before start
        before_timestamp = datetime(2023, 12, 31, 12, 0, 0).timestamp()
        assert state.should_execute(before_timestamp) is False

    def test_daily_between_times_within_range(self):
        """Test daily time-bounded execution within range."""
        start_time = time(9, 30, 0)
        end_time = time(16, 0, 0)
        state = RunInTimeExecutionState(start_time, end_time)

        # Create a timestamp at 12:00 today
        today = datetime.now().date()
        noon = datetime.combine(today, time(12, 0, 0))

        assert state.should_execute(noon.timestamp()) is True
        assert state.time_left > 0

    def test_daily_between_times_outside_range(self):
        """Test daily time-bounded execution outside range."""
        start_time = time(9, 30, 0)
        end_time = time(16, 0, 0)
        state = RunInTimeExecutionState(start_time, end_time)

        # Create a timestamp at 8:00 today (before start)
        today = datetime.now().date()
        early = datetime.combine(today, time(8, 0, 0))

        assert state.should_execute(early.timestamp()) is False

    def test_equality(self):
        """Test equality comparison."""
        start = datetime(2024, 1, 1, 9, 0, 0)
        end = datetime(2024, 12, 31, 17, 0, 0)

        state1 = RunInTimeExecutionState(start, end)
        state2 = RunInTimeExecutionState(start, end)
        state3 = RunInTimeExecutionState(start, datetime(2025, 1, 1, 0, 0, 0))

        assert state1 == state2
        assert state1 != state3


class TestCreateExecutionState:
    """Tests for the factory function."""

    def test_create_infinite_mode(self):
        """Test creating infinite execution state."""
        config = ExecutionTimeframeConfig(mode=ExecutionTimeframeMode.INFINITE)
        state = create_execution_state(config)

        assert isinstance(state, RunAlwaysExecutionState)

    def test_create_from_date_to_date_mode(self):
        """Test creating from_date_to_date execution state."""
        config = ExecutionTimeframeConfig(
            mode=ExecutionTimeframeMode.FROM_DATE_TO_DATE,
            start_datetime=datetime(2024, 1, 1, 9, 0, 0),
            end_datetime=datetime(2024, 12, 31, 17, 0, 0),
        )
        state = create_execution_state(config)

        assert isinstance(state, RunInTimeExecutionState)

    def test_create_daily_between_times_mode(self):
        """Test creating daily_between_times execution state."""
        config = ExecutionTimeframeConfig(
            mode=ExecutionTimeframeMode.DAILY_BETWEEN_TIMES,
            start_time=time(9, 30, 0),
            end_time=time(16, 0, 0),
        )
        state = create_execution_state(config)

        assert isinstance(state, RunInTimeExecutionState)

    def test_from_date_to_date_missing_start_raises(self):
        """Test that missing start_datetime raises error."""
        config = ExecutionTimeframeConfig(
            mode=ExecutionTimeframeMode.FROM_DATE_TO_DATE,
            start_datetime=None,
        )

        with pytest.raises(ValueError, match="start_datetime required"):
            create_execution_state(config)

    def test_daily_between_times_missing_times_raises(self):
        """Test that missing times raise error."""
        config = ExecutionTimeframeConfig(
            mode=ExecutionTimeframeMode.DAILY_BETWEEN_TIMES,
            start_time=None,
            end_time=None,
        )

        with pytest.raises(ValueError, match="start_time and end_time required"):
            create_execution_state(config)

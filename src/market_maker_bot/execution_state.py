"""
Execution state management for conditional trading timeframes.

Controls when the bot is allowed to execute trades based on time conditions:
- Infinite: Always trade (24/7)
- From date to date: Trade between specific datetime boundaries
- Daily between times: Trade during specific hours each day
"""

from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExecutionTimeframeMode(str, Enum):
    """Execution timeframe modes."""
    INFINITE = "infinite"
    FROM_DATE_TO_DATE = "from_date_to_date"
    DAILY_BETWEEN_TIMES = "daily_between_times"


@dataclass
class ExecutionTimeframeConfig:
    """Configuration for execution timeframe."""
    mode: ExecutionTimeframeMode = ExecutionTimeframeMode.INFINITE
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None


class ExecutionState(ABC):
    """
    Base class for execution state management.

    Controls whether the strategy should process ticks based on time conditions.
    """

    def __init__(self):
        self._closing_time: Optional[float] = None
        self._time_left: Optional[float] = None

    @property
    def time_left(self) -> Optional[float]:
        """Time left in milliseconds until end of trading window."""
        return self._time_left

    @time_left.setter
    def time_left(self, value: Optional[float]):
        self._time_left = value

    @property
    def closing_time(self) -> Optional[float]:
        """Total trading window duration in milliseconds."""
        return self._closing_time

    @closing_time.setter
    def closing_time(self, value: Optional[float]):
        self._closing_time = value

    def __eq__(self, other):
        return type(self) is type(other)

    @abstractmethod
    def should_execute(self, timestamp: float) -> bool:
        """
        Check if trading should be executed at the given timestamp.

        Args:
            timestamp: Current Unix timestamp in seconds

        Returns:
            True if trading is allowed, False otherwise
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class RunAlwaysExecutionState(ExecutionState):
    """
    Execution state that always allows trading.

    Used for infinite/24/7 trading mode.
    """

    def should_execute(self, timestamp: float) -> bool:
        self._closing_time = None
        self._time_left = None
        return True

    def __str__(self) -> str:
        return "run continuously"


class RunInTimeExecutionState(ExecutionState):
    """
    Execution state that allows trading only within specified time boundaries.

    Supports two modes:
    - datetime boundaries: Trade between specific dates
    - time boundaries: Trade during specific hours each day
    """

    def __init__(
        self,
        start_timestamp: Union[datetime, time],
        end_timestamp: Optional[Union[datetime, time]] = None,
    ):
        super().__init__()
        self._start_timestamp = start_timestamp
        self._end_timestamp = end_timestamp

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self._start_timestamp == other._start_timestamp
            and self._end_timestamp == other._end_timestamp
        )

    def __str__(self) -> str:
        if isinstance(self._start_timestamp, datetime):
            if self._end_timestamp is not None:
                return f"run between {self._start_timestamp} and {self._end_timestamp}"
            else:
                return f"run from {self._start_timestamp}"
        if isinstance(self._start_timestamp, time):
            if self._end_timestamp is not None:
                return f"run daily between {self._start_timestamp} and {self._end_timestamp}"
            return f"run daily from {self._start_timestamp}"
        return "run in time"

    def should_execute(self, timestamp: float) -> bool:
        """
        Check if trading should execute at the given timestamp.

        For datetime mode: checks if timestamp is within date range.
        For time mode: checks if current time of day is within time range.

        Args:
            timestamp: Current Unix timestamp in seconds

        Returns:
            True if trading is allowed
        """
        if isinstance(self._start_timestamp, datetime):
            return self._check_datetime_bounds(timestamp)
        elif isinstance(self._start_timestamp, time):
            return self._check_time_bounds(timestamp)
        return True

    def _check_datetime_bounds(self, timestamp: float) -> bool:
        """Check datetime boundaries (from date to date mode)."""
        if self._end_timestamp is not None:
            # From datetime to datetime
            start_ts = self._start_timestamp.timestamp()
            end_ts = self._end_timestamp.timestamp()

            self._closing_time = (end_ts - start_ts) * 1000

            if start_ts <= timestamp < end_ts:
                self._time_left = max((end_ts - timestamp) * 1000, 0)
                return True
            else:
                self._time_left = 0
                logger.debug(
                    f"Time span execution: tick will not be processed "
                    f"(executing between {self._start_timestamp.isoformat(sep=' ')} "
                    f"and {self._end_timestamp.isoformat(sep=' ')})"
                )
                return False
        else:
            # From datetime onwards (no end)
            self._closing_time = None
            self._time_left = None
            if self._start_timestamp.timestamp() <= timestamp:
                return True
            else:
                logger.debug(
                    f"Delayed start execution: tick will not be processed "
                    f"(executing from {self._start_timestamp.isoformat(sep=' ')})"
                )
                return False

    def _check_time_bounds(self, timestamp: float) -> bool:
        """Check daily time boundaries (daily between times mode)."""
        if self._end_timestamp is not None:
            # Daily between times
            today = datetime.today()
            start_dt = datetime.combine(today, self._start_timestamp)
            end_dt = datetime.combine(today, self._end_timestamp)

            self._closing_time = (end_dt - start_dt).total_seconds() * 1000
            current_time = datetime.fromtimestamp(timestamp).time()

            if self._start_timestamp <= current_time < self._end_timestamp:
                current_dt = datetime.combine(today, current_time)
                self._time_left = max((end_dt - current_dt).total_seconds() * 1000, 0)
                return True
            else:
                self._time_left = 0
                logger.debug(
                    f"Time span execution: tick will not be processed "
                    f"(executing between {self._start_timestamp} "
                    f"and {self._end_timestamp})"
                )
                return False

        return True


def create_execution_state(config: ExecutionTimeframeConfig) -> ExecutionState:
    """
    Factory function to create the appropriate execution state.

    Args:
        config: Execution timeframe configuration

    Returns:
        Appropriate ExecutionState instance
    """
    if config.mode == ExecutionTimeframeMode.INFINITE:
        return RunAlwaysExecutionState()

    elif config.mode == ExecutionTimeframeMode.FROM_DATE_TO_DATE:
        if config.start_datetime is None:
            raise ValueError("start_datetime required for from_date_to_date mode")
        return RunInTimeExecutionState(
            start_timestamp=config.start_datetime,
            end_timestamp=config.end_datetime,
        )

    elif config.mode == ExecutionTimeframeMode.DAILY_BETWEEN_TIMES:
        if config.start_time is None or config.end_time is None:
            raise ValueError(
                "start_time and end_time required for daily_between_times mode"
            )
        return RunInTimeExecutionState(
            start_timestamp=config.start_time,
            end_timestamp=config.end_time,
        )

    else:
        return RunAlwaysExecutionState()

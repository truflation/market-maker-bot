"""
Underlying stream volatility calculation.

Ported from market_generator/volatility.py.

Used for Black-Scholes pricing when no order book data exists.
Implements Yang-Zhang (for hourly streams) and Close-to-Close (for daily/monthly)
volatility estimators.
"""

import math
import logging
from typing import List, Dict, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StreamFrequency(Enum):
    """Stream frequency classifications based on event count in last 48 hours."""
    HOURLY = "HOURLY"
    DAILY = "DAILY"
    MONTHLY = "MONTHLY"


@dataclass
class VolatilityResult:
    """Result of volatility calculation."""
    inferred_frequency: StreamFrequency
    lookback_days: int
    daily_volatility: float
    annual_volatility: float
    method_used: str
    data_points: int


def infer_stream_frequency(
    records: List[Dict],
    hourly_threshold: int = 16,
    daily_threshold: int = 2,
) -> Tuple[StreamFrequency, int]:
    """
    Infer stream frequency from event count in last 48 hours.

    TRUF Network data older than 48 hours undergoes digest compression,
    so only the most recent 48 hours reflects true publishing frequency.

    Args:
        records: List of records with 'EventTime' field (Unix timestamp)
        hourly_threshold: Events >= this = HOURLY (default 16)
        daily_threshold: Events >= this and < hourly = DAILY (default 2)

    Returns:
        Tuple of (StreamFrequency, event_count)
    """
    now = datetime.now(timezone.utc)
    cutoff_time = int((now - timedelta(hours=48)).timestamp())

    # Count events in last 48 hours
    recent_count = sum(
        1 for r in records if int(r.get("EventTime", 0)) >= cutoff_time
    )

    if recent_count >= hourly_threshold:
        return StreamFrequency.HOURLY, recent_count
    elif recent_count >= daily_threshold:
        return StreamFrequency.DAILY, recent_count
    else:
        return StreamFrequency.MONTHLY, recent_count


def _aggregate_to_daily_ohlc(records: List[Dict]) -> List[Dict]:
    """
    Aggregate intraday records to daily OHLC format.

    Args:
        records: List of records with 'EventTime' and 'Value' fields

    Returns:
        List of daily OHLC dicts with 'date', 'open', 'high', 'low', 'close'
    """
    # Group by date
    daily_data: Dict[datetime, List[Tuple[int, float]]] = defaultdict(list)

    for record in records:
        event_time = int(record.get("EventTime", 0))
        value = float(record.get("Value", 0))
        if event_time and value:
            dt = datetime.fromtimestamp(event_time, tz=timezone.utc)
            date_key = dt.date()
            daily_data[date_key].append((event_time, value))

    # Calculate OHLC for each day
    ohlc_list = []
    for date_key in sorted(daily_data.keys()):
        day_records = sorted(daily_data[date_key], key=lambda x: x[0])
        if day_records:
            values = [v for _, v in day_records]
            ohlc_list.append({
                "date": date_key,
                "open": day_records[0][1],   # First value of day
                "high": max(values),
                "low": min(values),
                "close": day_records[-1][1],  # Last value of day
            })

    return ohlc_list


def calc_yang_zhang_volatility(
    records: List[Dict],
    lookback_days: int = 14,
) -> VolatilityResult:
    """
    Yang-Zhang volatility estimator for hourly frequency streams.

    Uses OHLC data to efficiently extract volatility. The TRUF Network digest
    compression retains High, Low, and Close values.

    Formula:
    σ²_YZ = σ²_overnight + k × σ²_open + (1-k) × σ²_RS

    Where:
    - σ²_overnight = Variance of overnight returns (close-to-open)
    - σ²_open = Variance of open-to-close returns
    - σ²_RS = Rogers-Satchell variance
    - k = 0.34 / (1.34 + (n+1)/(n-1))

    Args:
        records: List of records with 'EventTime' and 'Value' fields
        lookback_days: Number of days to look back (default 14)

    Returns:
        VolatilityResult with daily and annual volatility
    """
    # Filter to lookback period
    now = datetime.now(timezone.utc)
    cutoff_time = int((now - timedelta(days=lookback_days)).timestamp())
    filtered_records = [
        r for r in records if int(r.get("EventTime", 0)) >= cutoff_time
    ]

    # Aggregate to daily OHLC
    ohlc_data = _aggregate_to_daily_ohlc(filtered_records)

    if len(ohlc_data) < 3:
        logger.warning(
            f"Insufficient data for Yang-Zhang: {len(ohlc_data)} days, need at least 3"
        )
        # Fall back to Close-to-Close
        return calc_close_to_close_volatility(
            records, lookback_days, StreamFrequency.HOURLY
        )

    # Calculate returns
    r_overnight = []   # ln(Open_t / Close_t-1)
    r_open_close = []  # ln(Close_t / Open_t)
    rs_components = []  # Rogers-Satchell

    for i in range(1, len(ohlc_data)):
        prev_close = ohlc_data[i - 1]["close"]
        curr = ohlc_data[i]

        if prev_close <= 0 or curr["open"] <= 0 or curr["close"] <= 0:
            continue
        if curr["high"] <= 0 or curr["low"] <= 0:
            continue

        # Overnight return
        r_overnight.append(math.log(curr["open"] / prev_close))

        # Open-to-close return
        r_open_close.append(math.log(curr["close"] / curr["open"]))

        # Rogers-Satchell component
        # ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
        h, l, o, c = curr["high"], curr["low"], curr["open"], curr["close"]
        rs = math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l / o)
        rs_components.append(rs)

    n = len(r_overnight)
    if n < 2:
        logger.warning(f"Insufficient returns for Yang-Zhang: {n} data points")
        return calc_close_to_close_volatility(
            records, lookback_days, StreamFrequency.HOURLY
        )

    # Calculate variances
    def variance(data: List[float]) -> float:
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        return sum((x - mean) ** 2 for x in data) / (len(data) - 1)

    var_overnight = variance(r_overnight)
    var_open_close = variance(r_open_close)
    var_rs = sum(rs_components) / n if n > 0 else 0.0

    # Calculate k (optimal weighting factor)
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))

    # Yang-Zhang variance
    var_yz = var_overnight + k * var_open_close + (1 - k) * var_rs

    daily_vol = math.sqrt(max(var_yz, 0))
    annual_vol = daily_vol * math.sqrt(365)

    return VolatilityResult(
        inferred_frequency=StreamFrequency.HOURLY,
        lookback_days=lookback_days,
        daily_volatility=daily_vol,
        annual_volatility=annual_vol,
        method_used="Yang-Zhang (OHLC)",
        data_points=n,
    )


def calc_close_to_close_volatility(
    records: List[Dict],
    lookback_days: int = 365,
    frequency: StreamFrequency = StreamFrequency.DAILY,
) -> VolatilityResult:
    """
    Close-to-Close volatility for daily/monthly frequency streams.

    Uses standard volatility based on log returns:
    σ_daily = STDDEV(ln(Close_t / Close_t-1))
    σ_annual = σ_daily × √365

    Args:
        records: List of records with 'EventTime' and 'Value' fields
        lookback_days: Number of days to look back (default 365)
        frequency: Stream frequency for result

    Returns:
        VolatilityResult with daily and annual volatility
    """
    # Filter to lookback period
    now = datetime.now(timezone.utc)
    cutoff_time = int((now - timedelta(days=lookback_days)).timestamp())
    filtered_records = [
        r for r in records if int(r.get("EventTime", 0)) >= cutoff_time
    ]

    # Aggregate to daily closes
    ohlc_data = _aggregate_to_daily_ohlc(filtered_records)

    if len(ohlc_data) < 2:
        logger.warning(
            f"Insufficient data for Close-to-Close: {len(ohlc_data)} days"
        )
        return VolatilityResult(
            inferred_frequency=frequency,
            lookback_days=lookback_days,
            daily_volatility=0.0,
            annual_volatility=0.0,
            method_used="Close-to-Close (insufficient data)",
            data_points=len(ohlc_data),
        )

    # Calculate log returns
    log_returns = []
    for i in range(1, len(ohlc_data)):
        prev_close = ohlc_data[i - 1]["close"]
        curr_close = ohlc_data[i]["close"]

        if prev_close > 0 and curr_close > 0:
            log_returns.append(math.log(curr_close / prev_close))

    if len(log_returns) < 2:
        return VolatilityResult(
            inferred_frequency=frequency,
            lookback_days=lookback_days,
            daily_volatility=0.0,
            annual_volatility=0.0,
            method_used="Close-to-Close (insufficient returns)",
            data_points=len(log_returns),
        )

    # Calculate standard deviation
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    daily_vol = math.sqrt(variance)
    annual_vol = daily_vol * math.sqrt(365)

    return VolatilityResult(
        inferred_frequency=frequency,
        lookback_days=lookback_days,
        daily_volatility=daily_vol,
        annual_volatility=annual_vol,
        method_used="Close-to-Close",
        data_points=len(log_returns),
    )


def calculate_stream_volatility(
    records: List[Dict],
    hourly_lookback: int = 14,
    daily_lookback: int = 365,
    monthly_lookback: int = 9999,
    hourly_threshold: int = 16,
    daily_threshold: int = 2,
    min_volatility: float = 0.30,
) -> VolatilityResult:
    """
    Master function that automatically selects appropriate volatility calculation
    based on inferred frequency.

    Args:
        records: List of records with 'EventTime' and 'Value' fields
        hourly_lookback: Lookback days for hourly streams (default 14)
        daily_lookback: Lookback days for daily streams (default 365)
        monthly_lookback: Lookback days for monthly streams (default 9999)
        hourly_threshold: Threshold for hourly classification
        daily_threshold: Threshold for daily classification
        min_volatility: Minimum volatility floor (default 1%)

    Returns:
        VolatilityResult with daily and annual volatility
    """
    # Step 1: Infer frequency from last 48 hours
    frequency, event_count = infer_stream_frequency(
        records, hourly_threshold, daily_threshold
    )
    logger.info(
        f"Inferred frequency: {frequency.value} ({event_count} events in last 48h)"
    )

    # Step 2: Select method based on frequency
    if frequency == StreamFrequency.HOURLY:
        result = calc_yang_zhang_volatility(records, hourly_lookback)
    elif frequency == StreamFrequency.DAILY:
        result = calc_close_to_close_volatility(records, daily_lookback, frequency)
    else:  # MONTHLY
        result = calc_close_to_close_volatility(records, monthly_lookback, frequency)

    # Apply minimum volatility floor
    if result.annual_volatility < min_volatility:
        logger.warning(
            f"Calculated volatility {result.annual_volatility:.4f} below minimum, "
            f"using floor of {min_volatility}"
        )
        result = VolatilityResult(
            inferred_frequency=result.inferred_frequency,
            lookback_days=result.lookback_days,
            daily_volatility=min_volatility / math.sqrt(365),
            annual_volatility=min_volatility,
            method_used=f"{result.method_used} (floor applied)",
            data_points=result.data_points,
        )

    return result


def get_current_spot_value(records: List[Dict]) -> float:
    """
    Get the most recent value from stream records.

    Args:
        records: List of records with 'EventTime' and 'Value' fields

    Returns:
        Most recent value, or 0 if no records
    """
    if not records:
        return 0.0

    # Find record with latest EventTime
    latest = max(records, key=lambda r: int(r.get("EventTime", 0)))
    return float(latest.get("Value", 0))

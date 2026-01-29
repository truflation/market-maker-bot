"""Indicator modules for the Avellaneda Market Making Bot."""

from .volatility import InstantVolatilityIndicator
from .stream_volatility import (
    calculate_stream_volatility,
    infer_stream_frequency,
    calc_yang_zhang_volatility,
    calc_close_to_close_volatility,
    get_current_spot_value,
    StreamFrequency,
    VolatilityResult,
)
from .depth import OrderBookDepthAnalyzer

__all__ = [
    "InstantVolatilityIndicator",
    "calculate_stream_volatility",
    "infer_stream_frequency",
    "calc_yang_zhang_volatility",
    "calc_close_to_close_volatility",
    "get_current_spot_value",
    "StreamFrequency",
    "VolatilityResult",
    "OrderBookDepthAnalyzer",
]

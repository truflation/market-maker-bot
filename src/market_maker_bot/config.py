"""Configuration models for the Avellaneda Market Making Bot."""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Optional, Dict, Any, List

from .models import OutcomeMode
from .execution_state import ExecutionTimeframeMode, ExecutionTimeframeConfig


# =============================================================================
# Pre-Approved Streams for Market Making
# =============================================================================
# These are the official TrufNetwork streams that can be used for market making.
# Users can select which streams they want to market make from this list.

@dataclass
class ApprovedStream:
    """A pre-approved stream for market making."""
    stream_id: str
    name: str
    description: str = ""
    data_provider: str = "0x4710a8d8f0d845da110086812a32de6d90d7ff5c"  # Default Truflation provider


# Pre-approved streams available for market making
APPROVED_STREAMS: Dict[str, ApprovedStream] = {
    "us_inflation_yoy": ApprovedStream(
        stream_id="st1e321de22ece39a258bc2588dd2871",
        name="US Inflation YoY",
        description="US Year-over-Year Inflation Rate",
    ),
    "us_cpi_index": ApprovedStream(
        stream_id="st8f1e62d3a130572ec468dda082f889",
        name="US CPI Index",
        description="US Consumer Price Index",
    ),
    "us_cpi_index_alt": ApprovedStream(
        stream_id="st1d6d41423cd9746a81ea6063b1345e",
        name="US CPI Index Alt",
        description="US Consumer Price Index (Alternative)",
    ),
    "eu_inflation_yoy": ApprovedStream(
        stream_id="ste03c2844c591a10d8a524d14d23066",
        name="EU Inflation YoY",
        description="EU Year-over-Year Inflation Rate",
    ),
    "eu_cpi_index": ApprovedStream(
        stream_id="ste909219dce3f693c61a0f187758fb0",
        name="EU CPI Index",
        description="EU Consumer Price Index",
    ),
    "egg_price": ApprovedStream(
        stream_id="stf6584cf470744723c90130130cb7db",
        name="Egg Price",
        description="Egg Price Index",
    ),
}


TESTNET_APPROVED_STREAMS: Dict[str, ApprovedStream] = {
    "testnet_btc_price": ApprovedStream(
        stream_id="st9058219c3c3247faf2b0a738de7027",
        name="Testnet BTC-like Price",
        description="Testnet BTC-like Price Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
    "testnet_midcap_price": ApprovedStream(
        stream_id="st5cda3b42dc3db0e49af57d7bf14905",
        name="Testnet Mid-Cap Price",
        description="Testnet Mid-Cap Price Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
    "testnet_lowcap_price": ApprovedStream(
        stream_id="st361547d8b439502d3828d74ca679b5",
        name="Testnet Low-Cap Price",
        description="Testnet Low-Cap Price Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
    "testnet_rate_a": ApprovedStream(
        stream_id="st26e6f725c82630d2c5bd542883453f",
        name="Testnet Rate A",
        description="Testnet Rate A Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
    "testnet_rate_b": ApprovedStream(
        stream_id="stf826b74de25bcae10dcde294c25e87",
        name="Testnet Rate B",
        description="Testnet Rate B Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
    "testnet_midrange_price": ApprovedStream(
        stream_id="stde38e5fd701194ef8da203c8fb012b",
        name="Testnet Mid-Range Price",
        description="Testnet Mid-Range Price Stream",
        data_provider="0xe5252596672cd0208a881bdb67c9df429916ba92",
    ),
}


def get_approved_stream(key: str) -> Optional[ApprovedStream]:
    """Get an approved stream by its key (checks both mainnet and testnet)."""
    return APPROVED_STREAMS.get(key) or TESTNET_APPROVED_STREAMS.get(key)


def list_approved_streams() -> List[ApprovedStream]:
    """List all approved streams."""
    return list(APPROVED_STREAMS.values())


@dataclass
class MarketConfig:
    """Configuration for a single market."""
    query_id: int
    stream_id: str  # Underlying data stream for Black-Scholes pricing
    data_provider: str  # Data provider address for stream
    name: str = ""
    outcome_mode: OutcomeMode = OutcomeMode.YES_ONLY
    order_amount: int = 100  # Default order size in shares
    enabled: bool = True

    # Market-specific overrides (use global if None)
    gamma: Optional[float] = None
    min_spread: Optional[float] = None

    # Threshold data from generated_markets (for Black-Scholes pricing)
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    settle_time: Optional[int] = None

    def __post_init__(self):
        if not self.name:
            self.name = f"Market-{self.query_id}"


@dataclass
class AvellanedaConfig:
    """
    Avellaneda-Stoikov strategy parameters.

    Adapted from the original A-S paper for binary options markets
    where prices are in cents (1-99 range).
    """

    # Risk factor (γ) - gamma
    # Higher values = wider spreads, more conservative
    # Lower values = tighter spreads, more aggressive
    risk_factor: float = 1.0

    # Order amount shape factor (η) - eta
    # Range: 0.0 to 1.0
    # With eta=0, buy and sell orders have the same size regardless of inventory
    # With eta>0, order sizes decrease exponentially for orders against inventory target
    # Formula: size * exp(-eta * q) where q is inventory deviation
    order_amount_shape_factor: float = 0.0

    # Minimum spread as percentage of mid price
    # For binary options: 0 means no minimum, 2.0 means 2% of mid price
    # At mid=50 cents, 2% = 1 cent minimum spread
    min_spread: float = 0.0

    # Order refresh time in seconds
    # How often to re-evaluate and update orders
    order_refresh_time: float = 10.0

    # Maximum order age in seconds before forced refresh
    # Orders older than this are cancelled and replaced regardless of price changes
    max_order_age: float = 1800.0

    # Order refresh tolerance as percentage
    # Orders are only refreshed if price change exceeds this threshold
    # 0 = refresh every cycle, 1.0 = only refresh if price changed by 1%+
    order_refresh_tolerance_pct: float = 0.0

    # Delay in seconds after a fill before placing new orders
    filled_order_delay: float = 60.0

    # Inventory target as percentage of total value in base asset (shares)
    # 50 = aim for 50% in shares, 50% in USD
    inventory_target_base_pct: float = 50.0

    # Add transaction costs/fees to order prices
    # When true, buy prices are reduced and sell prices are increased by fee %
    add_transaction_costs: bool = False

    # Volatility buffer size (number of ticks/samples)
    # Used to calculate rolling volatility from order book mid-price changes
    volatility_buffer_size: int = 200

    # Trading intensity buffer size (number of ticks/samples)
    # Used to estimate alpha and kappa from order book liquidity
    trading_intensity_buffer_size: int = 200

    # Order book depth (kappa) parameters
    # Kappa represents order book liquidity - higher = more liquid book
    use_dynamic_kappa: bool = True  # Estimate kappa from order book depth
    default_kappa: float = 1.0  # Fallback kappa when not using dynamic estimation

    # Order optimization - jump to best bid+1 / best ask-1
    # When enabled, orders will improve on the best price in the book
    order_optimization_enabled: bool = True

    # Order levels mode
    # 1 = single order on each side
    # >1 = multiple orders at different price levels
    order_levels: int = 1

    # Distance between order levels as percentage of optimal spread
    # Only used when order_levels > 1
    # Example: 20.0 means each level is 20% of optimal spread apart
    level_distances: float = 0.0

    # Gamma multiplier per level for progressive spread widening
    # Each deeper level widens offset by gamma_mult^level_idx
    # 1.0 = linear spacing, 1.3 = ~30% wider per level
    level_gamma_multiplier: float = 1.3

    # Order size increase per level as percentage
    # Each deeper level increases order size by this percentage
    # 5.0 = 5% more per level (L0=base, L1=base*1.05, L2=base*1.1025, ...)
    level_amount_increase_pct: float = 0.0

    # Hanging orders - track orders that remain after opposite side fills
    # When enabled, unfilled orders on the opposite side are tracked and managed
    hanging_orders_enabled: bool = False

    # Percentage from mid price at which hanging orders are cancelled
    # Only used when hanging_orders_enabled=True
    hanging_orders_cancel_pct: float = 10.0

    # Wait for order cancellation confirmation before placing new orders
    should_wait_order_cancel_confirmation: bool = True

    # === Execution Timeframe ===

    # Execution timeframe mode: "infinite", "from_date_to_date", "daily_between_times"
    execution_timeframe_mode: str = "infinite"

    # For from_date_to_date mode: start and end datetimes (ISO format strings)
    execution_start_datetime: Optional[str] = None
    execution_end_datetime: Optional[str] = None

    # For daily_between_times mode: start and end times (HH:MM:SS format strings)
    execution_start_time: Optional[str] = None
    execution_end_time: Optional[str] = None

    # === Order Override ===

    # Custom order specifications that override calculated prices
    # Format: {"order_name": ["buy"|"sell", spread_pct, amount], ...}
    # Example: {"order_1": ["buy", 1.0, 100], "order_2": ["sell", 1.0, 100]}
    # spread_pct is the percentage distance from mid price
    order_override: Optional[Dict[str, Any]] = None

    # === Binary options specific parameters ===

    # Minimum samples before using calculated volatility (use default until then)
    volatility_min_samples: int = 10

    # Default volatility in cents when insufficient samples
    default_volatility: float = 5.0

    # Floor volatility in cents (never go below this)
    min_volatility: float = 1.0

    # Stream volatility parameters (for Black-Scholes initial pricing)
    stream_volatility_lookback_days: int = 14
    stream_volatility_min: float = 1.0  # Minimum annual volatility (100%) - wide quotes when no data

    # Maximum spread in cents (hard ceiling for binary options)
    max_spread: float = 20.0

    # Time horizon T in A-S formulas
    # 1.0 = infinite timeframe (recommended for 24/7 markets)
    time_horizon: float = 1.0


@dataclass
class BotConfig:
    """Main bot configuration."""

    # Network connection
    node_url: str = "http://localhost:8484"
    private_key: str = ""  # Will be loaded from environment

    # Markets to trade
    markets: list[MarketConfig] = field(default_factory=list)

    # Strategy parameters
    avellaneda: AvellanedaConfig = field(default_factory=AvellanedaConfig)

    # Polling intervals
    order_book_poll_interval: float = 2.0  # Seconds between order book polls
    inventory_refresh_interval: float = 30.0  # Seconds between inventory refresh

    # Operational settings
    dry_run: bool = False  # If True, log but don't execute orders
    debug: bool = False  # Enable verbose logging
    cancel_open_orders_on_exit: bool = True  # Cancel all open orders when bot shuts down
    order_state_file: str = "bot_order_state.json"  # File to persist order state for restart recovery
    pre_settlement_cutoff: float = 900.0  # Seconds before settle_time to pull all liquidity (default 15 min)


def load_config_from_dict(data: dict) -> BotConfig:
    """Load configuration from a dictionary (e.g., from YAML/JSON)."""
    import os

    avellaneda_data = data.get("avellaneda", {})
    avellaneda_config = AvellanedaConfig(**avellaneda_data)

    markets = []
    for market_data in data.get("markets", []):
        outcome_mode_str = market_data.pop("outcome_mode", "yes")
        market_data["outcome_mode"] = OutcomeMode(outcome_mode_str)
        markets.append(MarketConfig(**market_data))

    # Private key: YAML > TN_PRIVATE_KEY env var
    private_key = data.get("private_key", "") or os.environ.get("TN_PRIVATE_KEY", "")

    return BotConfig(
        node_url=data.get("node_url", "http://localhost:8484"),
        private_key=private_key,
        markets=markets,
        avellaneda=avellaneda_config,
        order_book_poll_interval=data.get("order_book_poll_interval", 2.0),
        inventory_refresh_interval=data.get("inventory_refresh_interval", 30.0),
        dry_run=data.get("dry_run", False),
        debug=data.get("debug", False),
        cancel_open_orders_on_exit=data.get("cancel_open_orders_on_exit", True),
        order_state_file=data.get("order_state_file", "bot_order_state.json"),
        pre_settlement_cutoff=data.get("pre_settlement_cutoff", 900.0),
    )

"""Data models for the Avellaneda Market Making Bot."""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class OutcomeMode(str, Enum):
    """Trading mode for market outcomes."""
    YES_ONLY = "yes"      # Trade only YES shares
    NO_ONLY = "no"        # Trade only NO shares
    BOTH = "both"         # Trade both YES and NO


class Side(str, Enum):
    """Order side."""
    BID = "bid"
    ASK = "ask"


@dataclass
class OrderLevel:
    """Single order level in the order book."""
    price: int  # -99 to 99 (negative = bid, positive = ask, 0 = holding)
    quantity: int
    age_seconds: float = 0.0
    wallet_address: Optional[bytes] = None


@dataclass
class MarketState:
    """Current state of a market's order book for one outcome."""
    query_id: int
    outcome: bool
    best_bid: Optional[int]  # Absolute positive price (1-99)
    best_ask: Optional[int]  # Absolute positive price (1-99)
    bid_levels: list[OrderLevel] = field(default_factory=list)
    ask_levels: list[OrderLevel] = field(default_factory=list)

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price if both sides have orders."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[int]:
        """Calculate spread in cents."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def has_liquidity(self) -> bool:
        """Check if market has orders on both sides."""
        return self.best_bid is not None and self.best_ask is not None


@dataclass
class PricingResult:
    """Result from Avellaneda-Stoikov pricing calculation."""
    reservation_price: float
    optimal_spread: float
    bid_price: float
    ask_price: float
    mid_price: float
    inventory_skew: float  # q value (-1 to +1)
    volatility: float
    kappa: float

    def to_int_prices(self) -> tuple[int, int]:
        """Convert to integer cents (1-99 range)."""
        bid_int = max(1, min(99, int(round(self.bid_price))))
        ask_int = max(1, min(99, int(round(self.ask_price))))
        return bid_int, ask_int


@dataclass
class BotOrder:
    """An order the bot intends to place or has placed."""
    query_id: int
    outcome: bool
    side: Side
    price: int  # 1-99 display price
    amount: int
    tx_hash: Optional[str] = None
    created_at: float = 0.0  # Unix timestamp


@dataclass
class ActiveOrders:
    """Tracks active orders for a market outcome, supporting multiple levels."""
    bids: list = field(default_factory=lambda: [None])
    asks: list = field(default_factory=lambda: [None])

    @property
    def bid(self) -> Optional[BotOrder]:
        """Get L0 bid (backwards compat)."""
        return self.bids[0] if self.bids else None

    @bid.setter
    def bid(self, value):
        if not self.bids:
            self.bids = [value]
        else:
            self.bids[0] = value

    @property
    def ask(self) -> Optional[BotOrder]:
        """Get L0 ask (backwards compat)."""
        return self.asks[0] if self.asks else None

    @ask.setter
    def ask(self, value):
        if not self.asks:
            self.asks = [value]
        else:
            self.asks[0] = value

    def get_bid(self, level: int = 0) -> Optional[BotOrder]:
        self._ensure_level(level)
        return self.bids[level]

    def get_ask(self, level: int = 0) -> Optional[BotOrder]:
        self._ensure_level(level)
        return self.asks[level]

    def set_bid(self, level: int, order: Optional[BotOrder]):
        self._ensure_level(level)
        self.bids[level] = order

    def set_ask(self, level: int, order: Optional[BotOrder]):
        self._ensure_level(level)
        self.asks[level] = order

    def _ensure_level(self, level: int):
        while len(self.bids) <= level:
            self.bids.append(None)
        while len(self.asks) <= level:
            self.asks.append(None)


@dataclass
class VolatilityEstimate:
    """Result of volatility estimation."""
    value: float
    source: str  # "order_book", "stream", "default"
    samples: int

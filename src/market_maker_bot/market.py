"""
Market state management for the Avellaneda Market Making Bot.

Handles order book parsing, state tracking, and order management
for individual markets.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field

from .models import OrderLevel, MarketState, BotOrder, ActiveOrders, Side
from .config import MarketConfig

logger = logging.getLogger(__name__)


def parse_order_book_entries(
    entries: list[dict],
    current_time: Optional[float] = None,
) -> tuple[list[OrderLevel], list[OrderLevel]]:
    """
    Parse SDK order book entries into bid and ask levels.

    The SDK returns entries with price encoding:
    - Negative price = open buy order (bid)
    - Positive price = open sell order (ask)
    - Zero = holding tokens (not in order book)

    Args:
        entries: List of OrderBookEntry dicts from SDK
        current_time: Current timestamp for age calculation

    Returns:
        Tuple of (bids, asks) sorted best to worst:
        - Bids: highest to lowest (best bid first)
        - Asks: lowest to highest (best ask first)
    """
    if current_time is None:
        current_time = time.time()

    bids = []
    asks = []

    for entry in entries:
        price = entry["price"]
        amount = entry["amount"]
        last_updated = entry.get("last_updated", current_time)

        # Calculate age in seconds
        age_seconds = max(0, current_time - last_updated)

        if price < 0:
            # Bid (buy order)
            bids.append(
                OrderLevel(
                    price=price,
                    quantity=amount,
                    age_seconds=age_seconds,
                    wallet_address=entry.get("wallet_address"),
                )
            )
        elif price > 0:
            # Ask (sell order)
            asks.append(
                OrderLevel(
                    price=price,
                    quantity=amount,
                    age_seconds=age_seconds,
                    wallet_address=entry.get("wallet_address"),
                )
            )
        # price == 0 means holding, skip for order book

    # Sort bids by absolute price value, highest first
    bids.sort(key=lambda x: abs(x.price), reverse=True)

    # Sort asks: lowest to highest
    asks.sort(key=lambda x: x.price)

    return bids, asks


def get_best_prices(
    bids: list[OrderLevel],
    asks: list[OrderLevel],
) -> tuple[Optional[int], Optional[int]]:
    """
    Extract best bid/ask from sorted level lists.

    Returns:
        Tuple of (best_bid, best_ask) in absolute positive values (1-99),
        or None if no orders on that side.
    """
    best_bid = abs(bids[0].price) if bids else None
    best_ask = asks[0].price if asks else None
    return best_bid, best_ask


def build_market_state(
    query_id: int,
    outcome: bool,
    order_book_entries: list[dict],
    current_time: Optional[float] = None,
) -> MarketState:
    """
    Build a MarketState from SDK order book data.

    Args:
        query_id: Market ID
        outcome: True for YES shares, False for NO shares
        order_book_entries: Raw entries from SDK's get_order_book()
        current_time: Current timestamp for age calculation

    Returns:
        MarketState with parsed bids, asks, and best prices
    """
    bids, asks = parse_order_book_entries(order_book_entries, current_time)
    best_bid, best_ask = get_best_prices(bids, asks)

    return MarketState(
        query_id=query_id,
        outcome=outcome,
        best_bid=best_bid,
        best_ask=best_ask,
        bid_levels=bids,
        ask_levels=asks,
    )


@dataclass
class MarketContext:
    """
    Runtime context for a single market.

    Tracks order book state, active orders, and timing information.
    """

    config: MarketConfig
    yes_state: Optional[MarketState] = None
    no_state: Optional[MarketState] = None
    yes_orders: ActiveOrders = field(default_factory=ActiveOrders)
    no_orders: ActiveOrders = field(default_factory=ActiveOrders)
    last_order_book_update: float = 0.0
    last_order_refresh: float = 0.0
    initial_price_yes: Optional[float] = None  # From Black-Scholes
    initial_price_no: Optional[float] = None

    @property
    def query_id(self) -> int:
        """Market ID."""
        return self.config.query_id

    def get_state(self, outcome: bool) -> Optional[MarketState]:
        """Get state for an outcome."""
        return self.yes_state if outcome else self.no_state

    def set_state(self, outcome: bool, state: MarketState) -> None:
        """Set state for an outcome."""
        if outcome:
            self.yes_state = state
        else:
            self.no_state = state

    def get_orders(self, outcome: bool) -> ActiveOrders:
        """Get active orders for an outcome."""
        return self.yes_orders if outcome else self.no_orders

    def get_mid_price(self, outcome: bool) -> Optional[float]:
        """
        Get mid price for an outcome.

        Falls back to initial Black-Scholes price if no order book data.
        """
        state = self.get_state(outcome)
        if state and state.mid_price is not None:
            return state.mid_price

        # Fall back to initial price
        return self.initial_price_yes if outcome else self.initial_price_no

    def needs_order_refresh(self, refresh_interval: float) -> bool:
        """Check if orders need to be refreshed based on time."""
        if self.last_order_refresh == 0:
            return True
        return time.time() - self.last_order_refresh >= refresh_interval


class OrderManager:
    """
    Manages order placement and updates for a single market.

    Handles the logic for determining when to place new orders,
    update existing orders, or cancel orders.
    """

    def __init__(
        self,
        context: MarketContext,
        refresh_tolerance_pct: float = 1.0,
        max_order_age: float = 300.0,
    ):
        """
        Initialize order manager.

        Args:
            context: Market context to manage
            refresh_tolerance_pct: Price change % to trigger refresh
            max_order_age: Maximum order age before forced refresh
        """
        self.context = context
        self.refresh_tolerance_pct = refresh_tolerance_pct
        self.max_order_age = max_order_age

    def should_update_order(
        self,
        outcome: bool,
        side: Side,
        new_price: int,
        level_idx: int = 0,
    ) -> tuple[bool, str]:
        """
        Determine if an order should be updated.

        Args:
            outcome: True for YES, False for NO
            side: Order side (bid or ask)
            new_price: Proposed new price in cents
            level_idx: Order level index (0 = tightest spread)

        Returns:
            Tuple of (should_update, reason)
        """
        current_order = self.get_current_order(outcome, side, level_idx)

        if current_order is None:
            return True, "no_existing_order"

        # Check age
        order_age = time.time() - current_order.created_at
        if order_age >= self.max_order_age:
            return True, f"max_age_exceeded ({order_age:.0f}s)"

        # Check price deviation
        price_diff = abs(new_price - current_order.price)
        price_pct = (price_diff / current_order.price) * 100

        if price_pct >= self.refresh_tolerance_pct:
            return True, f"price_deviation ({price_pct:.1f}%)"

        return False, "within_tolerance"

    def record_order(
        self,
        outcome: bool,
        side: Side,
        price: int,
        amount: int,
        tx_hash: str,
        level_idx: int = 0,
    ) -> BotOrder:
        """
        Record a placed order.

        Args:
            outcome: True for YES, False for NO
            side: Order side
            price: Order price in cents (1-99)
            amount: Order amount
            tx_hash: Transaction hash
            level_idx: Order level index

        Returns:
            Created BotOrder
        """
        order = BotOrder(
            query_id=self.context.query_id,
            outcome=outcome,
            side=side,
            price=price,
            amount=amount,
            tx_hash=tx_hash,
            created_at=time.time(),
        )

        orders = self.context.get_orders(outcome)
        if side == Side.BID:
            orders.set_bid(level_idx, order)
        else:
            orders.set_ask(level_idx, order)

        return order

    def clear_order(self, outcome: bool, side: Side, level_idx: int = 0) -> None:
        """Clear recorded order after cancellation."""
        orders = self.context.get_orders(outcome)
        if side == Side.BID:
            orders.set_bid(level_idx, None)
        else:
            orders.set_ask(level_idx, None)

    def get_current_order(
        self, outcome: bool, side: Side, level_idx: int = 0
    ) -> Optional[BotOrder]:
        """Get current order for outcome, side, and level."""
        orders = self.context.get_orders(outcome)
        return orders.get_bid(level_idx) if side == Side.BID else orders.get_ask(level_idx)


def convert_price_for_order(price: int, side: Side) -> int:
    """
    Convert display price (1-99) to SDK order price format.

    For bids: SDK expects negative price
    For asks: SDK expects positive price

    Args:
        price: Display price in cents (1-99)
        side: Order side

    Returns:
        Price in SDK format
    """
    if not 1 <= price <= 99:
        raise ValueError(f"price must be 1-99, got {price}")

    if side == Side.BID:
        return -price
    else:
        return price

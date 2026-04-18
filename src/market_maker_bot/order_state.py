"""
Persistent order state management.

Tracks orders placed by the bot so they can be recovered on restart.
Orders placed manually outside the bot are not tracked.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrackedOrder:
    """An order placed by the bot."""
    query_id: int
    outcome: bool  # True = YES, False = NO
    is_buy: bool
    price: int  # Price in cents (1-99)
    amount: int
    created_at: float  # Unix timestamp
    order_id: Optional[str] = None  # SDK order ID if available
    level_idx: int = 0  # Order level index (0 = tightest spread)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrackedOrder":
        # Backwards compat: old state files may not have level_idx
        if "level_idx" not in data:
            data["level_idx"] = 0
        return cls(**data)

    @property
    def key(self) -> str:
        """Unique key for this order position."""
        side = "buy" if self.is_buy else "sell"
        outcome_str = "yes" if self.outcome else "no"
        return f"{self.query_id}:{outcome_str}:{side}:{self.price}:{self.level_idx}"


class OrderStateManager:
    """
    Manages persistent order state for the bot.

    Stores order information in a JSON file so the bot can:
    1. Track which orders it placed (vs manual orders)
    2. Resume managing its orders after a restart
    3. Avoid interfering with manually placed orders
    """

    def __init__(self, state_file: str = "bot_order_state.json"):
        """
        Initialize the order state manager.

        Args:
            state_file: Path to the JSON file for persisting state
        """
        self.state_file = Path(state_file)
        self._orders: Dict[str, TrackedOrder] = {}  # key -> TrackedOrder
        self._load_state()

    def _load_state(self) -> None:
        """Load order state from file."""
        if not self.state_file.exists():
            logger.info(f"No existing state file at {self.state_file}")
            return

        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)

            for order_data in data.get("orders", []):
                order = TrackedOrder.from_dict(order_data)
                self._orders[order.key] = order

            logger.info(f"Loaded {len(self._orders)} tracked orders from {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to load order state: {e}")
            self._orders = {}

    def _save_state(self) -> None:
        """Save order state to file."""
        try:
            data = {
                "orders": [order.to_dict() for order in self._orders.values()],
                "last_updated": time.time(),
            }

            # Write atomically using temp file
            temp_file = self.state_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)

            temp_file.replace(self.state_file)
            logger.debug(f"Saved {len(self._orders)} tracked orders to {self.state_file}")

        except Exception as e:
            logger.error(f"Failed to save order state: {e}")

    def track_order(
        self,
        query_id: int,
        outcome: bool,
        is_buy: bool,
        price: int,
        amount: int,
        order_id: Optional[str] = None,
        level_idx: int = 0,
    ) -> TrackedOrder:
        """
        Track a new order placed by the bot.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            is_buy: True for buy order, False for sell order
            price: Order price in cents
            amount: Order amount
            order_id: Optional SDK order ID
            level_idx: Order level index (0 = tightest spread)

        Returns:
            The tracked order
        """
        order = TrackedOrder(
            query_id=query_id,
            outcome=outcome,
            is_buy=is_buy,
            price=price,
            amount=amount,
            created_at=time.time(),
            order_id=order_id,
            level_idx=level_idx,
        )

        self._orders[order.key] = order
        self._save_state()

        logger.debug(f"Tracking order: {order.key}")
        return order

    def update_order(
        self,
        query_id: int,
        outcome: bool,
        is_buy: bool,
        old_price: int,
        new_price: int,
        amount: int,
        order_id: Optional[str] = None,
        level_idx: int = 0,
    ) -> TrackedOrder:
        """
        Update a tracked order (price change).

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            is_buy: True for buy order, False for sell order
            old_price: Previous price
            new_price: New price
            amount: Order amount
            order_id: Optional SDK order ID
            level_idx: Order level index (0 = tightest spread)

        Returns:
            The updated tracked order
        """
        # Remove old order
        old_key = self._make_key(query_id, outcome, is_buy, old_price, level_idx)
        if old_key in self._orders:
            del self._orders[old_key]

        # Add new order
        return self.track_order(query_id, outcome, is_buy, new_price, amount, order_id, level_idx)

    def untrack_order(
        self,
        query_id: int,
        outcome: bool,
        is_buy: bool,
        price: int,
        level_idx: int = 0,
    ) -> bool:
        """
        Remove an order from tracking (cancelled or filled).

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            is_buy: True for buy order, False for sell order
            price: Order price
            level_idx: Order level index (0 = tightest spread)

        Returns:
            True if order was found and removed
        """
        key = self._make_key(query_id, outcome, is_buy, price, level_idx)
        if key in self._orders:
            del self._orders[key]
            self._save_state()
            logger.debug(f"Untracked order: {key}")
            return True
        return False

    def is_bot_order(
        self,
        query_id: int,
        outcome: bool,
        is_buy: bool,
        price: int,
        level_idx: int = 0,
    ) -> bool:
        """
        Check if an order was placed by the bot.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            is_buy: True for buy order, False for sell order
            price: Order price
            level_idx: Order level index (0 = tightest spread)

        Returns:
            True if this order is tracked by the bot
        """
        key = self._make_key(query_id, outcome, is_buy, price, level_idx)
        return key in self._orders

    def get_tracked_order(
        self,
        query_id: int,
        outcome: bool,
        is_buy: bool,
        price: int,
        level_idx: int = 0,
    ) -> Optional[TrackedOrder]:
        """Get a tracked order if it exists."""
        key = self._make_key(query_id, outcome, is_buy, price, level_idx)
        return self._orders.get(key)

    def get_market_orders(
        self, query_id: int, outcome: Optional[bool] = None
    ) -> List[TrackedOrder]:
        """
        Get all tracked orders for a market.

        Args:
            query_id: Market ID
            outcome: Optional filter by outcome (None = all)

        Returns:
            List of tracked orders
        """
        orders = []
        for order in self._orders.values():
            if order.query_id == query_id:
                if outcome is None or order.outcome == outcome:
                    orders.append(order)
        return orders

    def get_all_orders(self) -> List[TrackedOrder]:
        """Get all tracked orders."""
        return list(self._orders.values())

    def clear_market(self, query_id: int) -> int:
        """
        Clear all tracked orders for a market.

        Args:
            query_id: Market ID

        Returns:
            Number of orders removed
        """
        keys_to_remove = [
            key for key, order in self._orders.items()
            if order.query_id == query_id
        ]

        for key in keys_to_remove:
            del self._orders[key]

        if keys_to_remove:
            self._save_state()

        logger.info(f"Cleared {len(keys_to_remove)} tracked orders for market {query_id}")
        return len(keys_to_remove)

    def clear_all(self) -> int:
        """
        Clear all tracked orders.

        Returns:
            Number of orders removed
        """
        count = len(self._orders)
        self._orders.clear()
        self._save_state()
        logger.info(f"Cleared all {count} tracked orders")
        return count

    def reconcile_with_orderbook(
        self,
        query_id: int,
        outcome: bool,
        orderbook_prices: Dict[int, int],  # price -> amount for our orders
    ) -> Dict[str, List[TrackedOrder]]:
        """
        Reconcile tracked orders with actual order book state.

        This is called on startup to identify:
        - Orders that are still active (in both tracked and orderbook)
        - Orders that were filled/cancelled (tracked but not in orderbook)
        - Unknown orders (in orderbook but not tracked - these are ignored)

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            orderbook_prices: Dict of price -> amount for orders on the book

        Returns:
            Dict with 'active', 'stale' lists of TrackedOrder
        """
        result = {
            "active": [],  # Orders still on the book
            "stale": [],   # Orders no longer on the book (filled/cancelled)
        }

        tracked = self.get_market_orders(query_id, outcome)

        for order in tracked:
            if order.price in orderbook_prices:
                # Order is still on the book
                result["active"].append(order)
            else:
                # Order is no longer on the book - was filled or cancelled
                result["stale"].append(order)
                # Remove from tracking
                self.untrack_order(query_id, outcome, order.is_buy, order.price, order.level_idx)

        return result

    def _make_key(self, query_id: int, outcome: bool, is_buy: bool, price: int, level_idx: int = 0) -> str:
        """Create a unique key for an order."""
        side = "buy" if is_buy else "sell"
        outcome_str = "yes" if outcome else "no"
        return f"{query_id}:{outcome_str}:{side}:{price}:{level_idx}"

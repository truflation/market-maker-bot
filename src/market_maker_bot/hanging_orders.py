"""
Hanging orders tracker for the Avellaneda Market Making Bot.

Tracks orders that remain active after the opposite side of a pair is filled.
When a buy order is filled, the corresponding sell order becomes a "hanging order"
and vice versa.

Features:
- Track pairs of orders (buy/sell)
- Identify hanging orders when one side fills
- Cancel hanging orders that move too far from current price
- Renew hanging orders that exceed max age
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class HangingOrder:
    """Represents a hanging order that should be tracked."""
    order_id: Optional[str]
    query_id: int
    outcome: bool  # True for YES, False for NO
    is_buy: bool
    price: int  # Price in cents
    amount: int
    creation_timestamp: float

    def __hash__(self):
        return hash((self.order_id, self.query_id, self.outcome, self.is_buy, self.price))

    def __eq__(self, other):
        if not isinstance(other, HangingOrder):
            return False
        return (
            self.order_id == other.order_id
            and self.query_id == other.query_id
            and self.outcome == other.outcome
            and self.is_buy == other.is_buy
            and self.price == other.price
        )


@dataclass
class CreatedPairOfOrders:
    """
    Tracks a pair of orders (buy and sell) created together.

    When one side fills, the other becomes a hanging order.
    """
    buy_order: Optional[HangingOrder] = None
    sell_order: Optional[HangingOrder] = None
    filled_buy: bool = False
    filled_sell: bool = False

    def contains_order(self, order_id: str) -> bool:
        """Check if this pair contains the given order."""
        return (
            (self.buy_order is not None and self.buy_order.order_id == order_id)
            or (self.sell_order is not None and self.sell_order.order_id == order_id)
        )

    def partially_filled(self) -> bool:
        """Check if exactly one side of the pair has been filled."""
        return self.filled_buy != self.filled_sell

    def get_unfilled_order(self) -> Optional[HangingOrder]:
        """Get the unfilled order from a partially filled pair."""
        if self.partially_filled():
            if not self.filled_buy:
                return self.buy_order
            else:
                return self.sell_order
        return None


class HangingOrdersTracker:
    """
    Tracks and manages hanging orders.

    A hanging order is an order that remains active after its paired order
    (on the opposite side) has been filled. For example, if you place a buy
    at 48 and a sell at 52, and the buy gets filled, the sell at 52 becomes
    a hanging order.

    The tracker:
    - Monitors pairs of orders for partial fills
    - Converts unfilled sides to hanging orders when the other side fills
    - Cancels hanging orders that are too far from current price
    - Renews hanging orders that exceed maximum age
    """

    def __init__(
        self,
        hanging_orders_cancel_pct: float = 10.0,
        max_order_age: float = 1800.0,
    ):
        """
        Initialize the hanging orders tracker.

        Args:
            hanging_orders_cancel_pct: Cancel hanging orders when price moves
                this percentage from mid price (default 10%)
            max_order_age: Maximum age in seconds before renewing (default 30 min)
        """
        self._hanging_orders_cancel_pct = hanging_orders_cancel_pct / 100.0
        self._max_order_age = max_order_age

        # Current hanging orders being tracked
        self._hanging_orders: Set[HangingOrder] = set()

        # Completed hanging orders (filled)
        self._completed_hanging_orders: Set[HangingOrder] = set()

        # Orders currently being renewed (cancelled for recreation)
        self._orders_being_renewed: Set[HangingOrder] = set()

        # Orders pending cancellation
        self._orders_being_cancelled: Set[str] = set()

        # Pairs of orders created together
        self._current_pairs: List[CreatedPairOfOrders] = []

    @property
    def hanging_orders_cancel_pct(self) -> float:
        """Get the cancel percentage (as decimal, e.g., 0.1 for 10%)."""
        return self._hanging_orders_cancel_pct

    @hanging_orders_cancel_pct.setter
    def hanging_orders_cancel_pct(self, value: float):
        """Set the cancel percentage (as decimal)."""
        self._hanging_orders_cancel_pct = value

    @property
    def hanging_orders(self) -> Set[HangingOrder]:
        """Get current hanging orders."""
        return self._hanging_orders.copy()

    def add_order_pair(
        self,
        buy_order: Optional[HangingOrder],
        sell_order: Optional[HangingOrder],
    ) -> None:
        """
        Register a pair of orders for tracking.

        Args:
            buy_order: The buy order in the pair (or None)
            sell_order: The sell order in the pair (or None)
        """
        pair = CreatedPairOfOrders(buy_order=buy_order, sell_order=sell_order)
        self._current_pairs.append(pair)

    def on_order_filled(self, order_id: str, is_buy: bool) -> None:
        """
        Handle an order fill event.

        Marks the order as filled in any pair it belongs to.
        If this creates a partial fill, the unfilled side becomes a hanging order.

        Args:
            order_id: The filled order ID
            is_buy: True if the filled order was a buy
        """
        # Check if this is a hanging order being filled
        hanging_order = next(
            (ho for ho in self._hanging_orders if ho.order_id == order_id),
            None
        )

        if hanging_order:
            self._on_hanging_order_filled(hanging_order)
            return

        # Check if this is part of a tracked pair
        for pair in self._current_pairs:
            if pair.contains_order(order_id):
                if is_buy:
                    pair.filled_buy = True
                else:
                    pair.filled_sell = True

                # If now partially filled, the unfilled side is a hanging order
                if pair.partially_filled():
                    unfilled = pair.get_unfilled_order()
                    if unfilled:
                        self._hanging_orders.add(unfilled)
                        logger.info(
                            f"Order {order_id} filled, opposite side "
                            f"{unfilled.order_id} is now a hanging order"
                        )

    def _on_hanging_order_filled(self, order: HangingOrder) -> None:
        """Handle a hanging order being filled."""
        self._completed_hanging_orders.add(order)
        self._hanging_orders.discard(order)
        order_side = "BUY" if order.is_buy else "SELL"
        logger.info(
            f"Hanging {order_side} order {order.order_id} "
            f"(market {order.query_id}, {'YES' if order.outcome else 'NO'}, "
            f"{order.amount} @ {order.price}¢) has been filled"
        )

    def on_order_cancelled(self, order_id: str) -> None:
        """
        Handle an order cancellation event.

        Args:
            order_id: The cancelled order ID
        """
        self._orders_being_cancelled.discard(order_id)

        # Check if it was a hanging order
        order_to_remove = next(
            (ho for ho in self._hanging_orders if ho.order_id == order_id),
            None
        )
        if order_to_remove:
            self._hanging_orders.discard(order_to_remove)
            logger.info(f"Hanging order {order_id} cancelled")

        # Check if it was being renewed
        renewing_order = next(
            (ho for ho in self._orders_being_renewed if ho.order_id == order_id),
            None
        )
        if renewing_order:
            self._orders_being_renewed.discard(renewing_order)
            self._hanging_orders.discard(renewing_order)
            logger.info(
                f"Hanging order {order_id} cancelled as part of renewal process"
            )

    def is_hanging_order(self, order_id: str) -> bool:
        """Check if an order ID is a hanging order."""
        return any(ho.order_id == order_id for ho in self._hanging_orders)

    def is_completed_hanging_order(self, order_id: str) -> bool:
        """Check if an order ID is a completed (filled) hanging order."""
        return any(ho.order_id == order_id for ho in self._completed_hanging_orders)

    def get_orders_to_cancel(
        self, mid_price: float, current_timestamp: float
    ) -> List[HangingOrder]:
        """
        Get hanging orders that should be cancelled.

        Orders are cancelled if:
        - They are too far from the current mid price (cancel_pct)
        - They have exceeded the maximum order age

        Args:
            mid_price: Current mid price in cents
            current_timestamp: Current Unix timestamp

        Returns:
            List of hanging orders to cancel
        """
        orders_to_cancel: List[HangingOrder] = []

        for order in self._hanging_orders:
            if order.order_id in self._orders_being_cancelled:
                continue

            # Check if too far from price
            price_distance = abs(order.price - mid_price) / mid_price
            if price_distance > self._hanging_orders_cancel_pct:
                logger.info(
                    f"Hanging order {order.order_id} is {price_distance:.1%} "
                    f"from mid price (max {self._hanging_orders_cancel_pct:.1%}), cancelling"
                )
                orders_to_cancel.append(order)
                continue

            # Check if too old
            order_age = current_timestamp - order.creation_timestamp
            if order_age > self._max_order_age:
                logger.info(
                    f"Hanging order {order.order_id} age {order_age:.0f}s "
                    f"exceeds max {self._max_order_age}s, cancelling for renewal"
                )
                orders_to_cancel.append(order)
                self._orders_being_renewed.add(order)

        return orders_to_cancel

    def get_orders_to_recreate(self) -> List[HangingOrder]:
        """
        Get orders that were cancelled for renewal and need to be recreated.

        Returns:
            List of hanging orders to recreate
        """
        orders = list(self._orders_being_renewed)
        self._orders_being_renewed.clear()
        return orders

    def mark_cancellation_pending(self, order_id: str) -> None:
        """Mark an order as having a pending cancellation."""
        self._orders_being_cancelled.add(order_id)

    def process_tick(
        self, mid_price: float, current_timestamp: float
    ) -> Tuple[List[HangingOrder], List[HangingOrder]]:
        """
        Process a tick and return orders to cancel and recreate.

        Should be called each tick to maintain hanging orders.

        Args:
            mid_price: Current mid price in cents
            current_timestamp: Current Unix timestamp

        Returns:
            Tuple of (orders_to_cancel, orders_to_recreate)
        """
        # First, promote any partially filled pairs to hanging orders
        self._promote_partial_fills()

        # Get orders to cancel
        to_cancel = self.get_orders_to_cancel(mid_price, current_timestamp)

        # Get orders to recreate (from previous cancellations)
        to_recreate = self.get_orders_to_recreate()

        return to_cancel, to_recreate

    def _promote_partial_fills(self) -> None:
        """Promote unfilled orders from partially filled pairs to hanging orders."""
        for pair in self._current_pairs:
            if pair.partially_filled():
                unfilled = pair.get_unfilled_order()
                if unfilled and unfilled not in self._hanging_orders:
                    self._hanging_orders.add(unfilled)
                    logger.debug(f"Promoted {unfilled.order_id} to hanging order")

        # Clear fully processed pairs
        self._current_pairs = [
            p for p in self._current_pairs
            if not (p.filled_buy and p.filled_sell)
        ]

    def clear(self) -> None:
        """Clear all tracked orders."""
        self._hanging_orders.clear()
        self._completed_hanging_orders.clear()
        self._orders_being_renewed.clear()
        self._orders_being_cancelled.clear()
        self._current_pairs.clear()

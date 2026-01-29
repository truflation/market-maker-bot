"""Tests for hanging orders tracker."""

import pytest
import time

from market_maker_bot.hanging_orders import (
    HangingOrder,
    CreatedPairOfOrders,
    HangingOrdersTracker,
)


class TestHangingOrder:
    """Tests for HangingOrder dataclass."""

    def test_creation(self):
        """Test HangingOrder creation."""
        order = HangingOrder(
            order_id="order_123",
            query_id=1,
            outcome=True,
            is_buy=True,
            price=48,
            amount=100,
            creation_timestamp=time.time(),
        )

        assert order.order_id == "order_123"
        assert order.query_id == 1
        assert order.outcome is True
        assert order.is_buy is True
        assert order.price == 48
        assert order.amount == 100

    def test_equality(self):
        """Test HangingOrder equality."""
        ts = time.time()
        order1 = HangingOrder("id1", 1, True, True, 48, 100, ts)
        order2 = HangingOrder("id1", 1, True, True, 48, 100, ts)
        order3 = HangingOrder("id2", 1, True, True, 48, 100, ts)

        assert order1 == order2
        assert order1 != order3

    def test_hashable(self):
        """Test that HangingOrder is hashable."""
        order = HangingOrder("id1", 1, True, True, 48, 100, time.time())
        order_set = {order}
        assert order in order_set


class TestCreatedPairOfOrders:
    """Tests for CreatedPairOfOrders."""

    def test_creation(self):
        """Test pair creation."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        pair = CreatedPairOfOrders(buy_order=buy, sell_order=sell)

        assert pair.buy_order == buy
        assert pair.sell_order == sell
        assert pair.filled_buy is False
        assert pair.filled_sell is False

    def test_contains_order(self):
        """Test checking if pair contains an order."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)
        pair = CreatedPairOfOrders(buy_order=buy, sell_order=sell)

        assert pair.contains_order("buy_1") is True
        assert pair.contains_order("sell_1") is True
        assert pair.contains_order("other") is False

    def test_partially_filled(self):
        """Test partial fill detection."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)
        pair = CreatedPairOfOrders(buy_order=buy, sell_order=sell)

        # Neither filled
        assert pair.partially_filled() is False

        # Buy filled only
        pair.filled_buy = True
        assert pair.partially_filled() is True

        # Both filled
        pair.filled_sell = True
        assert pair.partially_filled() is False

    def test_get_unfilled_order(self):
        """Test getting the unfilled order from a partial fill."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)
        pair = CreatedPairOfOrders(buy_order=buy, sell_order=sell)

        # Neither filled - no unfilled order
        assert pair.get_unfilled_order() is None

        # Buy filled - sell is unfilled
        pair.filled_buy = True
        assert pair.get_unfilled_order() == sell

        # Reset and fill sell instead
        pair.filled_buy = False
        pair.filled_sell = True
        assert pair.get_unfilled_order() == buy


class TestHangingOrdersTracker:
    """Tests for HangingOrdersTracker."""

    @pytest.fixture
    def tracker(self):
        """Create a tracker with default settings."""
        return HangingOrdersTracker(
            hanging_orders_cancel_pct=10.0,
            max_order_age=1800.0,
        )

    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.hanging_orders_cancel_pct == 0.10  # Converted to decimal
        assert len(tracker.hanging_orders) == 0

    def test_add_order_pair(self, tracker):
        """Test adding an order pair."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)

        # Pair should be tracked but not yet hanging
        assert len(tracker.hanging_orders) == 0

    def test_on_order_filled_creates_hanging_order(self, tracker):
        """Test that filling one side creates a hanging order."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)
        tracker.on_order_filled("buy_1", is_buy=True)

        # Sell should now be a hanging order
        assert len(tracker.hanging_orders) == 1
        assert sell in tracker.hanging_orders

    def test_on_order_cancelled(self, tracker):
        """Test order cancellation removes from hanging orders."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)
        tracker.on_order_filled("buy_1", is_buy=True)

        assert sell in tracker.hanging_orders

        tracker.on_order_cancelled("sell_1")
        assert sell not in tracker.hanging_orders

    def test_is_hanging_order(self, tracker):
        """Test checking if an order is hanging."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)
        tracker.on_order_filled("buy_1", is_buy=True)

        assert tracker.is_hanging_order("sell_1") is True
        assert tracker.is_hanging_order("buy_1") is False

    def test_get_orders_to_cancel_far_from_price(self, tracker):
        """Test detecting orders too far from price."""
        ts = time.time()
        # Order at 48 cents
        sell = HangingOrder("sell_1", 1, True, False, 48, 100, ts)
        tracker._hanging_orders.add(sell)

        # Mid price at 50 cents - sell is 4% away (within 10%)
        to_cancel = tracker.get_orders_to_cancel(50.0, ts)
        assert len(to_cancel) == 0

        # Mid price at 60 cents - sell is 20% away (exceeds 10%)
        to_cancel = tracker.get_orders_to_cancel(60.0, ts)
        assert len(to_cancel) == 1
        assert sell in to_cancel

    def test_get_orders_to_cancel_too_old(self, tracker):
        """Test detecting orders that are too old."""
        old_ts = time.time() - 2000  # More than 1800 seconds ago
        sell = HangingOrder("sell_1", 1, True, False, 50, 100, old_ts)
        tracker._hanging_orders.add(sell)

        current_ts = time.time()
        to_cancel = tracker.get_orders_to_cancel(50.0, current_ts)

        assert len(to_cancel) == 1
        assert sell in to_cancel

    def test_process_tick(self, tracker):
        """Test processing a tick."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)
        tracker.on_order_filled("buy_1", is_buy=True)

        to_cancel, to_recreate = tracker.process_tick(50.0, ts)

        # Nothing should be cancelled yet
        assert len(to_cancel) == 0
        assert len(to_recreate) == 0

    def test_clear(self, tracker):
        """Test clearing all tracked orders."""
        ts = time.time()
        buy = HangingOrder("buy_1", 1, True, True, 48, 100, ts)
        sell = HangingOrder("sell_1", 1, True, False, 52, 100, ts)

        tracker.add_order_pair(buy, sell)
        tracker.on_order_filled("buy_1", is_buy=True)

        assert len(tracker.hanging_orders) == 1

        tracker.clear()
        assert len(tracker.hanging_orders) == 0

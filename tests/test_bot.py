"""Tests for bot components and integration."""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal

from market_maker_bot.config import (
    BotConfig,
    MarketConfig,
    AvellanedaConfig,
    load_config_from_dict,
)
from market_maker_bot.models import (
    OutcomeMode,
    Side,
    OrderLevel,
    MarketState,
    BotOrder,
)
from market_maker_bot.market import (
    parse_order_book_entries,
    get_best_prices,
    build_market_state,
    MarketContext,
    OrderManager,
    convert_price_for_order,
)
from market_maker_bot.pricing.inventory import (
    MarketInventory,
    InventoryManager,
)
from market_maker_bot.indicators.depth import (
    OrderBookDepthAnalyzer,
    DepthAnalysis,
)


class TestOrderBookParsing:
    """Tests for order book parsing."""

    def test_parse_empty_entries(self):
        """Test parsing empty order book."""
        bids, asks = parse_order_book_entries([])
        assert bids == []
        assert asks == []

    def test_parse_bids_and_asks(self):
        """Test parsing bids and asks correctly."""
        entries = [
            {"price": -50, "amount": 100},  # Bid at 50
            {"price": -48, "amount": 50},   # Bid at 48
            {"price": 55, "amount": 75},    # Ask at 55
            {"price": 60, "amount": 25},    # Ask at 60
            {"price": 0, "amount": 200},    # Holding (ignored)
        ]

        bids, asks = parse_order_book_entries(entries)

        assert len(bids) == 2
        assert len(asks) == 2

        # Bids sorted highest to lowest (best first)
        assert abs(bids[0].price) == 50
        assert abs(bids[1].price) == 48

        # Asks sorted lowest to highest (best first)
        assert asks[0].price == 55
        assert asks[1].price == 60

    def test_get_best_prices(self):
        """Test extracting best bid/ask."""
        bids = [
            OrderLevel(price=-50, quantity=100),
            OrderLevel(price=-48, quantity=50),
        ]
        asks = [
            OrderLevel(price=55, quantity=75),
            OrderLevel(price=60, quantity=25),
        ]

        best_bid, best_ask = get_best_prices(bids, asks)

        assert best_bid == 50
        assert best_ask == 55

    def test_get_best_prices_no_bids(self):
        """Test with no bids."""
        asks = [OrderLevel(price=55, quantity=75)]

        best_bid, best_ask = get_best_prices([], asks)

        assert best_bid is None
        assert best_ask == 55

    def test_build_market_state(self):
        """Test building MarketState from entries."""
        entries = [
            {"price": -50, "amount": 100},
            {"price": 55, "amount": 75},
        ]

        state = build_market_state(
            query_id=1,
            outcome=True,
            order_book_entries=entries,
        )

        assert state.query_id == 1
        assert state.outcome is True
        assert state.best_bid == 50
        assert state.best_ask == 55
        assert state.mid_price == 52.5
        assert state.has_liquidity


class TestMarketInventory:
    """Tests for per-market inventory tracking."""

    def test_initial_inventory(self):
        """Test initial inventory state."""
        inv = MarketInventory(query_id=1)

        assert inv.yes_shares == 0
        assert inv.no_shares == 0
        assert inv.usd_locked_yes_bids == Decimal("0")

    def test_update_from_positions(self):
        """Test updating inventory from position data."""
        inv = MarketInventory(query_id=1)
        inv.update_from_positions(
            yes_shares=100,
            no_shares=50,
            yes_bid_collateral=Decimal("10.00"),
            no_bid_collateral=Decimal("5.00"),
        )

        assert inv.yes_shares == 100
        assert inv.no_shares == 50
        assert inv.usd_locked_yes_bids == Decimal("10.00")

    def test_get_share_value(self):
        """Test share value calculation."""
        inv = MarketInventory(query_id=1, yes_shares=100, no_shares=50)

        # YES shares at 60 cents
        yes_value = inv.get_share_value(outcome=True, mid_price=60.0)
        assert yes_value == Decimal("60.0")  # 100 * 60/100

        # NO shares at 40 cents
        no_value = inv.get_share_value(outcome=False, mid_price=40.0)
        assert no_value == Decimal("20.0")  # 50 * 40/100

    def test_inventory_ratio_neutral(self):
        """Test inventory ratio when at target."""
        inv = MarketInventory(query_id=1, target_pct=50.0)
        inv.update_from_positions(yes_shares=100, no_shares=0)

        # At 50 cents, 100 shares = $50 value
        # If that's 50% of total, we're at target
        # But with only YES shares and no locked USD, the ratio depends on total
        ratio = inv.get_inventory_ratio(outcome=True, mid_price=50.0)

        # Should be near 0 if at target, or positive if over target
        assert -1.0 <= ratio <= 1.0

    def test_inventory_ratio_long(self):
        """Test inventory ratio with excess shares."""
        inv = MarketInventory(query_id=1, target_pct=50.0)
        inv.update_from_positions(
            yes_shares=200,
            no_shares=0,
            yes_bid_collateral=Decimal("0"),
            no_bid_collateral=Decimal("0"),
        )

        # With only YES shares, we're over target
        ratio = inv.get_inventory_ratio(outcome=True, mid_price=50.0)
        assert ratio > 0  # Positive = excess inventory


class TestInventoryManager:
    """Tests for InventoryManager."""

    def test_get_market_inventory(self):
        """Test getting/creating market inventory."""
        manager = InventoryManager(target_pct=50.0)

        inv1 = manager.get_market_inventory(1)
        inv2 = manager.get_market_inventory(2)
        inv1_again = manager.get_market_inventory(1)

        assert inv1.query_id == 1
        assert inv2.query_id == 2
        assert inv1 is inv1_again  # Same instance

    def test_update_from_user_positions(self):
        """Test updating from position list."""
        manager = InventoryManager()

        positions = [
            {"query_id": 1, "outcome": True, "price": 0, "amount": 100},   # YES holding
            {"query_id": 1, "outcome": False, "price": 0, "amount": 50},   # NO holding
            {"query_id": 1, "outcome": True, "price": -45, "amount": 20},  # YES bid
            {"query_id": 2, "outcome": True, "price": 0, "amount": 75},    # Different market
        ]

        manager.update_from_user_positions(positions)

        inv1 = manager.get_market_inventory(1)
        inv2 = manager.get_market_inventory(2)

        assert inv1.yes_shares == 100
        assert inv1.no_shares == 50
        assert inv2.yes_shares == 75


class TestOrderBookDepthAnalyzer:
    """Tests for depth analysis."""

    def test_analyze_empty_book(self):
        """Test analyzing empty order book."""
        analyzer = OrderBookDepthAnalyzer(default_kappa=0.5)

        result = analyzer.analyze([], [])

        assert result.kappa == 0.5
        assert result.bid_depth == 0
        assert result.ask_depth == 0

    def test_analyze_with_levels(self):
        """Test analyzing order book with levels."""
        analyzer = OrderBookDepthAnalyzer(default_kappa=0.5)

        bids = [
            OrderLevel(price=-50, quantity=100),
            OrderLevel(price=-48, quantity=50),
        ]
        asks = [
            OrderLevel(price=52, quantity=75),
            OrderLevel(price=55, quantity=25),
        ]

        result = analyzer.analyze(bids, asks, mid_price=51.0)

        assert result.bid_depth == 150
        assert result.ask_depth == 100
        assert result.bid_levels == 2
        assert result.ask_levels == 2
        assert result.kappa > 0

    def test_kappa_clamped(self):
        """Test that kappa is clamped to valid range."""
        analyzer = OrderBookDepthAnalyzer(
            min_kappa=0.01,
            max_kappa=10.0,
        )

        # Very deep book (high kappa)
        bids = [OrderLevel(price=-50, quantity=10000)]
        asks = [OrderLevel(price=51, quantity=10000)]

        result = analyzer.analyze(bids, asks, mid_price=50.5)

        assert result.kappa <= 10.0


class TestMarketContext:
    """Tests for MarketContext."""

    def test_market_context_properties(self):
        """Test MarketContext basic properties."""
        config = MarketConfig(
            query_id=1,
            stream_id="test_stream",
            data_provider="0x123",
            name="Test Market",
            outcome_mode=OutcomeMode.YES_ONLY,
        )

        context = MarketContext(config=config)

        assert context.query_id == 1
        assert context.yes_state is None
        assert context.no_state is None

    def test_get_set_state(self):
        """Test getting and setting state."""
        config = MarketConfig(
            query_id=1,
            stream_id="test",
            data_provider="0x123",
        )
        context = MarketContext(config=config)

        yes_state = MarketState(
            query_id=1,
            outcome=True,
            best_bid=50,
            best_ask=55,
        )

        context.set_state(True, yes_state)

        assert context.get_state(True) is yes_state
        assert context.get_state(False) is None


class TestOrderManager:
    """Tests for OrderManager."""

    def test_should_update_no_existing(self):
        """Test update check with no existing order."""
        config = MarketConfig(query_id=1, stream_id="test", data_provider="0x123")
        context = MarketContext(config=config)
        manager = OrderManager(context)

        should, reason = manager.should_update_order(True, Side.BID, 50)

        assert should is True
        assert reason == "no_existing_order"

    def test_should_update_price_deviation(self):
        """Test update triggered by price deviation."""
        config = MarketConfig(query_id=1, stream_id="test", data_provider="0x123")
        context = MarketContext(config=config)
        manager = OrderManager(context, refresh_tolerance_pct=1.0)

        # Record existing order
        manager.record_order(True, Side.BID, 50, 100, "0x123")

        # Small price change - no update
        should, reason = manager.should_update_order(True, Side.BID, 50)
        assert should is False
        assert "tolerance" in reason

        # Large price change - trigger update
        should, reason = manager.should_update_order(True, Side.BID, 55)
        assert should is True
        assert "deviation" in reason


class TestPriceConversion:
    """Tests for price conversion utilities."""

    def test_convert_bid_price(self):
        """Test converting bid price to SDK format."""
        assert convert_price_for_order(50, Side.BID) == -50
        assert convert_price_for_order(1, Side.BID) == -1
        assert convert_price_for_order(99, Side.BID) == -99

    def test_convert_ask_price(self):
        """Test converting ask price to SDK format."""
        assert convert_price_for_order(50, Side.ASK) == 50
        assert convert_price_for_order(1, Side.ASK) == 1
        assert convert_price_for_order(99, Side.ASK) == 99

    def test_invalid_price_raises(self):
        """Test that invalid prices raise ValueError."""
        with pytest.raises(ValueError):
            convert_price_for_order(0, Side.BID)

        with pytest.raises(ValueError):
            convert_price_for_order(100, Side.ASK)


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_from_dict(self):
        """Test loading config from dictionary."""
        data = {
            "node_url": "http://test:8484",
            "private_key": "test_key",
            "markets": [
                {
                    "query_id": 1,
                    "stream_id": "test_stream",
                    "data_provider": "0x123",
                    "name": "Test Market",
                    "outcome_mode": "both",
                    "order_amount": 50,
                }
            ],
            "avellaneda": {
                "risk_factor": 0.2,
                "min_spread": 3.0,
            },
            "dry_run": True,
        }

        config = load_config_from_dict(data)

        assert config.node_url == "http://test:8484"
        assert config.private_key == "test_key"
        assert len(config.markets) == 1
        assert config.markets[0].query_id == 1
        assert config.markets[0].outcome_mode == OutcomeMode.BOTH
        assert config.markets[0].order_amount == 50
        assert config.avellaneda.risk_factor == 0.2
        assert config.avellaneda.min_spread == 3.0
        assert config.dry_run is True

    def test_config_defaults(self):
        """Test config default values."""
        data = {
            "markets": [
                {
                    "query_id": 1,
                    "stream_id": "test",
                    "data_provider": "0x123",
                }
            ]
        }

        config = load_config_from_dict(data)

        assert config.node_url == "http://localhost:8484"
        assert config.avellaneda.risk_factor == 1.0  # Default
        assert config.markets[0].outcome_mode == OutcomeMode.YES_ONLY  # Default

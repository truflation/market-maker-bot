"""Tests for Avellaneda-Stoikov pricing model."""

import pytest
import math

from market_maker_bot.pricing.avellaneda import (
    AvellanedaPricing,
    AvellanedaParams,
)
from market_maker_bot.config import AvellanedaConfig
from market_maker_bot.models import PricingResult


class TestAvellanedaPricing:
    """Tests for the Avellaneda-Stoikov pricing model."""

    @pytest.fixture
    def pricing(self):
        """Create a pricing instance with default config."""
        config = AvellanedaConfig(
            risk_factor=0.1,
            min_spread=2.0,
            max_spread=20.0,
            min_volatility=1.0,
            time_horizon=1.0,
        )
        return AvellanedaPricing(config)

    @pytest.fixture
    def default_params(self):
        """Create default parameters for testing."""
        return AvellanedaParams(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=5.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
            min_spread=2.0,
        )

    def test_reservation_price_neutral_inventory(self, pricing):
        """Reservation price equals mid price with zero inventory."""
        r = pricing.calculate_reservation_price(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=5.0,
            gamma=0.1,
            time_horizon=1.0,
        )

        assert r == 50.0

    def test_reservation_price_long_inventory(self, pricing):
        """Reservation price lower than mid with excess inventory."""
        r = pricing.calculate_reservation_price(
            mid_price=50.0,
            inventory_skew=0.5,  # Long inventory
            volatility=5.0,
            gamma=0.1,
            time_horizon=1.0,
        )

        # r = 50 - 0.5 * 0.1 * 5 * 1 = 50 - 0.25 = 49.75
        assert r < 50.0
        assert abs(r - 49.75) < 0.01

    def test_reservation_price_short_inventory(self, pricing):
        """Reservation price higher than mid with deficit inventory."""
        r = pricing.calculate_reservation_price(
            mid_price=50.0,
            inventory_skew=-0.5,  # Short inventory
            volatility=5.0,
            gamma=0.1,
            time_horizon=1.0,
        )

        # r = 50 - (-0.5) * 0.1 * 5 * 1 = 50 + 0.25 = 50.25
        assert r > 50.0
        assert abs(r - 50.25) < 0.01

    def test_optimal_spread_basic(self, pricing):
        """Test optimal spread calculation."""
        spread = pricing.calculate_optimal_spread(
            volatility=5.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
        )

        # δ = γ × σ × T + (2/γ) × ln(1 + γ/κ)
        # δ = 0.1 * 5 * 1 + (2/0.1) * ln(1 + 0.1/0.5)
        # δ = 0.5 + 20 * ln(1.2)
        # δ = 0.5 + 20 * 0.182 ≈ 4.14
        expected = 0.1 * 5 * 1 + (2 / 0.1) * math.log(1 + 0.1 / 0.5)
        assert abs(spread - expected) < 0.01

    def test_spread_increases_with_volatility(self, pricing):
        """Higher volatility should increase spread."""
        low_vol = pricing.calculate_optimal_spread(
            volatility=2.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
        )

        high_vol = pricing.calculate_optimal_spread(
            volatility=10.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
        )

        assert high_vol > low_vol

    def test_spread_increases_with_gamma(self, pricing):
        """Higher risk aversion (gamma) should increase spread."""
        low_gamma = pricing.calculate_optimal_spread(
            volatility=5.0,
            kappa=0.5,
            gamma=0.05,
            time_horizon=1.0,
        )

        high_gamma = pricing.calculate_optimal_spread(
            volatility=5.0,
            kappa=0.5,
            gamma=0.2,
            time_horizon=1.0,
        )

        # Higher gamma increases volatility component but may decrease
        # the depth component, so test the overall spread
        # For most reasonable parameters, higher gamma = higher spread
        assert high_gamma > low_gamma

    def test_spread_decreases_with_kappa(self, pricing):
        """Higher liquidity (kappa) should decrease spread."""
        low_kappa = pricing.calculate_optimal_spread(
            volatility=5.0,
            kappa=0.1,  # Low liquidity
            gamma=0.1,
            time_horizon=1.0,
        )

        high_kappa = pricing.calculate_optimal_spread(
            volatility=5.0,
            kappa=2.0,  # High liquidity
            gamma=0.1,
            time_horizon=1.0,
        )

        assert high_kappa < low_kappa

    def test_calculate_prices_neutral(self, pricing, default_params):
        """Test full price calculation with neutral inventory."""
        result = pricing.calculate_prices(default_params)

        assert isinstance(result, PricingResult)
        assert result.mid_price == 50.0
        assert result.inventory_skew == 0.0

        # With zero inventory, bid and ask should be symmetric around mid
        mid = (result.bid_price + result.ask_price) / 2
        assert abs(mid - 50.0) < 1.0  # Allow some rounding

    def test_calculate_prices_long_inventory(self, pricing):
        """Test prices with long inventory bias (encourage selling)."""
        params = AvellanedaParams(
            mid_price=50.0,
            inventory_skew=0.8,  # Strong long bias
            volatility=5.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
            min_spread=2.0,
        )

        result = pricing.calculate_prices(params)

        # With long inventory, reservation price is lower
        # This means both bid and ask shift down (encourage selling)
        assert result.reservation_price < 50.0
        assert result.bid_price < 50.0
        assert result.ask_price < 50.0 + result.optimal_spread / 2

    def test_calculate_prices_short_inventory(self, pricing):
        """Test prices with short inventory bias (encourage buying)."""
        params = AvellanedaParams(
            mid_price=50.0,
            inventory_skew=-0.8,  # Strong short bias
            volatility=5.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
            min_spread=2.0,
        )

        result = pricing.calculate_prices(params)

        # With short inventory, reservation price is higher
        # This means both bid and ask shift up (encourage buying)
        assert result.reservation_price > 50.0

    def test_prices_clamped_to_valid_range(self, pricing):
        """Prices should be clamped to 1-99 range."""
        # Test at lower bound
        params_low = AvellanedaParams(
            mid_price=3.0,
            inventory_skew=0.5,
            volatility=10.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
            min_spread=2.0,
        )

        result_low = pricing.calculate_prices(params_low)
        assert result_low.bid_price >= 1.0
        assert result_low.ask_price >= 1.0

        # Test at upper bound
        params_high = AvellanedaParams(
            mid_price=97.0,
            inventory_skew=-0.5,
            volatility=10.0,
            kappa=0.5,
            gamma=0.1,
            time_horizon=1.0,
            min_spread=2.0,
        )

        result_high = pricing.calculate_prices(params_high)
        assert result_high.bid_price <= 99.0
        assert result_high.ask_price <= 99.0

    def test_bid_always_less_than_ask(self, pricing):
        """Bid price should always be less than ask price."""
        test_cases = [
            (50.0, 0.0),   # Neutral
            (50.0, 0.8),   # Long
            (50.0, -0.8),  # Short
            (5.0, 0.0),    # Low price
            (95.0, 0.0),   # High price
        ]

        for mid_price, inventory_skew in test_cases:
            params = AvellanedaParams(
                mid_price=mid_price,
                inventory_skew=inventory_skew,
                volatility=5.0,
                kappa=0.5,
                gamma=0.1,
                time_horizon=1.0,
                min_spread=2.0,
            )

            result = pricing.calculate_prices(params)
            assert result.bid_price < result.ask_price, (
                f"Failed for mid={mid_price}, q={inventory_skew}"
            )

    def test_min_spread_enforced(self, pricing):
        """Minimum spread should be enforced.

        min_spread is a percentage of mid price (Hummingbot style).
        With mid_price=50 and min_spread=8.0 (8%), the minimum spread is 4.0 cents.
        """
        params = AvellanedaParams(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=0.5,  # Very low volatility
            kappa=10.0,      # Very high liquidity
            gamma=0.01,      # Low risk aversion
            time_horizon=1.0,
            min_spread=8.0,  # 8% of mid price = 4.0 cents minimum spread
        )

        result = pricing.calculate_prices(params)
        actual_spread = result.ask_price - result.bid_price

        # 8% of 50 = 4.0 cents minimum spread
        assert actual_spread >= 4.0

    def test_to_int_prices(self, pricing, default_params):
        """Test conversion to integer prices."""
        result = pricing.calculate_prices(default_params)
        bid_int, ask_int = result.to_int_prices()

        assert isinstance(bid_int, int)
        assert isinstance(ask_int, int)
        assert 1 <= bid_int <= 99
        assert 1 <= ask_int <= 99
        assert bid_int < ask_int

    def test_calculate_from_config(self, pricing):
        """Test the convenience method using stored config."""
        result = pricing.calculate_from_config(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=5.0,
        )

        assert isinstance(result, PricingResult)
        assert result.mid_price == 50.0

    def test_max_spread_constraint(self, pricing):
        """Test that max spread constraint is applied."""
        # Use parameters that would generate very wide spread
        result = pricing.calculate_from_config(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=50.0,  # Very high volatility
            kappa=0.01,       # Very low liquidity
        )

        actual_spread = result.ask_price - result.bid_price
        assert actual_spread <= pricing.config.max_spread

    def test_gamma_override(self, pricing):
        """Test gamma override in calculate_from_config."""
        result_default = pricing.calculate_from_config(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=5.0,
        )

        result_high_gamma = pricing.calculate_from_config(
            mid_price=50.0,
            inventory_skew=0.0,
            volatility=5.0,
            gamma_override=0.5,  # Higher than default
        )

        # Higher gamma should result in wider spread
        spread_default = result_default.ask_price - result_default.bid_price
        spread_high = result_high_gamma.ask_price - result_high_gamma.bid_price

        assert spread_high > spread_default


class TestInvalidInputs:
    """Tests for handling invalid inputs."""

    @pytest.fixture
    def pricing(self):
        """Create a pricing instance."""
        return AvellanedaPricing(AvellanedaConfig())

    def test_zero_gamma_raises(self, pricing):
        """Zero gamma should raise ValueError."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            pricing.calculate_optimal_spread(
                volatility=5.0,
                kappa=0.5,
                gamma=0.0,
                time_horizon=1.0,
            )

    def test_negative_gamma_raises(self, pricing):
        """Negative gamma should raise ValueError."""
        with pytest.raises(ValueError, match="gamma must be positive"):
            pricing.calculate_optimal_spread(
                volatility=5.0,
                kappa=0.5,
                gamma=-0.1,
                time_horizon=1.0,
            )

    def test_zero_kappa_raises(self, pricing):
        """Zero kappa should raise ValueError."""
        with pytest.raises(ValueError, match="kappa must be positive"):
            pricing.calculate_optimal_spread(
                volatility=5.0,
                kappa=0.0,
                gamma=0.1,
                time_horizon=1.0,
            )

    def test_negative_kappa_raises(self, pricing):
        """Negative kappa should raise ValueError."""
        with pytest.raises(ValueError, match="kappa must be positive"):
            pricing.calculate_optimal_spread(
                volatility=5.0,
                kappa=-0.5,
                gamma=0.1,
                time_horizon=1.0,
            )

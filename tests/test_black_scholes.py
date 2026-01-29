"""Tests for Black-Scholes binary option pricing."""

import math
import pytest

from market_maker_bot.pricing.black_scholes import (
    price_binary_option,
    normal_cdf,
    normal_pdf,
    fair_value_to_cents,
    BinaryOptionPrice,
)


class TestNormalDistribution:
    """Tests for normal distribution functions."""

    def test_normal_cdf_at_zero(self):
        """CDF at 0 should be 0.5."""
        assert abs(normal_cdf(0) - 0.5) < 1e-10

    def test_normal_cdf_symmetry(self):
        """CDF should be symmetric: N(-x) = 1 - N(x)."""
        for x in [0.5, 1.0, 2.0, 3.0]:
            assert abs(normal_cdf(-x) - (1 - normal_cdf(x))) < 1e-10

    def test_normal_cdf_extreme_values(self):
        """CDF approaches 0 and 1 at extremes."""
        assert normal_cdf(-5) < 0.001
        assert normal_cdf(5) > 0.999

    def test_normal_pdf_at_zero(self):
        """PDF at 0 should be 1/sqrt(2*pi) ≈ 0.3989."""
        expected = 1 / math.sqrt(2 * math.pi)
        assert abs(normal_pdf(0) - expected) < 1e-10

    def test_normal_pdf_symmetry(self):
        """PDF should be symmetric: f(-x) = f(x)."""
        for x in [0.5, 1.0, 2.0]:
            assert abs(normal_pdf(-x) - normal_pdf(x)) < 1e-10

    def test_normal_pdf_always_positive(self):
        """PDF should always be positive."""
        for x in [-5, -2, -1, 0, 1, 2, 5]:
            assert normal_pdf(x) > 0


class TestBinaryOptionPricing:
    """Tests for binary option pricing."""

    def test_at_the_money_option(self):
        """ATM option should be close to 0.5 (adjusted for drift)."""
        result = price_binary_option(
            spot=100.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
            risk_free_rate=0.05,
        )

        # ATM binary option fair value should be around 0.5
        # but slightly adjusted for risk-free rate drift
        assert 0.45 < result.fair_value < 0.60
        assert isinstance(result, BinaryOptionPrice)

    def test_in_the_money_option(self):
        """ITM option (spot > strike) should have fair_value > 0.5."""
        result = price_binary_option(
            spot=110.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value > 0.5

    def test_out_of_the_money_option(self):
        """OTM option (spot < strike) should have fair_value < 0.5."""
        result = price_binary_option(
            spot=90.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value < 0.5

    def test_deep_itm_option(self):
        """Deep ITM option should be close to 1."""
        result = price_binary_option(
            spot=150.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value > 0.95

    def test_deep_otm_option(self):
        """Deep OTM option should be close to 0."""
        result = price_binary_option(
            spot=50.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value < 0.05

    def test_fair_value_clamped(self):
        """Fair value should be clamped to [0.001, 0.999]."""
        # Deep ITM
        result_itm = price_binary_option(
            spot=1000.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )
        assert result_itm.fair_value <= 0.999

        # Deep OTM
        result_otm = price_binary_option(
            spot=10.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )
        assert result_otm.fair_value >= 0.001

    def test_higher_volatility_spreads_probability(self):
        """Higher volatility should push fair value toward 0.5."""
        low_vol = price_binary_option(
            spot=110.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.10,
        )

        high_vol = price_binary_option(
            spot=110.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.50,
        )

        # ITM option: higher vol should lower fair value (closer to 0.5)
        assert high_vol.fair_value < low_vol.fair_value

    def test_longer_time_increases_uncertainty(self):
        """Longer time should push ITM options toward 0.5."""
        short_time = price_binary_option(
            spot=110.0,
            strike=100.0,
            time_years=0.1,
            volatility=0.20,
        )

        long_time = price_binary_option(
            spot=110.0,
            strike=100.0,
            time_years=1.0,
            volatility=0.20,
        )

        # More time = more uncertainty for ITM options
        assert long_time.fair_value < short_time.fair_value

    def test_zero_spot_returns_minimum(self):
        """Zero spot should return minimum fair value."""
        result = price_binary_option(
            spot=0.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value == 0.001

    def test_zero_strike_returns_maximum(self):
        """Zero strike should return maximum fair value."""
        result = price_binary_option(
            spot=100.0,
            strike=0.0,
            time_years=0.25,
            volatility=0.20,
        )

        assert result.fair_value == 0.999

    def test_minimum_volatility_applied(self):
        """Very low volatility should be floored to 1%."""
        result = price_binary_option(
            spot=100.0,
            strike=100.0,
            time_years=0.25,
            volatility=0.001,  # 0.1% - should be floored to 1%
        )

        # Should not crash and return valid result
        assert 0.001 <= result.fair_value <= 0.999


class TestFairValueToCents:
    """Tests for fair_value_to_cents conversion."""

    def test_conversion_middle_value(self):
        """50% probability should map to 50 cents."""
        assert fair_value_to_cents(0.50) == 50

    def test_conversion_high_value(self):
        """High probability should map to high cents."""
        assert fair_value_to_cents(0.95) == 95

    def test_conversion_low_value(self):
        """Low probability should map to low cents."""
        assert fair_value_to_cents(0.05) == 5

    def test_conversion_clamped_min(self):
        """Values below 0.01 should clamp to 1 cent."""
        assert fair_value_to_cents(0.001) == 1
        assert fair_value_to_cents(0.0) == 1

    def test_conversion_clamped_max(self):
        """Values above 0.99 should clamp to 99 cents."""
        assert fair_value_to_cents(0.999) == 99
        assert fair_value_to_cents(1.0) == 99

    def test_conversion_rounds_correctly(self):
        """Conversion should round to nearest cent."""
        assert fair_value_to_cents(0.554) == 55
        assert fair_value_to_cents(0.555) == 56
        assert fair_value_to_cents(0.556) == 56

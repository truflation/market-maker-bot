"""
Black-Scholes pricing for binary (digital) options.

Used for initial market pricing when no order book data exists.
Ported from market_generator/utils.py.
"""

import math
from dataclasses import dataclass


@dataclass
class BinaryOptionPrice:
    """Result of Black-Scholes binary option pricing."""
    fair_value: float  # Probability that S > K at expiry (0-1)
    delta: float
    gamma: float
    d1: float
    d2: float


def normal_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.
    Uses the error function for calculation.

    Args:
        x: The z-score value

    Returns:
        Probability P(Z <= x) where Z ~ N(0,1)
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    Args:
        x: The z-score value

    Returns:
        Density at x for standard normal distribution
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def price_binary_option(
    spot: float,
    strike: float,
    time_years: float,
    volatility: float,
    risk_free_rate: float = 0.05,
) -> BinaryOptionPrice:
    """
    Black-Scholes pricing for binary (digital) options.
    Returns fair value which equals implied probability that S > K at expiry.

    This is used to price new markets when there's no order book history.
    The fair_value can be directly mapped to a price in cents (multiply by 100).

    Args:
        spot: Current price/value of underlying
        strike: Strike/boundary price
        time_years: Time to expiry in years
        volatility: Annualized volatility (as decimal, e.g., 0.15 for 15%)
        risk_free_rate: Risk-free interest rate (default 5%)

    Returns:
        BinaryOptionPrice with fair_value (probability), greeks, and d1/d2

    Example:
        >>> result = price_binary_option(
        ...     spot=100.0,
        ...     strike=105.0,
        ...     time_years=0.25,  # 3 months
        ...     volatility=0.20,  # 20% annual vol
        ... )
        >>> price_cents = int(result.fair_value * 100)  # e.g., 42 cents
    """
    # Ensure minimum values for numerical stability
    T = max(time_years, 1.0 / 8760.0)  # Min 1 hour
    sigma = max(volatility, 0.01)  # Min 1%
    sqrt_T = math.sqrt(T)
    discount = math.exp(-risk_free_rate * T)

    # Handle edge cases
    if spot <= 0:
        return BinaryOptionPrice(
            fair_value=0.001,
            delta=0.0,
            gamma=0.0,
            d1=float("-inf"),
            d2=float("-inf"),
        )

    if strike <= 0:
        return BinaryOptionPrice(
            fair_value=0.999,
            delta=0.0,
            gamma=0.0,
            d1=float("inf"),
            d2=float("inf"),
        )

    # Calculate d2 (used for binary option pricing)
    # d2 = (ln(S/K) + (r - 0.5*σ²)*T) / (σ*√T)
    d2 = (math.log(spot / strike) + (risk_free_rate - 0.5 * sigma * sigma) * T) / (
        sigma * sqrt_T
    )
    d1 = d2 + sigma * sqrt_T

    # Binary option fair value = discounted probability P(S > K)
    pdf_d2 = normal_pdf(d2)
    fair_value = discount * normal_cdf(d2)

    # Clamp to valid probability range
    fair_value = max(0.001, min(0.999, fair_value))

    # Calculate Greeks
    if spot > 0 and sigma > 0 and sqrt_T > 0:
        delta = discount * pdf_d2 / (spot * sigma * sqrt_T)
        gamma = -discount * pdf_d2 * d1 / (spot * spot * sigma * sigma * T)
    else:
        delta = 0.0
        gamma = 0.0

    return BinaryOptionPrice(
        fair_value=fair_value,
        delta=delta,
        gamma=gamma,
        d1=d1,
        d2=d2,
    )


def fair_value_to_cents(fair_value: float) -> int:
    """
    Convert Black-Scholes fair value to price in cents.

    Args:
        fair_value: Probability (0-1)

    Returns:
        Price in cents (1-99)
    """
    return max(1, min(99, int(round(fair_value * 100))))

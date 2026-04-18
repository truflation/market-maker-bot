"""
Avellaneda-Stoikov pricing model adapted for binary options.

Implements the market making strategy from:
"High-frequency trading in a limit order book" by Avellaneda & Stoikov (2008)

Adapted for TrufNetwork prediction markets with prices in 1-99 cents range.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional

from ..models import PricingResult
from ..config import AvellanedaConfig

logger = logging.getLogger(__name__)


@dataclass
class AvellanedaParams:
    """Parameters for a single A-S calculation."""
    mid_price: float  # Current mid price in cents
    inventory_skew: float  # q: normalized inventory deviation (-1 to +1)
    volatility: float  # σ: volatility in cents (absolute, not percentage)
    kappa: float  # κ: order book depth factor
    gamma: float  # γ: risk aversion factor
    time_horizon: float  # T: time fraction (1.0 for infinite)
    min_spread: float  # Minimum spread in cents


class AvellanedaPricing:
    """
    Avellaneda-Stoikov market making pricing model.

    Core formulas (adapted for binary options):

    Reservation Price:
        r = mid_price - q × γ × σ × T

    Optimal Spread:
        δ = γ × σ × T + (2/γ) × ln(1 + γ/κ)

    Optimal Bid:
        bid = r - δ/2

    Optimal Ask:
        ask = r + δ/2

    Where:
        - q = inventory deviation from target (normalized -1 to +1)
        - γ = risk aversion factor
        - σ = volatility (in cents, absolute)
        - κ = order book depth factor
        - T = time fraction (1.0 for infinite timeframe)

    Binary Options Adaptations:
        - All prices clamped to 1-99 cents range
        - Volatility is absolute (cents), not percentage
        - Prices rounded to integers for on-chain compatibility
    """

    def __init__(self, config: Optional[AvellanedaConfig] = None):
        """
        Initialize pricing model.

        Args:
            config: Avellaneda strategy configuration
        """
        self.config = config or AvellanedaConfig()

    def calculate_reservation_price(
        self,
        mid_price: float,
        inventory_skew: float,
        volatility: float,
        gamma: float,
        time_horizon: float,
    ) -> float:
        """
        Calculate reservation price.

        The reservation price adjusts based on inventory:
        - Excess inventory (q > 0) → lower reservation price (encourage selling)
        - Deficit inventory (q < 0) → higher reservation price (encourage buying)

        Formula: r = mid_price - q × γ × σ × T

        Args:
            mid_price: Current mid price in cents
            inventory_skew: q value (-1 to +1)
            volatility: σ in cents
            gamma: γ risk aversion
            time_horizon: T time fraction

        Returns:
            Reservation price in cents
        """
        adjustment = inventory_skew * gamma * volatility * time_horizon
        return mid_price - adjustment

    def calculate_optimal_spread(
        self,
        volatility: float,
        kappa: float,
        gamma: float,
        time_horizon: float,
    ) -> float:
        """
        Calculate optimal spread.

        Formula: δ = γ × σ × T + (2/γ) × ln(1 + γ/κ)

        Args:
            volatility: σ in cents
            kappa: κ depth factor
            gamma: γ risk aversion
            time_horizon: T time fraction

        Returns:
            Optimal spread in cents
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if kappa <= 0:
            raise ValueError("kappa must be positive")

        # First term: volatility-based spread
        vol_spread = gamma * volatility * time_horizon

        # Second term: depth-based spread
        depth_spread = (2.0 / gamma) * math.log(1.0 + gamma / kappa)

        return vol_spread + depth_spread

    def calculate_prices(self, params: AvellanedaParams) -> PricingResult:
        """
        Calculate optimal bid and ask prices.

        Args:
            params: All parameters for the calculation

        Returns:
            PricingResult with reservation price, spread, and optimal prices
        """
        # Calculate reservation price
        reservation_price = self.calculate_reservation_price(
            mid_price=params.mid_price,
            inventory_skew=params.inventory_skew,
            volatility=params.volatility,
            gamma=params.gamma,
            time_horizon=params.time_horizon,
        )

        # Calculate optimal spread
        optimal_spread = self.calculate_optimal_spread(
            volatility=params.volatility,
            kappa=params.kappa,
            gamma=params.gamma,
            time_horizon=params.time_horizon,
        )

        # Calculate raw optimal bid and ask from reservation price
        half_spread = optimal_spread / 2
        raw_bid = reservation_price - half_spread
        raw_ask = reservation_price + half_spread

        # Apply minimum spread constraint (Hummingbot style)
        # min_spread is a percentage of mid price, creating limits around mid
        min_spread_distance = params.mid_price * (params.min_spread / 100) if params.min_spread > 0 else 0
        max_limit_bid = params.mid_price - min_spread_distance / 2
        min_limit_ask = params.mid_price + min_spread_distance / 2

        # Optimal ask = max(raw_ask, min_limit_ask) - ensures ask is at least min distance above mid
        # Optimal bid = min(raw_bid, max_limit_bid) - ensures bid is at least min distance below mid
        bid_price = min(raw_bid, max_limit_bid) if min_spread_distance > 0 else raw_bid
        ask_price = max(raw_ask, min_limit_ask) if min_spread_distance > 0 else raw_ask

        # Clamp to valid range [1, 99]
        bid_price = max(1.0, min(99.0, bid_price))
        ask_price = max(1.0, min(99.0, ask_price))

        # Ensure bid < ask (can happen at extremes)
        if bid_price >= ask_price:
            # Center around mid price with minimum spread
            effective_spread = max(2.0, min_spread_distance)  # At least 2 cents
            bid_price = max(1.0, params.mid_price - effective_spread / 2)
            ask_price = min(99.0, params.mid_price + effective_spread / 2)

        logger.debug(
            f"A-S pricing: mid={params.mid_price:.1f} q={params.inventory_skew:.2f} "
            f"σ={params.volatility:.2f} κ={params.kappa:.2f} γ={params.gamma:.2f} "
            f"→ r={reservation_price:.1f} δ={optimal_spread:.1f} "
            f"bid={bid_price:.1f} ask={ask_price:.1f}"
        )

        return PricingResult(
            reservation_price=reservation_price,
            optimal_spread=optimal_spread,
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=params.mid_price,
            inventory_skew=params.inventory_skew,
            volatility=params.volatility,
            kappa=params.kappa,
        )

    def calculate_from_config(
        self,
        mid_price: float,
        inventory_skew: float,
        volatility: float,
        kappa: Optional[float] = None,
        gamma_override: Optional[float] = None,
        min_spread_override: Optional[float] = None,
        time_horizon_override: Optional[float] = None,
    ) -> PricingResult:
        """
        Calculate prices using stored configuration.

        Args:
            mid_price: Current mid price in cents
            inventory_skew: q value (-1 to +1)
            volatility: σ in cents
            kappa: Optional kappa override (uses config default if None)
            gamma_override: Optional gamma override
            min_spread_override: Optional min spread override
            time_horizon_override: Optional time horizon override (e.g., from settle_time)

        Returns:
            PricingResult
        """
        params = AvellanedaParams(
            mid_price=mid_price,
            inventory_skew=inventory_skew,
            volatility=max(volatility, self.config.min_volatility),
            kappa=kappa if kappa is not None else 1.0,  # Default kappa if not provided
            gamma=gamma_override if gamma_override is not None else self.config.risk_factor,
            time_horizon=time_horizon_override if time_horizon_override is not None else self.config.time_horizon,
            min_spread=(
                min_spread_override
                if min_spread_override is not None
                else self.config.min_spread
            ),
        )

        result = self.calculate_prices(params)

        # Apply max spread constraint
        actual_spread = result.ask_price - result.bid_price
        if actual_spread > self.config.max_spread:
            excess = (actual_spread - self.config.max_spread) / 2
            result = PricingResult(
                reservation_price=result.reservation_price,
                optimal_spread=result.optimal_spread,
                bid_price=result.bid_price + excess,
                ask_price=result.ask_price - excess,
                mid_price=result.mid_price,
                inventory_skew=result.inventory_skew,
                volatility=result.volatility,
                kappa=result.kappa,
            )

        return result

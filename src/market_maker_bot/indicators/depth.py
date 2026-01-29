"""
Order book depth analysis for kappa estimation.

Kappa (κ) represents order book depth/liquidity factor in the
Avellaneda-Stoikov model. Higher kappa = more liquid market.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from ..models import OrderLevel

logger = logging.getLogger(__name__)


@dataclass
class DepthAnalysis:
    """Result of order book depth analysis."""
    kappa: float
    bid_depth: float  # Total volume on bid side
    ask_depth: float  # Total volume on ask side
    bid_levels: int
    ask_levels: int
    price_range: float


class OrderBookDepthAnalyzer:
    """
    Analyzes order book depth to estimate kappa for A-S pricing.

    Kappa represents the market depth factor:
    - Higher κ = more liquid market = tighter spreads justified
    - Lower κ = less liquid = wider spreads needed

    Simplified estimation:
    κ = total_volume / (price_range * mid_price)

    This normalizes volume by price range to get a depth "density" metric.
    """

    def __init__(
        self,
        default_kappa: float = 0.5,
        max_levels: int = 5,
        min_kappa: float = 0.01,
        max_kappa: float = 10.0,
    ):
        """
        Initialize analyzer.

        Args:
            default_kappa: Default value when analysis not possible
            max_levels: Maximum depth levels to consider
            min_kappa: Minimum kappa floor
            max_kappa: Maximum kappa ceiling
        """
        self._default_kappa = default_kappa
        self._max_levels = max_levels
        self._min_kappa = min_kappa
        self._max_kappa = max_kappa

    def analyze(
        self,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        mid_price: Optional[float] = None,
    ) -> DepthAnalysis:
        """
        Analyze order book depth and estimate kappa.

        Args:
            bid_levels: Bid levels (highest to lowest)
            ask_levels: Ask levels (lowest to highest)
            mid_price: Current mid price (calculated if not provided)

        Returns:
            DepthAnalysis with kappa estimate
        """
        # Limit to max levels
        bids = bid_levels[: self._max_levels]
        asks = ask_levels[: self._max_levels]

        if not bids and not asks:
            return DepthAnalysis(
                kappa=self._default_kappa,
                bid_depth=0.0,
                ask_depth=0.0,
                bid_levels=0,
                ask_levels=0,
                price_range=0.0,
            )

        # Calculate total depth
        bid_depth = sum(level.quantity for level in bids)
        ask_depth = sum(level.quantity for level in asks)
        total_depth = bid_depth + ask_depth

        if total_depth == 0:
            return DepthAnalysis(
                kappa=self._default_kappa,
                bid_depth=0.0,
                ask_depth=0.0,
                bid_levels=len(bids),
                ask_levels=len(asks),
                price_range=0.0,
            )

        # Calculate price range
        # For bids, prices are negative in SDK format
        if bids and asks:
            best_bid = abs(bids[0].price)
            best_ask = asks[0].price
            worst_bid = abs(bids[-1].price) if bids else best_bid
            worst_ask = asks[-1].price if asks else best_ask
            price_range = max(best_ask - best_bid, worst_ask - worst_bid, 1)
        elif bids:
            best_bid = abs(bids[0].price)
            worst_bid = abs(bids[-1].price)
            price_range = max(best_bid - worst_bid, 1)
            mid_price = mid_price or best_bid
        else:
            best_ask = asks[0].price
            worst_ask = asks[-1].price
            price_range = max(worst_ask - best_ask, 1)
            mid_price = mid_price or best_ask

        # Calculate mid price if not provided
        if mid_price is None:
            if bids and asks:
                mid_price = (abs(bids[0].price) + asks[0].price) / 2
            else:
                mid_price = 50.0  # Default center for binary options

        # Estimate kappa: volume density normalized by mid price
        # Higher total_depth and narrower price_range = higher kappa
        if price_range > 0 and mid_price > 0:
            # Normalize to make kappa comparable across price levels
            kappa = total_depth / (price_range * mid_price)
        else:
            kappa = self._default_kappa

        # Clamp to reasonable range
        kappa = max(self._min_kappa, min(self._max_kappa, kappa))

        logger.debug(
            f"Depth analysis: bids={bid_depth:.0f} asks={ask_depth:.0f} "
            f"range={price_range:.1f} mid={mid_price:.1f} → κ={kappa:.4f}"
        )

        return DepthAnalysis(
            kappa=kappa,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_levels=len(bids),
            ask_levels=len(asks),
            price_range=price_range,
        )

    def analyze_from_depth_data(
        self,
        depth_levels: list[dict],
        mid_price: Optional[float] = None,
    ) -> DepthAnalysis:
        """
        Analyze from SDK's get_market_depth() response.

        Args:
            depth_levels: List of DepthLevel dicts with 'price' and 'total_amount'
            mid_price: Current mid price (optional)

        Returns:
            DepthAnalysis with kappa estimate
        """
        bids = []
        asks = []

        for level in depth_levels:
            price = level["price"]
            amount = level["total_amount"]

            order = OrderLevel(price=price, quantity=amount)

            if price < 0:
                bids.append(order)
            elif price > 0:
                asks.append(order)

        # Sort: bids by absolute price descending, asks ascending
        bids.sort(key=lambda x: abs(x.price), reverse=True)
        asks.sort(key=lambda x: x.price)

        return self.analyze(bids, asks, mid_price)


class DepthTracker:
    """
    Tracks order book depth across multiple markets.
    """

    def __init__(
        self,
        default_kappa: float = 0.5,
        max_levels: int = 5,
    ):
        """
        Initialize tracker.

        Args:
            default_kappa: Default kappa when analysis not possible
            max_levels: Maximum depth levels to analyze
        """
        self._analyzer = OrderBookDepthAnalyzer(
            default_kappa=default_kappa,
            max_levels=max_levels,
        )
        self._cache: dict[tuple[int, bool], DepthAnalysis] = {}

    def update(
        self,
        query_id: int,
        outcome: bool,
        bid_levels: list[OrderLevel],
        ask_levels: list[OrderLevel],
        mid_price: Optional[float] = None,
    ) -> DepthAnalysis:
        """
        Update depth analysis for a market outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            bid_levels: Current bid levels
            ask_levels: Current ask levels
            mid_price: Current mid price

        Returns:
            Updated DepthAnalysis
        """
        analysis = self._analyzer.analyze(bid_levels, ask_levels, mid_price)
        self._cache[(query_id, outcome)] = analysis
        return analysis

    def get_kappa(self, query_id: int, outcome: bool) -> float:
        """
        Get cached kappa for a market outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO

        Returns:
            Cached kappa value or default
        """
        key = (query_id, outcome)
        if key in self._cache:
            return self._cache[key].kappa
        return self._analyzer._default_kappa

    def get_analysis(
        self, query_id: int, outcome: bool
    ) -> Optional[DepthAnalysis]:
        """Get cached analysis if available."""
        return self._cache.get((query_id, outcome))

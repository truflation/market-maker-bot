"""
Per-market inventory management for the Avellaneda Market Making Bot.

Tracks YES/NO share positions and calculates inventory deviation
for each market independently.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class MarketInventory:
    """
    Inventory tracking for a single market.

    YES and NO are separate tokens, but share USD as quote currency.
    Each market's inventory is tracked independently.
    """

    query_id: int
    target_pct: float = 50.0  # Target percentage of value in shares

    # Current positions
    yes_shares: int = 0
    no_shares: int = 0

    # Collateral locked in orders (not yet filled)
    usd_locked_yes_bids: Decimal = field(default_factory=lambda: Decimal("0"))
    usd_locked_no_bids: Decimal = field(default_factory=lambda: Decimal("0"))

    def update_from_positions(
        self,
        yes_shares: int,
        no_shares: int,
        yes_bid_collateral: Decimal = Decimal("0"),
        no_bid_collateral: Decimal = Decimal("0"),
    ) -> None:
        """
        Update inventory from position data.

        Args:
            yes_shares: Number of YES shares held
            no_shares: Number of NO shares held
            yes_bid_collateral: USD locked in YES bid orders
            no_bid_collateral: USD locked in NO bid orders
        """
        self.yes_shares = yes_shares
        self.no_shares = no_shares
        self.usd_locked_yes_bids = yes_bid_collateral
        self.usd_locked_no_bids = no_bid_collateral

    def get_share_value(self, outcome: bool, mid_price: float) -> Decimal:
        """
        Get value of shares for a specific outcome in USD.

        Args:
            outcome: True for YES, False for NO
            mid_price: Current mid price in cents (1-99)

        Returns:
            Value in USD (shares * price / 100)
        """
        shares = self.yes_shares if outcome else self.no_shares
        return Decimal(str(shares)) * Decimal(str(mid_price / 100))

    def get_market_value(self, yes_price: float, no_price: float) -> Decimal:
        """
        Total market value in USD for this market.

        Includes share values and locked collateral.

        Args:
            yes_price: YES mid price in cents
            no_price: NO mid price in cents

        Returns:
            Total USD value
        """
        yes_value = Decimal(str(self.yes_shares)) * Decimal(str(yes_price / 100))
        no_value = Decimal(str(self.no_shares)) * Decimal(str(no_price / 100))
        return yes_value + no_value + self.usd_locked_yes_bids + self.usd_locked_no_bids

    def get_inventory_ratio(self, outcome: bool, mid_price: float) -> float:
        """
        Calculate inventory deviation for specific outcome (Hummingbot-compatible).

        Uses the Hummingbot formula:
            q = (current_base - target_base) / total_inventory_in_base_units

        For binary options:
        - "base" = shares of the specific outcome
        - "quote" = USD (cash + locked collateral)
        - Total inventory in "base units" = (share_value + quote_value) / price

        Returns:
        - Positive = excess inventory (encourage selling)
        - Negative = deficit inventory (encourage buying)
        - Zero = at target

        The range is approximately -0.5 to +0.5 with 50% target,
        matching Hummingbot's scaling.

        Args:
            outcome: True for YES, False for NO
            mid_price: Current mid price in cents for this outcome

        Returns:
            Inventory ratio (q value)
        """
        if mid_price <= 0:
            return 0.0

        # Current shares for this outcome (base asset)
        shares = self.yes_shares if outcome else self.no_shares

        # Get quote value (cash/collateral for this side)
        # For simplicity, we use the locked collateral as "quote"
        quote_value = float(
            self.usd_locked_yes_bids if outcome else self.usd_locked_no_bids
        )

        # Calculate share value in USD (quote units)
        price_in_dollars = mid_price / 100.0  # Convert cents to dollars
        share_value = shares * price_in_dollars

        # Total inventory value in quote (USD)
        total_value_quote = share_value + quote_value

        if total_value_quote <= 0:
            return 0.0

        # Total inventory in base units (shares equivalent)
        total_inventory_base = total_value_quote / price_in_dollars

        # Target shares (target_pct% of total inventory in base units)
        target_shares = total_inventory_base * (self.target_pct / 100.0)

        # q = (current - target) / total_inventory_base
        # This gives approximately -0.5 to +0.5 range with 50% target
        q = (shares - target_shares) / total_inventory_base

        return q

    def get_net_exposure(self, yes_price: float, no_price: float) -> float:
        """
        Get net USD exposure for this market (YES value - NO value).

        Positive = long YES bias
        Negative = long NO bias

        Args:
            yes_price: YES mid price in cents
            no_price: NO mid price in cents

        Returns:
            Net exposure in USD
        """
        yes_value = self.yes_shares * yes_price / 100
        no_value = self.no_shares * no_price / 100
        return yes_value - no_value


class InventoryManager:
    """
    Manages inventory for each market independently.

    This class maintains per-market inventory tracking and provides
    methods to update from blockchain position data.
    """

    def __init__(self, target_pct: float = 50.0):
        """
        Initialize inventory manager.

        Args:
            target_pct: Default target percentage for shares (50 = neutral)
        """
        self._target_pct = target_pct
        self._inventories: dict[int, MarketInventory] = {}

    def get_market_inventory(self, query_id: int) -> MarketInventory:
        """
        Get inventory tracker for a specific market.

        Creates a new tracker if one doesn't exist.

        Args:
            query_id: Market ID

        Returns:
            MarketInventory for the market
        """
        if query_id not in self._inventories:
            self._inventories[query_id] = MarketInventory(
                query_id=query_id,
                target_pct=self._target_pct,
            )
        return self._inventories[query_id]

    def update_from_user_positions(self, positions: list[dict]) -> None:
        """
        Update all inventories from user positions data.

        Args:
            positions: List of position dicts from TNClient.get_user_positions()
                      Each has: query_id, outcome, price, amount
        """
        # Group by market
        by_market: dict[int, dict[str, int]] = {}

        for pos in positions:
            query_id = pos["query_id"]
            outcome = pos["outcome"]
            price = pos.get("price", 0)
            amount = pos.get("amount", 0)

            if query_id not in by_market:
                by_market[query_id] = {
                    "yes_shares": 0,
                    "no_shares": 0,
                    "yes_bid_value": 0,
                    "no_bid_value": 0,
                }

            # price == 0 means holding, price < 0 means bid
            if price == 0:
                # Holdings
                if outcome:
                    by_market[query_id]["yes_shares"] += amount
                else:
                    by_market[query_id]["no_shares"] += amount
            elif price < 0:
                # Open bid (buy order) - collateral locked
                collateral = abs(price) * amount  # In cents
                if outcome:
                    by_market[query_id]["yes_bid_value"] += collateral
                else:
                    by_market[query_id]["no_bid_value"] += collateral

        # Update each market's inventory
        for query_id, data in by_market.items():
            inventory = self.get_market_inventory(query_id)
            inventory.update_from_positions(
                yes_shares=data["yes_shares"],
                no_shares=data["no_shares"],
                yes_bid_collateral=Decimal(str(data["yes_bid_value"] / 100)),
                no_bid_collateral=Decimal(str(data["no_bid_value"] / 100)),
            )

        logger.debug(f"Updated inventory for {len(by_market)} markets")

    def get_inventory_skew(
        self, query_id: int, outcome: bool, mid_price: float
    ) -> float:
        """
        Get inventory skew (q value) for Avellaneda-Stoikov pricing.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO
            mid_price: Current mid price in cents

        Returns:
            q value from -1 to +1
        """
        inventory = self.get_market_inventory(query_id)
        return inventory.get_inventory_ratio(outcome, mid_price)

    def log_inventory_status(self, query_id: int, yes_price: float, no_price: float) -> None:
        """Log current inventory status for a market."""
        inv = self.get_market_inventory(query_id)
        market_value = inv.get_market_value(yes_price, no_price)
        net_exposure = inv.get_net_exposure(yes_price, no_price)

        logger.info(
            f"Market {query_id} inventory: "
            f"YES={inv.yes_shares} NO={inv.no_shares} "
            f"value=${market_value:.2f} net_exposure=${net_exposure:.2f}"
        )

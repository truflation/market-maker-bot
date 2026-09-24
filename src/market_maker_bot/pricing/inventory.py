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

# How long a sell that failed without proof it never rested stays out of the
# free count unless a read shows it listed: covers the write-confirm timeout
# plus the longest a blocked write pool can still broadcast (900s).
UNCONFIRMED_SELL_TTL_S = 900.0


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

    # Inventory-backed sells reserved against currently-listed asks. Each
    # reserved unit consumes one matched (YES, NO) pair regardless of which
    # outcome the ask is on, because backing a YES ask burns 1 YES from
    # inventory and backing a NO ask burns 1 NO. We track YES and NO
    # reservations separately so the available pool can shrink correctly
    # when asks on different outcomes are listed against the same pair pool.
    reserved_yes_sells: int = 0
    reserved_no_sells: int = 0

    # Shares CURRENTLY LISTED on chain as sell orders (any source: bot's own
    # split-mint asks, bot's own inventory-backed asks, or orphan auto-listed
    # legs from a partially-completed pre-mint). Distinct from yes_shares /
    # no_shares (which only count `price=0` held). Populated each refresh
    # from `get_user_positions` entries with `price > 0`.
    # paired_inventory() needs to count these because the underlying pair
    # collateral is still committed (the listed leg can be cancelled to
    # return to held inventory). Without this, a crash between split-mint
    # broadcast and the cancel of the auto-listed leg would cause the
    # next pre-mint to under-count owned pairs and double-mint.
    chain_listed_yes_sells: int = 0
    chain_listed_no_sells: int = 0

    # The same listed sells keyed by (outcome, price), the time the last
    # refresh's read started, and sells whose outcome is unknown (a timeout
    # may still land), as (outcome, price, amount, noted_at). With the bot's
    # tracked asks these give free_to_sell() without double-counting listed
    # shares.
    listed_by_price: dict = field(default_factory=dict)
    refreshed_at: float = 0.0
    unconfirmed_sells: list = field(default_factory=list)

    def reserve_pair(self, outcome: bool, n: int) -> None:
        """Vestigial: asks are sized by free_to_sell(), not by these
        counters or available_for_sell(); kept for logs and old callers.

        Reserve n shares for an inventory-backed ASK. outcome=True debits
        from YES side, outcome=False from NO side. No-op if n <= 0."""
        if n <= 0:
            return
        if outcome:
            self.reserved_yes_sells += n
        else:
            self.reserved_no_sells += n

    def release_pair(self, outcome: bool, n: int) -> None:
        """Return n reserved shares to available (cancel or fill). Clamped at
        zero so a stray double-release does not break the invariant."""
        if n <= 0:
            return
        if outcome:
            self.reserved_yes_sells = max(0, self.reserved_yes_sells - n)
        else:
            self.reserved_no_sells = max(0, self.reserved_no_sells - n)

    def available_for_sell(self, outcome: bool) -> int:
        """Shares available to back a new inventory-backed ASK on this side.
        Negative results are clamped to 0 so an inventory drift (e.g. from a
        partial fill between refresh cycles) does not produce nonsense."""
        held = self.yes_shares if outcome else self.no_shares
        reserved = self.reserved_yes_sells if outcome else self.reserved_no_sells
        return max(0, held - reserved)

    def note_unconfirmed_sell(
        self, outcome: bool, price: int, n: int, noted_at: float
    ) -> None:
        """A sell whose placement failed without proof it never rested (a
        timeout may still land, even after the next refresh). Its shares
        stay out of the free count until a read shows it listed or
        UNCONFIRMED_SELL_TTL_S passes."""
        if n > 0:
            self.unconfirmed_sells.append((outcome, price, n, noted_at))

    def free_to_sell(self, outcome: bool, tracked_asks: list) -> int:
        """Shares free to back a new inventory-backed ASK on `outcome`.

        `tracked_asks` is (price, amount, created_at) for each of the bot's
        inventory-backed asks on this outcome. Held shares (price=0
        positions) already exclude every listed sell, so:

        - an ask placed before the last refresh's read reduces the count
          only by the part that read did not see listed at its price (a
          stale read, or a partial fill);
        - an ask placed since reduces it in full, even if some other order
          of ours rests at the same price (the chain keeps one order per
          price, so the read cannot tell them apart).

        Every unknown rounds down. A cancelled ask's shares come back only
        with the next refresh; listings the bot does not track (legacy or
        orphan orders) are neither added nor subtracted.
        """
        held = self.yes_shares if outcome else self.no_shares
        seen: dict = {}
        pending = sum(n for o, _, n, _ in self.unconfirmed_sells if o == outcome)
        for price, amount, created_at in tracked_asks:
            if created_at >= self.refreshed_at:
                pending += amount
            else:
                seen[price] = seen.get(price, 0) + amount
        for price, amount in seen.items():
            listed = self.listed_by_price.get((outcome, price), 0)
            pending += max(0, amount - listed)
        return max(0, held - pending)

    def paired_inventory(self) -> int:
        """Number of fully-paired (1 YES + 1 NO) units we own on this market,
        including shares currently listed as sell orders (whose underlying
        collateral is still ours — a listed YES sell at 99c is just a held
        YES share with a pending sale; if we cancel the sell, the share
        returns to held inventory).

        Pre-mint deficit math runs against THIS, not held alone, so a crash
        between split-mint and auto-leg cancel does not cause the next
        startup to under-count owned pairs and double-mint."""
        total_yes = self.yes_shares + self.chain_listed_yes_sells
        total_no = self.no_shares + self.chain_listed_no_sells
        return min(total_yes, total_no)

    def update_from_positions(
        self,
        yes_shares: int,
        no_shares: int,
        yes_bid_collateral: Decimal = Decimal("0"),
        no_bid_collateral: Decimal = Decimal("0"),
        chain_listed_yes_sells: int = 0,
        chain_listed_no_sells: int = 0,
        listed_by_price: Optional[dict] = None,
        refreshed_at: float = 0.0,
    ) -> None:
        """
        Update inventory from position data. NOTE: held + listed totals
        come from chain truth and OVERWRITE the cached values; reservation
        counters (reserved_*_sells) are bot-side state and are NOT touched
        here, because a fresh chain snapshot does not invalidate the bot's
        in-flight intent to list shares.

        Args:
            yes_shares: Number of YES shares held (`price=0` entries).
            no_shares: Number of NO shares held.
            yes_bid_collateral: USD locked in YES bid orders.
            no_bid_collateral: USD locked in NO bid orders.
            chain_listed_yes_sells: Shares listed as YES sell orders on chain
                (`price>0, outcome=True` entries). Counted in paired_inventory
                but NOT in available_for_sell.
            chain_listed_no_sells: Same for NO sell orders.
        """
        self.yes_shares = yes_shares
        self.no_shares = no_shares
        self.usd_locked_yes_bids = yes_bid_collateral
        self.usd_locked_no_bids = no_bid_collateral
        self.chain_listed_yes_sells = chain_listed_yes_sells
        self.chain_listed_no_sells = chain_listed_no_sells
        self.listed_by_price = dict(listed_by_price or {})
        self.refreshed_at = refreshed_at
        # Drop an unconfirmed sell once this read shows it listed (it landed
        # and held already excludes it) or once it is too old to still land.
        self.unconfirmed_sells = [
            (o, price, n, noted_at)
            for o, price, n, noted_at in self.unconfirmed_sells
            if self.listed_by_price.get((o, price), 0) < n
            and refreshed_at - noted_at < UNCONFIRMED_SELL_TTL_S
        ]

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

    def update_from_user_positions(
        self, positions: list[dict], as_of: float = 0.0
    ) -> None:
        """
        Update all inventories from user positions data.

        Args:
            positions: List of position dicts from TNClient.get_user_positions()
                      Each has: query_id, outcome, price, amount
            as_of: Time the read started. Orders placed from then on are
                      not assumed visible in it.
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
                    "yes_listed_sells": 0,
                    "no_listed_sells": 0,
                    "listed_by_price": {},
                }

            # Signed-price convention from get_user_positions:
            #   price == 0  -> holding
            #   price <  0  -> open buy_order (signed-negative)
            #   price >  0  -> open sell_order
            if price == 0:
                if outcome:
                    by_market[query_id]["yes_shares"] += amount
                else:
                    by_market[query_id]["no_shares"] += amount
            elif price < 0:
                collateral = abs(price) * amount  # In cents
                if outcome:
                    by_market[query_id]["yes_bid_value"] += collateral
                else:
                    by_market[query_id]["no_bid_value"] += collateral
            else:
                # Open sell_order. Counted into pair-inventory accounting
                # (not into available_for_sell), so a partial pre-mint that
                # left an auto-listed leg on the book is not double-minted
                # on restart.
                if outcome:
                    by_market[query_id]["yes_listed_sells"] += amount
                else:
                    by_market[query_id]["no_listed_sells"] += amount
                listed = by_market[query_id]["listed_by_price"]
                key = (bool(outcome), int(price))
                listed[key] = listed.get(key, 0) + amount

        # A market with no positions at all holds nothing: reset it rather
        # than keep the last read's shares.
        for query_id in self._inventories:
            if query_id not in by_market:
                by_market[query_id] = {
                    "yes_shares": 0,
                    "no_shares": 0,
                    "yes_bid_value": 0,
                    "no_bid_value": 0,
                    "yes_listed_sells": 0,
                    "no_listed_sells": 0,
                    "listed_by_price": {},
                }

        # Update each market's inventory
        for query_id, data in by_market.items():
            inventory = self.get_market_inventory(query_id)
            inventory.update_from_positions(
                yes_shares=data["yes_shares"],
                no_shares=data["no_shares"],
                yes_bid_collateral=Decimal(str(data["yes_bid_value"] / 100)),
                no_bid_collateral=Decimal(str(data["no_bid_value"] / 100)),
                chain_listed_yes_sells=data["yes_listed_sells"],
                chain_listed_no_sells=data["no_listed_sells"],
                listed_by_price=data["listed_by_price"],
                refreshed_at=as_of,
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

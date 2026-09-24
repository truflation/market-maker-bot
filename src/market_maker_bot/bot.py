"""
Main bot orchestrator for the Avellaneda Market Making Bot.

Coordinates all components: pricing, indicators, inventory, and order management.
"""

import json
import math
import os
import sys
import time
import logging
import signal
from collections import deque
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set, TYPE_CHECKING
from dataclasses import dataclass

# Lazy import TNClient to avoid segfaults when SDK Go bindings aren't available
# (e.g., in dry-run mode or during testing)
if TYPE_CHECKING:
    from trufnetwork_sdk_py.client import TNClient

from .config import BotConfig, MarketConfig, AvellanedaConfig
from .models import OutcomeMode, Side, PricingResult, MarketState
from .market import (
    MarketContext,
    OrderManager,
    build_market_state,
    convert_price_for_order,
)
from .pricing import AvellanedaPricing, InventoryManager, price_binary_option
from .indicators import (
    InstantVolatilityIndicator,
    OrderBookDepthAnalyzer,
    calculate_stream_volatility,
    get_current_spot_value,
)
from .indicators.volatility import VolatilityTracker
from .indicators.depth import DepthTracker
from .execution_state import (
    ExecutionState,
    ExecutionTimeframeMode,
    ExecutionTimeframeConfig,
    RunAlwaysExecutionState,
    RunInTimeExecutionState,
    create_execution_state,
)
from .hanging_orders import HangingOrdersTracker, HangingOrder, CreatedPairOfOrders
from .order_state import OrderStateManager, TrackedOrder
from .bid_budget import BidBudget

logger = logging.getLogger(__name__)

SETTLED_MARKET_ERRORS = (
    "already settled",
    "settled market",
)

# Exit code taken when cancels loop on "order not found" (see the block
# below): the stale state behind that loop has no in-process fix, so the
# recovery is a clean process restart by systemd. Numbered alongside the
# pool exit codes used by the bounded-client builds (70, 71).
EXIT_STALE_CANCEL_LOOP = 72

# --- cancel "order not found" handling (2026-08-28 incident) ----------------
# When the chain rejects a cancel with "Order not found or does not belong to
# you", the order is definitively NOT on the book under our wallet. A stale
# order-book read can keep "showing" such orders for hours; retrying the
# cancel every cycle then produces a stream of failed transactions (observed:
# ~55k over 31 hours at ~0.5/sec). The chain error text is authoritative over
# any book read, so it is treated as "already cancelled" wherever it can be
# observed (wait=True).
CANCEL_NOT_FOUND_SNIPPET = "order not found or does not belong"

# Slot-guard stall escalation: a level skipping this many CONSECUTIVE
# cycles (its target price owned by another level) is not a passing grid
# shift, it is wedged tracking. ~5 min at a 2s cycle.
SLOT_GUARD_STALL_ERROR_EVERY = 150
# wait=False cancels (the reconcile orphan pass) never observe the error, so
# that loop is broken by attempt-counting instead: an orphan still on the book
# after this many cancel attempts is provably not cancellable from this
# process (the tx fails on chain, or the book read is stale) and is skipped.
ORPHAN_CANCEL_MAX_ATTEMPTS = 3
# A restart is the proven cure for the stale state behind either loop (fresh
# client, fresh connections). If this many DISTINCT orphans are stuck past
# their attempt budget, or this many observed not-found errors land inside
# the window, exit with EXIT_STALE_CANCEL_LOOP and restart clean.
STUCK_ORPHANS_EXIT_THRESHOLD = 10
CANCEL_NOT_FOUND_WINDOW_SEC = 600.0
CANCEL_NOT_FOUND_EXIT_COUNT = 50


# Minimum order notional enforced by TN's prediction-market protocol.
# All current markets (6-dec USDC mainnet and 18-dec TT2 testnet) use
# "1 token unit" as the minimum, which is exactly 100 in cent-shares
# (price * amount) for both decimals (1e6/1e4 == 1e18/1e16 == 100).
# Orders below this silently fail to land on chain: the SDK returns a
# tx_hash but the on-chain action reverts, leaving the bot's state file
# tracking phantom orders that don't exist.
#
# When TN starts allowing other per-market minimums we should source
# this from market_config.min_order_size (currently not threaded through
# from the orchestrator's market discovery; tracked as follow-up).
MIN_ORDER_NOTIONAL_CENT_SHARES = 100


def _meets_min_notional(price: int, amount: int) -> bool:
    """Does an order of `amount` shares at `price` cents clear the protocol's
    minimum notional? See the comment on MIN_ORDER_NOTIONAL_CENT_SHARES."""
    return price * amount >= MIN_ORDER_NOTIONAL_CENT_SHARES


def _compute_base_amount(market_config, price_cents: int) -> int:
    """Resolve the per-leg base order amount before eta/level adjustments.

    If `order_dollar_amount` is set on the market config, size the leg so
    its notional approximates that dollar target at the current leg price:
    amount = round(X * 100 / price). Otherwise fall back to the static
    `order_amount` share count. Floored at 1 share so we never compute 0.
    """
    dollar_target = getattr(market_config, "order_dollar_amount", None)
    if dollar_target is not None and price_cents > 0:
        return max(1, round(dollar_target * 100 / price_cents))
    return market_config.order_amount


def _pair_out_of_order(
    side: "Side", lvl_a: int, price_a: int, lvl_b: int, price_b: int
) -> bool:
    """True if two tracked levels violate the grid's ordering invariant:
    ask prices ascend with level index, bid prices descend (level 0 is the
    tightest quote on both sides). Equal prices are never out of order."""
    if lvl_a == lvl_b:
        return False
    if lvl_a > lvl_b:
        lvl_a, price_a, lvl_b, price_b = lvl_b, price_b, lvl_a, price_a
    if side == Side.ASK:
        return price_a > price_b
    return price_a < price_b


class MarketSettledError(Exception):
    """Raised when an operation targets a market that has already settled."""
    def __init__(self, query_id: int):
        self.query_id = query_id
        super().__init__(f"Market {query_id} has settled")


class ReadOnlyTNClient:
    """Wraps a TNClient. Read methods pass through; broadcast/write methods
    are logged and short-circuited so the bot can do a Phase-1-style dry run
    against a real chain without committing transactions.

    Distinct from `dry_run`, which skips client init entirely (so even reads
    fail). Use `read_only` when you want the bot to fully connect, fetch
    stream records and order books, compute prices, and log what it WOULD
    do, but not actually broadcast.
    """

    # SDK methods that produce on-chain transactions. Anything else passes
    # straight through to the wrapped client.
    _WRITE_METHODS = frozenset({
        "place_buy_order",
        "place_sell_order",
        "place_split_limit_order",
        "cancel_order",
        "settle_market",
    })

    def __init__(self, real_client):
        self._client = real_client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if name in self._WRITE_METHODS and callable(attr):
            def stub(*args, **kwargs):
                logger.info(
                    f"[READ-ONLY] suppressed {name}(args={args!r}, kwargs={kwargs!r})"
                )
                return f"read_only_stub_tx_{name}"
            return stub
        return attr


@dataclass
class BotStats:
    """Runtime statistics for the bot."""
    orders_placed: int = 0
    orders_updated: int = 0
    orders_cancelled: int = 0
    errors: int = 0
    cycles: int = 0


@dataclass
class TradingStats:
    """Cumulative P&L tracking."""
    total_bought_value: float = 0.0  # Total USD spent buying shares
    total_sold_value: float = 0.0    # Total USD received selling shares
    total_shares_bought: int = 0
    total_shares_sold: int = 0
    fills_detected: int = 0


class AvellanedaMarketMaker:
    """
    Avellaneda-Stoikov market making bot for TrufNetwork prediction markets.

    Main loop:
    1. Poll order book → update volatility indicator
    2. Refresh inventory periodically
    3. Estimate kappa from depth
    4. Calculate A-S prices
    5. Check order refresh tolerance
    6. Execute updates using change_bid()/change_ask() when possible
    """

    def __init__(self, config: BotConfig):
        """
        Initialize the market maker.

        Args:
            config: Bot configuration
        """
        self.config = config
        self._running = False
        self._shutdown_requested = False

        # Initialize TNClient (lazy loaded to avoid import issues with Go bindings)
        self._client: Optional["TNClient"] = None

        # Core components
        self._pricing = AvellanedaPricing(config.avellaneda)
        self._inventory = InventoryManager(
            target_pct=config.avellaneda.inventory_target_base_pct
        )
        self._volatility_tracker = VolatilityTracker(
            buffer_size=config.avellaneda.volatility_buffer_size,
            min_samples=config.avellaneda.volatility_min_samples,
            default_value=config.avellaneda.default_volatility,
        )
        self._depth_tracker = DepthTracker(
            default_kappa=1.0,  # Dynamic kappa estimated from trading intensity
            max_levels=5,
        )

        # Order timing state
        self._last_fill_time: dict[tuple[int, bool], float] = {}  # (query_id, outcome) -> time

        # Market contexts
        self._markets: dict[int, MarketContext] = {}

        # Statistics
        self.stats = BotStats()
        self._trading_stats = TradingStats()

        # Timing
        self._last_inventory_refresh = 0.0
        self._last_reconcile = 0.0
        # Cancel-loop defenses (see the CANCEL_NOT_FOUND_SNIPPET block above).
        # (query_id, outcome, signed_sdk_price) -> wait=False cancel attempts
        self._reconcile_cancel_attempts: Dict[Tuple[int, bool, int], int] = {}
        # Timestamps of observed on-chain "order not found" cancel failures.
        self._cancel_not_found_times: "deque[float]" = deque()
        # Level loop breaker (2026-09-09 Eggs 72c storm): recent "order not
        # found" CLEAR timestamps per exact quote slot, and the epoch until
        # which a tripped slot is suppressed from quoting.
        # Key: (query_id, outcome, is_buy, display_price).
        self._level_not_found_times: Dict[
            Tuple[int, bool, bool, int], "deque[float]"
        ] = {}
        self._level_cooldown_until: Dict[Tuple[int, bool, bool, int], float] = {}
        # Consecutive slot-guard skips per (query_id, outcome, is_buy,
        # level_idx). A grid shift stalls a level for a cycle or two; a
        # level stalled for hundreds of cycles means tracking is wedged
        # (e.g. a zombie duplicate slot) and must page, not whisper INFO.
        self._slot_guard_skips: Dict[Tuple[int, bool, bool, int], int] = {}

        # Execution state (timeframe control)
        self._execution_state = self._create_execution_state()

        # Hanging orders tracker (per market/outcome)
        self._hanging_trackers: dict[tuple[int, bool], HangingOrdersTracker] = {}

        # In-flight cancellations (for should_wait_order_cancel_confirmation)
        self._in_flight_cancels: Set[str] = set()
        self._pre_settlement_pulled: Set[int] = set()
        # Markets whose earnings cutoff has passed and whose resting orders have
        # been pulled THIS session. In-memory on purpose (not persisted): the
        # cutoff timestamp is the deterministic source of truth, so every
        # (re)start re-cancels a past-earnings market's orders on the first cycle
        # before it can quote -- a persisted "pulled" flag could go stale and
        # leave leaked orders live (fail-open). This re-asserts each session.
        self._earnings_pulled_session: Set[int] = set()

        # Order state persistence (for restart recovery)
        self._order_state = OrderStateManager(config.order_state_file)

        # Derive pre_settlement_pulled persistence path from order_state_file.
        # Include the state file's stem so a second MM bot instance with a
        # different order_state_file (e.g. mainnet) gets its own
        # pre_settlement file rather than sharing one with testnet.
        state_path = Path(config.order_state_file)
        self._pre_settlement_file = str(
            state_path.parent / f"{state_path.stem}_pre_settlement_pulled.json"
        )
        # One-time migration: if an old shared `pre_settlement_pulled.json`
        # exists at the same parent and the new per-stem file does not,
        # rename it into place. Keeps the existing testnet bot from losing
        # its pre-settlement state on the upgrade.
        legacy = state_path.parent / "pre_settlement_pulled.json"
        new_path = Path(self._pre_settlement_file)
        if legacy.exists() and not new_path.exists():
            try:
                legacy.rename(new_path)
                logger.info(
                    f"Migrated pre_settlement_pulled file: {legacy.name} -> {new_path.name}"
                )
            except OSError as e:
                logger.warning(f"Failed to migrate {legacy}: {e}")
        self._load_pre_settlement_pulled()

    def _create_execution_state(self) -> ExecutionState:
        """
        Create execution state from configuration.

        Returns:
            Appropriate ExecutionState instance based on config
        """
        mode_str = self.config.avellaneda.execution_timeframe_mode
        try:
            mode = ExecutionTimeframeMode(mode_str)
        except ValueError:
            logger.warning(f"Unknown execution mode '{mode_str}', using infinite")
            mode = ExecutionTimeframeMode.INFINITE

        config = ExecutionTimeframeConfig(mode=mode)

        if mode == ExecutionTimeframeMode.FROM_DATE_TO_DATE:
            if self.config.avellaneda.execution_start_datetime:
                config.start_datetime = datetime.fromisoformat(
                    self.config.avellaneda.execution_start_datetime
                )
            if self.config.avellaneda.execution_end_datetime:
                config.end_datetime = datetime.fromisoformat(
                    self.config.avellaneda.execution_end_datetime
                )

        elif mode == ExecutionTimeframeMode.DAILY_BETWEEN_TIMES:
            if self.config.avellaneda.execution_start_time:
                config.start_time = dt_time.fromisoformat(
                    self.config.avellaneda.execution_start_time
                )
            if self.config.avellaneda.execution_end_time:
                config.end_time = dt_time.fromisoformat(
                    self.config.avellaneda.execution_end_time
                )

        return create_execution_state(config)

    def _load_pre_settlement_pulled(self) -> None:
        """Load persisted pre_settlement_pulled set from disk."""
        try:
            path = Path(self._pre_settlement_file)
            if path.exists():
                data = json.loads(path.read_text())
                self._pre_settlement_pulled = set(data)
                logger.info(
                    f"Loaded {len(self._pre_settlement_pulled)} pre-settlement pulled markets from {self._pre_settlement_file}"
                )
        except Exception as e:
            logger.warning(f"Failed to load pre_settlement_pulled: {e}")

    def _save_pre_settlement_pulled(self) -> None:
        """Persist pre_settlement_pulled set to disk."""
        try:
            Path(self._pre_settlement_file).write_text(
                json.dumps(sorted(self._pre_settlement_pulled))
            )
        except Exception as e:
            logger.warning(f"Failed to save pre_settlement_pulled: {e}")

    def _get_hanging_tracker(
        self, query_id: int, outcome: bool
    ) -> HangingOrdersTracker:
        """
        Get or create hanging orders tracker for a market/outcome.

        Args:
            query_id: Market ID
            outcome: True for YES, False for NO

        Returns:
            HangingOrdersTracker instance
        """
        key = (query_id, outcome)
        if key not in self._hanging_trackers:
            self._hanging_trackers[key] = HangingOrdersTracker(
                hanging_orders_cancel_pct=self.config.avellaneda.hanging_orders_cancel_pct,
                max_order_age=self.config.avellaneda.max_order_age,
            )
        return self._hanging_trackers[key]

    def _can_create_orders(self) -> bool:
        """
        Check if we can create new orders.

        Respects should_wait_order_cancel_confirmation setting.

        Returns:
            True if order creation is allowed
        """
        if not self.config.avellaneda.should_wait_order_cancel_confirmation:
            return True

        # Wait for all in-flight cancellations to complete
        return len(self._in_flight_cancels) == 0

    def _create_proposal_from_order_override(
        self, mid_price: float
    ) -> Optional[List[Tuple[str, int, int]]]:
        """
        Create order proposals from order_override configuration.

        Args:
            mid_price: Current mid price in cents

        Returns:
            List of (side, price, amount) tuples, or None if no override
        """
        order_override = self.config.avellaneda.order_override
        if not order_override:
            return None

        proposals = []
        for key, value in order_override.items():
            if len(value) != 3:
                logger.warning(f"Invalid order_override entry '{key}': {value}")
                continue

            side_str, spread_pct, amount = value
            if side_str not in ["buy", "sell"]:
                logger.warning(f"Invalid side '{side_str}' in order_override '{key}'")
                continue

            try:
                spread_pct = float(spread_pct)
                amount = int(amount)
            except (ValueError, TypeError):
                logger.warning(f"Invalid values in order_override '{key}': {value}")
                continue

            if side_str == "buy":
                price = int(mid_price * (1 - spread_pct / 100))
            else:
                price = int(mid_price * (1 + spread_pct / 100))

            # Clamp to valid range
            price = max(1, min(99, price))

            if amount > 0 and price > 0:
                proposals.append((side_str, price, amount))

        return proposals if proposals else None

    def _init_client(self) -> None:
        """Initialize the TNClient connection."""
        if self._client is not None:
            return

        if self.config.dry_run:
            logger.info("[DRY RUN] Skipping TNClient initialization")
            return

        # Import TNClient at runtime to avoid segfaults when Go bindings aren't available
        from trufnetwork_sdk_py.client import TNClient

        # Pre-flight: derive the wallet address from the private key and
        # log it alongside the gateway URL. Op-time sanity check — if these
        # don't match what the operator expected (wrong env file, wrong
        # network, copy-paste error), this is the last log line before any
        # broadcast so it's easy to ctrl-C.
        wallet_addr = self._derive_wallet_address(self.config.private_key)
        logger.info(
            "Connecting to %s as wallet %s (read_only=%s)",
            self.config.node_url, wallet_addr, self.config.read_only,
        )

        real_client = TNClient(
            url=self.config.node_url,
            token=self.config.private_key,
        )

        if self.config.read_only:
            logger.info(
                "[READ-ONLY] Wrapping TNClient: reads pass through, writes "
                "(place/cancel/settle) are logged and suppressed."
            )
            self._client = ReadOnlyTNClient(real_client)
        else:
            self._client = real_client

    @staticmethod
    def _derive_wallet_address(private_key: str) -> str:
        """Derive the 0x-prefixed Ethereum address from a hex private key.
        Returns "<unknown>" on any failure rather than raising — the address
        is for logging only, not authorization."""
        try:
            from eth_account import Account
            return Account.from_key(private_key).address
        except Exception:
            return "<unknown>"

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown handlers."""
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    def _init_markets(self) -> None:
        """Initialize market contexts from configuration."""
        for market_config in self.config.markets:
            if not market_config.enabled:
                logger.info(f"Skipping disabled market {market_config.query_id}")
                continue

            context = MarketContext(config=market_config)
            self._markets[market_config.query_id] = context

            logger.info(
                f"Initialized market {market_config.query_id} ({market_config.name}) "
                f"mode={market_config.outcome_mode.value}"
            )

        # Per-market bid-collateral caps (2026-09-07 incident, #43): the cap
        # is what honest quoting could ever need, times a safety multiplier.
        # No order-book read can raise it.
        caps: dict[int, int] = {}
        mult = getattr(self.config, "bid_budget_multiplier", 1.5)
        if mult and mult > 0:
            levels = max(1, self.config.avellaneda.order_levels)
            for qid, ctx in self._markets.items():
                oda = ctx.config.order_dollar_amount
                if not oda:
                    continue
                per_outcome = int(oda * 100) * levels
                backstop = 0
                if getattr(self.config, "backstop_amount", 0):
                    backstop = (
                        self.config.backstop_price_cents
                        * self.config.backstop_amount
                    )
                caps[qid] = int((per_outcome + backstop) * 2 * mult)
        self._bid_budget = BidBudget(caps)
        if caps:
            if self.config.reconcile_interval <= 0:
                # The budget's only true-down is the periodic reconcile's
                # sync; without it every fill/cancel leaks committed budget
                # until each market silently stops bidding at its cap
                # (review finding 7). Refuse the combination loudly.
                raise ValueError(
                    "bid budget requires the periodic reconcile: set "
                    "reconcile_interval > 0 (recommended 60) or disable "
                    "the budget with bid_budget_multiplier: 0"
                )
            logger.info(
                f"Bid budget enabled on {len(caps)} markets "
                f"(multiplier {mult})"
            )

    def _reconcile_orders_on_startup(self) -> None:
        """
        Reconcile tracked orders with the order book on startup.

        This allows the bot to resume managing its own orders after a restart,
        while ignoring any orders placed manually outside the bot.
        """
        if self.config.dry_run:
            logger.info("[DRY RUN] Skipping order reconciliation")
            return

        tracked_orders = self._order_state.get_all_orders()
        if not tracked_orders:
            logger.info("No tracked orders from previous session")
            return

        logger.info(f"Reconciling {len(tracked_orders)} tracked orders from previous session...")

        # Group tracked orders by market
        orders_by_market: Dict[int, List[TrackedOrder]] = {}
        for order in tracked_orders:
            if order.query_id not in orders_by_market:
                orders_by_market[order.query_id] = []
            orders_by_market[order.query_id].append(order)

        recovered = 0
        stale = 0

        now_ts = int(time.time())
        for query_id, orders in orders_by_market.items():
            context = self._markets.get(query_id)
            if context is None:
                # Market not configured anymore, clear its orders
                logger.info(f"Market {query_id} no longer configured, clearing tracked orders")
                self._order_state.clear_market(query_id)
                stale += len(orders)
                continue

            # Prune already-settled markets up front so we don't try to query
            # an order book that no longer exists. Without this guard, settled
            # markets accumulate stale tracked orders forever across restarts
            # (root cause of the orchestrator-shutdown bloat that wedged the
            # main thread on 2026-05-01).
            settle_time = context.config.settle_time
            if settle_time is not None and now_ts >= settle_time:
                logger.info(
                    f"Market {query_id} already settled "
                    f"(settle_time={settle_time}, now={now_ts}); clearing "
                    f"{len(orders)} tracked orders"
                )
                self._order_state.clear_market(query_id)
                stale += len(orders)
                continue

            # Fetch current order book for this market.
            # The chain holds ONE order per (outcome, side, price); state
            # files written before the slot-collision guard existed can
            # carry two levels at the same price (the 09-09 Eggs 72c
            # incident wrote such duplicates). Recovering both would
            # re-seed the collision loop with the guard unable to see it,
            # so only the first tracked entry per slot is recovered and
            # the rest are dropped as stale.
            recovered_slots: set[tuple[bool, bool, int]] = set()
            for outcome in [True, False]:
                outcome_orders = [o for o in orders if o.outcome == outcome]
                if not outcome_orders:
                    continue

                try:
                    # Get order book from SDK
                    order_book = self._client.get_order_book(query_id, outcome)
                    state = build_market_state(
                        query_id=query_id,
                        outcome=outcome,
                        order_book_entries=order_book,
                    )
                    context.set_state(outcome, state)

                    # Get our tracked prices
                    for tracked in outcome_orders:
                        # Check if order is still on the book
                        is_on_book = False
                        if tracked.is_buy and state.bid_levels:
                            # bid_levels have negative prices (SDK format), tracked.price is positive
                            is_on_book = any(
                                abs(entry.price) == tracked.price
                                for entry in state.bid_levels
                            )
                        elif not tracked.is_buy and state.ask_levels:
                            is_on_book = any(
                                entry.price == tracked.price
                                for entry in state.ask_levels
                            )

                        slot = (outcome, tracked.is_buy, tracked.price)
                        if is_on_book and slot in recovered_slots:
                            self._order_state.untrack_order(
                                query_id, outcome, tracked.is_buy,
                                tracked.price, tracked.level_idx,
                            )
                            stale += 1
                            logger.warning(
                                f"Duplicate tracked slot dropped: market "
                                f"{query_id} {'YES' if outcome else 'NO'} "
                                f"{'buy' if tracked.is_buy else 'sell'} "
                                f"@{tracked.price}¢ L{tracked.level_idx} "
                                f"(another level already recovered this "
                                f"chain price slot)"
                            )
                            continue

                        if is_on_book:
                            recovered_slots.add(slot)
                            # Order is still active - record it in context
                            side = Side.BID if tracked.is_buy else Side.ASK
                            order_mgr = OrderManager(
                                context,
                                refresh_tolerance_pct=self.config.avellaneda.order_refresh_tolerance_pct,
                                max_order_age=self.config.avellaneda.max_order_age,
                            )
                            order_mgr.record_order(
                                outcome, side, tracked.price, tracked.amount, tracked.order_id,
                                tracked.level_idx,
                                is_inventory_backed=tracked.is_inventory_backed,
                            )
                            # Re-populate inventory reservation so available
                            # inventory accounting is correct from the first
                            # cycle. Without this, a recovered inventory-backed
                            # ASK would be invisible to _place_ask's available
                            # check and the bot could double-list the same
                            # shares on a refresh.
                            if side == Side.ASK and tracked.is_inventory_backed:
                                self._inventory.get_market_inventory(query_id).reserve_pair(
                                    outcome, tracked.amount
                                )
                            recovered += 1
                            logger.debug(
                                f"Recovered order: market {query_id} "
                                f"{'YES' if outcome else 'NO'} "
                                f"{'buy' if tracked.is_buy else 'sell'} @{tracked.price}¢ L{tracked.level_idx}"
                            )
                        else:
                            # Order is no longer on the book - was filled or cancelled externally
                            self._order_state.untrack_order(
                                query_id, outcome, tracked.is_buy, tracked.price,
                                tracked.level_idx,
                            )
                            stale += 1
                            logger.debug(
                                f"Stale order removed: market {query_id} "
                                f"{'YES' if outcome else 'NO'} "
                                f"{'buy' if tracked.is_buy else 'sell'} @{tracked.price}¢"
                            )

                except Exception as e:
                    # Order book query failed (settled, market gone, RPC hiccup).
                    # Untrack the tracked orders for this outcome rather than
                    # carry them forward across restarts. Better to lose a few
                    # legitimate tracked entries on a transient failure than
                    # to accumulate thousands of stale ones over time.
                    # 2026-09-07 (#43): but the orders may STILL REST on chain
                    # (the read failed, they were not confirmed gone), so seed
                    # the bid budget with their cost. This stops a restart
                    # into a degraded gateway from re-placing on top of them;
                    # the first successful periodic reconcile trues it up.
                    seed_cents = sum(
                        t.price * t.amount for t in outcome_orders if t.is_buy
                    )
                    if seed_cents:
                        self._bid_budget.seed(query_id, seed_cents)
                        logger.warning(
                            f"Market {query_id}: seeding bid budget with "
                            f"{seed_cents} cent-shares of unverifiable "
                            f"previously-tracked bids"
                        )
                    logger.error(
                        f"Failed to reconcile orders for market {query_id} "
                        f"outcome={outcome}: {e}; untracking {len(outcome_orders)} "
                        f"orders defensively"
                    )
                    for tracked in outcome_orders:
                        self._order_state.untrack_order(
                            query_id, outcome, tracked.is_buy, tracked.price,
                            tracked.level_idx,
                        )
                        stale += 1

        logger.info(f"Order reconciliation complete: {recovered} recovered, {stale} stale")

    def _calculate_initial_price(
        self, market_config: MarketConfig
    ) -> Optional[float]:
        """
        Calculate initial fair-YES price (in cents, 1-99) when no order book
        exists yet.

        Two paths:
        1. If ``market_config.initial_probability`` is set, use it directly:
           fair_yes_cents = round(initial_probability * 100), clamped to [1, 99].
           Used by Hormuz markets that ship a per-bucket prior because their
           underlying stream has too little history (and a degenerate spot=0
           in the closed-strait regime) for B-S to produce a useful prior.
        2. Otherwise, run Black-Scholes against the stream history. Used by
           CPI markets with rich monthly history.

        Note on fallback: under ``pricing_source="black_scholes"`` (the
        mainnet setting), MarketContext.get_mid_price always returns the
        initial price, NOT the order-book mid. So this prior is effectively
        a permanent fair-value override for Hormuz, not a transient seed.
        Only ``pricing_source="order_book"`` falls back to mid as the order
        book accumulates liquidity.
        """
        # Fixed-prior short-circuit. No SDK call needed; same value every cycle.
        if market_config.initial_probability is not None:
            prior = market_config.initial_probability
            if not (0.0 <= prior <= 1.0):
                logger.warning(
                    f"Market {market_config.query_id}: initial_probability "
                    f"{prior} out of [0, 1]; falling back to B-S"
                )
            else:
                fair_cents = max(1, min(99, int(round(prior * 100))))
                logger.info(
                    f"Market {market_config.query_id}: prior fair-YES "
                    f"= {prior:.4f} -> {fair_cents}c (no B-S)"
                )
                return float(fair_cents)

        try:
            # Fetch stream records with explicit date range.
            # 365 days ensures enough history for monthly streams;
            # the vol calculator applies its own per-frequency lookback internally.
            now_ts = int(time.time())
            date_from = now_ts - 365 * 86400
            records = self._client.get_records(
                stream_id=market_config.stream_id,
                data_provider=market_config.data_provider,
                date_from=date_from,
                date_to=now_ts,
            )

            if not records:
                logger.warning(
                    f"No stream records for market {market_config.query_id}"
                )
                return None

            # Convert StreamRecord objects to dicts if needed (SDK returns Pydantic models)
            if records and hasattr(records[0], "dict"):
                records = [r.dict() if hasattr(r, "dict") else r for r in records]

            # Get current spot value. Negative values are still rejected as
            # nonsense, but zero is a valid spot for streams whose underlying
            # observable can legitimately be zero (e.g. the Hormuz Index ship
            # count during a closed-strait regime). Clamp to a tiny positive
            # epsilon so Black-Scholes, which needs log(spot), stays defined.
            spot = get_current_spot_value(records)
            if spot < 0:
                logger.warning(
                    f"Invalid spot value {spot} for market {market_config.query_id}"
                )
                return None
            if spot == 0:
                spot = 1e-9

            # Calculate stream volatility
            vol_result = calculate_stream_volatility(
                records,
                hourly_lookback=self.config.avellaneda.stream_volatility_lookback_days,
                min_volatility=self.config.avellaneda.stream_volatility_min,
            )

            # Time to expiry from settle_time, or default 3 months
            if market_config.settle_time:
                seconds_left = max(market_config.settle_time - int(time.time()), 3600)
                time_years = seconds_left / (365.25 * 86400)
            else:
                time_years = 0.25

            vol = vol_result.annual_volatility
            has_lower = market_config.lower_bound is not None
            has_upper = market_config.upper_bound is not None

            if has_lower and has_upper:
                # Range market: "between X and Y"
                # P(X <= S < Y) = P(S > X) - P(S > Y)
                p_above_lower = price_binary_option(
                    spot, market_config.lower_bound, time_years, vol
                ).fair_value
                p_above_upper = price_binary_option(
                    spot, market_config.upper_bound, time_years, vol
                ).fair_value
                fair_value = max(0.001, min(0.999, p_above_lower - p_above_upper))
                strike_desc = f"range [{market_config.lower_bound:.1f}, {market_config.upper_bound:.1f}]"
            elif has_upper:
                # "Below X" market: P(S < X) = 1 - P(S > X)
                bs_result = price_binary_option(
                    spot, market_config.upper_bound, time_years, vol
                )
                fair_value = 1.0 - bs_result.fair_value
                strike_desc = f"below {market_config.upper_bound:.1f}"
            elif has_lower:
                # "Above X" market: P(S >= X) = P(S > X)
                bs_result = price_binary_option(
                    spot, market_config.lower_bound, time_years, vol
                )
                fair_value = bs_result.fair_value
                strike_desc = f"above {market_config.lower_bound:.1f}"
            else:
                # Fallback: at-the-money (no threshold data)
                bs_result = price_binary_option(
                    spot, spot, time_years, vol
                )
                fair_value = bs_result.fair_value
                strike_desc = f"ATM {spot:.1f}"

            price_cents = max(1, min(99, int(round(fair_value * 100))))

            logger.info(
                f"Market {market_config.query_id}: Black-Scholes initial price "
                f"spot={spot:.2f} vol={vol:.2%} T={time_years:.4f}y "
                f"strike={strike_desc} "
                f"-> fair_value={fair_value:.3f} -> {price_cents}c"
            )

            return float(price_cents)

        except Exception as e:
            logger.error(
                f"Failed to calculate initial price for market {market_config.query_id}: {e}"
            )
            return None

    def _periodic_reconcile_against_chain(self) -> None:
        """In-loop reconcile: untracks local state with no chain
        counterpart AND cancels chain orders the bot does not track.

        The orphan-cancel direction is the fix for the known cancel-
        then-place silent-failure race in `change_bid` / `_cancel_ask`
        update paths, where the cancel succeeds, the replacement
        silently fails, and the on-chain order persists until the
        bot restarts. Without this pass the orphan rate grows
        unboundedly on a degraded gateway.

        Wallet-scoped: the on-chain order book entries are filtered to
        the bot's own wallet address. If the address cannot be derived
        from the configured private key, the reconcile aborts rather
        than risk cancelling other participants' orders.

        False-positive risk on freshly-placed orders: all placements use
        `wait=True` and `track_order` runs on the same thread before the
        next cycle, so by the time periodic reconcile runs every order
        placed in the cycle is both on chain and in local state.

        Skipped in dry_run.
        """
        if self.config.dry_run:
            return

        # MAA mode: our orders are owned by the MAA (execute_agent_action runs
        # as the MAA), not the agent key that signs. Filtering the order book to
        # the agent key matches nothing, so orphans never get cancelled and the
        # bot retry-cancels non-existent orders forever. Use the MAA address.
        wallet_addr: Optional[str] = None
        if getattr(self.config, "maa_address", ""):
            wallet_addr = self.config.maa_address.strip().lower()
        else:
            try:
                key = self.config.private_key.strip()
                if key.startswith("0x"):
                    key = key[2:]
                if len(key) != 64:
                    raise ValueError("private_key must be 64 hex chars")
                from eth_account import Account
                wallet_addr = Account.from_key("0x" + key).address.lower()
            except Exception:
                wallet_addr = None
        if wallet_addr is None:
            logger.warning(
                "Periodic reconcile: could not derive wallet address; "
                "aborting rather than cancelling other participants' orders"
            )
            return

        # Per-pass cap on cancel calls to avoid a nonce-storm on a
        # backlog of accumulated orphans (e.g. first pass after enabling
        # the flag on a long-running bot). Excess orphans are deferred
        # to subsequent passes; reconcile is idempotent so the next pass
        # will pick them up.
        MAX_CANCELS_PER_PASS = 20
        # Backstop placements are wait=True broadcasts; cap them per pass so
        # first activation on a large config cannot park the loop past the
        # watchdog timeout (review finding 5). Reconcile is idempotent, so
        # later passes finish the job.
        MAX_BACKSTOPS_PER_PASS = 10
        backstops_placed = 0
        # Budget refunds require a fresh gateway (review finding 1).
        gateway_fresh = self._gateway_fresh()
        if gateway_fresh:
            self._fresh_probe_failures = 0
        else:
            self._fresh_probe_failures = getattr(
                self, "_fresh_probe_failures", 0
            ) + 1
            if self._fresh_probe_failures in (10, 100, 1000):
                logger.warning(
                    f"Gateway freshness probe has failed "
                    f"{self._fresh_probe_failures} consecutive passes; "
                    f"budget refunds are paused. If node_url does not serve "
                    f"/api/v1/health this pauses them FOREVER and markets "
                    f"will ratchet to their bid caps."
                )
        total_orphans = 0
        total_stale = 0
        # Orphans enumerated this pass and orphans past their attempt budget.
        pass_orphans: set[tuple[int, bool, int]] = set()
        stuck_orphans: set[tuple[int, bool, int]] = set()
        for query_id, context in self._markets.items():
            # Settled market (roller configs can carry a just-settled rung
            # for up to one roller tick): its book is gone or frozen, orphan
            # cancels against it are guaranteed failed txs, and the budget
            # has nothing left to true up. Skip entirely.
            # ALSO skip the PRE-SETTLEMENT CUTOFF window: the pull empties
            # the market by design with wait=False cancels and untracks
            # unconditionally, so a reconcile pass that reads the book
            # before those cancels confirm sees every resting order as an
            # orphan and double-cancels it - ~60-90 guaranteed not-found
            # failed txs around every daily ladder settle (the recurring
            # ~05:30-06:30 Eggs alert pages, 2026-09-22 diagnosis).
            settle_ts_mkt = context.config.settle_time
            if settle_ts_mkt is not None and time.time() >= (
                settle_ts_mkt - self.config.pre_settlement_cutoff
            ):
                continue
            bids: dict[bool, dict[int, int]] = {True: {}, False: {}}
            asks: dict[bool, dict[int, int]] = {True: {}, False: {}}
            # 2026-09-07 incident (#43): a failed read left bids/asks empty,
            # and the stale pass below then untracked EVERY order for that
            # outcome as if the chain had confirmed them gone. The quote
            # engine re-placed on top of the invisible resting orders each
            # pass until the wallet drained. An unreadable book is UNKNOWN,
            # not empty: skip both the untrack and orphan passes for any
            # outcome whose read failed.
            read_ok: dict[bool, bool] = {True: False, False: False}
            market_min_ask: dict[bool, int] = {}

            for outcome in (True, False):
                try:
                    entries = self._client.get_order_book(query_id, outcome)
                    read_ok[outcome] = True
                except Exception as exc:
                    logger.warning(
                        f"Periodic reconcile: get_order_book("
                        f"{query_id}, {outcome}) failed: {exc}; treating book "
                        f"as UNREADABLE (no untrack/orphan action this pass)"
                    )
                    continue
                for entry in entries:
                    raw_p = entry.get("price")
                    if raw_p is not None and raw_p > 0:
                        # Min ask across ALL wallets: the backstop cross-guard
                        # must see third-party asks too (re-review finding 2);
                        # the dicts below are wallet-filtered.
                        cur_min = market_min_ask.get(outcome)
                        if cur_min is None or raw_p < cur_min:
                            market_min_ask[outcome] = raw_p
                    owner = entry.get("wallet_address")
                    if owner is None:
                        continue
                    if isinstance(owner, (bytes, bytearray)):
                        owner_hex = "0x" + owner.hex()
                    else:
                        owner_hex = str(owner)
                    if owner_hex.lower() != wallet_addr:
                        continue
                    raw_price = entry.get("price")
                    amount = entry.get("amount", 0)
                    if raw_price is None:
                        continue
                    if raw_price < 0:
                        bids[outcome][abs(raw_price)] = (
                            bids[outcome].get(abs(raw_price), 0) + amount
                        )
                    elif raw_price > 0:
                        asks[outcome][raw_price] = (
                            asks[outcome].get(raw_price, 0) + amount
                        )

            # Build the set of (outcome, is_buy, price) the bot
            # considers its own. Untrack stale local entries.
            tracked_keys: set[tuple[bool, bool, int]] = set()
            tracked_orders = self._order_state.get_market_orders(query_id)
            stale = 0
            for tracked in tracked_orders:
                if not read_ok[tracked.outcome]:
                    # Book unreadable: keep the order tracked (see above).
                    tracked_keys.add(
                        (tracked.outcome, tracked.is_buy, tracked.price)
                    )
                    continue
                if tracked.is_buy:
                    is_active = tracked.price in bids[tracked.outcome]
                else:
                    is_active = tracked.price in asks[tracked.outcome]

                if is_active:
                    tracked_keys.add((tracked.outcome, tracked.is_buy, tracked.price))
                else:
                    stale += 1
                    self._order_state.untrack_order(
                        query_id, tracked.outcome, tracked.is_buy,
                        tracked.price, tracked.level_idx,
                    )

            # Orphan pass: anything in bids/asks for this wallet that is
            # not in tracked_keys is an orphan. Cancel it. Respect the
            # per-pass cap to avoid a nonce-storm.
            orphan_cancels = 0
            for outcome in (True, False):
                if total_orphans + orphan_cancels >= MAX_CANCELS_PER_PASS:
                    break
                for price in bids[outcome]:
                    if total_orphans + orphan_cancels >= MAX_CANCELS_PER_PASS:
                        break
                    if (outcome, True, price) in tracked_keys:
                        continue
                    if (
                        getattr(self.config, "backstop_amount", 0)
                        and price == self.config.backstop_price_cents
                        and bids[outcome][price] <= self.config.backstop_amount
                    ):
                        # Backstop bids are deliberately untracked by the
                        # quote engine; they are not orphans. Bounded by
                        # amount so incident-leftover bids that happen to
                        # rest at this price stay cancellable.
                        continue
                    akey = (query_id, outcome, -price)
                    pass_orphans.add(akey)
                    if (
                        self._reconcile_cancel_attempts.get(akey, 0)
                        >= ORPHAN_CANCEL_MAX_ATTEMPTS
                    ):
                        stuck_orphans.add(akey)
                        continue
                    try:
                        self._client.cancel_order(
                            query_id=query_id, outcome=outcome,
                            price=-price, wait=False,
                        )
                        orphan_cancels += 1
                        self._reconcile_cancel_attempts[akey] = (
                            self._reconcile_cancel_attempts.get(akey, 0) + 1
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Periodic reconcile: failed to cancel orphan "
                            f"bid qid={query_id} {'YES' if outcome else 'NO'} "
                            f"{price}c: {exc}"
                        )
                for price in asks[outcome]:
                    if total_orphans + orphan_cancels >= MAX_CANCELS_PER_PASS:
                        break
                    if (outcome, False, price) in tracked_keys:
                        continue
                    akey = (query_id, outcome, price)
                    pass_orphans.add(akey)
                    if (
                        self._reconcile_cancel_attempts.get(akey, 0)
                        >= ORPHAN_CANCEL_MAX_ATTEMPTS
                    ):
                        stuck_orphans.add(akey)
                        continue
                    try:
                        self._client.cancel_order(
                            query_id=query_id, outcome=outcome,
                            price=price, wait=False,
                        )
                        orphan_cancels += 1
                        self._reconcile_cancel_attempts[akey] = (
                            self._reconcile_cancel_attempts.get(akey, 0) + 1
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Periodic reconcile: failed to cancel orphan "
                            f"ask qid={query_id} {'YES' if outcome else 'NO'} "
                            f"{price}c: {exc}"
                        )

            # Budget truth-up + backstop maintenance, only from a fully
            # successful read of both outcomes (#43).
            if read_ok[True] and read_ok[False]:
                resting_bid_cents = sum(
                    price * amount
                    for outcome in (True, False)
                    for price, amount in bids[outcome].items()
                )
                self._bid_budget.sync(
                    query_id, resting_bid_cents, fresh=gateway_fresh
                )

                bs_amount = getattr(self.config, "backstop_amount", 0)
                if bs_amount and not self.config.read_only:
                    bs_price = self.config.backstop_price_cents
                    now_bs = int(time.time())
                    settle_ts = context.config.settle_time
                    cutoff_ok = (
                        settle_ts is None
                        or now_bs < settle_ts - self.config.pre_settlement_cutoff
                    )
                    if (
                        cutoff_ok
                        and query_id not in self._pre_settlement_pulled
                        and query_id not in self._earnings_pulled_session
                    ):
                        for outcome in (True, False):
                            if backstops_placed >= MAX_BACKSTOPS_PER_PASS:
                                break
                            if bs_price in bids[outcome]:
                                continue
                            # Never cross an ask (review finding 2): deep-OTM
                            # books legitimately quote 1-2c asks (our own
                            # included); a crossing backstop self-fills every
                            # pass, a perpetual drain. The book was read this
                            # pass; use it.
                            min_ask = market_min_ask.get(outcome)
                            if min_ask is not None and min_ask <= bs_price:
                                logger.info(
                                    f"Backstop skipped market {query_id} "
                                    f"{'YES' if outcome else 'NO'}: best ask "
                                    f"{min_ask}c <= backstop {bs_price}c "
                                    f"(any wallet)"
                                )
                                continue
                            if not self._bid_budget.try_reserve(
                                query_id, bs_price, bs_amount
                            ):
                                continue
                            try:
                                self._client.place_buy_order(
                                    query_id=query_id, outcome=outcome,
                                    price=bs_price, amount=bs_amount,
                                    wait=True,
                                )
                                backstops_placed += 1
                                logger.info(
                                    f"Backstop bid placed: market {query_id} "
                                    f"{'YES' if outcome else 'NO'} @{bs_price}c "
                                    f"x{bs_amount}"
                                )
                            except Exception as exc:
                                if self._is_definitive_rejection(exc):
                                    self._bid_budget.release(
                                        query_id, bs_price, bs_amount
                                    )
                                logger.warning(
                                    f"Backstop bid failed: market {query_id} "
                                    f"{'YES' if outcome else 'NO'}: {exc}"
                                )

            total_orphans += orphan_cancels
            total_stale += stale

        # Attempt bookkeeping. Prune entries for orphans that finally left
        # the book -- but only when the pass enumerated every book (a pass
        # that hit the cancel cap broke out early, so absence from
        # pass_orphans proves nothing on such a pass). A pruned key that
        # reappears gets a fresh attempt budget on purpose: a NEW order at
        # the same price is a new orphan.
        if total_orphans < MAX_CANCELS_PER_PASS:
            for akey in list(self._reconcile_cancel_attempts):
                if akey not in pass_orphans:
                    del self._reconcile_cancel_attempts[akey]

        if stuck_orphans:
            logger.error(
                f"Periodic reconcile: {len(stuck_orphans)} orphan(s) still "
                f"on the book after {ORPHAN_CANCEL_MAX_ATTEMPTS} cancel "
                f"attempts each; suppressing further cancels for them (the "
                f"chain rejects the cancel, or the book read is stale)"
            )
            if len(stuck_orphans) >= STUCK_ORPHANS_EXIT_THRESHOLD:
                logger.critical(
                    f"{len(stuck_orphans)} distinct orphans are "
                    f"uncancellable (stale-state pattern). Exiting "
                    f"{EXIT_STALE_CANCEL_LOOP} for a clean restart."
                )
                sys.exit(EXIT_STALE_CANCEL_LOOP)

        logger.info(
            f"Periodic reconcile pass: "
            f"{total_orphans} on-chain orphans cancelled, "
            f"{total_stale} stale local entries untracked, "
            f"{len(stuck_orphans)} stuck orphans suppressed across "
            f"{len(self._markets)} markets"
        )

    def _pre_mint_all_markets(self) -> None:
        """
        Walk every configured market and bring pair inventory up to the
        per-market `initial_mint_pairs` target via a one-time split-mint
        whose only auto-listed leg parks at an unreachable price.

        SDK semantics (per place_split_limit_order doc): mints `amount`
        YES+NO pairs from collateral and auto-lists the NO side at
        `100 - true_price`. Setting true_price=1 puts the NO leg at 99c
        — no rational counterparty buys NO at 99c on a sub-1.0-prior
        market — so any race between mint and cancel cannot bleed shares.

        Idempotent: per market the deficit is `target - paired_inventory()`
        clamped to 0. Subsequent restarts with full inventory mint nothing.
        Pre-mint is skipped if a market is within `pre_settlement_cutoff +
        300s` of settling (no point minting into a market we're about to
        liquidate).

        Skipped in dry_run / read_only modes (the SDK is either uninit'd or
        broadcasts are short-circuited, so split_limit_order would either
        crash or no-op silently — neither is what we want).

        Honors `pre_mint_max_total_collateral_usd` as a wallet circuit
        breaker: if total deficit across all markets would exceed this
        cap, abort startup before broadcasting any mint.
        """
        if self.config.dry_run or self.config.read_only:
            logger.info(
                "[%s] skipping pre-mint",
                "DRY RUN" if self.config.dry_run else "READ-ONLY",
            )
            return

        # Compute per-market deficits up front so we can apply the global
        # collateral cap before any broadcast.
        now_ts = int(time.time())
        # Skip markets that will start liquidating soon. The +300s buffer
        # over pre_settlement_cutoff is to keep the mint+cancel sequence
        # well clear of the liquidation window even if RPCs are slow.
        cutoff_buffer = self.config.pre_settlement_cutoff + 300
        deficits: dict[int, int] = {}
        total_deficit_pairs = 0
        for query_id, context in self._markets.items():
            target = context.config.initial_mint_pairs
            if not target or target <= 0:
                continue
            settle_time = context.config.settle_time
            if settle_time is not None and (settle_time - now_ts) <= cutoff_buffer:
                logger.info(
                    "Pre-mint skip market %d: settle_time=%d is within "
                    "%ds of cutoff",
                    query_id, settle_time, cutoff_buffer,
                )
                continue
            inv = self._inventory.get_market_inventory(query_id)
            paired = inv.paired_inventory()
            deficit = max(0, int(target) - int(paired))
            if deficit > 0:
                deficits[query_id] = deficit
                total_deficit_pairs += deficit

        if not deficits:
            logger.info("Pre-mint: no deficit across %d markets", len(self._markets))
            return

        cap = self.config.pre_mint_max_total_collateral_usd
        if cap is not None and total_deficit_pairs > cap:
            logger.error(
                "Pre-mint aborted: total deficit %d pairs ($%d) exceeds "
                "pre_mint_max_total_collateral_usd=%.2f. Either lower per-"
                "market initial_mint_pairs in orchestrator config or raise "
                "the cap if intentional.",
                total_deficit_pairs, total_deficit_pairs, cap,
            )
            raise RuntimeError("pre_mint_max_total_collateral_usd exceeded")

        park_price = int(self.config.avellaneda.pre_mint_listing_price_yes_cents)
        if not (1 <= park_price <= 99):
            logger.error("pre_mint_listing_price_yes_cents must be 1-99, got %d", park_price)
            return

        # Pre-flight log: gateway URL + estimated capital + park price. If
        # any of these don't match what the operator expected, this is the
        # last chance to ctrl-C before broadcasting.
        logger.info(
            "Pre-mint pre-flight: gateway=%s, deficit=%d pairs ($%d collateral) "
            "across %d markets, park price true_price=%d (auto-lists NO at %dc)",
            self.config.node_url, total_deficit_pairs, total_deficit_pairs,
            len(deficits), park_price, 100 - park_price,
        )

        for query_id, deficit in deficits.items():
            # Honor SIGTERM mid-pre-mint. Without this, an early shutdown
            # signal is queued but the pre-mint loop runs to completion,
            # which on 35 markets can take long enough for systemd to
            # SIGKILL the bot. Check between markets.
            if self._shutdown_requested:
                logger.info(
                    "Pre-mint interrupted by shutdown request after "
                    "broadcasting %d/%d markets",
                    len(deficits) - sum(1 for q in deficits if q >= query_id),
                    len(deficits),
                )
                break
            try:
                self._client.place_split_limit_order(
                    query_id=query_id,
                    true_price=park_price,
                    amount=deficit,
                    wait=True,
                )
            except Exception as e:
                logger.error(
                    "Pre-mint failed for market %d (deficit=%d pairs): %s. "
                    "Asks on this market will quote from existing inventory only.",
                    query_id, deficit, e,
                )
                continue

            # Cancel only the auto-listed NO leg at 100-park_price. The
            # SDK does NOT auto-list a YES leg here (split-mint lists the
            # "unwanted side" only — see place_split_limit_order doc).
            # An earlier draft also cancelled YES at park_price; that
            # call always failed (no such order existed) and was a wasted
            # round-trip.
            try:
                self._client.cancel_order(
                    query_id=query_id,
                    outcome=False,
                    price=100 - park_price,
                    wait=True,
                )
            except Exception as e:
                logger.warning(
                    "Pre-mint cancel of auto-listed NO@%dc on market %d "
                    "failed: %s. Leg may sit on book until next refresh; "
                    "since park price is unreachable, fill risk is low.",
                    100 - park_price, query_id, e,
                )

            logger.info(
                "Pre-minted %d pairs on market %d (NO@%dc auto-list cancelled)",
                deficit, query_id, 100 - park_price,
            )

        # Refresh inventory once so the new shares show up in available()
        # before any ASK placement runs.
        self._refresh_inventory()

    def _refresh_inventory(self) -> None:
        """Refresh inventory from user positions."""
        if self.config.dry_run:
            # Use empty positions in dry-run mode
            self._inventory.update_from_user_positions([])
            self._last_inventory_refresh = time.time()
            logger.debug("[DRY RUN] Using empty inventory")
            return

        try:
            positions = self._client.get_user_positions()
            self._inventory.update_from_user_positions(positions)
            self._last_inventory_refresh = time.time()
            logger.debug(f"Refreshed inventory from {len(positions)} positions")
        except Exception as e:
            logger.error(f"Failed to refresh inventory: {e}")
            self.stats.errors += 1

    def _update_order_book(
        self, context: MarketContext, outcome: bool
    ) -> bool:
        """
        Update order book state for a market outcome.

        Args:
            context: Market context
            outcome: True for YES, False for NO

        Returns:
            True if update successful
        """
        try:
            if self.config.dry_run:
                # Use mock order book data in dry-run mode
                # Simulate a market with mid price around 50 cents
                mock_entries = [
                    {"price": -48, "amount": 100},  # Bid at 48 cents
                    {"price": 52, "amount": 100},   # Ask at 52 cents
                ]
                entries = mock_entries
                logger.debug(f"[DRY RUN] Using mock order book for market {context.query_id}")
            else:
                entries = self._client.get_order_book(context.query_id, outcome)

            state = build_market_state(
                query_id=context.query_id,
                outcome=outcome,
                order_book_entries=entries,
            )
            context.set_state(outcome, state)
            context.last_order_book_update = time.time()

            # Update volatility indicator if we have mid price
            if state.mid_price is not None:
                self._volatility_tracker.add_sample(
                    context.query_id, outcome, state.mid_price
                )

            # Update depth tracker
            self._depth_tracker.update(
                context.query_id,
                outcome,
                state.bid_levels,
                state.ask_levels,
                state.mid_price,
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to update order book for market {context.query_id} "
                f"outcome={outcome}: {e}"
            )
            self.stats.errors += 1
            return False

    def _calculate_prices(
        self, context: MarketContext, outcome: bool
    ) -> Optional[PricingResult]:
        """
        Calculate optimal bid/ask prices using Avellaneda-Stoikov.

        Args:
            context: Market context
            outcome: True for YES, False for NO

        Returns:
            PricingResult or None if calculation not possible
        """
        # Get mid price (from order book or Black-Scholes fallback)
        mid_price = context.get_mid_price(outcome, self.config.pricing_source)
        if mid_price is None:
            logger.warning(
                f"No mid price available for market {context.query_id} "
                f"outcome={outcome}"
            )
            return None

        # Get volatility
        vol_estimate = self._volatility_tracker.get_volatility(
            context.query_id, outcome
        )

        # Get kappa
        if self.config.avellaneda.use_dynamic_kappa:
            kappa = self._depth_tracker.get_kappa(context.query_id, outcome)
        else:
            kappa = self.config.avellaneda.default_kappa

        # Get inventory skew
        inventory_skew = self._inventory.get_inventory_skew(
            context.query_id, outcome, mid_price
        )

        # Get market-specific overrides
        gamma = context.config.gamma or self.config.avellaneda.risk_factor

        # Calculate min_spread: config is percentage of mid price
        # Convert to cents for binary options
        min_spread_pct = context.config.min_spread or self.config.avellaneda.min_spread
        min_spread_cents = mid_price * (min_spread_pct / 100.0) if min_spread_pct > 0 else 0.0

        # Derive Avellaneda time horizon from settle_time when available
        time_horizon_override = None
        if context.config.settle_time:
            seconds_left = max(context.config.settle_time - int(time.time()), 3600)
            time_horizon_override = seconds_left / (365.25 * 86400)

        # Calculate prices
        result = self._pricing.calculate_from_config(
            mid_price=mid_price,
            inventory_skew=inventory_skew,
            volatility=vol_estimate.value,
            kappa=kappa,
            gamma_override=gamma,
            min_spread_override=min_spread_cents,
            time_horizon_override=time_horizon_override,
        )

        logger.debug(
            f"Market {context.query_id} {('YES' if outcome else 'NO')}: "
            f"mid={mid_price:.1f} vol={vol_estimate.value:.2f}({vol_estimate.source}) "
            f"κ={kappa:.3f} q={inventory_skew:.2f} "
            f"→ bid={result.bid_price:.1f} ask={result.ask_price:.1f}"
        )

        return result

    def _apply_eta_transformation(
        self, base_amount: int, inventory_skew: float, is_buy: bool
    ) -> int:
        """
        Apply eta transformation to order amount.

        From the Avellaneda-Stoikov paper, eta controls asymmetric order sizing
        based on inventory. When we have excess inventory (q > 0), we want to
        reduce buy order sizes. When we have deficit (q < 0), we reduce sell
        order sizes.

        Formula: size * exp(-eta * q) for orders going against inventory target

        Args:
            base_amount: Original order amount
            inventory_skew: q value (-1 to +1), positive = excess inventory
            is_buy: True for buy orders, False for sell orders

        Returns:
            Adjusted order amount
        """
        eta = self.config.avellaneda.order_amount_shape_factor
        if eta <= 0:
            return base_amount

        # Apply eta transformation only for orders against inventory target
        # q > 0 (excess inventory) → reduce buy size
        # q < 0 (deficit inventory) → reduce sell size
        if is_buy and inventory_skew > 0:
            adjusted = base_amount * math.exp(-eta * inventory_skew)
        elif not is_buy and inventory_skew < 0:
            adjusted = base_amount * math.exp(eta * inventory_skew)  # note: q is negative
        else:
            adjusted = base_amount

        return max(1, int(round(adjusted)))

    def _apply_order_optimization(
        self,
        context: MarketContext,
        outcome: bool,
        bid_price: int,
        ask_price: int,
    ) -> Tuple[int, int]:
        """
        Apply order optimization - cap prices at best bid+1 / best ask-1.

        When enabled, prevents placing orders too aggressively:
        - Buy orders are capped at best_bid + 1 (don't overpay)
        - Sell orders are floored at best_ask - 1 (don't undersell)

        This matches Hummingbot's order_optimization behavior.

        Args:
            context: Market context
            outcome: True for YES, False for NO
            bid_price: Proposed bid price
            ask_price: Proposed ask price

        Returns:
            Tuple of (optimized_bid, optimized_ask)
        """
        if not self.config.avellaneda.order_optimization_enabled:
            return bid_price, ask_price

        state = context.get_state(outcome)
        if state is None:
            return bid_price, ask_price

        optimized_bid = bid_price
        optimized_ask = ask_price

        # For buys: If our bid price > best_bid + 1, cap it at best_bid + 1
        # This prevents us from paying more than 1 tick above the best bid
        if state.best_bid is not None:
            price_above_bid = state.best_bid + 1
            if bid_price > price_above_bid:
                optimized_bid = price_above_bid

        # For sells: If our ask price < best_ask - 1, raise it to best_ask - 1
        # This prevents us from selling for less than 1 tick below the best ask
        if state.best_ask is not None:
            price_below_ask = state.best_ask - 1
            if ask_price < price_below_ask:
                optimized_ask = price_below_ask

        # Clamp to valid range
        optimized_bid = max(1, min(99, optimized_bid))
        optimized_ask = max(1, min(99, optimized_ask))

        # Ensure bid < ask
        if optimized_bid >= optimized_ask:
            # Revert to original prices
            return bid_price, ask_price

        return optimized_bid, optimized_ask

    def _apply_transaction_costs(
        self, bid_price: int, ask_price: int, fee_pct: float = 0.0
    ) -> Tuple[int, int]:
        """
        Apply transaction costs to order prices.

        When enabled, adjusts prices to account for trading fees:
        - Buy price reduced by fee percentage
        - Sell price increased by fee percentage

        Args:
            bid_price: Proposed bid price
            ask_price: Proposed ask price
            fee_pct: Fee percentage (e.g., 0.1 for 0.1%)

        Returns:
            Tuple of (adjusted_bid, adjusted_ask)
        """
        if not self.config.avellaneda.add_transaction_costs or fee_pct <= 0:
            return bid_price, ask_price

        # Reduce bid price by fee
        adjusted_bid = int(bid_price * (1 - fee_pct / 100))
        # Increase ask price by fee
        adjusted_ask = int(math.ceil(ask_price * (1 + fee_pct / 100)))

        # Clamp to valid range
        adjusted_bid = max(1, min(98, adjusted_bid))
        adjusted_ask = max(2, min(99, adjusted_ask))

        # Ensure bid < ask
        if adjusted_bid >= adjusted_ask:
            return bid_price, ask_price

        return adjusted_bid, adjusted_ask

    def _create_order_levels(
        self, base_bid: int, base_ask: int, optimal_spread: float
    ) -> List[Tuple[int, int]]:
        """
        Create multiple order levels at different price points.

        When order_levels > 1, creates orders at progressively wider spreads.

        Args:
            base_bid: Base bid price (level 0)
            base_ask: Base ask price (level 0)
            optimal_spread: Optimal spread for calculating level distances

        Returns:
            List of (bid, ask) tuples for each level
        """
        order_levels = self.config.avellaneda.order_levels
        if order_levels <= 1:
            return [(base_bid, base_ask)]

        level_distances_pct = self.config.avellaneda.level_distances
        # Ensure at least 1 cent per level (prices are integers 1-99)
        level_step = max(1, int(round((optimal_spread / 2) * (level_distances_pct / 100))))
        gamma_mult = self.config.avellaneda.level_gamma_multiplier

        levels = []
        seen_bids: set[int] = set()
        seen_asks: set[int] = set()
        for i in range(order_levels):
            if i == 0:
                level_offset = 0
            else:
                # Each level gets progressively wider spread via gamma scaling
                level_offset = int(round(level_step * i * (gamma_mult ** i)))
            bid = max(1, base_bid - level_offset)
            ask = min(99, base_ask + level_offset)

            if bid < ask:  # Only add valid levels
                # Skip levels where bid or ask duplicates a previous level
                if bid in seen_bids or ask in seen_asks:
                    logger.debug(
                        f"Skipping duplicate order level {i}: bid={bid} ask={ask}"
                    )
                    continue
                seen_bids.add(bid)
                seen_asks.add(ask)
                levels.append((bid, ask))

        return levels if levels else [(base_bid, base_ask)]

    def _should_delay_after_fill(
        self, context: MarketContext, outcome: bool
    ) -> bool:
        """
        Check if we should delay order placement after a recent fill.

        Args:
            context: Market context
            outcome: True for YES, False for NO

        Returns:
            True if we should delay
        """
        key = (context.query_id, outcome)
        last_fill = self._last_fill_time.get(key, 0)
        delay = self.config.avellaneda.filled_order_delay

        return time.time() - last_fill < delay

    def _record_fill(self, context: MarketContext, outcome: bool) -> None:
        """Record a fill event for delay tracking."""
        key = (context.query_id, outcome)
        self._last_fill_time[key] = time.time()

    def _execute_order_updates(
        self, context: MarketContext, outcome: bool, pricing: PricingResult
    ) -> None:
        """
        Execute order placements/updates based on pricing result.

        Applies the following transformations in order:
        1. Check filled order delay
        2. Apply eta transformation to order amounts
        3. Apply order optimization (jump to best bid+1 / best ask-1)
        4. Apply transaction costs
        5. Create multiple order levels if configured

        Args:
            context: Market context
            outcome: True for YES, False for NO
            pricing: Calculated prices
        """
        # Check if we should delay after a recent fill
        if self._should_delay_after_fill(context, outcome):
            logger.debug(
                f"Market {context.query_id}: delaying orders after recent fill"
            )
            return

        bid_price, ask_price = pricing.to_int_prices()
        order_mgr = OrderManager(
            context,
            refresh_tolerance_pct=self.config.avellaneda.order_refresh_tolerance_pct,
            max_order_age=self.config.avellaneda.max_order_age,
        )
        # Static (share-count) base amount; only used when order_dollar_amount
        # is unset. With order_dollar_amount set, base is recomputed per leg
        # below using each level's price so each individual placed order
        # targets a fixed dollar notional.
        base_amount = context.config.order_amount

        # Apply order optimization (jump to best bid+1 / best ask-1)
        bid_price, ask_price = self._apply_order_optimization(
            context, outcome, bid_price, ask_price
        )

        # Apply transaction costs (currently no fee info available, placeholder)
        # In practice, this would use the actual fee from the exchange
        bid_price, ask_price = self._apply_transaction_costs(
            bid_price, ask_price, fee_pct=0.0
        )

        # Defensive self-match check. After all the optimization+transaction
        # steps above, verify bid < ask within this (market, outcome). The
        # pricing module already widens crossing extremes inline (see
        # avellaneda.py around the bid_price/ask_price clamp + recovery), but
        # apply a belt-and-suspenders gate here in case a downstream step
        # (rounding, order optimization, fee adjustment) crosses the prices.
        # Skip the whole update for this outcome rather than place a
        # self-crossing pair.
        if bid_price >= ask_price:
            logger.warning(
                f"Market {context.query_id} outcome={outcome}: bid {bid_price}c "
                f">= ask {ask_price}c after all adjustments. Skipping update to "
                f"avoid self-matching."
            )
            return

        # Create order levels
        order_levels = self._create_order_levels(
            bid_price, ask_price, pricing.optimal_spread
        )

        # Execute orders for each level
        for level_idx, (level_bid, level_ask) in enumerate(order_levels):
            # Write heartbeat during long placement cycles to prevent orchestrator kills
            if level_idx > 0 and level_idx % 5 == 0:
                self._write_heartbeat()

            # Per-leg base sizing. With order_dollar_amount unset this is just
            # the configured share count (flat across levels). With it set, the
            # base is rescaled to each level's PRICE so each placed order has
            # the configured dollar notional on the order book (price * amount).
            #
            # Both legs use the leg price directly. For ASKs at extreme priors
            # this produces small share counts that may fail the protocol's
            # split-mint min-notional check on the low leg (split_price *
            # amount < 100 cent-shares); those orders are skipped at the place
            # call site rather than inflating amount to clear min, which would
            # otherwise blow notional far past the user's $X target.
            bid_base = _compute_base_amount(context.config, level_bid)
            ask_base = _compute_base_amount(context.config, level_ask)
            bid_amt = self._apply_eta_transformation(
                bid_base, pricing.inventory_skew, is_buy=True
            )
            ask_amt = self._apply_eta_transformation(
                ask_base, pricing.inventory_skew, is_buy=False
            )

            # Track order pair for hanging orders (first level only)
            buy_order_info = None
            sell_order_info = None

            # Update bid
            buy_result = self._update_single_order(
                context, outcome, Side.BID, level_bid, bid_amt, order_mgr, level_idx
            )
            if buy_result and self.config.avellaneda.hanging_orders_enabled:
                buy_order_info = HangingOrder(
                    order_id=buy_result,
                    query_id=context.query_id,
                    outcome=outcome,
                    is_buy=True,
                    price=level_bid,
                    amount=bid_amt,
                    creation_timestamp=time.time(),
                )

            # Update ask
            sell_result = self._update_single_order(
                context, outcome, Side.ASK, level_ask, ask_amt, order_mgr, level_idx
            )
            if sell_result and self.config.avellaneda.hanging_orders_enabled:
                sell_order_info = HangingOrder(
                    order_id=sell_result,
                    query_id=context.query_id,
                    outcome=outcome,
                    is_buy=False,
                    price=level_ask,
                    amount=ask_amt,
                    creation_timestamp=time.time(),
                )

            # Register pair for hanging order tracking
            if self.config.avellaneda.hanging_orders_enabled:
                if buy_order_info or sell_order_info:
                    tracker = self._get_hanging_tracker(context.query_id, outcome)
                    tracker.add_order_pair(buy_order_info, sell_order_info)

        context.last_order_refresh = time.time()

    def _place_ask(
        self,
        context: MarketContext,
        outcome: bool,
        new_price: int,
        amount: int,
    ) -> tuple[Optional[str], bool]:
        """
        Place an ASK from existing held inventory.

        Returns (tx_hash, is_inventory_backed). On a skip returns
        (None, False) and the caller is expected to abort that order.

        Inventory path (preferred): when paired YES+NO inventory is on hand
        and the visible (price, amount) clears the protocol's per-order
        min-notional, place a single-leg place_sell_order at the actual
        quote price and reserve the consumed shares so subsequent levels in
        the same cycle don't double-book them.

        No fallback when inventory is short: the ask is skipped. The former
        split-mint fallback (mint pairs, keep the auto-listed opposite leg at
        100 - price, sell this outcome at price) never rested: the chain
        matches a YES sell at p against a NO sell at 100 - p as a burn, so
        the sell redeemed its own auto-listed leg on arrival, and where one
        of our opposite bids sat at or above that leg it took part of it
        first (a self-trade). The bot tracked both legs anyway, and
        cancelling them failed on chain as "order not found" (2026-09-24:
        22 failed cancels in one pre-settlement pull). Skipping leaves the
        real book as it was and stops the wasted and failed transactions.
        """
        inv = self._inventory.get_market_inventory(context.query_id)
        available = inv.available_for_sell(outcome)

        if available >= amount and _meets_min_notional(new_price, amount):
            try:
                tx_hash = self._client.place_sell_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    price=new_price,
                    amount=amount,
                    wait=True,
                )
            except Exception as e:
                logger.error(
                    f"Inventory-backed sell failed (qid={context.query_id} "
                    f"outcome={'YES' if outcome else 'NO'} {new_price}c x{amount}): {e}"
                )
                raise
            inv.reserve_pair(outcome, amount)
            logger.info(
                f"Inventory-backed ask qid={context.query_id} "
                f"outcome={'YES' if outcome else 'NO'} {new_price}c x{amount} "
                f"(avail before/after: {available}/{available - amount})"
            )
            return tx_hash, True

        logger.info(
            f"Market {context.query_id} ask "
            f"outcome={'YES' if outcome else 'NO'} @{new_price}c x{amount}: "
            f"skip (inventory available={available})"
        )
        return None, False

    def _leg_still_on_book(
        self, query_id: int, outcome: bool, price: int
    ) -> bool:
        """
        Return True if an ASK at ``price`` for (query_id, outcome) is still
        resting on chain. Used by _cancel_ask to distinguish a split-mint leg
        that is genuinely gone (self-filled on placement, or lifted by a taker
        between refreshes) from a real cancel failure.

        Asks are stored with positive prices, so a resting leg matches the
        cancel price exactly. On any error querying the book, return True
        (conservative: treat as still-resting so the caller raises and
        reconcile picks it up), preserving pre-existing behavior.
        """
        try:
            entries = self._client.get_order_book(query_id, outcome)
        except Exception:
            return True
        for e in entries:
            try:
                if int(e.get("price")) == price:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _is_definitive_rejection(exc: BaseException) -> bool:
        """True only for errors that prove the order NEVER rested (so its
        budget reservation may be released). A timeout / dropped connection /
        unconfirmed-tx error is UNKNOWN - the broadcast may still land - and
        must keep the reservation (review finding 4); the next fresh
        reconcile sync trues it up."""
        s = str(exc).lower()
        return (
            "insufficient balance" in s
            or "below min" in s
            or "must be" in s
            or ("invalid" in s and "nonce" not in s)
        )

    def _gateway_fresh(self) -> bool:
        """One cheap health probe: is the gateway serving CURRENT state?
        Used to gate budget refunds (review finding 1) - a stale replica
        answers reads with old snapshots while returning 200s. Any failure
        counts as not-fresh."""
        try:
            import urllib.request as _ur
            url = self.config.node_url.rstrip("/") + "/api/v1/health"
            with _ur.urlopen(url, timeout=5) as resp:
                h = json.loads(resp.read())
            u = h.get("services", {}).get("user", {})
            return bool(u.get("healthy")) and float(
                u.get("block_age", 9e9)
            ) < 120_000  # ms
        except Exception:
            return False

    @staticmethod
    def _is_cancel_not_found(exc: BaseException) -> bool:
        """True if a chain cancel failure says the order is not on the book.

        The node's own error text is decisive: unlike a get_order_book
        recheck it cannot be stale.
        """
        return CANCEL_NOT_FOUND_SNIPPET in str(exc).lower()

    def _note_cancel_not_found(self, query_id: int, detail: str) -> None:
        """Record an observed not-found cancel failure; exit for a clean
        restart if they storm (stale in-process state has no in-process
        fix, and occasional single not-founds are normal self-fill races).
        """
        now = time.time()
        self._cancel_not_found_times.append(now)
        while (
            self._cancel_not_found_times
            and now - self._cancel_not_found_times[0]
            > CANCEL_NOT_FOUND_WINDOW_SEC
        ):
            self._cancel_not_found_times.popleft()
        count = len(self._cancel_not_found_times)
        logger.warning(
            f"Market {query_id}: cancel target already gone on chain "
            f"({detail}); treating as cancelled ({count} not-found "
            f"cancels in the last {CANCEL_NOT_FOUND_WINDOW_SEC:.0f}s)"
        )
        if count >= CANCEL_NOT_FOUND_EXIT_COUNT:
            logger.critical(
                f"{count} 'order not found' cancel failures within "
                f"{CANCEL_NOT_FOUND_WINDOW_SEC:.0f}s: local state or reads "
                f"are stale beyond in-process repair. "
                f"Exiting {EXIT_STALE_CANCEL_LOOP} for a clean restart."
            )
            sys.exit(EXIT_STALE_CANCEL_LOOP)

    def _note_level_not_found_clear(
        self, query_id: int, outcome: bool, side: Side, price: int
    ) -> None:
        """Record a not-found CLEAR on one exact quote slot; trip a per-slot
        cooldown when they repeat inside the window.

        Distinct from _note_cancel_not_found (process-wide storm counter):
        this one targets the single-slot placement/clear loop where two
        levels fight over one chain price. A restart does not fix that
        loop; refusing to re-quote the contested slot does.
        NOT reset on successful placement -- the loop's signature is
        alternating success/not-found, so a success-reset would blind it.
        """
        threshold = getattr(self.config, "level_loop_threshold", 4)
        if threshold <= 0:
            return
        window = getattr(self.config, "level_loop_window", 120.0)
        key = (query_id, outcome, side == Side.BID, price)
        now = time.time()
        times = self._level_not_found_times.setdefault(key, deque())
        times.append(now)
        while times and now - times[0] > window:
            times.popleft()
        if len(times) >= threshold:
            cooldown = getattr(self.config, "level_loop_cooldown", 300.0)
            self._level_cooldown_until[key] = now + cooldown
            times.clear()
            logger.error(
                f"LEVEL LOOP BREAKER: market {query_id} "
                f"{'YES' if outcome else 'NO'} {side.value} @{price}c hit "
                f"{threshold} 'order not found' clears in {window:.0f}s; "
                f"suppressing quotes on this slot for {cooldown:.0f}s"
            )

    def _level_slot_cooling(
        self, query_id: int, outcome: bool, side: Side, price: int
    ) -> bool:
        """True if the loop breaker is suppressing this exact quote slot."""
        key = (query_id, outcome, side == Side.BID, price)
        until = self._level_cooldown_until.get(key)
        if until is None:
            return False
        if time.time() >= until:
            del self._level_cooldown_until[key]
            return False
        logger.info(
            f"Market {query_id} {side.value}: skip @{price}c, level loop "
            f"breaker cooling this slot for another "
            f"{until - time.time():.0f}s"
        )
        return True

    def _cancel_ask(
        self,
        context: MarketContext,
        outcome: bool,
        price: int,
        amount: int,
        is_inventory_backed: bool,
        wait: bool = True,
    ) -> None:
        """
        Cancel an ASK previously placed by the bot. The inventory path
        listed only the single side at the quote price. A legacy split-mint
        ask (retired 2026-09-24, may still be recovered from a state file)
        listed two legs, but only its YES sell leg at split_price can still
        be resting; see the comment in the split branch below.

        Behavior on chain-cancel failure:
        - wait=True: propagate the exception. Caller (refresh path or
          synchronous shutdown) decides whether to untrack/release. This
          is critical for keeping bot state consistent with chain when a
          cancel fails — otherwise an order on chain would be silently
          marked gone in the bot's state and its inventory reservation
          released, leaving the next placement free to double-list the
          same shares (silent-revert pattern from PR#13).
        - wait=False: best-effort fire-and-forget; failures are swallowed
          because we cannot distinguish broadcast-failed-on-chain from
          confirmed-cancelled when not waiting. Reservation is released
          regardless. Caller is expected to be a bulk cancel (off-hours /
          shutdown) with reconcile-on-startup handling any orphans.

        wait=True is required when this cancel is followed by a new ASK
        placement against the same shares — otherwise the new place
        broadcasts before the chain has accepted the cancel, and the
        second order silently reverts on insufficient inventory.
        """
        inv = self._inventory.get_market_inventory(context.query_id)
        if is_inventory_backed:
            try:
                self._client.cancel_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    price=price,
                    wait=wait,
                )
            except Exception as exc:
                if wait and not self._is_cancel_not_found(exc):
                    # Do not release the reservation: the order may still
                    # be on chain; let the caller's exception handler skip
                    # untrack so reconcile picks it up next cycle.
                    raise
                if wait:
                    # "Order not found": the chain says the order is gone
                    # (filled, or already cancelled). Retrying it forever
                    # produces a failed-tx stream; treat as cancelled and
                    # let the periodic inventory refresh absorb any drift
                    # from a fill.
                    self._note_cancel_not_found(
                        context.query_id,
                        f"ask {'YES' if outcome else 'NO'} {price}c",
                    )
                # wait=False: best-effort, fall through to release.
            inv.release_pair(outcome, amount)
        else:
            # Legacy split-mint ask recovered from state. Its auto-listed leg
            # (NO @ 100 - split_price) was always consumed on placement,
            # burned by the YES sell or taken by one of our NO bids, so
            # cancelling it only produced a guaranteed "order not found"
            # failed tx. Only the YES sell leg can still be resting.
            split_price = price if outcome else (100 - price)
            any_failed = False
            for cancel_out, cancel_p in [(True, split_price)]:
                try:
                    self._client.cancel_order(
                        query_id=context.query_id,
                        outcome=cancel_out,
                        price=cancel_p,
                        wait=wait,
                    )
                except Exception as exc:
                    # A cancel can fail because the leg is genuinely gone (it
                    # self-filled on placement, or was lifted by a taker
                    # between refreshes), not only because of a real chain
                    # error. The chain's own "order not found" is decisive
                    # and beats any book recheck (a stale read can keep a
                    # dropped leg "on book"). Otherwise re-check the book:
                    # only a leg still resting on chain is a true failure
                    # worth surfacing (and leaving for reconcile). On any
                    # doubt (still resting, or the recheck itself errors)
                    # fall back to the prior behavior and raise.
                    if self._is_cancel_not_found(exc):
                        self._note_cancel_not_found(
                            context.query_id,
                            f"split leg {'YES' if cancel_out else 'NO'} "
                            f"{cancel_p}c",
                        )
                    elif wait and self._leg_still_on_book(
                        context.query_id, cancel_out, cancel_p
                    ):
                        any_failed = True
            if any_failed and wait:
                raise RuntimeError(
                    f"split-mint cancel partially failed for "
                    f"qid={context.query_id} outcome={outcome} price={price}; "
                    f"orders may remain on book and reconcile will recover them"
                )

    def _update_single_order(
        self,
        context: MarketContext,
        outcome: bool,
        side: Side,
        new_price: int,
        amount: int,
        order_mgr: OrderManager,
        level_idx: int = 0,
    ) -> Optional[str]:
        """
        Update a single order (bid or ask).

        Uses atomic change_bid/change_ask when possible, otherwise
        cancels and places new order.

        Args:
            context: Market context
            outcome: True for YES, False for NO
            side: Order side
            new_price: New price in cents (1-99)
            amount: Order amount
            order_mgr: Order manager instance
            level_idx: Order level index (0 = tightest spread)

        Returns:
            Order ID (tx_hash) if order was placed/updated, None otherwise
        """
        should_update, reason = order_mgr.should_update_order(
            outcome, side, new_price, level_idx
        )

        if not should_update:
            logger.debug(
                f"Market {context.query_id} {side.value} L{level_idx}: no update needed ({reason})"
            )
            return None

        # Position limit enforcement: skip bids when at max inventory
        if side == Side.BID:
            inv = self._inventory.get_market_inventory(context.query_id)
            current_shares = inv.yes_shares if outcome else inv.no_shares
            max_pos = self.config.avellaneda.max_position_per_outcome
            if max_pos > 0 and current_shares >= max_pos:
                logger.warning(
                    f"Market {context.query_id}: position limit reached "
                    f"({current_shares}/{max_pos}), skipping bid"
                )
                return None

        current_order = order_mgr.get_current_order(outcome, side, level_idx)

        # Slot-collision guard: the chain keys orders by (wallet, outcome,
        # signed price), so acting at a price another level is tracked at
        # would move/clobber THAT level's on-chain order and leave its
        # tracking stale -- its next change hits "Old order not found",
        # clears, re-places, and the two levels loop (Eggs 72c, 09-09).
        # Skip this level for the cycle; the grid separates as pricing moves.
        owner_lvl = order_mgr.level_owning_price(
            outcome, side, new_price, level_idx
        )
        skip_key = (context.query_id, outcome, side == Side.BID, level_idx)
        if owner_lvl is not None:
            # Level-inversion healer: when this level's target price is the
            # OWNER's tracked price and the pair is tracked strictly out of
            # order (asks must ascend with level, bids descend), the labels
            # are swapped relative to what pricing wants and the guard would
            # block both directions forever (observed: 17 mag7 ask pairs
            # wedged for days, 2026-09-15). Levels are bot-local bookkeeping
            # - the chain keys orders by price - so exchanging the two
            # records heals the wedge with zero chain writes. A correctly
            # ordered pair (the normal transient grid shift) is never
            # touched, and a swap strictly sorts the pair so it cannot
            # flap; longer cycles sort pairwise over successive passes.
            owner_order = order_mgr.get_current_order(outcome, side, owner_lvl)
            if (
                current_order is not None
                and owner_order is not None
                and _pair_out_of_order(
                    side, level_idx, current_order.price,
                    owner_lvl, owner_order.price,
                )
            ):
                order_mgr.swap_levels(outcome, side, level_idx, owner_lvl)
                is_buy = side == Side.BID
                self._order_state.untrack_order(
                    context.query_id, outcome, is_buy,
                    current_order.price, level_idx,
                )
                self._order_state.untrack_order(
                    context.query_id, outcome, is_buy,
                    owner_order.price, owner_lvl,
                )
                self._order_state.track_order(
                    query_id=context.query_id, outcome=outcome, is_buy=is_buy,
                    price=owner_order.price, amount=owner_order.amount,
                    order_id=owner_order.tx_hash, level_idx=level_idx,
                    is_inventory_backed=owner_order.is_inventory_backed,
                )
                self._order_state.track_order(
                    query_id=context.query_id, outcome=outcome, is_buy=is_buy,
                    price=current_order.price, amount=current_order.amount,
                    order_id=current_order.tx_hash, level_idx=owner_lvl,
                    is_inventory_backed=current_order.is_inventory_backed,
                )
                self._slot_guard_skips.pop(skip_key, None)
                self._slot_guard_skips.pop(
                    (context.query_id, outcome, is_buy, owner_lvl), None
                )
                logger.warning(
                    f"LEVEL RELABEL: market {context.query_id} {side.value} "
                    f"L{level_idx}@{current_order.price}c <-> "
                    f"L{owner_lvl}@{owner_order.price}c were tracked out of "
                    f"order; swapped labels locally (no chain action). "
                    f"Quoting resumes next cycle."
                )
                return None
            skips = self._slot_guard_skips.get(skip_key, 0) + 1
            self._slot_guard_skips[skip_key] = skips
            # Benign hold vs real stall. When this level HAS a resting quote
            # and the pair is correctly ordered, the blocked target is
            # already quoted by the neighbor and our own quote rests
            # consistently beside it: the book covers the proposed grid,
            # nothing is broken. A saturated grid near the 1c/99c caps parks
            # here for hours (market 787 asks resting 97/98/99 vs proposals
            # 98/99, 2026-09-16) - that must not page. The ERROR escalation
            # is reserved for a level that cannot PLACE at all (no resting
            # quote, book thinner than designed) or a state the inversion
            # healer above did not recognize.
            benign_hold = current_order is not None
            if benign_hold:
                if skips % SLOT_GUARD_STALL_ERROR_EVERY == 0:
                    logger.info(
                        f"SLOT GUARD HOLD: market {context.query_id} "
                        f"{side.value} L{level_idx} holding @"
                        f"{current_order.price}c for {skips} cycles (target "
                        f"@{new_price}c already quoted by L{owner_lvl}; book "
                        f"covers the proposed grid)"
                    )
                else:
                    logger.debug(
                        f"Market {context.query_id} {side.value} "
                        f"L{level_idx}: hold @{current_order.price}c, target "
                        f"@{new_price}c quoted by L{owner_lvl}"
                    )
            elif skips % SLOT_GUARD_STALL_ERROR_EVERY == 0:
                logger.error(
                    f"SLOT GUARD STALL: market {context.query_id} "
                    f"{side.value} L{level_idx} has skipped {skips} "
                    f"consecutive cycles and has NO resting quote (target "
                    f"@{new_price}c owned by L{owner_lvl}); the book is "
                    f"thinner than designed"
                )
            else:
                logger.info(
                    f"Market {context.query_id} {side.value} L{level_idx}: "
                    f"skip @{new_price}c, slot owned by L{owner_lvl} (chain "
                    f"keys orders by price; acting would corrupt that level)"
                )
            return None
        self._slot_guard_skips.pop(skip_key, None)

        # Level loop breaker: a slot that recently cleared "order not found"
        # repeatedly is in a placement/clear loop; stop feeding it.
        if self._level_slot_cooling(context.query_id, outcome, side, new_price):
            return None

        if self.config.dry_run:
            logger.info(
                f"[DRY RUN] Market {context.query_id} {side.value}: "
                f"would {'update' if current_order else 'place'} "
                f"@{new_price}¢ x{amount} ({reason})"
            )
            return None

        try:
            if current_order is not None:
                # Skip if price hasn't actually changed
                if current_order.price == new_price:
                    logger.debug(
                        f"Market {context.query_id} {side.value} L{level_idx}: "
                        f"price unchanged at {new_price}¢, skipping"
                    )
                    return None

                # Use atomic update
                old_sdk_price = convert_price_for_order(current_order.price, side)
                new_sdk_price = convert_price_for_order(new_price, side)

                if side == Side.BID:
                    # change_bid has the same silent-failure mode as
                    # place_buy_order: if new_price * amount falls below the
                    # protocol minimum, the SDK returns a tx_hash but the
                    # on-chain action reverts. Pre-check; if below min, leave
                    # the old order alone (don't attempt update) and let the
                    # next refresh cycle decide.
                    if not _meets_min_notional(new_price, amount):
                        logger.info(
                            f"Market {context.query_id} {side.value} L{level_idx} "
                            f"outcome={'YES' if outcome else 'NO'}: skip change_bid "
                            f"(notional {new_price}c x {amount} = "
                            f"{new_price * amount} < min {MIN_ORDER_NOTIONAL_CENT_SHARES}). "
                            f"Keeping old bid @{current_order.price}c on book."
                        )
                        return None
                    if (
                        getattr(self.config, "backstop_amount", 0)
                        and new_price <= self.config.backstop_price_cents
                    ):
                        logger.info(
                            f"Market {context.query_id} BID L{level_idx}: "
                            f"skip quote at {new_price}c (backstop price "
                            f"slot {self.config.backstop_price_cents}c is "
                            f"reserved; chain keys orders by price)"
                        )
                        return None
                    if not self._bid_budget.reserve_delta(
                        context.query_id, current_order.price,
                        current_order.amount, new_price, amount,
                    ):
                        logger.warning(
                            f"Market {context.query_id} BID L{level_idx}: bid "
                            f"budget exhausted for change_bid; skipping update"
                        )
                        return None
                    try:
                        tx_hash = self._client.change_bid(
                            query_id=context.query_id,
                            outcome=outcome,
                            old_price=old_sdk_price,
                            new_price=new_sdk_price,
                            new_amount=amount,
                            wait=True,
                        )
                    except Exception:
                        # Reverse the delta (review finding 3): the swap did
                        # not happen, so the OLD bid's accounting must stand.
                        # Without this, a shrinking update re-applies its
                        # negative delta on every failed retry and ratchets
                        # committed toward zero while the big bid still rests.
                        self._bid_budget.reserve_delta(
                            context.query_id, new_price, amount,
                            current_order.price, current_order.amount,
                        )
                        raise
                else:
                    # Pre-check: can the inventory path place this ask at
                    # all? If not, leave the old ask on the
                    # book (still strictly better than an empty level until
                    # next refresh).
                    inv_for_check = self._inventory.get_market_inventory(context.query_id)
                    avail = inv_for_check.available_for_sell(outcome)
                    # If the OLD ask was inventory-backed, its amount is
                    # currently locked in reservations; we'd release it on
                    # cancel and that capacity would be available to the new
                    # ask. Account for that here so we don't false-negative.
                    old_inv_release = (
                        current_order.amount
                        if current_order.is_inventory_backed
                        else 0
                    )
                    will_inv = (
                        (avail + old_inv_release) >= amount
                        and _meets_min_notional(new_price, amount)
                    )
                    if not will_inv:
                        logger.info(
                            f"Market {context.query_id} {side.value} L{level_idx} "
                            f"outcome={'YES' if outcome else 'NO'}: skip refresh "
                            f"(inventory cannot place {new_price}c x{amount}; "
                            f"avail={avail}). Keeping old."
                        )
                        return None

                    # Cancel old (path-aware), then place new (path-decided
                    # at runtime by _place_ask based on now-current inventory).
                    self._cancel_ask(
                        context=context,
                        outcome=outcome,
                        price=current_order.price,
                        amount=current_order.amount,
                        is_inventory_backed=current_order.is_inventory_backed,
                    )
                    tx_hash, is_inv_backed = self._place_ask(
                        context=context,
                        outcome=outcome,
                        new_price=new_price,
                        amount=amount,
                    )
                    if tx_hash is None:
                        # Pre-check said it would place but actual placement
                        # was rejected (rare; e.g. inventory shifted between
                        # check and place due to a fill). Old ask is gone;
                        # next refresh will retry.
                        logger.warning(
                            f"Market {context.query_id} {side.value} L{level_idx}: "
                            f"refresh placement returned None after old cancel "
                            f"(level now empty until next cycle)"
                        )
                        return None

                order_mgr.record_order(
                    outcome, side, new_price, amount, tx_hash, level_idx,
                    is_inventory_backed=(is_inv_backed if side == Side.ASK else False),
                )
                self.stats.orders_updated += 1

                # Track P&L
                if side == Side.BID:
                    self._trading_stats.total_bought_value += new_price * amount / 100.0
                    self._trading_stats.total_shares_bought += amount
                else:
                    self._trading_stats.total_sold_value += new_price * amount / 100.0
                    self._trading_stats.total_shares_sold += amount

                # Update order state tracking (for restart recovery)
                self._order_state.update_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    is_buy=(side == Side.BID),
                    old_price=current_order.price,
                    new_price=new_price,
                    amount=amount,
                    order_id=tx_hash,
                    level_idx=level_idx,
                    is_inventory_backed=(is_inv_backed if side == Side.ASK else False),
                )

                logger.info(
                    f"Market {context.query_id} {side.value} L{level_idx}: updated "
                    f"{current_order.price}→{new_price}¢ x{amount} ({reason})"
                )

                return tx_hash

            else:
                # Place new order
                is_inv_backed = False
                if side == Side.BID:
                    if not _meets_min_notional(new_price, amount):
                        logger.info(
                            f"Market {context.query_id} {side.value} L{level_idx} "
                            f"outcome={'YES' if outcome else 'NO'}: skip place "
                            f"(notional {new_price}c x {amount} = "
                            f"{new_price * amount} < min {MIN_ORDER_NOTIONAL_CENT_SHARES})"
                        )
                        return None
                    if (
                        getattr(self.config, "backstop_amount", 0)
                        and new_price <= self.config.backstop_price_cents
                    ):
                        logger.info(
                            f"Market {context.query_id} BID L{level_idx}: "
                            f"skip quote at {new_price}c (backstop price "
                            f"slot {self.config.backstop_price_cents}c is "
                            f"reserved; chain keys orders by price)"
                        )
                        return None
                    if not self._bid_budget.try_reserve(
                        context.query_id, new_price, amount
                    ):
                        logger.warning(
                            f"Market {context.query_id} BID L{level_idx}: bid "
                            f"budget exhausted ({self._bid_budget.committed(context.query_id)}"
                            f"/{self._bid_budget.cap(context.query_id)} cent-shares "
                            f"committed); skipping placement until reconcile "
                            f"confirms resting state"
                        )
                        return None
                    try:
                        tx_hash = self._client.place_buy_order(
                            query_id=context.query_id,
                            outcome=outcome,
                            price=new_price,
                            amount=amount,
                            wait=True,
                        )
                    except Exception as place_exc:
                        # Release only when the chain PROVED the order never
                        # rested; a timeout may still land on chain, and its
                        # reservation must stand until a fresh sync (review
                        # finding 4).
                        if self._is_definitive_rejection(place_exc):
                            self._bid_budget.release(
                                context.query_id, new_price, amount
                            )
                        raise
                else:
                    # Inventory-aware ASK: prefer existing held YES/NO shares
                    # (single-leg, no new collateral). _place_ask returns
                    # (None, _) when inventory is short or the order would
                    # miss the protocol's min-notional, in which case skip
                    # this order rather than placing something that will
                    # silently revert on chain.
                    tx_hash, is_inv_backed = self._place_ask(
                        context=context,
                        outcome=outcome,
                        new_price=new_price,
                        amount=amount,
                    )
                    if tx_hash is None:
                        return None

                order_mgr.record_order(
                    outcome, side, new_price, amount, tx_hash, level_idx,
                    is_inventory_backed=is_inv_backed,
                )
                self.stats.orders_placed += 1

                # Track P&L
                if side == Side.BID:
                    self._trading_stats.total_bought_value += new_price * amount / 100.0
                    self._trading_stats.total_shares_bought += amount
                else:
                    self._trading_stats.total_sold_value += new_price * amount / 100.0
                    self._trading_stats.total_shares_sold += amount

                # Track order state (for restart recovery)
                self._order_state.track_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    is_buy=(side == Side.BID),
                    price=new_price,
                    amount=amount,
                    order_id=tx_hash,
                    level_idx=level_idx,
                    is_inventory_backed=is_inv_backed,
                )

                logger.info(
                    f"Market {context.query_id} {side.value} L{level_idx}: placed "
                    f"@{new_price}¢ x{amount}"
                    f"{' (inv-backed)' if is_inv_backed else ''}"
                )

                return tx_hash

        except Exception as e:
            err_str = str(e).lower()
            if any(msg in err_str for msg in SETTLED_MARKET_ERRORS):
                raise MarketSettledError(context.query_id) from e

            # Clear stale order state so next cycle places a fresh order
            # instead of retrying a failed update forever. level_idx is
            # load-bearing: without it clear_order defaults to level 0, so
            # an L1+ not-found wiped L0's LIVE tracking (invisible resting
            # order, double placement) while L1's stale entry retried the
            # dead change_bid every cycle, keyed by its old price where the
            # loop breaker (which gates on the NEW price) never sees it.
            if current_order is not None and (
                "order not found" in err_str or "old order not found" in err_str
            ):
                order_mgr.clear_order(outcome, side, level_idx)
                self._order_state.untrack_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    is_buy=(side == Side.BID),
                    price=current_order.price,
                    level_idx=level_idx,
                )
                logger.warning(
                    f"Market {context.query_id} {side.value}: "
                    f"order not found on-chain, cleared stale state "
                    f"(was @{current_order.price}¢). Will re-place next cycle."
                )
                self._note_level_not_found_clear(
                    context.query_id, outcome, side, current_order.price
                )
                self.stats.errors += 1
                return None

            logger.error(
                f"Failed to update {side.value} for market {context.query_id}: {e}"
            )
            self.stats.errors += 1
            return None

    def _cancel_market_orders(self, context: MarketContext) -> None:
        """Cancel all orders for a market during shutdown / pre-settlement.

        Uses wait=False for the cancels themselves: with order counts in
        the 100s, sequential wait=True cancels at ~1-2s each can blow past
        the orchestrator's market_maker_timeout (300s) and trigger SIGKILL
        mid-cancel. wait=False broadcasts and returns; reconcile-on-startup
        on the next bot lifetime catches any cancel that didn't actually
        land on chain (the order will still appear in get_order_book and
        get re-recorded). Untrack runs unconditionally because we cannot
        distinguish broadcast-but-failed from confirmed-cancelled in this
        mode — accepting the small state-vs-chain divergence is strictly
        better than the SIGKILL alternative.
        """
        # Settled market: settlement REMOVES resting orders from the book,
        # so every cancel broadcast afterwards fails on chain (verified
        # from the indexer, 2026-09-10 06:03 UTC Eggs settle: 148
        # cancel_order failures, all "Order not found or does not belong
        # to you", zero successes -- the daily post-settle alert wave).
        # Holdings settle for value regardless; drop local tracking
        # without touching the chain.
        settle_time = context.config.settle_time
        if settle_time is not None and int(time.time()) >= settle_time:
            cleared = 0
            for outcome in (True, False):
                orders = context.get_orders(outcome)
                for side, level_list in (
                    (Side.BID, orders.bids),
                    (Side.ASK, orders.asks),
                ):
                    for lvl_idx, order in enumerate(level_list):
                        if order is None:
                            continue
                        if side == Side.BID:
                            orders.set_bid(lvl_idx, None)
                        else:
                            orders.set_ask(lvl_idx, None)
                        self._order_state.untrack_order(
                            query_id=context.query_id,
                            outcome=outcome,
                            is_buy=(side == Side.BID),
                            price=order.price,
                            level_idx=lvl_idx,
                        )
                        cleared += 1
            if cleared:
                logger.info(
                    f"Market {context.query_id} already settled "
                    f"(settle_time={settle_time}); dropped {cleared} tracked "
                    f"orders locally, no cancels broadcast"
                )
            return

        for outcome in [True, False]:
            orders = context.get_orders(outcome)

            all_orders = (
                [(Side.BID, i, o) for i, o in enumerate(orders.bids) if o is not None] +
                [(Side.ASK, i, o) for i, o in enumerate(orders.asks) if o is not None]
            )
            for side, lvl_idx, order in all_orders:
                if order is None:
                    continue

                try:
                    if side == Side.ASK:
                        # Branch on the recorded path: inventory-backed asks
                        # were placed as a single leg at the quote price;
                        # split-mint asks have two on-chain orders to remove.
                        self._cancel_ask(
                            context=context,
                            outcome=outcome,
                            price=order.price,
                            amount=order.amount,
                            is_inventory_backed=order.is_inventory_backed,
                            wait=False,
                        )
                    else:
                        cancel_outcome = outcome
                        cancel_price = convert_price_for_order(order.price, side)
                        self._client.cancel_order(
                            query_id=context.query_id,
                            outcome=cancel_outcome,
                            price=cancel_price,
                            wait=False,
                        )
                    self.stats.orders_cancelled += 1

                    # Untrack the order
                    self._order_state.untrack_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        is_buy=(side == Side.BID),
                        price=order.price,
                        level_idx=lvl_idx,
                    )

                    logger.info(
                        f"Cancelled {side.value} for market {context.query_id} "
                        f"outcome={'YES' if outcome else 'NO'}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to cancel {side.value} for market "
                        f"{context.query_id}: {e}"
                    )

    def _process_market(self, context: MarketContext) -> None:
        """
        Process a single market for one cycle.

        Args:
            context: Market context to process
        """
        # Earnings pull (evaluated first; fires ~1 week before settle_time). Once
        # an EPS market's number is public (its earnings cutoff has passed), the
        # 24/7 book is pick-off-able until settlement, so stop quoting this market
        # and hold inventory (it settles for value at settle_time). Deterministic
        # on earnings_cutoff_time. Re-asserted each session: the first cycle after
        # any (re)start cancels resting orders before quoting, so a restart can
        # never re-expose a pulled market. No config rewrite, no process restart.
        ec = context.config.earnings_cutoff_time
        if ec is not None and int(time.time()) >= ec:
            if context.query_id not in self._earnings_pulled_session:
                logger.info(
                    f"Market {context.query_id}: past earnings cutoff "
                    f"(cutoff={ec}, now={int(time.time())}). Pulling liquidity and "
                    f"holding inventory until settlement."
                )
                self._cancel_market_orders(context)
                self._earnings_pulled_session.add(context.query_id)
            return

        # Liquidation mode: widen spreads and reduce inventory when T < 30 min
        liquidation_mode = False
        liquidation_skew_threshold = 0.3
        if context.config.settle_time:
            seconds_left_liq = context.config.settle_time - int(time.time())
            if seconds_left_liq < 1800:
                liquidation_mode = True
                logger.info(
                    f"Market {context.query_id}: liquidation mode, "
                    f"{seconds_left_liq}s to settlement"
                )

        # Pull liquidity before settlement to protect capital
        if context.config.settle_time and self.config.pre_settlement_cutoff > 0:
            seconds_left = context.config.settle_time - int(time.time())
            if seconds_left <= self.config.pre_settlement_cutoff:
                if context.query_id not in self._pre_settlement_pulled:
                    logger.info(
                        f"Market {context.query_id}: within pre-settlement cutoff "
                        f"({seconds_left}s left, cutoff={self.config.pre_settlement_cutoff}s). "
                        f"Pulling liquidity."
                    )
                    self._cancel_market_orders(context)
                    self._pre_settlement_pulled.add(context.query_id)
                    self._save_pre_settlement_pulled()
                return

        mode = context.config.outcome_mode

        # Determine which outcomes to trade
        outcomes = []
        if mode in (OutcomeMode.YES_ONLY, OutcomeMode.BOTH):
            outcomes.append(True)
        if mode in (OutcomeMode.NO_ONLY, OutcomeMode.BOTH):
            outcomes.append(False)

        for outcome in outcomes:
            # Update order book
            if not self._update_order_book(context, outcome):
                continue

            # Process hanging orders if enabled
            if self.config.avellaneda.hanging_orders_enabled:
                self._process_hanging_orders(context, outcome)

            # Refresh pricing from Black-Scholes or order book
            if self.config.pricing_source == "black_scholes":
                # Always refresh B-S pricing each cycle
                initial_price = self._calculate_initial_price(context.config)
                if initial_price is not None:
                    context.initial_price_yes = initial_price
                    context.initial_price_no = 100 - initial_price
            else:
                # Original behavior: only calculate when no order book data
                state = context.get_state(outcome)
                if state and not state.has_liquidity:
                    if context.get_mid_price(outcome) is None:
                        initial_price = self._calculate_initial_price(context.config)
                        if initial_price is not None:
                            if outcome:
                                context.initial_price_yes = initial_price
                            else:
                                context.initial_price_no = 100 - initial_price

            # Check for order_override
            mid_price = context.get_mid_price(outcome, self.config.pricing_source)
            if mid_price is not None:
                override_proposals = self._create_proposal_from_order_override(mid_price)
                if override_proposals:
                    self._execute_order_override(context, outcome, override_proposals)
                    continue

            # Calculate prices using Avellaneda-Stoikov
            # In liquidation mode, temporarily boost gamma by 5x for wider spreads
            original_gamma = context.config.gamma
            if liquidation_mode:
                base_gamma = context.config.gamma or self.config.avellaneda.risk_factor
                context.config.gamma = base_gamma * 5.0

            pricing = self._calculate_prices(context, outcome)

            # Restore original gamma
            if liquidation_mode:
                context.config.gamma = original_gamma

            if pricing is None:
                continue

            # In liquidation mode with high inventory skew, only quote the
            # side that reduces inventory (no new accumulation)
            if liquidation_mode and abs(pricing.inventory_skew) > liquidation_skew_threshold:
                q = pricing.inventory_skew
                # q > 0 means long inventory -> only place asks (sell to reduce)
                # q < 0 means short inventory -> only place bids (buy to reduce)
                if q > 0:
                    # Zero out bid so only ask is placed
                    pricing = PricingResult(
                        reservation_price=pricing.reservation_price,
                        optimal_spread=pricing.optimal_spread,
                        bid_price=0.0, ask_price=pricing.ask_price,
                        mid_price=pricing.mid_price,
                        inventory_skew=pricing.inventory_skew,
                        volatility=pricing.volatility,
                        kappa=pricing.kappa,
                    )
                else:
                    # Zero out ask so only bid is placed
                    pricing = PricingResult(
                        reservation_price=pricing.reservation_price,
                        optimal_spread=pricing.optimal_spread,
                        bid_price=pricing.bid_price, ask_price=100.0,
                        mid_price=pricing.mid_price,
                        inventory_skew=pricing.inventory_skew,
                        volatility=pricing.volatility,
                        kappa=pricing.kappa,
                    )
                logger.info(
                    f"Market {context.query_id}: liquidation skew q={q:.2f}, "
                    f"quoting {'asks only' if q > 0 else 'bids only'}"
                )

            # Execute order updates
            self._execute_order_updates(context, outcome, pricing)

    def _process_hanging_orders(
        self, context: MarketContext, outcome: bool
    ) -> None:
        """
        Process hanging orders for a market/outcome.

        Cancels orders that are too far from price or too old.

        Args:
            context: Market context
            outcome: True for YES, False for NO
        """
        tracker = self._get_hanging_tracker(context.query_id, outcome)
        mid_price = context.get_mid_price(outcome, self.config.pricing_source)

        if mid_price is None:
            return

        # Get orders to cancel and recreate
        to_cancel, to_recreate = tracker.process_tick(mid_price, time.time())

        # Cancel far/old hanging orders
        for order in to_cancel:
            if self.config.dry_run:
                logger.info(
                    f"[DRY RUN] Would cancel hanging order {order.order_id}"
                )
                continue

            try:
                side = Side.BID if order.is_buy else Side.ASK
                sdk_price = convert_price_for_order(order.price, side)
                self._client.cancel_order(
                    query_id=context.query_id,
                    outcome=outcome,
                    price=sdk_price,
                    wait=False,
                )
                tracker.mark_cancellation_pending(order.order_id)
                self._in_flight_cancels.add(order.order_id)
                self.stats.orders_cancelled += 1
                logger.info(f"Cancelled hanging order {order.order_id}")
            except Exception as e:
                logger.error(f"Failed to cancel hanging order: {e}")

        # Recreate renewed hanging orders
        for order in to_recreate:
            if self.config.dry_run:
                logger.info(
                    f"[DRY RUN] Would recreate hanging order at {order.price}¢"
                )
                continue

            try:
                if not _meets_min_notional(order.price, order.amount):
                    logger.info(
                        f"Skipping hanging order recreate (price {order.price}c "
                        f"x amount {order.amount} below min "
                        f"{MIN_ORDER_NOTIONAL_CENT_SHARES})"
                    )
                    continue
                if order.is_buy:
                    if not self._bid_budget.try_reserve(
                        context.query_id, order.price, order.amount
                    ):
                        logger.warning(
                            f"Market {context.query_id}: bid budget exhausted; "
                            f"skipping hanging order recreate"
                        )
                        continue
                    tx_hash = self._client.place_buy_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        price=order.price,
                        amount=order.amount,
                        wait=True,
                    )
                else:
                    tx_hash = self._client.place_sell_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        price=order.price,
                        amount=order.amount,
                        wait=True,
                    )

                self.stats.orders_placed += 1
                logger.info(
                    f"Recreated hanging order at {order.price}¢ (was {order.order_id})"
                )
            except Exception as e:
                if order.is_buy and self._is_definitive_rejection(e):
                    self._bid_budget.release(
                        context.query_id, order.price, order.amount
                    )
                logger.error(f"Failed to recreate hanging order: {e}")

    def _execute_order_override(
        self,
        context: MarketContext,
        outcome: bool,
        proposals: List[Tuple[str, int, int]],
    ) -> None:
        """
        Execute orders from order_override configuration.

        Args:
            context: Market context
            outcome: True for YES, False for NO
            proposals: List of (side_str, price, amount) tuples
        """
        for side_str, price, amount in proposals:
            side = Side.BID if side_str == "buy" else Side.ASK

            if self.config.dry_run:
                logger.info(
                    f"[DRY RUN] Would place override {side_str} @{price}¢ x{amount}"
                )
                continue

            try:
                if not _meets_min_notional(price, amount):
                    logger.info(
                        f"Skipping override {side_str} @{price}c x{amount} "
                        f"(below min notional {MIN_ORDER_NOTIONAL_CENT_SHARES})"
                    )
                    continue
                if side_str == "buy":
                    if not self._bid_budget.try_reserve(
                        context.query_id, price, amount
                    ):
                        logger.warning(
                            f"Market {context.query_id}: bid budget exhausted; "
                            f"skipping override buy"
                        )
                        continue
                    tx_hash = self._client.place_buy_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        price=price,
                        amount=amount,
                        wait=True,
                    )
                else:
                    tx_hash = self._client.place_sell_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        price=price,
                        amount=amount,
                        wait=True,
                    )

                self.stats.orders_placed += 1
                logger.info(
                    f"Placed override {side_str} @{price}¢ x{amount}"
                )
            except Exception as e:
                if side_str == "buy" and self._is_definitive_rejection(e):
                    self._bid_budget.release(context.query_id, price, amount)
                logger.error(f"Failed to place override order: {e}")

    def _main_loop(self) -> None:
        """Main trading loop."""
        logger.info("Starting main loop")
        logger.info(f"Execution mode: {self._execution_state}")

        # Write initial heartbeat so the orchestrator can detect first-cycle hangs.
        # Without this, the heartbeat file doesn't exist until after the first complete
        # cycle, and the health check skips detection when no file is present.
        self._write_heartbeat()

        while not self._shutdown_requested:
            cycle_start = time.time()

            # Check execution state - should we trade right now?
            if not self._execution_state.should_execute(cycle_start):
                # Outside trading window - cancel active orders
                logger.debug("Outside execution timeframe, skipping cycle")
                self._cancel_all_active_orders()
                self.stats.cycles += 1
                self._write_heartbeat()
                time.sleep(self.config.order_book_poll_interval)
                continue

            # Check if we can create orders (respects should_wait_order_cancel_confirmation)
            if not self._can_create_orders():
                logger.debug(
                    f"Waiting for {len(self._in_flight_cancels)} cancellation(s) to complete"
                )
                time.sleep(1.0)  # Brief wait before retry
                continue

            # Refresh inventory periodically
            if (
                time.time() - self._last_inventory_refresh
                >= self.config.inventory_refresh_interval
            ):
                self._refresh_inventory()

            # Process each market
            settled_markets: list[int] = []
            for query_id, context in self._markets.items():
                if self._shutdown_requested:
                    break

                try:
                    self._process_market(context)
                except MarketSettledError:
                    logger.warning(f"Market {query_id} has settled, removing from active set")
                    settled_markets.append(query_id)
                except Exception as e:
                    logger.error(f"Error processing market {query_id}: {e}")
                    self.stats.errors += 1

            for qid in settled_markets:
                del self._markets[qid]

            self.stats.cycles += 1
            self._write_heartbeat()

            # Periodic on-chain orphan reconcile. Opt-in via
            # `config.reconcile_interval > 0`. Cancels chain orders the
            # bot is not tracking (orphans from the change_bid /
            # _cancel_ask silent-failure race) and untracks local state
            # with no chain counterpart.
            if (
                self.config.reconcile_interval > 0
                and not self.config.dry_run
                and time.time() - self._last_reconcile >= self.config.reconcile_interval
            ):
                try:
                    self._periodic_reconcile_against_chain()
                except Exception as e:
                    logger.warning(f"Periodic reconcile failed: {e}")
                finally:
                    self._last_reconcile = time.time()

            # Sleep until next poll interval
            elapsed = time.time() - cycle_start
            sleep_time = max(0, self.config.order_book_poll_interval - elapsed)
            if sleep_time > 0 and not self._shutdown_requested:
                time.sleep(sleep_time)

    def _write_heartbeat(self) -> None:
        """Write a heartbeat file so the orchestrator can detect if we're stuck."""
        heartbeat_path = os.environ.get("MM_HEARTBEAT_FILE")
        if not heartbeat_path:
            return
        try:
            Path(heartbeat_path).write_text(str(time.time()))
        except OSError:
            pass

        # Log trading stats every 10 cycles
        if self.stats.cycles % 10 == 0:
            ts = self._trading_stats
            logger.info(
                f"Trading stats: bought={ts.total_shares_bought} "
                f"(${ts.total_bought_value:.0f}), "
                f"sold={ts.total_shares_sold} "
                f"(${ts.total_sold_value:.0f}), "
                f"net=${ts.total_sold_value - ts.total_bought_value:.0f}"
            )

    def _cancel_all_active_orders(self) -> None:
        """Cancel all active orders across ALL levels (off-hours / outside
        execution timeframe). Mirrors the multi-level walk in
        `_cancel_market_orders` (shutdown), but uses `wait=False` so this
        non-shutdown path doesn't block on-chain confirmations.

        Previously this only walked `orders.bid` / `orders.ask` (level 0),
        leaving levels 1..N-1 on chain. Combined with periodic off-hours
        cycles, that produced the stale-order state-bloat that wedged the
        orchestrator on 2026-05-01 (#3).
        """
        if self.config.dry_run:
            return

        for context in self._markets.values():
            # Same settled-market rule as _cancel_market_orders: the chain
            # rejects every cancel past settle_time, so broadcasting them
            # is failed-tx noise. Latent today (all live configs run
            # execution_timeframe_mode "infinite") but guarded for parity.
            settle_time = context.config.settle_time
            if settle_time is not None and int(time.time()) >= settle_time:
                continue
            for outcome in [True, False]:
                orders = context.get_orders(outcome)

                # Walk multi-level bids and asks (levels 0..N-1).
                all_orders = (
                    [(Side.BID, i, o) for i, o in enumerate(orders.bids) if o is not None] +
                    [(Side.ASK, i, o) for i, o in enumerate(orders.asks) if o is not None]
                )
                for side, lvl_idx, order in all_orders:
                    try:
                        if side == Side.ASK:
                            # Inventory-backed asks are single-leg; split-mint
                            # asks live on both sides of the book. Branch via
                            # the recorded flag on the BotOrder. wait=False
                            # because this is a bulk off-hours cancel; no new
                            # placement follows.
                            self._cancel_ask(
                                context=context,
                                outcome=outcome,
                                price=order.price,
                                amount=order.amount,
                                is_inventory_backed=order.is_inventory_backed,
                                wait=False,
                            )
                        else:
                            cancel_outcome = outcome
                            cancel_price = convert_price_for_order(order.price, side)
                            self._client.cancel_order(
                                query_id=context.query_id,
                                outcome=cancel_outcome,
                                price=cancel_price,
                                wait=False,
                            )
                        self.stats.orders_cancelled += 1

                        # Untrack the order at the correct level so the
                        # state file does not accumulate stale entries
                        # for levels we just cancelled.
                        self._order_state.untrack_order(
                            query_id=context.query_id,
                            outcome=outcome,
                            is_buy=(side == Side.BID),
                            price=order.price,
                            level_idx=lvl_idx,
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to cancel {side.value} L{lvl_idx} for market "
                            f"{context.query_id}: {e}"
                        )

    def _shutdown(self) -> None:
        """Graceful shutdown - optionally cancel all orders based on config."""
        if self.config.cancel_open_orders_on_exit:
            logger.info("Shutting down - cancelling all open orders...")

            if self.config.dry_run:
                logger.info("[DRY RUN] Would cancel all orders")
            else:
                for context in self._markets.values():
                    self._cancel_market_orders(context)
        else:
            logger.info("Shutting down - leaving orders open (cancel_open_orders_on_exit=False)")

        logger.info(
            f"Shutdown complete. Stats: "
            f"placed={self.stats.orders_placed} "
            f"updated={self.stats.orders_updated} "
            f"cancelled={self.stats.orders_cancelled} "
            f"errors={self.stats.errors} "
            f"cycles={self.stats.cycles}"
        )

    def run(self) -> None:
        """Start the market maker bot."""
        logger.info("Starting Avellaneda Market Maker")

        try:
            self._setup_signal_handlers()
            self._init_client()
            self._init_markets()

            if not self._markets:
                logger.error("No markets configured")
                return

            # Write heartbeat before reconciliation so the orchestrator doesn't
            # kill us during the potentially long reconciliation phase.
            self._write_heartbeat()

            # Reconcile orders from previous session (recover bot's own orders)
            self._reconcile_orders_on_startup()

            # Initial inventory refresh
            self._refresh_inventory()

            # Pre-mint pair inventory per market so subsequent ASKs can be
            # backed by held shares (place_sell_order) instead of minting
            # new pairs every cycle. Idempotent: a restart with sufficient
            # inventory will compute zero deficit and no-op. Skipped in
            # dry_run / read_only modes.
            self._pre_mint_all_markets()

            # Run main loop
            self._running = True
            self._main_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.stats.errors += 1

        finally:
            self._running = False
            self._shutdown()

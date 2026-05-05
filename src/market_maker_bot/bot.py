"""
Main bot orchestrator for the Avellaneda Market Making Bot.

Coordinates all components: pricing, indicators, inventory, and order management.
"""

import json
import math
import os
import time
import logging
import signal
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

logger = logging.getLogger(__name__)

SETTLED_MARKET_ERRORS = (
    "already settled",
    "settled market",
)


class MarketSettledError(Exception):
    """Raised when an operation targets a market that has already settled."""
    def __init__(self, query_id: int):
        self.query_id = query_id
        super().__init__(f"Market {query_id} has settled")


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

        # Execution state (timeframe control)
        self._execution_state = self._create_execution_state()

        # Hanging orders tracker (per market/outcome)
        self._hanging_trackers: dict[tuple[int, bool], HangingOrdersTracker] = {}

        # In-flight cancellations (for should_wait_order_cancel_confirmation)
        self._in_flight_cancels: Set[str] = set()
        self._pre_settlement_pulled: Set[int] = set()

        # Order state persistence (for restart recovery)
        self._order_state = OrderStateManager(config.order_state_file)

        # Derive pre_settlement_pulled persistence path from order_state_file
        state_path = Path(config.order_state_file)
        self._pre_settlement_file = str(
            state_path.parent / "pre_settlement_pulled.json"
        )
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

        logger.info(f"Connecting to {self.config.node_url}")
        self._client = TNClient(
            url=self.config.node_url,
            token=self.config.private_key,
        )

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

            # Fetch current order book for this market
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

                        if is_on_book:
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
        Calculate initial price using Black-Scholes when no order book exists.

        Args:
            market_config: Market configuration with stream info

        Returns:
            Fair value in cents (1-99), or None if calculation fails
        """
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

            # Get current spot value
            spot = get_current_spot_value(records)
            if spot <= 0:
                logger.warning(
                    f"Invalid spot value {spot} for market {market_config.query_id}"
                )
                return None

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
        base_amount = context.config.order_amount

        # Apply eta transformation to order amounts
        bid_amount = self._apply_eta_transformation(
            base_amount, pricing.inventory_skew, is_buy=True
        )
        ask_amount = self._apply_eta_transformation(
            base_amount, pricing.inventory_skew, is_buy=False
        )

        # Apply order optimization (jump to best bid+1 / best ask-1)
        bid_price, ask_price = self._apply_order_optimization(
            context, outcome, bid_price, ask_price
        )

        # Apply transaction costs (currently no fee info available, placeholder)
        # In practice, this would use the actual fee from the exchange
        bid_price, ask_price = self._apply_transaction_costs(
            bid_price, ask_price, fee_pct=0.0
        )

        # Create order levels
        order_levels = self._create_order_levels(
            bid_price, ask_price, pricing.optimal_spread
        )

        # Execute orders for each level
        for level_idx, (level_bid, level_ask) in enumerate(order_levels):
            # Write heartbeat during long placement cycles to prevent orchestrator kills
            if level_idx > 0 and level_idx % 5 == 0:
                self._write_heartbeat()

            # All levels get the same eta-adjusted amount (A-S model: flat sizing,
            # inventory rebalancing handled by eta, risk by spread widening per level)
            bid_amt = bid_amount
            ask_amt = ask_amount

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
                    tx_hash = self._client.change_bid(
                        query_id=context.query_id,
                        outcome=outcome,
                        old_price=old_sdk_price,
                        new_price=new_sdk_price,
                        new_amount=amount,
                        wait=True,
                    )
                else:
                    # Cancel old asks on both sides, then re-mint and re-place.
                    old_split_price = current_order.price if outcome else (100 - current_order.price)
                    # Cancel NO-side ask
                    try:
                        self._client.cancel_order(
                            query_id=context.query_id,
                            outcome=False,
                            price=100 - old_split_price,
                            wait=True,
                        )
                    except Exception:
                        pass  # May already be filled/cancelled
                    # Cancel YES-side ask
                    try:
                        self._client.cancel_order(
                            query_id=context.query_id,
                            outcome=True,
                            price=old_split_price,
                            wait=True,
                        )
                    except Exception:
                        pass  # May already be filled/cancelled
                    # Mint new pairs and place asks on both sides
                    split_price = new_price if outcome else (100 - new_price)
                    self._client.place_split_limit_order(
                        query_id=context.query_id,
                        true_price=split_price,
                        amount=amount,
                        wait=True,
                    )
                    try:
                        tx_hash = self._client.place_sell_order(
                            query_id=context.query_id,
                            outcome=True,
                            price=split_price,
                            amount=amount,
                            wait=True,
                        )
                    except Exception as sell_err:
                        # Split succeeded but sell failed - try to cancel the
                        # orphaned NO-side order placed by the split to recover.
                        logger.error(f"place_sell_order failed after split mint: {sell_err}")
                        try:
                            cancel_price = 100 - split_price
                            self._client.cancel_order(
                                query_id=context.query_id,
                                outcome=False,
                                price=cancel_price,
                                wait=True,
                            )
                            logger.info("Cancelled orphaned split order after sell failure")
                        except Exception:
                            logger.error("Failed to cancel orphaned split order")
                        raise sell_err

                order_mgr.record_order(outcome, side, new_price, amount, tx_hash, level_idx)
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
                )

                logger.info(
                    f"Market {context.query_id} {side.value} L{level_idx}: updated "
                    f"{current_order.price}→{new_price}¢ x{amount} ({reason})"
                )

                return tx_hash

            else:
                # Place new order
                if side == Side.BID:
                    tx_hash = self._client.place_buy_order(
                        query_id=context.query_id,
                        outcome=outcome,
                        price=new_price,
                        amount=amount,
                        wait=True,
                    )
                else:
                    # Mint share pairs and place asks on BOTH sides of the book.
                    # place_split_limit_order mints pairs and sells NO at (100 - true_price).
                    # We then also sell the YES shares we retained via place_sell_order.
                    split_price = new_price if outcome else (100 - new_price)
                    self._client.place_split_limit_order(
                        query_id=context.query_id,
                        true_price=split_price,
                        amount=amount,
                        wait=True,
                    )
                    # Place YES sell order with the shares we just minted
                    try:
                        tx_hash = self._client.place_sell_order(
                            query_id=context.query_id,
                            outcome=True,
                            price=split_price,
                            amount=amount,
                            wait=True,
                        )
                    except Exception as sell_err:
                        # Split succeeded but sell failed - try to cancel the
                        # orphaned NO-side order placed by the split to recover.
                        logger.error(f"place_sell_order failed after split mint: {sell_err}")
                        try:
                            cancel_price = 100 - split_price
                            self._client.cancel_order(
                                query_id=context.query_id,
                                outcome=False,
                                price=cancel_price,
                                wait=True,
                            )
                            logger.info("Cancelled orphaned split order after sell failure")
                        except Exception:
                            logger.error("Failed to cancel orphaned split order")
                        raise sell_err

                order_mgr.record_order(outcome, side, new_price, amount, tx_hash, level_idx)
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
                )

                logger.info(
                    f"Market {context.query_id} {side.value} L{level_idx}: placed "
                    f"@{new_price}¢ x{amount}"
                )

                return tx_hash

        except Exception as e:
            err_str = str(e).lower()
            if any(msg in err_str for msg in SETTLED_MARKET_ERRORS):
                raise MarketSettledError(context.query_id) from e

            # Clear stale order state so next cycle places a fresh order
            # instead of retrying a failed update forever.
            if current_order is not None and (
                "order not found" in err_str or "old order not found" in err_str
            ):
                order_mgr.clear_order(outcome, side)
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
                self.stats.errors += 1
                return None

            logger.error(
                f"Failed to update {side.value} for market {context.query_id}: {e}"
            )
            self.stats.errors += 1
            return None

    def _cancel_market_orders(self, context: MarketContext) -> None:
        """Cancel all orders for a market during shutdown."""
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
                        # Ask orders exist on both sides (split mint + sell).
                        # Cancel NO-side ask and YES-side ask.
                        split_price = order.price if outcome else (100 - order.price)
                        for cancel_out, cancel_p in [(False, 100 - split_price), (True, split_price)]:
                            try:
                                self._client.cancel_order(
                                    query_id=context.query_id,
                                    outcome=cancel_out,
                                    price=cancel_p,
                                    wait=True,
                                )
                            except Exception:
                                pass  # May already be filled/cancelled
                    else:
                        cancel_outcome = outcome
                        cancel_price = convert_price_for_order(order.price, side)
                        self._client.cancel_order(
                            query_id=context.query_id,
                            outcome=cancel_outcome,
                            price=cancel_price,
                            wait=True,
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
                if order.is_buy:
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
                if side_str == "buy":
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
                            # Ask orders exist on both sides (split mint + sell).
                            # Cancel NO-side and YES-side legs. Best-effort:
                            # an inner-leg failure is swallowed because the
                            # other leg may still need to be cancelled.
                            split_price = order.price if outcome else (100 - order.price)
                            for cancel_out, cancel_p in [(False, 100 - split_price), (True, split_price)]:
                                try:
                                    self._client.cancel_order(
                                        query_id=context.query_id,
                                        outcome=cancel_out,
                                        price=cancel_p,
                                        wait=False,
                                    )
                                except Exception:
                                    pass  # may already be filled/cancelled
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

            # Run main loop
            self._running = True
            self._main_loop()

        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            self.stats.errors += 1

        finally:
            self._running = False
            self._shutdown()

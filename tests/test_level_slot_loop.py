"""Regression tests for the 2026-09-09 Eggs 72c change_bid loop and the
daily post-settle cancel noise.

The chain keys resting orders by (wallet, outcome, signed price) while the
bot tracks them per level_idx. A cross-cycle grid shift could land one
level's NEW price on another level's CURRENT price: the first level's
change_bid moved the on-chain order out from under the second, whose next
change hit "Old order not found", cleared its state, and re-placed at the
same price -- observed cycling every ~2s (~1,500 failed
txs per 30 min). These tests pin the three defenses:

  - slot guard: _update_single_order refuses to act at a price another
    level is tracked at;
  - level loop breaker: repeated not-found clears on one exact slot put
    that slot on cooldown instead of feeding the loop;
  - settle-aware suppression: no cancel broadcasts or reconcile passes
    against a market whose settle_time has passed (the chain rejects
    them all; they were the daily ~06:07 failed-tx alert wave).
"""

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import AvellanedaMarketMaker, _pair_out_of_order
from market_maker_bot.market import OrderManager, Side
from market_maker_bot.models import ActiveOrders


class _Ctx:
    """Minimal MarketContext stand-in: real ActiveOrders, stub config."""

    def __init__(self, query_id=7, settle_time=None):
        self.config = SimpleNamespace(query_id=query_id, settle_time=settle_time)
        self.yes_orders = ActiveOrders()
        self.no_orders = ActiveOrders()

    @property
    def query_id(self):
        return self.config.query_id

    def get_orders(self, outcome):
        return self.yes_orders if outcome else self.no_orders

    def set_state(self, outcome, state):
        pass


def _mgr(ctx):
    return OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)


def _quote_bot(threshold=3, window=120.0, cooldown=300.0):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False,
        backstop_amount=0,
        backstop_price_cents=2,
        level_loop_threshold=threshold,
        level_loop_window=window,
        level_loop_cooldown=cooldown,
        avellaneda=SimpleNamespace(max_position_per_outcome=0),
    )
    bot._funds_blocked.return_value = False
    bot._level_not_found_times = {}
    bot._level_cooldown_until = {}
    bot._slot_guard_skips = {}
    # Bind the real helpers so guard + breaker logic actually runs.
    bot._level_slot_cooling = AvellanedaMarketMaker._level_slot_cooling.__get__(bot)
    bot._note_level_not_found_clear = (
        AvellanedaMarketMaker._note_level_not_found_clear.__get__(bot)
    )
    bot._is_definitive_rejection.return_value = False
    return bot


def _update(bot, ctx, mgr, price, amount=10, level_idx=0, outcome=True,
            side=Side.BID):
    return AvellanedaMarketMaker._update_single_order(
        bot, ctx, outcome, side, price, amount, mgr, level_idx
    )


# --- OrderManager.level_owning_price ----------------------------------------

def test_level_owning_price_finds_other_level():
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    assert mgr.level_owning_price(True, Side.BID, 72, exclude_level=0) == 1


def test_level_owning_price_excludes_own_level():
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=0)
    assert mgr.level_owning_price(True, Side.BID, 72, exclude_level=0) is None


def test_level_owning_price_free_slot_and_other_side():
    ctx = _Ctx()
    mgr = _mgr(ctx)
    # An ASK at 72 must not block a BID at 72: on chain they are distinct
    # keys (bids carry negative SDK prices).
    mgr.record_order(True, Side.ASK, 72, 10, "tx1", level_idx=1)
    assert mgr.level_owning_price(True, Side.BID, 72, exclude_level=0) is None


# --- slot guard in _update_single_order -------------------------------------

def test_place_skipped_when_other_level_owns_slot():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    bot._client.place_buy_order.assert_not_called()
    # L1's tracking is untouched.
    assert ctx.yes_orders.get_bid(1).price == 72


def test_change_skipped_when_target_slot_owned():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 73, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    bot._client.change_bid.assert_not_called()
    assert ctx.yes_orders.get_bid(0).price == 73


def test_free_slot_still_places():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    _update(bot, ctx, mgr, 71, level_idx=0)
    bot._client.place_buy_order.assert_called_once()


def test_grid_shift_resolves_without_collision():
    """The 72c scenario: a downward grid shift stalls one cycle instead of
    corrupting tracking, then completes once the neighbor has vacated."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 73, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)

    # Cycle 1 proposes L0=72 (occupied by L1 -> stall), L1=71 (free -> move).
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    _update(bot, ctx, mgr, 71, level_idx=1)
    bot._client.change_bid.assert_called_once()

    # Cycle 2: slot 72 is now free, L0 completes its move.
    _update(bot, ctx, mgr, 72, level_idx=0)
    assert bot._client.change_bid.call_count == 2
    prices = [o.price for o in ctx.yes_orders.bids if o is not None]
    assert sorted(prices) == [71, 72]
    assert len(prices) == len(set(prices))


# --- level loop breaker ------------------------------------------------------

def test_not_found_clear_feeds_breaker_counter():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx0", level_idx=0)
    bot._client.change_bid.side_effect = RuntimeError("Old order not found")
    assert _update(bot, ctx, mgr, 71, level_idx=0) is None
    # State cleared (existing behavior) AND the clear was counted per-slot.
    assert ctx.yes_orders.get_bid(0) is None
    assert len(bot._level_not_found_times[(7, True, True, 72)]) == 1


def test_breaker_trips_at_threshold_and_blocks_slot():
    bot = _quote_bot(threshold=3)
    ctx = _Ctx()
    mgr = _mgr(ctx)
    for _ in range(3):
        bot._note_level_not_found_clear(7, True, Side.BID, 72)
    key = (7, True, True, 72)
    assert bot._level_cooldown_until[key] > time.time()
    # Quoting that exact slot is refused while cooling.
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    bot._client.place_buy_order.assert_not_called()
    # A different price on the same market is unaffected.
    _update(bot, ctx, mgr, 71, level_idx=0)
    bot._client.place_buy_order.assert_called_once()


def test_breaker_success_does_not_reset_count():
    """The loop's signature is alternating success/not-found; a placement
    between clears must not blind the breaker."""
    bot = _quote_bot(threshold=3)
    ctx = _Ctx()
    mgr = _mgr(ctx)
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    _update(bot, ctx, mgr, 72, level_idx=0)  # successful re-place
    ctx.yes_orders.set_bid(0, None)  # simulate the next clear's precondition
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    assert (7, True, True, 72) in bot._level_cooldown_until


def test_breaker_window_prunes_old_clears(monkeypatch):
    bot = _quote_bot(threshold=3, window=120.0)
    t = [1000.0]
    monkeypatch.setattr(time, "time", lambda: t[0])
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    t[0] += 200.0  # outside the window
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    t[0] += 200.0
    bot._note_level_not_found_clear(7, True, Side.BID, 72)
    assert (7, True, True, 72) not in bot._level_cooldown_until


def test_breaker_cooldown_expires():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    key = (7, True, True, 72)
    bot._level_cooldown_until[key] = time.time() - 1
    _update(bot, ctx, mgr, 72, level_idx=0)
    bot._client.place_buy_order.assert_called_once()
    assert key not in bot._level_cooldown_until


def test_breaker_disabled_by_zero_threshold():
    bot = _quote_bot(threshold=0)
    for _ in range(10):
        bot._note_level_not_found_clear(7, True, Side.BID, 72)
    assert bot._level_cooldown_until == {}
    assert bot._level_not_found_times == {}


# --- settle-aware cancel suppression -----------------------------------------

def _cancel_bot():
    bot = MagicMock()
    bot.config = SimpleNamespace(dry_run=False)
    return bot


def _ctx_with_orders(settle_time):
    ctx = _Ctx(settle_time=settle_time)
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 40, 10, "tx0", level_idx=0)
    mgr.record_order(False, Side.ASK, 60, 10, "tx1", level_idx=0)
    return ctx


def test_settled_market_cancels_nothing_on_chain():
    bot = _cancel_bot()
    ctx = _ctx_with_orders(settle_time=int(time.time()) - 60)
    AvellanedaMarketMaker._cancel_market_orders(bot, ctx)
    bot._client.cancel_order.assert_not_called()
    bot._cancel_ask.assert_not_called()
    # Local tracking is dropped so nothing acts on these orders again.
    assert ctx.yes_orders.get_bid(0) is None
    assert ctx.no_orders.get_ask(0) is None
    assert bot._order_state.untrack_order.call_count == 2


def test_unsettled_market_still_broadcasts_cancels():
    bot = _cancel_bot()
    ctx = _ctx_with_orders(settle_time=int(time.time()) + 3600)
    AvellanedaMarketMaker._cancel_market_orders(bot, ctx)
    bot._client.cancel_order.assert_called_once()  # the bid
    bot._cancel_ask.assert_called_once()           # the ask


def test_no_settle_time_still_broadcasts_cancels():
    bot = _cancel_bot()
    ctx = _ctx_with_orders(settle_time=None)
    AvellanedaMarketMaker._cancel_market_orders(bot, ctx)
    bot._client.cancel_order.assert_called_once()
    bot._cancel_ask.assert_called_once()


# --- settle-aware periodic reconcile skip ------------------------------------

def _reconcile_bot(settle_time):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False,
        read_only=False,
        maa_address="0xAA00000000000000000000000000000000000aa0",
        private_key="",
        backstop_amount=0,
        backstop_price_cents=2,
        pre_settlement_cutoff=900.0,
    )
    ctx = SimpleNamespace(config=SimpleNamespace(settle_time=settle_time))
    bot._markets = {7: ctx}
    bot._order_state.get_market_orders.return_value = []
    bot._reconcile_cancel_attempts = {}
    bot._cancel_not_found_times = deque()
    bot._bid_budget = MagicMock()
    bot._pre_settlement_pulled = set()
    bot._earnings_pulled_session = set()
    bot._client.get_order_book.return_value = []
    return bot


def test_periodic_reconcile_skips_settled_market():
    bot = _reconcile_bot(settle_time=int(time.time()) - 60)
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    bot._client.get_order_book.assert_not_called()
    bot._client.cancel_order.assert_not_called()


def test_periodic_reconcile_still_runs_before_settle():
    bot = _reconcile_bot(settle_time=int(time.time()) + 3600)
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    assert bot._client.get_order_book.call_count == 2  # both outcomes


# --- review round 2: not-found clear must target the RIGHT level -------------

def test_not_found_clear_targets_its_own_level_only():
    """An L1 change_bid not-found must clear L1, not L0. The old
    clear_order(outcome, side) defaulted to level 0: L0's LIVE chain order
    went locally invisible (re-placed on top next cycle) while L1's stale
    entry retried the dead change forever, keyed by its OLD price where
    the breaker (which gates on the NEW price) never sees it."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 75, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    bot._client.change_bid.side_effect = RuntimeError("Old order not found")
    assert _update(bot, ctx, mgr, 71, level_idx=1) is None
    assert ctx.yes_orders.get_bid(1) is None          # L1 cleared
    assert ctx.yes_orders.get_bid(0).price == 75      # L0 untouched
    # The clear was counted against L1's old price slot.
    assert (7, True, True, 72) in bot._level_not_found_times
    # And the untrack hit L1, not L0.
    kwargs = bot._order_state.untrack_order.call_args[1]
    assert kwargs["level_idx"] == 1
    assert kwargs["price"] == 72


def test_cleared_level_recovers_next_cycle_without_freezing_neighbors():
    """Finding-2 scenario: after a correct L1 clear, L1 re-places freely
    and L0 keeps updating; no mutual slot-guard freeze forms."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 75, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    bot._client.change_bid.side_effect = RuntimeError("Old order not found")
    _update(bot, ctx, mgr, 71, level_idx=1)           # L1 clears
    bot._client.change_bid.side_effect = None
    _update(bot, ctx, mgr, 71, level_idx=1)           # L1 re-places
    bot._client.place_buy_order.assert_called_once()
    _update(bot, ctx, mgr, 74, level_idx=0)           # L0 still updates
    bot._client.change_bid.assert_called()
    prices = [o.price for o in ctx.yes_orders.bids if o is not None]
    assert sorted(prices) == [71, 74]


def test_slot_guard_skip_counter_tracks_and_resets():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    for _ in range(3):
        _update(bot, ctx, mgr, 72, level_idx=0)       # blocked by L1
    assert bot._slot_guard_skips[(7, True, True, 0)] == 3
    _update(bot, ctx, mgr, 71, level_idx=0)           # slot free -> proceeds
    assert (7, True, True, 0) not in bot._slot_guard_skips


# --- review round 2: startup dedupe of incident-era duplicate slots ----------

def _startup_bot(tracked, book_entries):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False,
        avellaneda=SimpleNamespace(
            order_refresh_tolerance_pct=1.0, max_order_age=300.0
        ),
    )
    ctx = _Ctx()
    bot._markets = {7: ctx}
    bot._order_state.get_all_orders.return_value = tracked
    bot._client.get_order_book.return_value = book_entries
    return bot, ctx


def _tracked(price, level_idx, outcome=True, is_buy=True):
    return SimpleNamespace(
        query_id=7, outcome=outcome, is_buy=is_buy, price=price,
        amount=10, level_idx=level_idx, order_id=f"tx-{price}-{level_idx}",
        is_inventory_backed=False,
    )


def test_startup_recovery_drops_duplicate_price_slots():
    """A pre-fix state file can hold two levels at one price; the chain
    holds ONE order there. Recovering both re-seeds the collision loop,
    so only the first entry per (outcome, side, price) is recovered."""
    tracked = [_tracked(72, 0), _tracked(72, 1)]
    bot, ctx = _startup_bot(tracked, [{"price": -72, "amount": 10}])
    AvellanedaMarketMaker._reconcile_orders_on_startup(bot)
    recovered = [o for o in ctx.yes_orders.bids if o is not None]
    assert len(recovered) == 1
    # The duplicate was untracked from the persistent state too.
    untracked = [c[0] for c in bot._order_state.untrack_order.call_args_list]
    assert (7, True, True, 72, 1) in untracked


def test_startup_recovery_keeps_distinct_prices():
    tracked = [_tracked(72, 0), _tracked(71, 1)]
    bot, ctx = _startup_bot(
        tracked, [{"price": -72, "amount": 10}, {"price": -71, "amount": 10}]
    )
    AvellanedaMarketMaker._reconcile_orders_on_startup(bot)
    recovered = [o for o in ctx.yes_orders.bids if o is not None]
    assert sorted(o.price for o in recovered) == [71, 72]
    bot._order_state.untrack_order.assert_not_called()


# --- level-inversion healer (2026-09-15: 17 mag7 ask pairs wedged) -----------

def test_pair_out_of_order_semantics():
    # Asks ascend with level; bids descend; equal prices are ordered.
    assert _pair_out_of_order(Side.ASK, 0, 85, 1, 84)
    assert not _pair_out_of_order(Side.ASK, 0, 84, 1, 85)
    assert _pair_out_of_order(Side.BID, 0, 70, 1, 72)
    assert not _pair_out_of_order(Side.BID, 0, 73, 1, 72)
    assert not _pair_out_of_order(Side.ASK, 0, 84, 1, 84)
    # Argument order must not matter.
    assert _pair_out_of_order(Side.ASK, 1, 84, 0, 85)


def test_ask_inversion_heals_with_zero_chain_writes():
    """The 09-15 wedge: asks tracked L0@85 / L1@84, pricing wants L0@84 /
    L1@85. The healer swaps the labels locally and quoting resumes."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.ASK, 85, 7, "txA", level_idx=0)
    mgr.record_order(True, Side.ASK, 84, 9, "txB", level_idx=1)
    assert _update(bot, ctx, mgr, 84, level_idx=0, side=Side.ASK) is None
    # Labels swapped: records (price, amount, tx) travel intact.
    a0, a1 = ctx.yes_orders.get_ask(0), ctx.yes_orders.get_ask(1)
    assert (a0.price, a0.amount, a0.tx_hash) == (84, 9, "txB")
    assert (a1.price, a1.amount, a1.tx_hash) == (85, 7, "txA")
    # Persistent state re-labeled the same way.
    assert bot._order_state.untrack_order.call_count == 2
    tracked = [c[1] for c in bot._order_state.track_order.call_args_list]
    assert {(t["price"], t["level_idx"]) for t in tracked} == {(84, 0), (85, 1)}
    # Zero chain writes.
    bot._cancel_ask.assert_not_called()
    bot._place_ask.assert_not_called()
    bot._client.change_bid.assert_not_called()
    # Next cycle both levels are consistent: no update needed, still no writes.
    _update(bot, ctx, mgr, 84, level_idx=0, side=Side.ASK)
    _update(bot, ctx, mgr, 85, level_idx=1, side=Side.ASK)
    bot._cancel_ask.assert_not_called()
    bot._place_ask.assert_not_called()


def test_bid_inversion_heals():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 70, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    assert ctx.yes_orders.get_bid(0).price == 72
    assert ctx.yes_orders.get_bid(1).price == 70
    bot._client.change_bid.assert_not_called()
    bot._client.place_buy_order.assert_not_called()


def test_ordered_pair_is_never_relabeled():
    """The normal transient grid shift (correctly ordered pair) must keep
    the plain skip behavior - no swap, no state writes."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 73, 10, "tx0", level_idx=0)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    assert ctx.yes_orders.get_bid(0).price == 73
    assert ctx.yes_orders.get_bid(1).price == 72
    bot._order_state.track_order.assert_not_called()
    assert bot._slot_guard_skips[(7, True, True, 0)] == 1


def test_untracked_level_targeting_owned_slot_does_not_relabel():
    """A level with NO current order proposing an occupied price is the
    place-collision case, not an inversion: plain skip."""
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.ASK, 84, 9, "txB", level_idx=1)
    assert _update(bot, ctx, mgr, 84, level_idx=0, side=Side.ASK) is None
    assert ctx.yes_orders.get_ask(1).price == 84
    bot._order_state.track_order.assert_not_called()


def test_three_level_cycle_sorts_pairwise():
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.ASK, 97, 5, "t0", level_idx=0)
    mgr.record_order(True, Side.ASK, 99, 5, "t1", level_idx=1)
    mgr.record_order(True, Side.ASK, 98, 5, "t2", level_idx=2)
    # L1 targets 98 (owned by L2); pair (L1@99, L2@98) is out of order.
    assert _update(bot, ctx, mgr, 98, level_idx=1, side=Side.ASK) is None
    prices = [ctx.yes_orders.get_ask(i).price for i in range(3)]
    assert prices == [97, 98, 99]
    bot._cancel_ask.assert_not_called()
    bot._place_ask.assert_not_called()


# --- benign hold vs real stall (2026-09-16, market 787 saturated grid) -------

def test_benign_hold_never_logs_error(caplog):
    """A level WITH a resting quote whose target is quoted by its neighbor
    is a covered book, not a wedge: no ERROR escalation at the threshold."""
    import logging
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.ASK, 97, 5, "t0", level_idx=0)
    mgr.record_order(True, Side.ASK, 98, 5, "t1", level_idx=1)
    with caplog.at_level(logging.INFO):
        for _ in range(300):
            assert _update(bot, ctx, mgr, 98, level_idx=0, side=Side.ASK) is None
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)
    assert any("SLOT GUARD HOLD" in r.message for r in caplog.records)
    # Nothing moved and nothing was broadcast.
    assert ctx.yes_orders.get_ask(0).price == 97
    bot._cancel_ask.assert_not_called()
    bot._place_ask.assert_not_called()


def test_placement_blocked_still_escalates_to_error(caplog):
    """A level with NO resting quote blocked from placing means the book is
    thinner than designed: the ERROR escalation must survive."""
    import logging
    bot = _quote_bot()
    ctx = _Ctx()
    mgr = _mgr(ctx)
    mgr.record_order(True, Side.BID, 72, 10, "tx1", level_idx=1)
    with caplog.at_level(logging.INFO):
        for _ in range(150):
            assert _update(bot, ctx, mgr, 72, level_idx=0) is None
    assert any(
        r.levelno == logging.ERROR and "SLOT GUARD STALL" in r.message
        for r in caplog.records
    )


# --- reconcile skips the pre-settlement cutoff window (2026-09-22) -----------

def test_reconcile_skips_market_inside_cutoff():
    """The pull empties a market by design; reconcile orphan-hunting there
    double-cancels the same orders (~60-90 failed txs per daily settle)."""
    bot = _reconcile_bot(settle_time=int(time.time()) + 900)  # 15 min out
    bot.config.pre_settlement_cutoff = 1800.0
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    bot._client.get_order_book.assert_not_called()
    bot._client.cancel_order.assert_not_called()


def test_reconcile_runs_outside_cutoff():
    bot = _reconcile_bot(settle_time=int(time.time()) + 7200)  # 2h out
    bot.config.pre_settlement_cutoff = 1800.0
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    assert bot._client.get_order_book.call_count == 2

"""Inventory-backed asks move with one atomic change_ask.

Before, an ask move was cancel-then-place. The cancelled shares only count
as free after the next inventory refresh, so on a book with few spare shares
an up-move pulled the ask and re-placed it ~30s later (CESR 2026-09-29: 162
pull/re-place cycles and ~50 txs a minute). change_ask moves the resting sell
in one transaction and only pulls from holdings what the new amount adds.
"""

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import AvellanedaMarketMaker
from market_maker_bot.market import OrderManager, Side
from market_maker_bot.models import ActiveOrders
from market_maker_bot.pricing.inventory import MarketInventory

SHORT = Exception(
    "ERROR: Insufficient shares in holdings. Need 4 more shares, but only "
    "have 0 in holdings."
)


class _Ctx:
    def __init__(self):
        self.config = SimpleNamespace(query_id=5, settle_time=None)
        self.yes_orders = ActiveOrders()
        self.no_orders = ActiveOrders()

    @property
    def query_id(self):
        return self.config.query_id

    def get_orders(self, outcome):
        return self.yes_orders if outcome else self.no_orders

    def set_state(self, outcome, state):
        pass


def _setup(held=0, listed=None, old_price=38, old_amount=6, inventory_backed=True):
    inv = MarketInventory(query_id=5)
    inv.yes_shares = held
    inv.listed_by_price = dict(listed or {(True, old_price): old_amount})
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False, backstop_amount=0, backstop_price_cents=2,
        level_loop_threshold=3, level_loop_window=120.0,
        level_loop_cooldown=300.0,
        avellaneda=SimpleNamespace(max_position_per_outcome=0),
    )
    bot._funds_blocked.return_value = False
    bot._level_not_found_times = {}
    bot._level_cooldown_until = {}
    bot._slot_guard_skips = {}
    bot._cancel_not_found_times = deque()
    bot._inventory.get_market_inventory.return_value = inv
    for name in (
        "_free_to_sell", "_move_ask", "_place_ask", "_cancel_ask",
        "_level_slot_cooling", "_note_level_not_found_clear",
        "_note_cancel_not_found",
    ):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    bot._is_definitive_rejection.return_value = False
    bot._client.change_ask.return_value = "tx-change"
    bot._client.place_sell_order.return_value = "tx-sell"
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, old_price, old_amount, "old", level_idx=0,
                     is_inventory_backed=inventory_backed)
    inv.refreshed_at = time.time() + 60  # the old ask was seen by the read
    return bot, ctx, mgr, inv


def _update(bot, ctx, mgr, price, amount):
    return AvellanedaMarketMaker._update_single_order(
        bot, ctx, True, Side.ASK, price, amount, mgr, 0
    )


def test_same_size_move_needs_no_free_shares():
    bot, ctx, mgr, _ = _setup(held=0)

    assert _update(bot, ctx, mgr, 39, 6) == "tx-change"
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 6
    bot._client.cancel_order.assert_not_called()
    assert ctx.yes_orders.get_ask(0).price == 39


def test_short_book_moves_at_a_capped_size_instead_of_pulling():
    # Target x9, but only the 6 listed plus 2 free are ours to use.
    bot, ctx, mgr, _ = _setup(held=2)

    assert _update(bot, ctx, mgr, 39, 9) == "tx-change"
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 8
    assert ctx.yes_orders.get_ask(0).amount == 8
    bot._client.cancel_order.assert_not_called()


def test_smaller_target_moves_at_the_target():
    bot, ctx, mgr, _ = _setup(held=0)

    _update(bot, ctx, mgr, 45, 4)
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 4
    assert ctx.yes_orders.get_ask(0).amount == 4


def test_move_below_protocol_minimum_falls_back():
    # 10c x6 = 60 cent-shares: change_ask would revert, so the old path
    # (keep or pull) decides instead.
    bot, ctx, mgr, _ = _setup(held=0, old_price=12)

    _update(bot, ctx, mgr, 10, 6)
    bot._client.change_ask.assert_not_called()


def test_partly_filled_ask_is_corrected_not_retried_forever():
    # The chain measured the extra shares against the 2 that still rest.
    bot, ctx, mgr, _ = _setup(held=0)
    bot._client.change_ask.side_effect = SHORT

    assert _update(bot, ctx, mgr, 39, 6) is None
    order = ctx.yes_orders.get_ask(0)
    assert (order.price, order.amount) == (38, 2)
    bot._order_state.update_order.assert_called_once()
    bot._client.cancel_order.assert_not_called()


def test_old_order_gone_takes_the_existing_not_found_path():
    # Held shares and a larger target, so a wrongly noted unconfirmed move
    # would show up in unconfirmed_sells.
    bot, ctx, mgr, inv = _setup(held=4)
    bot._client.change_ask.side_effect = Exception(
        "ERROR: Old order not found at specified price"
    )

    assert _update(bot, ctx, mgr, 39, 10) is None
    assert ctx.yes_orders.get_ask(0) is None
    assert inv.unconfirmed_sells == []


def test_refresh_clears_the_holdings_cap():
    inv = MarketInventory(query_id=5)
    inv.note_holdings(True, 0)
    inv.update_from_positions(yes_shares=4, no_shares=0, refreshed_at=1.0)
    assert inv.holdings_cap == {}
    assert inv.free_to_sell(True, []) == 4


def test_legacy_split_record_does_not_use_change_ask():
    bot, ctx, mgr, _ = _setup(held=6, inventory_backed=False)

    _update(bot, ctx, mgr, 39, 6)
    bot._client.change_ask.assert_not_called()
    assert [c.kwargs["price"] for c in bot._client.cancel_order.call_args_list] == [38]


def test_moved_ask_counts_as_unseen_until_the_next_refresh():
    bot, ctx, mgr, inv = _setup(held=0)

    _update(bot, ctx, mgr, 39, 6)
    moved = ctx.yes_orders.get_ask(0)
    assert inv.free_to_sell(True, [(39, 6, moved.created_at)]) == 0


def _seen_then_fresh(ctx, inv):
    """The tracked ask predates the last read; anything moved later is new."""
    ctx.yes_orders.get_ask(0).created_at = time.time() - 100
    inv.refreshed_at = time.time() - 50


def test_second_move_before_refresh_does_not_spend_the_same_shares():
    # Review finding 1: 4 held, ask 38c x6. Move 1 to 39c x10 takes the 4.
    # Move 2 (down to 37c, target x14) before the next read must not count
    # those 4 again: it moves at x10 with nothing extra from holdings.
    bot, ctx, mgr, inv = _setup(held=4)
    _seen_then_fresh(ctx, inv)

    assert _update(bot, ctx, mgr, 39, 10) == "tx-change"
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 10
    assert _update(bot, ctx, mgr, 37, 14) == "tx-change"
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 10


def test_partial_fill_seen_by_the_read_moves_what_rests():
    # Tracked x6 but the read listed only 2 (4 filled), nothing held: move
    # x2 instead of asking the chain for 4 shares that do not exist.
    bot, ctx, mgr, inv = _setup(held=0, old_price=58, listed={(True, 58): 2})

    _update(bot, ctx, mgr, 60, 6)  # 60c x2 clears the protocol minimum
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 2


def test_holdings_short_without_a_fill_does_not_loop():
    # Review finding 3: the read says 4 held, the chain has 0.
    bot, ctx, mgr, inv = _setup(held=4)
    bot._client.change_ask.side_effect = [
        Exception("ERROR: Insufficient shares in holdings. Need 4 more "
                  "shares, but only have 0 in holdings."),
        "tx-change",
    ]

    assert _update(bot, ctx, mgr, 39, 10) is None
    assert ctx.yes_orders.get_ask(0).amount == 6  # nothing filled
    assert _update(bot, ctx, mgr, 39, 10) == "tx-change"
    assert bot._client.change_ask.call_args.kwargs["new_amount"] == 6


def test_timed_out_move_holds_back_the_extra_shares():
    # Review finding 4: the move may still land and pull 4 from holdings.
    bot, ctx, mgr, inv = _setup(held=4)
    bot._client.change_ask.side_effect = Exception(
        "tx abc unconfirmed after 30s (node likely dropped it)"
    )

    _update(bot, ctx, mgr, 39, 10)
    assert inv.free_to_sell(True, []) == 0

"""Regression tests for the free-share count behind inventory-backed asks.

available_for_sell() was held - reserved. Held shares (price=0 positions)
already exclude every listed sell, and reservations counted the same listed
asks again, so each resting inventory-backed ask was subtracted twice; a
filled ask also kept its reservation until restart. On mainnet (2026-09-24)
a market holding 21 free YES shares with 11 listed across two asks showed
10 free, which pushed new asks onto the split-mint fallback.

free_to_sell() subtracts an ask placed before the last refresh read only for
the part that read did not see listed, subtracts an ask placed since in
full, and rounds every unknown down (a cancel's shares count again only
once a refresh sees them held).
"""

import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import AvellanedaMarketMaker
from market_maker_bot.market import OrderManager, Side
from market_maker_bot.models import ActiveOrders
from market_maker_bot.pricing.inventory import (
    UNCONFIRMED_SELL_TTL_S,
    InventoryManager,
    MarketInventory,
)


SEEN = 100.0  # placed before the refresh read (refreshed_at = 200)
NEW = 300.0  # placed after it


def _inv(yes=0, no=0, listed=None):
    inv = MarketInventory(query_id=5)
    inv.yes_shares = yes
    inv.no_shares = no
    inv.listed_by_price = dict(listed or {})
    inv.refreshed_at = 200.0
    return inv


# --- MarketInventory.free_to_sell ------------------------------------------

def test_listed_asks_are_not_subtracted_twice():
    # The mainnet case: 21 held, our asks 41c x6 and 43c x5 listed, plus a
    # 39c x4 listing the bot does not track as inventory-backed.
    inv = _inv(yes=21, listed={(True, 39): 4, (True, 41): 6, (True, 43): 5})
    inv.reserve_pair(True, 11)

    assert inv.available_for_sell(True) == 10  # the old count
    assert inv.free_to_sell(True, [(41, 6, SEEN), (43, 5, SEEN)]) == 21


def test_ask_placed_since_the_refresh_is_subtracted():
    inv = _inv(yes=21)

    assert inv.free_to_sell(True, [(41, 6, NEW)]) == 15


def test_ask_placed_since_the_refresh_is_subtracted_even_over_an_orphan():
    # An untracked x6 already rests at 41c. The chain keeps one order per
    # price, so the read cannot tell the orphan from the new ask.
    inv = _inv(yes=21, listed={(True, 41): 6})

    assert inv.free_to_sell(True, [(41, 6, NEW)]) == 15


def test_stale_read_that_misses_a_listing_rounds_down():
    # The read still counts the 6 listed shares as held.
    inv = _inv(yes=27)

    assert inv.free_to_sell(True, [(41, 6, SEEN)]) == 21


def test_filled_ask_counts_only_while_tracked():
    inv = _inv(yes=15)  # the ask filled: no longer listed, not held

    assert inv.free_to_sell(True, [(41, 6, SEEN)]) == 9  # rounds down
    assert inv.free_to_sell(True, []) == 15  # no leak once untracked


def test_partly_filled_ask_subtracts_its_filled_part():
    inv = _inv(yes=15, listed={(True, 41): 4})

    assert inv.free_to_sell(True, [(41, 6, SEEN)]) == 13


def test_cancelled_ask_frees_nothing_until_the_next_refresh():
    # The ask may have partly filled since the read: only a refresh knows
    # how many shares the cancel returned.
    inv = _inv(yes=0, listed={(True, 38): 6})

    assert inv.free_to_sell(True, [(38, 6, SEEN)]) == 0
    assert inv.free_to_sell(True, []) == 0


def test_unconfirmed_sell_is_held_back_until_a_read_shows_it():
    inv = _inv(yes=10)
    inv.note_unconfirmed_sell(True, 41, 6, noted_at=250.0)

    assert inv.free_to_sell(True, []) == 4
    assert inv.free_to_sell(False, []) == 0

    # The next read comes before it lands: still held back.
    inv.update_from_positions(yes_shares=10, no_shares=0, refreshed_at=300.0)
    assert inv.free_to_sell(True, []) == 4

    # It landed: the read lists it and held already excludes it.
    inv.update_from_positions(yes_shares=4, no_shares=0,
                              listed_by_price={(True, 41): 6},
                              refreshed_at=330.0)
    assert inv.unconfirmed_sells == []
    assert inv.free_to_sell(True, []) == 4


def test_unconfirmed_sell_expires_after_the_ttl():
    inv = _inv(yes=10)
    inv.note_unconfirmed_sell(True, 41, 6, noted_at=250.0)

    inv.update_from_positions(yes_shares=10, no_shares=0,
                              refreshed_at=250.0 + UNCONFIRMED_SELL_TTL_S)
    assert inv.free_to_sell(True, []) == 10


def test_untracked_listings_are_ignored():
    inv = _inv(yes=5, listed={(True, 39): 4})

    assert inv.free_to_sell(True, []) == 5


def test_outcomes_are_counted_separately():
    inv = _inv(yes=5, no=7, listed={(False, 70): 3})

    assert inv.free_to_sell(True, [(70, 3, SEEN)]) == 2
    assert inv.free_to_sell(False, [(70, 3, SEEN)]) == 7


def test_refresh_rebuilds_listings_and_resets_absent_markets():
    mgr = InventoryManager()
    gone = mgr.get_market_inventory(9)
    gone.yes_shares = 10
    gone.listed_by_price = {(True, 50): 4}

    mgr.update_from_user_positions([
        {"query_id": 5, "outcome": True, "price": 0, "amount": 21},
        {"query_id": 5, "outcome": True, "price": 43, "amount": 5},
        {"query_id": 5, "outcome": True, "price": 43, "amount": 1},
        {"query_id": 5, "outcome": False, "price": -60, "amount": 4},
    ], as_of=500.0)

    inv = mgr.get_market_inventory(5)
    assert inv.listed_by_price == {(True, 43): 6}
    assert inv.refreshed_at == 500.0
    assert inv.free_to_sell(True, [(43, 6, 400.0)]) == 21
    assert (gone.yes_shares, gone.listed_by_price) == (0, {})


# --- bot wiring --------------------------------------------------------------

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


def _bot(inv):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False,
        backstop_amount=0,
        backstop_price_cents=2,
        level_loop_threshold=3,
        level_loop_window=120.0,
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
        "_free_to_sell",
        "_place_ask",
        "_cancel_ask",
        "_level_slot_cooling",
        "_note_level_not_found_clear",
        "_note_cancel_not_found",
    ):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    bot._is_definitive_rejection.return_value = False
    bot._client.place_sell_order.return_value = "tx-new"
    return bot


def _seen(inv):
    # Orders recorded now count as placed before the last refresh read.
    inv.refreshed_at = time.time() + 60


def test_new_ask_places_from_shares_the_old_count_hid():
    # 21 held, two of our asks listed: the old count said 10 < 12.
    inv = _inv(yes=21, listed={(True, 41): 6, (True, 43): 5})
    bot = _bot(inv)
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, 41, 6, "a", level_idx=1,
                     is_inventory_backed=True)
    mgr.record_order(True, Side.ASK, 43, 5, "b", level_idx=2,
                     is_inventory_backed=True)
    _seen(inv)

    assert bot._place_ask(ctx, True, 39, 12) == ("tx-new", True)


def test_refresh_moves_an_ask_from_held_shares():
    inv = _inv(yes=6, listed={(True, 38): 6})
    bot = _bot(inv)
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, 38, 6, "old", level_idx=0,
                     is_inventory_backed=True)
    _seen(inv)

    result = AvellanedaMarketMaker._update_single_order(
        bot, ctx, True, Side.ASK, 39, 6, mgr, 0
    )

    assert result == "tx-new"
    assert [c.kwargs["price"] for c in bot._client.cancel_order.call_args_list] == [38]
    assert ctx.yes_orders.get_ask(0).price == 39
    bot._order_state.untrack_order.assert_called_once_with(
        query_id=5, outcome=True, is_buy=False, price=38, level_idx=0
    )


def test_refresh_with_everything_listed_pulls_rather_than_oversells():
    # A taker may have filled part of the 38c ask since the read, so the
    # cancel's returned shares are not counted: the ask is pulled and the
    # level waits for the next refresh instead of selling shares that may
    # not exist.
    inv = _inv(yes=0, listed={(True, 38): 6})
    bot = _bot(inv)
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, 38, 6, "old", level_idx=0,
                     is_inventory_backed=True)
    _seen(inv)

    assert AvellanedaMarketMaker._update_single_order(
        bot, ctx, True, Side.ASK, 39, 6, mgr, 0
    ) is None
    bot._client.place_sell_order.assert_not_called()
    assert ctx.yes_orders.get_ask(0) is None


def test_failed_place_after_cancel_leaves_no_stale_record():
    inv = _inv(yes=6, listed={(True, 38): 6})
    bot = _bot(inv)
    bot._client.place_sell_order.side_effect = Exception("gateway timeout")
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, 38, 6, "old", level_idx=0,
                     is_inventory_backed=True)
    _seen(inv)

    AvellanedaMarketMaker._update_single_order(
        bot, ctx, True, Side.ASK, 39, 6, mgr, 0
    )

    assert ctx.yes_orders.get_ask(0) is None
    bot._order_state.untrack_order.assert_called_once_with(
        query_id=5, outcome=True, is_buy=False, price=38, level_idx=0
    )
    # The timed-out sell may still land: its shares stay out of the count.
    assert inv.free_to_sell(True, []) == 0


def test_definitive_rejection_is_not_held_back():
    inv = _inv(yes=6)
    bot = _bot(inv)
    bot._is_definitive_rejection.return_value = True
    bot._client.place_sell_order.side_effect = Exception("rejected")

    try:
        bot._place_ask(_Ctx(), True, 39, 6)
    except Exception:
        pass
    assert inv.unconfirmed_sells == []


def test_refresh_stamps_the_read_before_it_starts():
    bot = MagicMock()
    bot.config = SimpleNamespace(dry_run=False)
    stamps = []

    def read():
        stamps.append(time.time())
        return []

    bot._client.get_user_positions.side_effect = read
    before = time.time()
    AvellanedaMarketMaker._refresh_inventory(bot)

    as_of = bot._inventory.update_from_user_positions.call_args.kwargs["as_of"]
    assert before <= as_of <= stamps[0]

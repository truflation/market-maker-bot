"""Burn held YES+NO pairs once a market is pulled before settlement.

A pair carried to settlement pays the winner 98c (2c fee) and the loser 0.
A YES sell at X resting against a NO sell at 100 - X burns for the full
100c with no fee, whichever side wins (match_burn, node migration 032).
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import (
    BURN_RETRY_INTERVAL,
    PULL_RECONCILE_GRACE,
    AvellanedaMarketMaker,
)
from market_maker_bot.config import MarketConfig
from market_maker_bot.market import MarketContext
from market_maker_bot.pricing.inventory import MarketInventory

OURS = "0xaa00000000000000000000000000000000000aa0"
THEM = "0xbb00000000000000000000000000000000000bb0"


def _setup(yes=12, no=12, books=None, settle_in=3600, pulled_ago=600, refreshed_ago=10):
    now = time.time()
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False, read_only=False, maa_address=OURS, private_key="",
        burn_pairs_on_pull=True, decided_mode=False,
    )
    bot._pre_settlement_pulled_at = {7: now - pulled_ago}
    bot._burn_attempt_at = {}
    bot._pulled_listings = {}
    inv = MarketInventory(query_id=7)
    inv.update_from_positions(yes_shares=yes, no_shares=no, refreshed_at=now - refreshed_ago)
    bot._inventory.get_market_inventory.return_value = inv
    books = books or {True: [], False: []}
    bot._client.get_order_book.side_effect = lambda q, o: books[o]
    for name in ("_burn_held_pairs", "_order_owner_address", "_burn_cancel_leg",
                 "_burn_reference_fair"):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    ctx = MarketContext(config=MarketConfig(
        query_id=7, stream_id="s", data_provider="p", settle_time=int(now) + settle_in))
    return bot, ctx


def _sells(bot):
    return [c.args for c in bot._client.place_sell_order.call_args_list]


def test_burns_every_held_pair_at_par():
    bot, ctx = _setup(yes=12, no=12)
    assert bot._burn_held_pairs(ctx) == 12
    assert _sells(bot) == [(7, True, 50, 12), (7, False, 50, 12)]


def test_only_complete_pairs_are_burned():
    bot, ctx = _setup(yes=15, no=9)
    assert bot._burn_held_pairs(ctx) == 9
    assert _sells(bot)[0] == (7, True, 50, 9)


def test_price_clears_other_wallets_bids_so_nothing_direct_fills():
    # A foreign YES bid at 60 would take a YES sell at 50 directly.
    books = {True: [{"wallet_address": THEM, "price": -60, "amount": 5}],
             False: [{"wallet_address": THEM, "price": -30, "amount": 5}]}
    bot, ctx = _setup(books=books)
    bot._burn_held_pairs(ctx)
    yes_price = _sells(bot)[0][2]
    assert yes_price > 60 and 100 - yes_price > 30
    assert _sells(bot)[1][2] == 100 - yes_price


def test_price_avoids_other_wallets_asks_so_the_burn_pairs_our_own_legs():
    books = {True: [{"wallet_address": THEM, "price": 50, "amount": 5}], False: []}
    bot, ctx = _setup(books=books)
    bot._burn_held_pairs(ctx)
    assert _sells(bot)[0][2] != 50


def test_too_few_pairs_for_the_minimum_order_is_left_alone():
    bot, ctx = _setup(yes=1, no=1)
    assert bot._burn_held_pairs(ctx) == 0
    bot._client.place_sell_order.assert_not_called()


def test_our_own_resting_orders_are_cleared_first():
    books = {True: [{"wallet_address": OURS, "price": 50, "amount": 3}], False: []}
    bot, ctx = _setup(books=books)
    assert bot._burn_held_pairs(ctx) == 0
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=True, price=50, wait=True)
    bot._client.place_sell_order.assert_not_called()


def test_failed_no_leg_pulls_both_legs():
    bot, ctx = _setup()
    bot._client.place_sell_order.side_effect = ["tx", RuntimeError("gateway timeout")]
    assert bot._burn_held_pairs(ctx) == 0
    cancels = [c.kwargs for c in bot._client.cancel_order.call_args_list]
    assert dict(query_id=7, outcome=True, price=50, wait=True) in cancels
    assert dict(query_id=7, outcome=False, price=50, wait=True) in cancels


def test_failed_yes_leg_is_pulled_in_case_it_landed():
    # Review: a timed-out wait=True write can still land.
    bot, ctx = _setup()
    bot._client.place_sell_order.side_effect = RuntimeError("tx unconfirmed after 30s")
    assert bot._burn_held_pairs(ctx) == 0
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=True, price=50, wait=True)
    assert bot._inventory.get_market_inventory.return_value.free_to_sell(True, []) == 0


def test_price_is_centred_on_fair():
    # Review: a leg another wallet takes between our placements sells near
    # value, not at 50. Fair 90, 12 pairs: NO leg needs 12 x (100-X) >= 100.
    bot, ctx = _setup()
    ctx.initial_price_yes = 90.0
    bot._burn_held_pairs(ctx)
    assert _sells(bot)[0][2] == 90


def test_a_leftover_leg_is_cleared_even_with_under_two_pairs():
    # Review: after a race leaves < 2 pairs, our lone leg must still go.
    books = {True: [{"wallet_address": OURS, "price": 50, "amount": 11}], False: []}
    bot, ctx = _setup(yes=1, no=12, books=books)
    bot._burn_held_pairs(ctx)
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=True, price=50, wait=True)


def test_waits_for_the_pull_grace_and_a_fresh_holdings_read():
    bot, ctx = _setup(pulled_ago=PULL_RECONCILE_GRACE - 10)
    assert bot._burn_held_pairs(ctx) == 0
    bot, ctx = _setup(pulled_ago=600, refreshed_ago=700)   # read predates the pull
    assert bot._burn_held_pairs(ctx) == 0
    bot._client.place_sell_order.assert_not_called()


def test_attempts_are_spaced_out():
    bot, ctx = _setup()
    bot._client.place_sell_order.side_effect = RuntimeError("down")
    bot._burn_held_pairs(ctx)
    bot._burn_held_pairs(ctx)
    assert bot._client.place_sell_order.call_count == 1
    bot._burn_attempt_at[7] -= BURN_RETRY_INTERVAL
    inv = bot._inventory.get_market_inventory.return_value
    inv.refreshed_at = time.time()
    inv.unconfirmed_sells = []   # the failed leg's hold-back has expired
    bot._burn_held_pairs(ctx)
    assert bot._client.place_sell_order.call_count == 2


def test_never_burns_close_to_settle():
    bot, ctx = _setup(settle_in=150)
    assert bot._burn_held_pairs(ctx) == 0
    bot._client.place_sell_order.assert_not_called()


def test_pull_loop_burns_only_markets_already_pulled():
    now = time.time()
    bot = MagicMock()
    bot.config = SimpleNamespace(pre_settlement_cutoff=1800.0, burn_pairs_on_pull=True,
                                 decided_mode=False)
    bot._pre_settlement_pulled = {7}
    bot._earnings_pulled_session = set()
    ctx = MarketContext(config=MarketConfig(
        query_id=7, stream_id="s", data_provider="p", settle_time=int(now) + 900))
    AvellanedaMarketMaker._process_market(bot, ctx)
    bot._burn_held_pairs.assert_called_once_with(ctx)

    bot.config.burn_pairs_on_pull = False
    bot._burn_held_pairs.reset_mock()
    AvellanedaMarketMaker._process_market(bot, ctx)
    bot._burn_held_pairs.assert_not_called()

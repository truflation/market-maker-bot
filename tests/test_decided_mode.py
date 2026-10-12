"""decided_mode: keep held shares listed on a pulled market.

Settlement pays a listed winning ask 98c, the same as a holding, and a
loser 0 (node 033 process_settlement). So a held share is never worse off
listed: at 99c it beats settlement whatever the outcome or any revision,
and once the band is decided a losing share sells for something instead of
nothing, instead of every share sitting at price 0 from the pull to
`settled`.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import AvellanedaMarketMaker
from market_maker_bot.config import MarketConfig
from market_maker_bot.market import MarketContext
from market_maker_bot.pricing.inventory import MarketInventory

OURS = "0xaa00000000000000000000000000000000000aa0"
DAY = 86400


def _setup(yes=0, no=3, books=None, value=2.85, lo=2.79, hi=2.91, margin=0.03,
           record_age=3600, settle_in=5400, burn=True):
    now = time.time()
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False, read_only=False, maa_address=OURS, private_key="",
        burn_pairs_on_pull=burn, decided_mode=True, decided_loser_ask_cents=3,
        decided_edge_margin=margin, decided_min_age=600.0, stream_max_staleness=600.0,
    )
    bot._pre_settlement_pulled_at = {7: now - 600}
    bot._burn_attempt_at, bot._pulled_listings, bot._decided_since = {}, {}, {}
    inv = MarketInventory(query_id=7)
    inv.update_from_positions(yes_shares=yes, no_shares=no, refreshed_at=now - 5)
    bot._inventory.get_market_inventory.return_value = inv
    books = books or {True: [], False: []}
    bot._client.get_order_book.side_effect = lambda q, o: books[o]
    t_last = int(now) - record_age
    times = [t_last - i * DAY for i in range(150)][::-1]
    bot._stream_series.return_value = (times, [value] * 150, now)
    for name in ("_burn_held_pairs", "_order_owner_address", "_burn_cancel_leg",
                 "_burn_reference_fair", "_decided_winner", "_pulled_targets",
                 "_list_held_shares", "_guard_decided_listings"):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    ctx = MarketContext(config=MarketConfig(
        query_id=7, stream_id="s", data_provider="p", lower_bound=lo, upper_bound=hi,
        settle_time=int(now) + settle_in, fair_model="stream_bands"))
    return bot, ctx


def _sells(bot):
    return [c.args for c in bot._client.place_sell_order.call_args_list]


def test_unpaired_shares_are_listed_at_99_before_the_band_is_trusted():
    # First sighting of the decided value: not yet stable, so 99c both ways.
    # (1 NO share at 99c is under the minimum order, so only YES lists.)
    bot, ctx = _setup(yes=40, no=1)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, True, 99, 40)]
    assert bot._pulled_listings[7] == {True: 99}


def test_a_stable_decided_band_offers_its_losing_side_low():
    # Value 2.85 sits inside [2.79, 2.91] by 0.06 > margin 0.03: YES won.
    bot, ctx = _setup(yes=0, no=40)
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 3, 40)]


def test_a_value_near_the_band_edge_keeps_everything_at_99():
    bot, ctx = _setup(yes=0, no=40, value=2.90)   # 0.01 from the upper edge
    bot._decided_since[7] = (2.90, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 99, 40)]


def test_a_changed_value_restarts_the_stability_clock():
    bot, ctx = _setup(yes=0, no=40)
    bot._decided_since[7] = (2.86, time.time() - 700)   # revised since
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 99, 40)]
    assert bot._decided_since[7][0] == 2.85


def test_an_undecided_band_stays_at_99():
    bot, ctx = _setup(yes=0, no=40, record_age=30 * 3600)   # next record not landed
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 99, 40)]


def test_a_listing_at_its_target_is_kept_and_a_stale_one_is_replaced():
    books = {True: [{"wallet_address": OURS, "price": 99, "amount": 5}],
             False: [{"wallet_address": OURS, "price": 99, "amount": 40}]}
    bot, ctx = _setup(yes=0, no=0, books=books)
    bot._decided_since[7] = (2.85, time.time() - 700)   # NO now targets 3c
    bot._burn_held_pairs(ctx)
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=False, price=99, wait=True)
    assert bot._pulled_listings[7] == {True: 99}


def test_pairs_are_burned_before_anything_is_listed():
    bot, ctx = _setup(yes=12, no=12)
    bot._burn_held_pairs(ctx)
    yes_leg, no_leg = _sells(bot)
    assert yes_leg[3] == no_leg[3] == 12 and yes_leg[2] + no_leg[2] == 100


def test_without_burning_held_pairs_are_listed_at_99():
    bot, ctx = _setup(yes=12, no=12, burn=False)
    bot._burn_held_pairs(ctx)
    assert sorted(_sells(bot)) == [(7, False, 99, 12), (7, True, 99, 12)]


def test_listings_below_the_minimum_order_are_skipped():
    bot, ctx = _setup(yes=0, no=20)   # 20 x 3c < 100 cent-shares
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    bot._client.place_sell_order.assert_not_called()


def test_reconcile_leaves_decided_listings_alone():
    from tests.test_level_slot_loop import _OURS, _reconcile_bot

    bot = _reconcile_bot(settle_time=int(time.time()) + 900)
    bot.config.pre_settlement_cutoff = 1800.0
    bot._pulled_listings = {7: {False: 99}}
    bot._client.get_order_book.side_effect = lambda q, o: (
        [] if o else [{"wallet_address": _OURS, "price": 99, "amount": 2},
                      {"wallet_address": _OURS, "price": 40, "amount": 2}])
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=False, price=40, wait=False)


def test_a_record_stamped_after_settle_does_not_decide_the_band():
    # Review: the shared stream read reaches later rungs' settles. Settlement
    # reads the latest record at or before settle (2.85: YES wins), not the
    # later 3.10.
    bot, ctx = _setup(yes=40, no=40, burn=False)
    times, values, fetched = bot._stream_series.return_value
    S = ctx.config.settle_time
    bot._stream_series.return_value = (times + [S + 3600], values + [3.10], fetched)
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert sorted(_sells(bot)) == [(7, False, 3, 40), (7, True, 99, 40)]


def test_a_stale_stream_read_trusts_no_decision():
    bot, ctx = _setup(yes=0, no=40)
    times, values, _ = bot._stream_series.return_value
    bot._stream_series.return_value = (times, values, time.time() - 6 * 3600)
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 99, 40)]


def test_a_revision_pulls_the_loser_listing_at_once_without_the_burn_gates():
    # Review: the cancel must not wait for the retry spacing or a fresh
    # holdings read. The record now reads 2.95: outside the band, NO wins,
    # so the resting NO@3 is the winner.
    bot, ctx = _setup(yes=0, no=0, value=2.95)
    bot._pulled_listings[7] = {False: 3}
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._inventory.get_market_inventory.return_value.refreshed_at = 0   # reads failing
    bot._guard_decided_listings(ctx)
    bot._client.cancel_order.assert_called_once_with(query_id=7, outcome=False, price=3, wait=True)
    assert bot._pulled_listings[7] == {}


def test_a_confirmed_loser_listing_is_left_alone():
    bot, ctx = _setup(yes=0, no=0)
    bot._pulled_listings[7] = {False: 3}
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._guard_decided_listings(ctx)
    bot._client.cancel_order.assert_not_called()


def test_a_zero_margin_never_decides():
    bot, ctx = _setup(yes=0, no=40, margin=0)
    bot._decided_since[7] = (2.85, time.time() - 700)
    bot._burn_held_pairs(ctx)
    assert _sells(bot) == [(7, False, 99, 40)]


def test_pull_loop_guards_listings_every_cycle():
    now = time.time()
    bot = MagicMock()
    bot.config = SimpleNamespace(pre_settlement_cutoff=1800.0, burn_pairs_on_pull=False,
                                 decided_mode=True)
    bot._pre_settlement_pulled = {7}
    bot._earnings_pulled_session = set()
    ctx = MarketContext(config=MarketConfig(
        query_id=7, stream_id="s", data_provider="p", settle_time=int(now) + 900))
    AvellanedaMarketMaker._process_market(bot, ctx)
    bot._guard_decided_listings.assert_called_once_with(ctx)
    bot._burn_held_pairs.assert_called_once_with(ctx)


def test_the_guard_stops_at_settle():
    bot, ctx = _setup(yes=0, no=0, value=2.95, settle_in=-30)
    bot._pulled_listings[7] = {False: 3}
    bot._guard_decided_listings(ctx)
    bot._client.cancel_order.assert_not_called()

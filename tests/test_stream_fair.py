"""fair_model="stream_bands": the bot reprices bands from the stream.

2026-10-08: daily ladders quoted a fixed per-band prior (centre 36%) for a
rung's whole life. The centre band paid almost every day, and kept being
offered at 38-41c after the deciding print was on chain; one informed taker
made 53 fills on it. These tests pin the replacement: price from the
stream every cycle, stop quoting a decided band, fail closed on stale
reads, never quote through fair.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from market_maker_bot.bot import (
    FAIR_PULL_RECONCILE_GRACE,
    FAIR_PULL_RETRY,
    AvellanedaMarketMaker,
)
from market_maker_bot.config import MarketConfig, load_config_from_dict
from market_maker_bot.market import MarketContext
from market_maker_bot.market import OrderManager
from market_maker_bot.models import OutcomeMode, Side

DAY = 86400


def _records(n=150, step=0.0005, start=2.80, last_age=3600):
    t_last = int(time.time()) - last_age
    return [{"EventTime": str(t_last - (n - 1 - i) * DAY), "Value": str(start + i * step)}
            for i in range(n)]


def _cfg(qid=7, lo=2.67, hi=2.91, settle_in=26 * 3600, prior=0.36, **kw):
    return MarketConfig(
        query_id=qid, stream_id="st1", data_provider="0xABC",
        outcome_mode=OutcomeMode.BOTH, lower_bound=lo, upper_bound=hi,
        settle_time=int(time.time()) + settle_in, initial_probability=prior,
        fair_model="stream_bands", **kw,
    )


def _bot(records):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        stream_refresh_interval=30.0, stream_max_staleness=600.0,
        stream_max_record_age=144000.0,
        stream_lookback_days=200, pre_settlement_cutoff=1800.0,
        pricing_source="black_scholes",
        avellaneda=SimpleNamespace(min_spread_cents=4.0, hanging_orders_enabled=False),
    )
    bot._stream_cache, bot._stream_attempt = {}, {}
    bot._fair_pulled, bot._last_fair, bot._fair_pull_retry_at = {}, {}, {}
    bot._pre_settlement_pulled = set()
    bot._markets = {}
    bot._earnings_pulled_session = set()
    bot._client.get_records.return_value = records
    bot._create_proposal_from_order_override.return_value = None
    for name in ("_stream_series", "_fair_gate", "_clamp_to_fair"):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    return bot


def _process(bot, ctx):
    AvellanedaMarketMaker._process_market(bot, ctx)


def test_quotes_the_centre_band_from_the_stream_not_the_prior():
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg())
    _process(bot, ctx)
    assert ctx.initial_price_yes >= 90 and ctx.initial_price_no <= 10
    bot._cancel_market_orders.assert_not_called()
    bot._execute_order_updates.assert_called()


def test_decided_band_is_pulled_once_and_not_quoted():
    # The deciding record (stamped an hour ago) is on chain; settle in 2h.
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg(settle_in=2 * 3600))
    _process(bot, ctx)
    _process(bot, ctx)
    bot._cancel_market_orders.assert_called_once_with(ctx)
    bot._execute_order_updates.assert_not_called()
    assert bot._fair_pulled[ctx.query_id][0] == "decided"


def test_failed_reads_fail_closed_then_resume():
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg())
    bot._client.get_records.side_effect = RuntimeError("gateway down")
    _process(bot, ctx)
    assert bot._fair_pulled[ctx.query_id][0] == "stale"
    bot._cancel_market_orders.assert_called_once()
    bot._execute_order_updates.assert_not_called()

    bot._client.get_records.side_effect = None
    bot._stream_attempt.clear()  # next refresh interval
    _process(bot, ctx)
    assert ctx.query_id not in bot._fair_pulled
    bot._execute_order_updates.assert_called()


def test_old_read_goes_stale_even_without_errors():
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg())
    _process(bot, ctx)
    key = ("st1", "0xabc")
    times, values, _ = bot._stream_cache[key]
    bot._stream_cache[key] = (times, values, time.time() - 700)
    bot._stream_attempt[key] = time.time()  # retry not due, cache too old
    assert AvellanedaMarketMaker._fair_gate(bot, ctx) == "stale"


def test_one_read_per_stream_shared_by_its_markets():
    bot = _bot(_records())
    ctxs = [MarketContext(config=_cfg(qid=q, lo=lo, hi=hi)) for q, lo, hi in
            ((1, None, 2.67), (2, 2.67, 2.79), (3, 2.79, 2.91), (4, 2.91, None))]
    for _ in range(3):
        for c in ctxs:
            _process(bot, c)
    assert bot._client.get_records.call_count == 1


def test_markets_without_fair_model_never_read_the_stream():
    bot = _bot(_records())
    cfg = _cfg()
    cfg.fair_model = None
    bot._calculate_initial_price.return_value = 36.0
    ctx = MarketContext(config=cfg)
    _process(bot, ctx)
    bot._client.get_records.assert_not_called()
    assert ctx.initial_price_yes == 36.0


@pytest.mark.parametrize("outcome,fair_yes,quotes,expected", [
    (True, 95, (93, 94), (93, 97)),   # skewed ask below fair -> lifted to fair+2
    (True, 95, (97, 99), (93, 99)),   # skewed bid above fair -> cut to fair-2
    (False, 95, (1, 3), (1, 7)),      # NO side mirrors (NO fair 5)
    (True, 98, (90, 99), (90, 99)),   # never past 99
])
def test_quotes_never_cross_fair(outcome, fair_yes, quotes, expected):
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg())
    ctx.initial_price_yes, ctx.initial_price_no = float(fair_yes), float(100 - fair_yes)
    assert AvellanedaMarketMaker._clamp_to_fair(bot, ctx, outcome, *quotes) == expected


@pytest.mark.parametrize("kw", [
    dict(lower_bound=None, upper_bound=None, settle_time=1, fair_model="stream_bands"),
    dict(upper_bound=1.0, fair_model="stream_bands"),  # no settle_time
    dict(upper_bound=1.0, settle_time=1, fair_model="bs"),
])
def test_config_rejects_unusable_fair_model_specs(kw):
    with pytest.raises(ValueError):
        MarketConfig(query_id=1, stream_id="s", data_provider="p", **kw)


def test_loader_reads_the_stream_settings():
    cfg = load_config_from_dict({
        "stream_refresh_interval": 15, "stream_max_staleness": 120,
        "stream_lookback_days": 90,
        "markets": [{"query_id": 1, "stream_id": "s", "data_provider": "p",
                     "upper_bound": 2.0, "settle_time": 2_000_000_000,
                     "fair_model": "stream_bands", "outcome_mode": "both"}],
    })
    assert (cfg.stream_refresh_interval, cfg.stream_max_staleness,
            cfg.stream_lookback_days) == (15, 120, 90)
    assert cfg.markets[0].fair_model == "stream_bands"


def test_reconcile_skips_a_just_pulled_market_then_sweeps_it():
    from tests.test_reconcile import TEST_WALLET, _book_entry, _bot_mock

    qid = 100
    bot = _bot_mock([], {(qid, True): [_book_entry(TEST_WALLET, 40)], (qid, False): []})
    bot._fair_pulled = {qid: ("decided", time.time())}
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    bot._client.cancel_order.assert_not_called()

    bot._fair_pulled = {qid: ("decided", time.time() - FAIR_PULL_RECONCILE_GRACE - 1)}
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)
    bot._client.cancel_order.assert_called_once_with(
        query_id=qid, outcome=True, price=40, wait=False)


def test_a_failed_pull_cancel_is_retried_until_the_market_is_clear():
    # Review finding 1: a cancel that raises leaves the order tracked and
    # resting; reconcile never sweeps a tracked order, so the gate retries.
    bot = _bot(_records())
    ctx = MarketContext(config=_cfg(settle_in=2 * 3600))
    OrderManager(ctx, 0.0, 1e9).record_order(True, Side.ASK, 40, 6, "tx", level_idx=0)
    _process(bot, ctx)                       # pull; the mock cancel clears nothing
    _process(bot, ctx)                       # inside the retry interval: no retry
    assert bot._cancel_market_orders.call_count == 1
    bot._fair_pull_retry_at[ctx.query_id] -= FAIR_PULL_RETRY
    _process(bot, ctx)
    assert bot._cancel_market_orders.call_count == 2
    ctx.yes_orders.asks[0] = None            # cleared
    bot._fair_pull_retry_at[ctx.query_id] -= FAIR_PULL_RETRY
    _process(bot, ctx)
    assert bot._cancel_market_orders.call_count == 2


def test_a_stream_that_stopped_printing_is_stale_not_live():
    # Review finding 2: last record 50h old, settle in 20h -> k would be 2
    # and the (already decided) band would be quoted.
    bot = _bot(_records(last_age=50 * 3600))
    ctx = MarketContext(config=_cfg(settle_in=20 * 3600))
    _process(bot, ctx)
    assert bot._fair_pulled[ctx.query_id][0] == "stale"
    bot._execute_order_updates.assert_not_called()


def test_a_new_stream_does_not_quote_a_fixed_prior():
    bot = _bot(_records(n=10))
    ctx = MarketContext(config=_cfg())
    _process(bot, ctx)
    assert bot._fair_pulled[ctx.query_id][0] == "short history"
    bot._execute_order_updates.assert_not_called()


def test_stream_read_reaches_settle_so_an_early_record_counts():
    bot = _bot(_records())
    cfg = _cfg()
    AvellanedaMarketMaker._fair_gate(bot, MarketContext(config=cfg))
    assert bot._client.get_records.call_args.kwargs["date_to"] == cfg.settle_time


def test_stream_priced_markets_never_take_the_order_override_path():
    bot = _bot(_records())
    bot._create_proposal_from_order_override.return_value = ["proposal"]
    ctx = MarketContext(config=_cfg())
    _process(bot, ctx)
    bot._execute_order_override.assert_not_called()
    bot._execute_order_updates.assert_called()


def test_a_clean_pull_is_not_cancelled_again():
    # Round-2 review: successful cancels must free the slots, or the retry
    # re-broadcasts cancels for orders already gone (guaranteed failed txs).
    bot = _bot(_records())
    bot._cancel_market_orders = AvellanedaMarketMaker._cancel_market_orders.__get__(bot)
    ctx = MarketContext(config=_cfg(settle_in=2 * 3600))
    mgr = OrderManager(ctx, 0.0, 1e9)
    mgr.record_order(True, Side.ASK, 40, 6, "a", level_idx=0, is_inventory_backed=True)
    mgr.record_order(True, Side.BID, 30, 6, "b", level_idx=0)
    _process(bot, ctx)
    assert bot._cancel_ask.call_count == 1 and bot._client.cancel_order.call_count == 1
    bot._fair_pull_retry_at[ctx.query_id] -= FAIR_PULL_RETRY
    _process(bot, ctx)
    assert bot._cancel_ask.call_count == 1 and bot._client.cancel_order.call_count == 1


def test_a_shared_read_covers_every_settle_on_the_stream():
    bot = _bot(_records())
    near, far = _cfg(qid=1, settle_in=26 * 3600), _cfg(qid=2, settle_in=74 * 3600)
    bot._markets = {1: MarketContext(config=near), 2: MarketContext(config=far)}
    AvellanedaMarketMaker._fair_gate(bot, bot._markets[1])
    assert bot._client.get_records.call_args.kwargs["date_to"] == far.settle_time

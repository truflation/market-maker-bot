"""Regression tests for the 2026-09-07 over-placement incident (#43).

A gateway outage made get_order_book fail; the periodic reconcile treated the
unreadable books as EMPTY, untracked every resting order, and the quote engine
re-placed bids on top of the invisible ones each pass until the wallet
drained (~30x intended size on the UK books, free balance $1.07). These tests
pin the two independent defenses:

  - reconcile: an unreadable book is UNKNOWN, not empty - no untrack, no
    orphan cancels for that outcome, no budget sync;
  - BidBudget: a hard per-market cap on committed bid collateral that no
    order-book read can raise, released only on chain-confirmed evidence.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bid_budget import BidBudget
from market_maker_bot.bot import AvellanedaMarketMaker


# --- BidBudget unit tests ----------------------------------------------------

def test_reserve_up_to_cap_then_refuse():
    b = BidBudget({7: 1000})
    assert b.try_reserve(7, 40, 10)          # 400
    assert b.try_reserve(7, 60, 10)          # 1000 total
    assert not b.try_reserve(7, 1, 1)        # over cap
    assert b.committed(7) == 1000


def test_uncapped_market_always_reserves():
    b = BidBudget({})
    assert b.try_reserve(99, 90, 1000)


def test_release_floors_at_zero():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 40, 10)
    b.release(7, 40, 10)
    b.release(7, 40, 10)
    assert b.committed(7) == 0


def test_change_bid_delta_accounting():
    b = BidBudget({7: 1000})
    assert b.try_reserve(7, 40, 10)              # 400
    # replace 40x10 with 90x10: delta +500 -> 900, fits
    assert b.reserve_delta(7, 40, 10, 90, 10)
    assert b.committed(7) == 900
    # replace 90x10 with 95x11: delta +145 -> over cap
    assert not b.reserve_delta(7, 90, 10, 95, 11)
    # shrinking always succeeds
    assert b.reserve_delta(7, 90, 10, 10, 10)
    assert b.committed(7) == 100


def test_sync_upward_is_unconditional():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 40, 10)                     # 400
    b.sync(7, 900)                               # chain shows MORE resting
    assert b.committed(7) == 900


def test_sync_shrink_needs_fresh_quiet_consecutive_passes():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 90, 11)                     # 990 committed
    # Pass 1: fresh, but first shrink proposal -> deferred.
    b.sync(7, 200, fresh=True)
    assert b.committed(7) == 990
    # Pass 2: consecutive agreeing fresh quiet pass -> applied.
    b.sync(7, 200, fresh=True)
    assert b.committed(7) == 200


def test_sync_shrink_never_applies_without_freshness():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 90, 11)
    for _ in range(5):
        b.sync(7, 0, fresh=False)                # stale replica refunds: no
    assert b.committed(7) == 990


def test_sync_shrink_blocked_by_interleaved_reservation():
    b = BidBudget({7: 2000})
    b.try_reserve(7, 90, 11)                     # 990
    b.sync(7, 200, fresh=True)                   # deferred (first proposal)
    b.try_reserve(7, 40, 10)                     # engine placed in between
    b.sync(7, 200, fresh=True)                   # reserved-since -> deferred
    assert b.committed(7) == 1390


def test_seed_is_monotonic_max():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 40, 10)                     # 400
    b.seed(7, 300)                               # lower: keeps 400
    assert b.committed(7) == 400
    b.seed(7, 800)                               # higher: raises
    assert b.committed(7) == 800


def test_negative_sdk_prices_count_absolute():
    b = BidBudget({7: 1000})
    assert b.try_reserve(7, -40, 10)             # signed SDK price
    assert b.committed(7) == 400


# --- periodic reconcile: unreadable book guard -------------------------------

def _reconcile_bot(get_order_book, tracked, backstop_amount=0):
    bot = MagicMock()
    bot.config = SimpleNamespace(
        dry_run=False,
        read_only=False,
        maa_address="0xAA00000000000000000000000000000000000aa0",
        private_key="",
        backstop_amount=backstop_amount,
        backstop_price_cents=2,
        pre_settlement_cutoff=900.0,
    )
    ctx = SimpleNamespace(config=SimpleNamespace(settle_time=None))
    bot._markets = {7: ctx}
    bot._order_state.get_market_orders.return_value = tracked
    bot._reconcile_cancel_attempts = {}
    bot._cancel_not_found_times = deque()
    bot._bid_budget = MagicMock()
    bot._pre_settlement_pulled = set()
    bot._earnings_pulled_session = set()
    bot._client.get_order_book.side_effect = get_order_book
    return bot


def _tracked_bid(price=40, amount=10, outcome=True):
    return SimpleNamespace(
        query_id=7, outcome=outcome, is_buy=True, price=price,
        amount=amount, level_idx=0,
    )


def _run(bot):
    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)


def test_unreadable_book_untracks_nothing_and_cancels_nothing():
    def boom(query_id, outcome):
        raise RuntimeError("backend is offline")

    bot = _reconcile_bot(boom, [_tracked_bid()])
    _run(bot)
    bot._order_state.untrack_order.assert_not_called()
    bot._client.cancel_order.assert_not_called()
    bot._bid_budget.sync.assert_not_called()


def test_readable_empty_book_still_untracks_stale_orders():
    bot = _reconcile_bot(lambda q, o: [], [_tracked_bid()])
    _run(bot)
    bot._order_state.untrack_order.assert_called_once()


def test_half_readable_book_only_touches_readable_outcome():
    def half(query_id, outcome):
        if outcome:
            raise RuntimeError("backend is offline")
        return []

    tracked = [_tracked_bid(outcome=True), _tracked_bid(outcome=False)]
    bot = _reconcile_bot(half, tracked)
    _run(bot)
    # Only the NO order (readable, empty book) gets untracked.
    assert bot._order_state.untrack_order.call_count == 1
    args = bot._order_state.untrack_order.call_args[0]
    assert args[1] is False  # outcome
    # Budget sync requires BOTH outcomes readable.
    bot._bid_budget.sync.assert_not_called()


def test_successful_read_syncs_budget_from_chain():
    wallet = "0xAA00000000000000000000000000000000000aa0".lower()
    entries = [
        {"wallet_address": wallet, "price": -40, "amount": 10},   # our bid
        {"wallet_address": "0xother", "price": -90, "amount": 5},  # not ours
    ]
    bot = _reconcile_bot(
        lambda q, o: entries if o else [], [_tracked_bid(price=40)]
    )
    _run(bot)
    bot._bid_budget.sync.assert_called_once_with(7, 400, fresh=bot._gateway_fresh.return_value)


def test_backstop_placed_when_missing_and_skipped_as_orphan():
    wallet = "0xAA00000000000000000000000000000000000aa0".lower()
    # An untracked resting bid AT the backstop price must not be treated as
    # an orphan; a missing backstop on the other outcome must be placed.
    entries_yes = [{"wallet_address": wallet, "price": -2, "amount": 50}]
    bot = _reconcile_bot(
        lambda q, o: entries_yes if o else [], [], backstop_amount=50
    )
    bot._bid_budget.try_reserve.return_value = True
    _run(bot)
    bot._client.cancel_order.assert_not_called()
    # Placed exactly once: the NO side (YES already has one resting).
    assert bot._client.place_buy_order.call_count == 1
    kwargs = bot._client.place_buy_order.call_args[1]
    assert kwargs["outcome"] is False and kwargs["price"] == 2



def test_backstop_skipped_when_ask_at_or_below_backstop_price():
    wallet = "0xAA00000000000000000000000000000000000aa0".lower()
    # YES book: resting 2c ASK (ours or anyone's) -> backstop must skip.
    entries_yes = [{"wallet_address": wallet, "price": 2, "amount": 5}]
    bot = _reconcile_bot(
        lambda q, o: entries_yes if o else [], [], backstop_amount=50
    )
    bot._bid_budget.try_reserve.return_value = True
    _run(bot)
    # NO side has no asks -> placed there only.
    assert bot._client.place_buy_order.call_count == 1
    assert bot._client.place_buy_order.call_args[1]["outcome"] is False


def test_orphan_exclusion_bounded_by_amount():
    wallet = "0xAA00000000000000000000000000000000000aa0".lower()
    # An untracked 2c bid LARGER than the backstop amount is an incident
    # leftover and must still be cancelled.
    entries_yes = [{"wallet_address": wallet, "price": -2, "amount": 500}]
    bot = _reconcile_bot(
        lambda q, o: entries_yes if o else [], [], backstop_amount=50
    )
    bot._bid_budget.try_reserve.return_value = True
    _run(bot)
    bot._client.cancel_order.assert_called_once()


def test_definitive_rejection_classifier():
    from market_maker_bot.bot import AvellanedaMarketMaker as M
    assert M._is_definitive_rejection(Exception("ERROR: Insufficient balance. Required: 42"))
    assert not M._is_definitive_rejection(Exception("tx abc unconfirmed after 30s"))
    assert not M._is_definitive_rejection(Exception("connection reset by peer"))
    assert not M._is_definitive_rejection(Exception("RPC timeout"))


def test_backstop_skipped_on_third_party_ask():
    # Re-review finding 2: the cross-guard must see asks from ANY wallet,
    # not just ours (the taker or an external user can rest a 1-2c ask).
    entries_yes = [{"wallet_address": "0xsomeoneelse", "price": 1, "amount": 5}]
    bot = _reconcile_bot(
        lambda q, o: entries_yes if o else [], [], backstop_amount=50
    )
    bot._bid_budget.try_reserve.return_value = True
    _run(bot)
    assert bot._client.place_buy_order.call_count == 1
    assert bot._client.place_buy_order.call_args[1]["outcome"] is False


def test_invalid_nonce_is_not_a_definitive_rejection():
    from market_maker_bot.bot import AvellanedaMarketMaker as M
    assert not M._is_definitive_rejection(
        Exception("broadcast error: invalid nonce")
    )
    assert M._is_definitive_rejection(Exception("invalid price range"))


def test_stale_pass_cannot_prime_the_shrink_agreement():
    b = BidBudget({7: 1000})
    b.try_reserve(7, 90, 11)                     # 990
    b.sync(7, 200, fresh=False)                  # stale: must not prime
    b.sync(7, 200, fresh=True)                   # first FRESH proposal only
    assert b.committed(7) == 990                 # not applied yet
    b.sync(7, 200, fresh=True)
    assert b.committed(7) == 200

"""Tests for the MM bot's periodic on-chain orphan reconcile path.

Specifically targets `_periodic_reconcile_against_chain`, the new opt-in
method that detects two directions of drift between local tracked state
and the on-chain order book:
  - tracked-but-not-on-chain  -> untrack locally
  - on-chain-but-not-tracked  -> CANCEL the orphan

Mocks TNClient and the bot's own state so the method can run without
instantiating the full LiquidityProviderBot class graph. Calls the method
as an unbound function on a MagicMock with the relevant attributes set,
which is the lightest-weight way to exercise the matching logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from market_maker_bot.bot import AvellanedaMarketMaker
from market_maker_bot.order_state import TrackedOrder

# Recompute the real address from the test key once at import. We could
# hardcode it but recomputing makes the test resilient to library updates.
from eth_account import Account as _Account
TEST_WALLET_DERIVED = _Account.from_key("0x" + "1" * 64).address.lower()

# A real wallet derived from a deterministic 64-char hex key. Used so
# the address-derivation guard inside the reconcile does NOT abort.
TEST_PRIVATE_KEY = "0x" + "1" * 64
# eth_account derives this address from the all-1s key above.
TEST_WALLET = TEST_WALLET_DERIVED


def _book_entry(wallet: str, price: int, amount: int = 10) -> dict:
    """Shape returned by client.get_order_book(qid, outcome)."""
    return {"wallet_address": wallet, "price": price, "amount": amount}


def _tracked(qid: int, outcome: bool, is_buy: bool, price: int,
             level_idx: int = 0) -> TrackedOrder:
    return TrackedOrder(
        query_id=qid, outcome=outcome, is_buy=is_buy, price=price,
        amount=10, created_at=0.0, order_id="",
        level_idx=level_idx, is_inventory_backed=False,
    )


def _bot_mock(tracked_orders: list[TrackedOrder],
              order_books: dict[tuple[int, bool], list[dict]]) -> MagicMock:
    """Build a MagicMock that has just enough attributes for
    `_periodic_reconcile_against_chain` to run. The unbound method is
    invoked via `AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)`.
    """
    bot = MagicMock()
    bot.config.dry_run = False
    bot.config.private_key = TEST_PRIVATE_KEY
    qids = sorted({qid for qid, _ in order_books.keys()})
    bot._markets = {qid: MagicMock() for qid in qids}

    state = MagicMock()
    by_market: dict[int, list[TrackedOrder]] = {}
    for t in tracked_orders:
        by_market.setdefault(t.query_id, []).append(t)
    state.get_market_orders.side_effect = lambda qid: by_market.get(qid, [])
    bot._order_state = state

    def _get_ob(qid: int, outcome: bool) -> list[dict]:
        return order_books.get((qid, outcome), [])
    bot._client.get_order_book.side_effect = _get_ob

    return bot


def test_reconcile_cancels_chain_order_not_in_local_state():
    """ORPHAN-PRESENT: bot has no local tracking for an on-chain order
    => the new code path cancels it. This is the core regression guard
    for the cancel-then-place silent-failure race."""
    qid = 100
    bot = _bot_mock(
        tracked_orders=[],  # local state empty
        order_books={
            (qid, True): [_book_entry(TEST_WALLET, -50)],  # bid at 50c
            (qid, False): [],
        },
    )

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._client.cancel_order.assert_called_once_with(
        query_id=qid, outcome=True, price=-50, wait=False,
    )


def test_reconcile_skips_tracked_orders():
    """ORPHAN-ABSENT: every on-chain order matches a tracked entry =>
    no cancels (bot leaves its own orders alone)."""
    qid = 100
    bot = _bot_mock(
        tracked_orders=[
            _tracked(qid, outcome=True, is_buy=True, price=50),
        ],
        order_books={
            (qid, True): [_book_entry(TEST_WALLET, -50)],
            (qid, False): [],
        },
    )

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._client.cancel_order.assert_not_called()


def test_reconcile_filters_other_wallets():
    """SAFETY: on-chain orders owned by another wallet must NOT be
    cancelled, even though they have no local tracking."""
    qid = 100
    other_wallet = "0xabc" + "1" * 37
    bot = _bot_mock(
        tracked_orders=[],
        order_books={
            (qid, True): [_book_entry(other_wallet, -50)],
            (qid, False): [],
        },
    )

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._client.cancel_order.assert_not_called()


def test_reconcile_aborts_when_wallet_derivation_fails():
    """SAFETY: if the wallet address cannot be derived from the
    private key (e.g. malformed env), the reconcile aborts rather
    than over-matching against other participants' orders."""
    bot = _bot_mock(tracked_orders=[], order_books={(100, True): [
        _book_entry(TEST_WALLET, -50)]})
    bot.config.private_key = "not-a-hex-key"

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._client.cancel_order.assert_not_called()
    # And we should NOT have even fetched any order books
    bot._client.get_order_book.assert_not_called()


def test_reconcile_skipped_in_dry_run():
    bot = _bot_mock(tracked_orders=[], order_books={(100, True): [
        _book_entry(TEST_WALLET, -50)]})
    bot.config.dry_run = True

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._client.cancel_order.assert_not_called()


def test_reconcile_cap_caps_first_pass_at_20():
    """CANCEL-STORM GUARD: even with 50+ orphans on a single market,
    the per-pass cap fires AT MOST MAX_CANCELS_PER_PASS=20 cancels.
    The remainder is deferred to subsequent (idempotent) passes."""
    qid = 100
    bot = _bot_mock(
        tracked_orders=[],
        order_books={
            (qid, True): [_book_entry(TEST_WALLET, -p) for p in range(1, 51)],
            (qid, False): [],
        },
    )

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    assert bot._client.cancel_order.call_count == 20


def test_reconcile_untracks_stale_local_entries():
    """The other direction: local state has an entry that the chain
    doesn't show. Untrack it (existing startup-reconcile behavior,
    now also exercised on every periodic pass)."""
    qid = 100
    bot = _bot_mock(
        tracked_orders=[_tracked(qid, outcome=True, is_buy=True, price=50)],
        order_books={(qid, True): [], (qid, False): []},
    )

    AvellanedaMarketMaker._periodic_reconcile_against_chain(bot)

    bot._order_state.untrack_order.assert_called_once_with(
        qid, True, True, 50, 0,
    )
    bot._client.cancel_order.assert_not_called()

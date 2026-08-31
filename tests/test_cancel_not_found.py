"""Regression tests for the failed-cancel spam loop.

A stale order-book read can keep "showing" orders the chain has already
dropped; retrying their cancels every cycle then produces a stream of failed
transactions ("ERROR: Order not found or does not belong to you"), observed
at ~0.5/sec for 31 hours before these defenses existed. The chain's error
text is authoritative: an order it reports as not found IS gone, no matter
what the book read claims. These tests pin:

  - _cancel_ask (wait=True) treats a not-found cancel as already-cancelled
    instead of raising, so the caller untracks and moves on;
  - the book recheck in the split path cannot override a not-found error;
  - other cancel errors still raise (the PR#13 double-list guard);
  - a storm of observed not-found failures exits EXIT_STALE_CANCEL_LOOP for
    a clean restart, the only proven cure for the stale state behind it.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from market_maker_bot.bot import (
    AvellanedaMarketMaker,
    CANCEL_NOT_FOUND_EXIT_COUNT,
    EXIT_STALE_CANCEL_LOOP,
)

NOT_FOUND = Exception("ERROR: Order not found or does not belong to you")


def _bot():
    bot = MagicMock()
    bot._cancel_not_found_times = deque()
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    bot._note_cancel_not_found = (
        lambda qid, detail: AvellanedaMarketMaker._note_cancel_not_found(
            bot, qid, detail
        )
    )
    return bot


def _ctx(qid=5):
    return SimpleNamespace(query_id=qid)


def _cancel_ask(bot, **kwargs):
    defaults = dict(
        context=_ctx(), outcome=True, price=40, amount=10,
        is_inventory_backed=True, wait=True,
    )
    defaults.update(kwargs)
    return AvellanedaMarketMaker._cancel_ask(bot, **defaults)


def test_not_found_on_inventory_cancel_is_treated_as_cancelled():
    bot = _bot()
    bot._client.cancel_order.side_effect = NOT_FOUND
    inv = MagicMock()
    bot._inventory.get_market_inventory.return_value = inv

    _cancel_ask(bot)  # must not raise

    inv.release_pair.assert_called_once_with(True, 10)
    assert len(bot._cancel_not_found_times) == 1


def test_other_cancel_errors_still_raise_and_keep_reservation():
    bot = _bot()
    bot._client.cancel_order.side_effect = Exception("gateway timeout")
    inv = MagicMock()
    bot._inventory.get_market_inventory.return_value = inv

    with pytest.raises(Exception, match="gateway timeout"):
        _cancel_ask(bot)

    inv.release_pair.assert_not_called()


def test_not_found_beats_stale_book_recheck_on_split_legs():
    """The split path rechecks the book on failure; a STALE book that still
    shows the leg must not turn a definitive not-found into a raise."""
    bot = _bot()
    bot._client.cancel_order.side_effect = NOT_FOUND
    bot._leg_still_on_book = MagicMock(return_value=True)  # stale read
    inv = MagicMock()
    bot._inventory.get_market_inventory.return_value = inv

    _cancel_ask(bot, is_inventory_backed=False)  # must not raise

    bot._leg_still_on_book.assert_not_called()
    assert len(bot._cancel_not_found_times) == 2  # both split legs


def test_not_found_storm_exits_for_restart():
    bot = _bot()
    with pytest.raises(SystemExit) as exc:
        for _ in range(CANCEL_NOT_FOUND_EXIT_COUNT):
            AvellanedaMarketMaker._note_cancel_not_found(bot, 5, "test")
    assert exc.value.code == EXIT_STALE_CANCEL_LOOP

"""Regression tests for the 2026-09-24 split-mint self-burn.

The split-mint ask fallback minted pairs, kept the auto-listed opposite leg
(NO @ 100 - p for a YES ask at p) and sold YES at p. The chain matches a YES
sell at p against a NO sell at 100 - p as a burn, so the sell redeemed its
own auto-listed leg on arrival; where one of our NO bids sat at or above
100 - p it took part of the leg first (a self-trade). Mainnet evidence: a
YES ask at 39c tracked x6 rested x4, its NO@61 leg was gone, and so was our
NO bid at 62c. The pre-settlement pull then cancelled both legs of every
such ask and 22 cancels failed as "order not found".

These tests pin the retirement: short inventory skips the ask (no mint, no
sell); a refresh keeps the old ask only while it is priced at or above the
target and otherwise pulls it; legacy split records cancel only their
tracked slot, the one leg that can still rest.
"""

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from market_maker_bot.bot import AvellanedaMarketMaker
from market_maker_bot.market import OrderManager, Side
from market_maker_bot.models import ActiveOrders
from market_maker_bot.pricing.inventory import MarketInventory


class _Ctx:
    """Minimal MarketContext stand-in: real ActiveOrders, stub config."""

    def __init__(self, query_id=5):
        self.config = SimpleNamespace(query_id=query_id, settle_time=None)
        self.yes_orders = ActiveOrders()
        self.no_orders = ActiveOrders()

    @property
    def query_id(self):
        return self.config.query_id

    def get_orders(self, outcome):
        return self.yes_orders if outcome else self.no_orders

    def set_state(self, outcome, state):
        pass


def _inv(yes=0, no=0):
    inv = MarketInventory(query_id=5)
    inv.yes_shares = yes
    inv.no_shares = no
    return inv


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
        "_place_ask",
        "_cancel_ask",
        "_level_slot_cooling",
        "_note_level_not_found_clear",
        "_note_cancel_not_found",
    ):
        setattr(bot, name, getattr(AvellanedaMarketMaker, name).__get__(bot))
    bot._is_cancel_not_found = AvellanedaMarketMaker._is_cancel_not_found
    bot._is_definitive_rejection.return_value = False
    bot._client.place_sell_order.return_value = "tx-sell"
    return bot


def _cancels(bot):
    return [
        (c.kwargs["outcome"], c.kwargs["price"])
        for c in bot._client.cancel_order.call_args_list
    ]


def test_short_inventory_skips_without_minting_or_selling():
    bot = _bot(_inv(yes=4, no=4))

    assert bot._place_ask(_Ctx(), True, 39, 6) == (None, False)
    bot._client.place_split_limit_order.assert_not_called()
    bot._client.place_sell_order.assert_not_called()


def test_enough_inventory_sells_one_leg():
    inv = _inv(yes=10, no=10)
    bot = _bot(inv)

    assert bot._place_ask(_Ctx(), True, 39, 6) == ("tx-sell", True)
    bot._client.place_split_limit_order.assert_not_called()
    assert [c.kwargs for c in bot._client.place_sell_order.call_args_list] == [
        dict(query_id=5, outcome=True, price=39, amount=6, wait=True)
    ]
    assert inv.reserved_yes_sells == 6


def _refresh(bot, ctx, old_price, new_price, old_amount=3, new_amount=6,
             inventory_backed=True):
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, old_price, old_amount, "tx-old",
                     level_idx=0, is_inventory_backed=inventory_backed)
    return AvellanedaMarketMaker._update_single_order(
        bot, ctx, True, Side.ASK, new_price, new_amount, mgr, 0
    )


def _short_inv():
    # Only 3 shares would be free after cancelling the old x3 ask.
    inv = _inv(yes=0, no=0)
    inv.reserve_pair(True, 3)
    return inv


def test_refresh_keeps_old_ask_priced_at_or_above_target():
    bot = _bot(_short_inv())
    ctx = _Ctx()

    assert _refresh(bot, ctx, old_price=40, new_price=39) is None
    bot._client.cancel_order.assert_not_called()
    bot._client.place_split_limit_order.assert_not_called()
    assert ctx.yes_orders.get_ask(0).price == 40


def test_refresh_pulls_old_ask_priced_below_target():
    # Fair moved up: a kept 38c ask would be lifted cheap or crossed by our
    # own re-priced bid. Pull it and leave the level empty.
    bot = _bot(_short_inv())
    ctx = _Ctx()

    assert _refresh(bot, ctx, old_price=38, new_price=39) is None
    assert _cancels(bot) == [(True, 38)]
    bot._client.place_sell_order.assert_not_called()
    assert ctx.yes_orders.get_ask(0) is None
    bot._order_state.untrack_order.assert_called_once_with(
        query_id=5, outcome=True, is_buy=False, price=38, level_idx=0
    )


def test_refresh_pulls_legacy_split_ask_even_when_priced_above_target():
    # Its shares are locked in the listing, so it could never be re-placed;
    # keeping it would pin a stale price until settlement.
    bot = _bot(_short_inv())
    ctx = _Ctx()

    assert _refresh(bot, ctx, old_price=40, new_price=39,
                    inventory_backed=False) is None
    assert _cancels(bot) == [(True, 40)]
    assert ctx.yes_orders.get_ask(0) is None


def test_legacy_yes_split_ask_cancels_only_its_tracked_slot():
    bot = _bot(_inv())

    bot._cancel_ask(context=_Ctx(), outcome=True, price=39, amount=6,
                    is_inventory_backed=False, wait=False)

    assert _cancels(bot) == [(True, 39)]


def test_legacy_no_split_ask_cancels_only_its_tracked_slot():
    # A NO split ask at 70c listed NO@70 and sold YES@30. If one of our YES
    # bids at 30c or above took part of the YES sell, what survives the
    # burn is NO@70, the tracked slot.
    bot = _bot(_inv())

    bot._cancel_ask(context=_Ctx(), outcome=False, price=70, amount=4,
                    is_inventory_backed=False, wait=False)

    assert _cancels(bot) == [(False, 70)]


def test_legacy_cancel_not_found_is_treated_as_cancelled():
    bot = _bot(_inv())
    bot._client.cancel_order.side_effect = Exception(
        "ERROR: Order not found or does not belong to you"
    )

    bot._cancel_ask(context=_Ctx(), outcome=True, price=39, amount=6,
                    is_inventory_backed=False, wait=True)  # must not raise


def test_legacy_cancel_error_raises_only_while_leg_still_rests():
    bot = _bot(_inv())
    bot._client.cancel_order.side_effect = Exception("gateway timeout")

    bot._leg_still_on_book.return_value = False
    bot._cancel_ask(context=_Ctx(), outcome=True, price=39, amount=6,
                    is_inventory_backed=False, wait=True)  # gone: no raise

    bot._leg_still_on_book.return_value = True
    try:
        bot._cancel_ask(context=_Ctx(), outcome=True, price=39, amount=6,
                        is_inventory_backed=False, wait=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError while the leg rests")


NOT_FOUND = Exception("ERROR: Order not found or does not belong to you")


def test_pull_releases_the_reservation():
    inv = _short_inv()
    bot = _bot(inv)

    _refresh(bot, _Ctx(), old_price=38, new_price=39)

    assert inv.reserved_yes_sells == 0


def test_pull_cancel_error_keeps_state_and_reservation():
    inv = _short_inv()
    bot = _bot(inv)
    ctx = _Ctx()
    bot._client.cancel_order.side_effect = Exception("gateway timeout")

    assert _refresh(bot, ctx, old_price=38, new_price=39) is None
    assert ctx.yes_orders.get_ask(0).price == 38
    assert inv.reserved_yes_sells == 3
    bot._order_state.untrack_order.assert_not_called()


def test_pull_cancel_not_found_clears_and_releases_once():
    inv = _short_inv()
    bot = _bot(inv)
    ctx = _Ctx()
    bot._client.cancel_order.side_effect = NOT_FOUND

    assert _refresh(bot, ctx, old_price=38, new_price=39) is None
    assert ctx.yes_orders.get_ask(0) is None
    assert inv.reserved_yes_sells == 0
    assert bot._order_state.untrack_order.call_count == 1


def test_pull_of_resting_legacy_ask_that_errors_keeps_state():
    bot = _bot(_short_inv())
    ctx = _Ctx()
    bot._client.cancel_order.side_effect = Exception("gateway timeout")
    bot._leg_still_on_book.return_value = True

    _refresh(bot, ctx, old_price=40, new_price=39, inventory_backed=False)

    assert ctx.yes_orders.get_ask(0).price == 40
    bot._order_state.untrack_order.assert_not_called()


def test_next_cycle_after_a_pull_places_nothing():
    bot = _bot(_short_inv())
    ctx = _Ctx()
    mgr = OrderManager(ctx, refresh_tolerance_pct=0.0, max_order_age=1e9)
    mgr.record_order(True, Side.ASK, 38, 3, "tx-old", level_idx=0,
                     is_inventory_backed=True)

    for _ in range(2):
        AvellanedaMarketMaker._update_single_order(
            bot, ctx, True, Side.ASK, 39, 6, mgr, 0
        )

    assert ctx.yes_orders.get_ask(0) is None
    bot._client.place_sell_order.assert_not_called()
    assert bot._client.cancel_order.call_count == 1

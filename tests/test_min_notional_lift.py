"""Dollar-sized orders are lifted to the protocol minimum, not skipped.

At $1.50/level a bid at 47c sized 2 shares (94 cent-shares < 100) was
skipped every pass, leaving that side of the book empty.
"""

from types import SimpleNamespace

from market_maker_bot.bot import _compute_base_amount, _lift_to_min_notional


def _cfg(dollars=None, shares=10):
    return SimpleNamespace(order_dollar_amount=dollars, order_amount=shares)


def test_a_skew_shrunk_order_is_lifted_back_to_the_minimum():
    # 1.5 at 47c: base 3 shares, the eta shrink takes it to 2 (94 < 100).
    assert _lift_to_min_notional(_cfg(1.5), 47, 2, 3) == 3


def test_an_order_already_over_the_minimum_is_unchanged():
    assert _lift_to_min_notional(_cfg(1.5), 47, 4, 4) == 4
    assert _lift_to_min_notional(_cfg(3.0), 1, 300, 300) == 300


def test_a_lift_never_exceeds_the_unskewed_size():
    # 1.0 at 67c rounds to 1 share (67 < 100); lifting to 2 would put the
    # order above its own dollar size, so it stays skipped.
    base = _compute_base_amount(_cfg(1.0), 67)
    assert base == 1
    assert _lift_to_min_notional(_cfg(1.0), 67, base, base) == 1


def test_share_count_configs_keep_the_skip():
    assert _lift_to_min_notional(_cfg(None, 2), 47, 2, 2) == 2


def test_lifted_orders_clear_the_minimum_and_stay_within_their_base():
    for dollars in (1.5, 1.8, 2.0, 3.0):
        for price in range(1, 100):
            base = _compute_base_amount(_cfg(dollars), price)
            for shrunk in range(1, base + 1):
                got = _lift_to_min_notional(_cfg(dollars), price, shrunk, base)
                assert got <= base
                if base * price >= 100:
                    assert got * price >= 100

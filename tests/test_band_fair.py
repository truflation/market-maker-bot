"""Stream-priced bands: the pure model in pricing/band_fair.py."""

import pytest

from market_maker_bot.pricing import band_fair as bf

DAY = bf.DAY
T0 = 1_790_000_000 + 4 * 3600  # records stamped at a fixed time of day


def series(n, step=0.0, start=2.80):
    return [T0 + i * DAY for i in range(n)], [start + i * step for i in range(n)]


def test_band_edges_follow_the_sdk_market_definitions():
    assert bf.in_band(2.66, None, 2.67) and not bf.in_band(2.67, None, 2.67)
    assert bf.in_band(2.67, 2.67, 2.79) and bf.in_band(2.79, 2.67, 2.79)
    assert bf.in_band(3.04, 3.03, None) and not bf.in_band(3.03, 3.03, None)
    with pytest.raises(ValueError):
        bf.in_band(1.0, None, None)


def test_steps_remaining_counts_whole_days_to_settle():
    assert bf.steps_remaining(T0, T0 + 25 * 3600) == 1
    assert bf.steps_remaining(T0, T0 + 2 * 3600) == 0
    assert bf.steps_remaining(T0, T0 + 73 * 3600) == 3
    assert bf.steps_remaining(T0, T0 - 10) == 0


def test_decided_band_is_near_certain_either_way():
    t, v = series(60)
    assert bf.band_probability(t, v, 2.67, 2.91, t[-1] + 3600, 0.36) == (1 - bf.FLOOR, 0)
    assert bf.band_probability(t, v, 2.91, None, t[-1] + 3600, 0.10) == (bf.FLOOR, 0)


def test_slow_stream_puts_a_wide_centre_band_near_certain():
    # The 2026-10-08 case: a fixed prior said 36%, the band paid every day.
    t, v = series(150, step=0.0005)
    p, k = bf.band_probability(t, v, 2.67, 2.91, t[-1] + 26 * 3600, 0.36)
    assert k == 1 and p >= 0.9


def test_choppy_stream_spreads_mass_across_bands():
    t, _ = series(150)
    v = [2.80 + (0.08 if i % 2 else -0.08) for i in range(150)]
    p_mid, _ = bf.band_probability(t, v, 2.75, 2.85, t[-1] + 26 * 3600, 0.36)
    p_up, _ = bf.band_probability(t, v, 2.85, None, t[-1] + 26 * 3600, 0.10)
    assert p_mid < 0.2 and p_up > 0.3


def test_short_history_gives_no_fair():
    t, v = series(10)
    assert bf.band_probability(t, v, 2.67, 2.91, t[-1] + 50 * 3600, 0.36) == (None, 2)


def test_k_across_dst_and_stamp_drift():
    # A stream stamped 4pm ET settling 5pm ET. Fall back: Oct 31 20:00 UTC
    # record, Nov 1 22:00 UTC settle = 26h -> 1; spring forward: Mar 7
    # 21:00 UTC record, Mar 8 21:00 UTC settle = 24h -> 1.
    assert bf.steps_remaining(1_793_476_800, 1_793_476_800 + 26 * 3600) == 1
    assert bf.steps_remaining(1_804_446_000, 1_804_446_000 + 24 * 3600) == 1
    # Stamps drifting 02:30 -> 03:10; a 16:00 settle still counts 1 then 0.
    assert bf.steps_remaining(T0, T0 + 36 * 3600) == 1
    assert bf.steps_remaining(T0 + DAY, T0 + 36 * 3600) == 0
    # A record stamped after settle is already decided.
    assert bf.steps_remaining(T0 + 3600, T0) == 0


def test_fair_never_reaches_zero_or_one():
    t, v = series(150)
    for lo, hi in ((None, 2.67), (2.67, 2.91), (2.91, None)):
        p, _ = bf.band_probability(t, v, lo, hi, t[-1] + 26 * 3600, 0.2)
        assert bf.FLOOR <= p <= 1 - bf.FLOOR


def test_steps_across_a_long_gap_are_ignored():
    t, v = series(150)
    t = t[:100] + [x + 30 * DAY for x in t[100:]]
    v = v[:100] + [3.5] * 50  # the jump across the gap is not a 1-day move
    p, _ = bf.band_probability(t, v, 3.45, 3.55, t[-1] + 26 * 3600, 0.36)
    assert p >= 0.9


def test_clean_series_sorts_dedups_and_drops_junk_stamps():
    t, v = bf.clean_series([
        {"EventTime": "100", "Value": "1"},
        {"EventTime": "1700000000", "Value": "2"},
        {"EventTime": "1700000000", "Value": "3"},
        {"EventTime": "1690000000", "Value": "4"},
    ])
    assert t == [1690000000, 1700000000] and v == [4.0, 3.0]

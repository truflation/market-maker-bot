"""Fair YES for a value band from the underlying stream's own history.

For daily-printing streams split into value bands (below / range / above),
the chance a band pays is mostly a question of how many prints are still to
land before settlement and how far the value usually moves over that many
prints. A fixed per-band prior ignores both: it under-prices a wide centre
band from birth and keeps offering the winning band after the deciding
print is on chain (2026-10-08: an informed taker made 53 fills at 38-41c on
bands that paid 100).

Model:
  k = whole days between the latest record's event time and settle_time
      (records still to land, for a stream that stamps once a day).
  k == 0 -> the deciding record is on chain; the band is decided.
  k >= 1 -> recency-weighted empirical distribution of k-step changes,
            applied to the latest value, blended with a prior as a few
            pseudo-counts so a short history cannot give 0 or 1.

Band edges follow the SDK market definitions: below = value < upper,
range = lower <= value <= upper, above = value > lower.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

DAY = 86400
WINDOW = 120          # k-step changes used
HALF_LIFE = 30.0      # recency half-life, in records
PRIOR_WEIGHT = 3.0    # pseudo-counts given to the prior
MIN_SAMPLES = 20      # fewer usable changes -> return the prior
MAX_GAP_DAYS = 3      # skip steps spanning longer gaps (outages, cadence changes)
FLOOR = 0.03          # fair stays inside [FLOOR, 1 - FLOOR]
EPS = 1e-12


def in_band(value: float, lower: Optional[float], upper: Optional[float]) -> bool:
    if lower is None and upper is None:
        raise ValueError("band needs a lower or an upper bound")
    if lower is None:
        return value < upper - EPS
    if upper is None:
        return value > lower + EPS
    return lower - EPS <= value <= upper + EPS


def steps_remaining(last_event_ts: int, settle_ts: int) -> int:
    """Daily records still to land at or before settle."""
    return max(0, int((settle_ts - last_event_ts) // DAY))


def clean_series(records: Iterable[dict], min_ts: int = 1_600_000_000
                 ) -> Tuple[List[int], List[float]]:
    """(event times, values) ascending, one value per event time (last wins)."""
    by_t = {}
    for r in records:
        t = int(r["EventTime"])
        if t >= min_ts:
            by_t[t] = float(r["Value"])
    times = sorted(by_t)
    return times, [by_t[t] for t in times]


def band_probability(times: Sequence[int], values: Sequence[float],
                     lower: Optional[float], upper: Optional[float],
                     settle_ts: int, prior: float) -> Tuple[Optional[float], int]:
    """(fair YES in [FLOOR, 1-FLOOR], k). Caller handles k == 0 as decided.

    Fair is None when the history is too short to say anything (a new
    stream): the caller should not quote rather than fall back to a fixed
    prior, which is the failure this module replaces.
    """
    if not values:
        return None, -1
    k = steps_remaining(times[-1], settle_ts)
    latest = values[-1]
    if k == 0:
        return (1 - FLOOR if in_band(latest, lower, upper) else FLOOR), 0

    n = len(values)
    hit = total = 0.0
    used = 0
    for i in range(max(0, n - k - WINDOW), n - k):
        if times[i + k] - times[i] > (k + MAX_GAP_DAYS) * DAY:
            continue
        w = 0.5 ** (((n - k - 1) - i) / HALF_LIFE)
        total += w
        used += 1
        if in_band(latest + values[i + k] - values[i], lower, upper):
            hit += w
    if used < MIN_SAMPLES:
        return None, k
    p = (hit + PRIOR_WEIGHT * prior) / (total + PRIOR_WEIGHT)
    return min(1 - FLOOR, max(FLOOR, p)), k

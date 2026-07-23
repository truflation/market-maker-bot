"""Earnings-aware quote cutoff for EPS markets.

Each EPS market's underlying number goes public at an earnings announcement,
which for the Mag7 lands ~1 week BEFORE our on-chain ``settle_time``. The
TrufNetwork prediction book trades 24/7, so the moment the number prints our
resting orders are pick-off-able for the whole week until settlement. The MM
must therefore STOP quoting a ticker before its earnings print (keeping
inventory, which still settles for value at ``settle_time``).

Rule (Roland 2026-07-22). Given the announcement day ``T`` and whether the
release is after-market-close (AMC) or before-market-open (BMO):

    AMC on day T:  stop quoting QUOTE_CUTOFF_BUFFER before the market close on T.
    BMO on day T:  stop quoting at the market close of the PREVIOUS trading day.

Why the asymmetry is correct for a 24/7 book:
  - AMC: the print happens shortly after T's 16:00 close, so we pull with a
    buffer before that close to be safely flat before the number can drop.
  - BMO: the print happens before T's open, i.e. during the overnight gap after
    T-1's close. The last moment our book faces a not-yet-public number is
    T-1's close, so we stop exactly there. No extra buffer is needed because the
    underlying is closed from then until the print (there is no window in which
    quoting is both safe and forgone). Pulling earlier would only forgo safe
    spread.

"Market close" is 16:00 America/New_York (DST-aware; 20:00 UTC in EDT, 21:00 UTC
in EST). Half-days (e.g. day after Thanksgiving, Christmas Eve) close at 13:00
ET and are handled via ``early_closes``. "Previous trading day" skips weekends
and the US market ``holidays`` set. Both calendars are inputs (not hardcoded)
so they can be maintained without touching this logic; the market-creation side
supplies the current year's NYSE calendar.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from zoneinfo import ZoneInfo

EXCHANGE_TZ = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

REGULAR_CLOSE = _dt.time(16, 0)   # 4:00 PM ET regular-session close
EARLY_CLOSE = _dt.time(13, 0)     # 1:00 PM ET half-day close
QUOTE_CUTOFF_BUFFER = _dt.timedelta(minutes=30)

# Bundled NYSE calendars. Update per year (or externalize to config). Used as a
# convenience default by callers (e.g. MarketConfig) that don't thread a calendar
# through. quote_cutoff_time itself still defaults holidays/early_closes to empty
# (explicit), so a caller must opt into these.
DEFAULT_HOLIDAYS = frozenset(_dt.date(2026, m, d) for m, d in [
    (1, 1), (1, 19), (2, 16), (4, 3), (5, 25), (6, 19),
    (7, 3), (9, 7), (11, 26), (12, 25),
])
DEFAULT_EARLY_CLOSES = frozenset({_dt.date(2026, 11, 27), _dt.date(2026, 12, 24)})


class EarningsTiming(str, Enum):
    """When, relative to the underlying's trading session, the number prints."""

    AMC = "amc"   # after market close on day T
    BMO = "bmo"   # before market open on day T

    @classmethod
    def parse(cls, value: "str | EarningsTiming") -> "EarningsTiming":
        if isinstance(value, cls):
            return value
        v = str(value).strip().lower()
        aliases = {
            "amc": cls.AMC, "after": cls.AMC, "after_close": cls.AMC,
            "aftermarket": cls.AMC, "after-hours": cls.AMC, "afterhours": cls.AMC,
            "bmo": cls.BMO, "before": cls.BMO, "before_open": cls.BMO,
            "premarket": cls.BMO, "pre-market": cls.BMO, "pre-open": cls.BMO,
        }
        if v not in aliases:
            raise ValueError(
                f"unrecognized earnings_timing {value!r}; expected one of "
                f"amc/after or bmo/before"
            )
        return aliases[v]


def _as_date(d: "str | _dt.date") -> _dt.date:
    # Reject datetime: a datetime's calendar day depends on its tzinfo, so
    # ``.date()`` on an aware UTC datetime can be a day off from the ET date and
    # silently shift the whole cutoff by a trading day. Earnings dates are ET
    # calendar days; require them as a date or ISO string. (datetime is a
    # subclass of date, so this check must come first.)
    if isinstance(d, _dt.datetime):
        raise TypeError(
            "earnings_date must be a date or 'YYYY-MM-DD' string, not a "
            "datetime (its calendar day is timezone-dependent and would mis-date "
            "the cutoff); pass the ET calendar date explicitly"
        )
    if isinstance(d, _dt.date):
        return d
    return _dt.date.fromisoformat(str(d).strip())


def is_trading_day(
    d: _dt.date,
    holidays: "frozenset[_dt.date]" = frozenset(),
) -> bool:
    """True if ``d`` is a weekday and not in the supplied holiday set."""
    return d.weekday() < 5 and d not in holidays


def market_close_dt(
    d: _dt.date,
    early_closes: "frozenset[_dt.date]" = frozenset(),
) -> _dt.datetime:
    """The exchange close instant on calendar date ``d`` as an aware datetime.

    16:00 ET normally, 13:00 ET on a half-day. Constructed with a zoneinfo
    tzinfo so the wall-clock time is interpreted DST-correctly (the 16:00 and
    13:00 closes never fall in a DST transition gap, so there is no ambiguity).
    """
    close = EARLY_CLOSE if d in early_closes else REGULAR_CLOSE
    return _dt.datetime.combine(d, close, tzinfo=EXCHANGE_TZ)


def previous_trading_day(
    d: _dt.date,
    holidays: "frozenset[_dt.date]" = frozenset(),
) -> _dt.date:
    """The trading day strictly before ``d``, skipping weekends and holidays."""
    cur = d - _dt.timedelta(days=1)
    # Bounded walk-back (2 weeks) guards against a mis-specified holiday set that
    # would otherwise loop forever.
    for _ in range(14):
        if cur.weekday() < 5 and cur not in holidays:
            return cur
        cur -= _dt.timedelta(days=1)
    raise ValueError(
        f"no trading day found within 14 days before {d}; check the holidays set"
    )


def quote_cutoff_time(
    earnings_date: "str | _dt.date",
    timing: "str | EarningsTiming",
    holidays: "frozenset[_dt.date]" = frozenset(),
    early_closes: "frozenset[_dt.date]" = frozenset(),
    buffer: _dt.timedelta = QUOTE_CUTOFF_BUFFER,
) -> int:
    """Unix timestamp (UTC seconds) at which the MM must stop quoting the market.

    AMC: close(T) - buffer.  BMO: close(previous_trading_day(T)).
    ``buffer`` applies to AMC only, by design (see module docstring).
    """
    d = _as_date(earnings_date)
    t = EarningsTiming.parse(timing)
    if buffer < _dt.timedelta(0):
        raise ValueError(f"buffer must be non-negative, got {buffer!r}")
    # An earnings announcement date must itself be a trading day. If it is not
    # (config typo, wrong quarter, or a half-day mistakenly listed in holidays),
    # the AMC path would otherwise compute a close on a day with no session and
    # silently produce a too-late cutoff (fail-open). Raise loudly instead.
    if not is_trading_day(d, holidays):
        raise ValueError(
            f"earnings_date {d} is not a trading day (weekend or in holidays); "
            f"an earnings announcement date must fall on a trading day. Check the "
            f"date, the quarter, and that a half-day is in early_closes not holidays."
        )
    if t is EarningsTiming.AMC:
        cutoff = market_close_dt(d, early_closes) - buffer
    else:  # BMO
        prev = previous_trading_day(d, holidays)
        cutoff = market_close_dt(prev, early_closes)
    return int(cutoff.timestamp())

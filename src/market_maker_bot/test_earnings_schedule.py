"""Tests for earnings_schedule.quote_cutoff_time.

Each case asserts the human-readable UTC wall clock of the cutoff, which is
unambiguous. Weekday/holiday facts used below (2026 US/NYSE calendar):
  - 2026-07-21 Tue, 2026-07-22 Wed, 2026-07-24 Fri, 2026-07-27 Mon
  - July 4 2026 is a Saturday -> observed holiday Fri July 3
  - Thanksgiving 2026 = Thu Nov 26 -> half-day Fri Nov 27; 2026-11-30 Mon
"""

import datetime as dt
from zoneinfo import ZoneInfo

from earnings_schedule import (
    EarningsTiming,
    quote_cutoff_time,
    previous_trading_day,
)

UTC = ZoneInfo("UTC")


def _utc(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def check(label, got_ts, expected_str):
    got = _utc(got_ts)
    status = "OK" if got == expected_str else "FAIL"
    print(f"  [{status}] {label}: got {got} | expected {expected_str}")
    return got == expected_str


def main():
    ok = True
    HOL = frozenset({dt.date(2026, 7, 3)})            # July 4 observed
    HALF = frozenset({dt.date(2026, 11, 27)})         # day after Thanksgiving

    # 1. AMC, EDT summer: close 16:00 EDT = 20:00 UTC, -30m = 19:30 UTC
    ok &= check("AMC EDT (2026-07-22)",
                quote_cutoff_time("2026-07-22", "amc"), "2026-07-22 19:30 UTC")

    # 2. AMC, EST winter: close 16:00 EST = 21:00 UTC, -30m = 20:30 UTC
    ok &= check("AMC EST (2026-01-15)",
                quote_cutoff_time("2026-01-15", "amc"), "2026-01-15 20:30 UTC")

    # 3. BMO weekday: T=Wed 07-22 -> prev Tue 07-21 close 20:00 UTC (no buffer)
    ok &= check("BMO weekday (T=2026-07-22)",
                quote_cutoff_time("2026-07-22", "bmo"), "2026-07-21 20:00 UTC")

    # 4. BMO Monday -> prev Friday
    ok &= check("BMO Monday (T=2026-07-27)",
                quote_cutoff_time("2026-07-27", "bmo"), "2026-07-24 20:00 UTC")

    # 5. BMO day after holiday+weekend: T=Mon 07-06, skip Sat/Sun + Fri 07-03 holiday -> Thu 07-02
    ok &= check("BMO after holiday (T=2026-07-06)",
                quote_cutoff_time("2026-07-06", "bmo", holidays=HOL),
                "2026-07-02 20:00 UTC")

    # 6. AMC half-day: close 13:00 EST = 18:00 UTC, -30m = 17:30 UTC
    ok &= check("AMC half-day (2026-11-27)",
                quote_cutoff_time("2026-11-27", "amc", early_closes=HALF),
                "2026-11-27 17:30 UTC")

    # 7. BMO where prev day is a half-day: T=Mon 11-30 -> prev Fri 11-27 (half) close 13:00 EST = 18:00 UTC
    ok &= check("BMO prev-day-half (T=2026-11-30)",
                quote_cutoff_time("2026-11-30", "bmo", early_closes=HALF),
                "2026-11-27 18:00 UTC")

    # 8. parse aliases
    assert EarningsTiming.parse("after") is EarningsTiming.AMC
    assert EarningsTiming.parse("BMO") is EarningsTiming.BMO
    assert EarningsTiming.parse("premarket") is EarningsTiming.BMO
    assert EarningsTiming.parse(EarningsTiming.AMC) is EarningsTiming.AMC
    try:
        EarningsTiming.parse("whenever"); print("  [FAIL] bad timing did not raise"); ok = False
    except ValueError:
        print("  [OK] parse rejects unknown timing")

    # 9. previous_trading_day sanity
    assert previous_trading_day(dt.date(2026, 7, 27)) == dt.date(2026, 7, 24)  # Mon->Fri
    assert previous_trading_day(dt.date(2026, 7, 6), HOL) == dt.date(2026, 7, 2)
    print("  [OK] previous_trading_day weekend+holiday skips")

    # 10. weekday facts hold (guards the assumptions above)
    assert dt.date(2026, 7, 22).weekday() == 2, "07-22 should be Wed"
    assert dt.date(2026, 7, 27).weekday() == 0, "07-27 should be Mon"
    assert dt.date(2026, 11, 30).weekday() == 0, "11-30 should be Mon"
    print("  [OK] 2026 weekday facts confirmed")

    # 11. HARDENING (post-verification): fail-safe input validation
    import earnings_schedule as es
    # B1: reject datetime (tz-dependent calendar day would mis-date the cutoff)
    try:
        quote_cutoff_time(dt.datetime(2026, 7, 22, 2, 0, tzinfo=UTC), "amc")
        print("  [FAIL] datetime input not rejected"); ok = False
    except TypeError:
        print("  [OK] rejects datetime input (B1)")
    # B2: earnings_date must be a trading day -> weekend raises
    try:
        quote_cutoff_time("2026-07-25", "amc")  # Saturday
        print("  [FAIL] weekend earnings_date not rejected"); ok = False
    except ValueError:
        print("  [OK] rejects non-trading earnings_date / weekend (B2)")
    # B2: earnings_date on a holiday raises
    try:
        quote_cutoff_time("2026-07-03", "amc", holidays=HOL)
        print("  [FAIL] holiday earnings_date not rejected"); ok = False
    except ValueError:
        print("  [OK] rejects earnings_date on a holiday (B2)")
    # B4: negative buffer raises
    try:
        quote_cutoff_time("2026-07-22", "amc", buffer=dt.timedelta(minutes=-30))
        print("  [FAIL] negative buffer not rejected"); ok = False
    except ValueError:
        print("  [OK] rejects negative buffer (B4)")
    # is_trading_day helper
    assert es.is_trading_day(dt.date(2026, 7, 22))            # Wed
    assert not es.is_trading_day(dt.date(2026, 7, 25))        # Sat
    assert not es.is_trading_day(dt.date(2026, 7, 3), HOL)    # holiday
    print("  [OK] is_trading_day helper")

    print("ALL PASS" if ok else "SOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)

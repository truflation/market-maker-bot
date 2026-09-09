"""Per-market bid-collateral budget (2026-09-07 incident, prediction-bots #43).

During a gateway outage the bot cannot read its own resting orders. The
reconcile passes then see an "empty" book, untrack everything, and the quote
engine re-places bids on top of the invisible resting ones, converting a read
failure directly into unbounded collateral commitment (observed: ~30x intended
size on one market group, free balance drained to zero).

This ledger is the hard stop: it counts the cent-shares committed to open bids
per market from the bot's own actions, and refuses placements past a cap that
no order-book read can raise. Releases only happen on evidence that money came
back (a cancel the chain confirmed, a chain read that succeeded) - never on
"order not found", which a degraded gateway reports for orders that still rest.

The ledger deliberately fails safe in both directions: over-counting can only
pause quoting on a market until the next successful reconcile trues it up;
under-counting is prevented by never releasing on unverifiable signals.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict

logger = logging.getLogger(__name__)


class BidBudget:
    """Tracks cent-shares committed to open bids per market, with a hard cap.

    All prices are positive cents, amounts are shares; a bid of 40c x 25
    commits 1000 cent-shares ($10). Thread-safe: the bot's quote loop and
    reconcile run on the same thread today, but the lock keeps that a
    non-assumption.
    """

    def __init__(self, cap_cents: Dict[int, int]):
        # query_id -> cap in cent-shares. Markets absent from the map are
        # uncapped (budget disabled for them).
        self._cap = dict(cap_cents)
        self._committed: Dict[int, int] = {}
        # Shrink gating (review finding 1): a merely-successful read must not
        # refund the budget - a stale replica answers 200s with an old
        # snapshot, and refunding from it re-arms the exact over-placement
        # loop this ledger exists to stop. A shrink is applied only when the
        # caller vouches the read was FRESH, no reservation happened since
        # the previous sync for that market, and the previous sync proposed
        # the same-or-lower value (two consecutive agreeing quiet passes).
        self._pending_shrink: Dict[int, int] = {}
        self._reserved_since_sync: Dict[int, bool] = {}
        self._lock = threading.Lock()

    def cap(self, query_id: int) -> int | None:
        return self._cap.get(query_id)

    def committed(self, query_id: int) -> int:
        with self._lock:
            return self._committed.get(query_id, 0)

    def try_reserve(self, query_id: int, price: int, amount: int) -> bool:
        """Reserve budget for a new bid. False = would exceed the cap; skip
        the placement (and log at the call site)."""
        cost = abs(price) * amount
        cap = self._cap.get(query_id)
        with self._lock:
            cur = self._committed.get(query_id, 0)
            if cap is not None and cur + cost > cap:
                return False
            self._committed[query_id] = cur + cost
            self._reserved_since_sync[query_id] = True
            return True

    def reserve_delta(self, query_id: int, old_price: int, old_amount: int,
                      new_price: int, new_amount: int) -> bool:
        """Budget accounting for change_bid: the old bid's collateral is
        replaced by the new bid's. Negative deltas always succeed."""
        delta = abs(new_price) * new_amount - abs(old_price) * old_amount
        cap = self._cap.get(query_id)
        with self._lock:
            cur = self._committed.get(query_id, 0)
            if delta > 0 and cap is not None and cur + delta > cap:
                return False
            self._committed[query_id] = max(0, cur + delta)
            self._reserved_since_sync[query_id] = True
            return True

    def release(self, query_id: int, price: int, amount: int) -> None:
        """Release budget for a bid the chain CONFIRMED cancelled or filled.
        Never call this on an 'order not found' result: a degraded gateway
        says that about orders that still rest."""
        cost = abs(price) * amount
        with self._lock:
            self._committed[query_id] = max(
                0, self._committed.get(query_id, 0) - cost
            )

    def sync(self, query_id: int, resting_bid_cents: int,
             fresh: bool = False) -> None:
        """Correction from a successful chain read of both outcomes.

        UPWARD corrections (chain shows more resting than we thought) apply
        unconditionally - over-counting is the safe direction. DOWNWARD
        corrections (refunds) apply only when `fresh` is True (the caller
        verified gateway freshness, e.g. block age), no reservation was made
        since the previous sync, and the previous sync proposed the same or
        a lower value. A stale-but-responding replica therefore cannot
        refund the budget and re-arm the over-placement loop.
        """
        with self._lock:
            prev = self._committed.get(query_id, 0)
            value = max(0, resting_bid_cents)
            if value >= prev:
                self._committed[query_id] = value
                self._pending_shrink.pop(query_id, None)
                self._reserved_since_sync[query_id] = False
                return
            # Proposed shrink.
            reserved = self._reserved_since_sync.get(query_id, False)
            pending = self._pending_shrink.get(query_id)
            apply = (
                fresh
                and not reserved
                and pending is not None
                and value >= pending
            )
            if apply:
                self._committed[query_id] = value
                self._pending_shrink.pop(query_id, None)
                logger.info(
                    "BidBudget shrink applied market %d: %d -> %d cent-shares",
                    query_id, prev, value,
                )
            elif fresh:
                # Only a FRESH pass may prime the two-pass shrink agreement;
                # otherwise a stale read primes what a later fresh pass
                # confirms (re-review residual 1a).
                self._pending_shrink[query_id] = value
                logger.debug(
                    "BidBudget shrink deferred market %d: %d -> %d "
                    "(fresh=%s reserved=%s pending=%s)",
                    query_id, prev, value, fresh, reserved, pending,
                )
            self._reserved_since_sync[query_id] = False

    def seed(self, query_id: int, resting_bid_cents: int) -> None:
        """Conservative startup seeding when the book CANNOT be read: count
        previously-tracked bids as still committed so a restart into a
        degraded gateway cannot re-place on top of them."""
        with self._lock:
            self._committed[query_id] = max(
                self._committed.get(query_id, 0), resting_bid_cents
            )

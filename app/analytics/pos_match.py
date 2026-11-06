"""CCTV <-> POS correlation.

POS records carry no customer identity, so a transaction is attributed to the session
whose billing activity (queue join / billing-zone entry) happened within
``window_before`` minutes before the POS timestamp (plus a small ``window_after`` grace
for clock skew). Assignment is one-to-one and follows queue discipline (FIFO):
transactions are processed chronologically and each takes the *earliest* still-unmatched
billing signal in its window, so two transactions never claim the same visitor and
shoppers are not "crossed" in the queue.
A queue join that was abandoned before the transaction is not a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from .sessions import Session


@dataclass
class Match:
    transaction_id: str
    pos_timestamp: datetime
    basket_value_inr: float
    session_id: Optional[str]
    visitor_id: Optional[str]
    billing_timestamp: Optional[datetime]
    gap_s: Optional[float]
    confidence: float

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _billing_signals(s: Session) -> list[datetime]:
    sig = []
    for j in s.queue_joins:
        # a join followed by an abandon (before the next join) is not a purchase signal
        later_abandon = [a for a in s.queue_abandons if a >= j]
        later_join = [k for k in s.queue_joins if k > j]
        if later_abandon and (not later_join or min(later_abandon) < min(later_join)):
            continue
        sig.append(j)
    sig.extend(s.billing_zone_times)
    return sig


def match_transactions(sessions: list[Session], transactions: list[dict],
                       window_before_min: float = 5.0, window_after_min: float = 1.0) -> list[Match]:
    wb, wa = timedelta(minutes=window_before_min), timedelta(minutes=window_after_min)
    signals = [(s, t) for s in sessions for t in _billing_signals(s)]
    used_tx: dict[int, Match] = {}
    used_sessions: set[str] = set()
    order = sorted(range(len(transactions)), key=lambda i: transactions[i]["timestamp"])
    for ti in order:
        tx = transactions[ti]
        ts = tx["timestamp"]
        cands = [(bt, s) for s, bt in signals
                 if ts - wb <= bt <= ts + wa and s.session_id not in used_sessions]
        if not cands:
            continue
        bt, s = min(cands, key=lambda c: (c[0], c[1].session_id))
        gap = (ts - bt).total_seconds()
        span = (wb if gap >= 0 else wa).total_seconds() or 1.0
        conf = round(max(0.05, 1.0 - 0.5 * abs(gap) / span) * s.min_confidence, 3)
        used_tx[ti] = Match(tx["transaction_id"], ts, float(tx["basket_value_inr"]),
                            s.session_id, s.visitor_id, bt, round(gap, 1), conf)
        used_sessions.add(s.session_id)
        s.converted, s.transaction_id = True, tx["transaction_id"]
        s.basket_value_inr = float(tx["basket_value_inr"])
    out = []
    for ti, tx in enumerate(transactions):
        out.append(used_tx.get(ti) or Match(tx["transaction_id"], tx["timestamp"],
                                            float(tx["basket_value_inr"]), None, None, None, None, 0.0))
    return out

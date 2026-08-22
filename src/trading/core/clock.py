"""Time, injected.

Nothing in this codebase calls ``datetime.now()``.  A single injected clock is
what lets the identical strategy and risk code run under a simulated clock in a
backtest and a real one in live trading (Phase 0 §4.2).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "SimulatedClock", "SystemClock", "ensure_utc"]


def ensure_utc(moment: datetime) -> datetime:
    """Reject naive datetimes and normalise to UTC.

    Timezone-naive timestamps are the root of a whole family of silent
    off-by-one-session bugs, so they are an error rather than an assumption.
    """
    if moment.tzinfo is None:
        raise ValueError(f"Naive datetime {moment!r}: all timestamps must be timezone-aware")
    return moment.astimezone(UTC)


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        """Current time, always timezone-aware and in UTC."""
        ...


class SystemClock:
    """Wall-clock time. Used in PAPER and LIVE."""

    def now(self) -> datetime:
        return datetime.now(UTC)


class SimulatedClock:
    """A clock the backtester advances by hand.

    Time only moves forward: an attempt to set it backwards is a bug in the
    caller, and silently allowing it would let a strategy observe the future.
    """

    def __init__(self, start: datetime) -> None:
        self._now = ensure_utc(start)

    def now(self) -> datetime:
        return self._now

    def advance_to(self, moment: datetime) -> None:
        moment = ensure_utc(moment)
        if moment < self._now:
            raise ValueError(f"Clock cannot move backwards: {self._now} -> {moment}")
        self._now = moment

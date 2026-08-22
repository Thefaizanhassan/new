"""Market calendars.

Sessions are data, never ``if weekday < 5``.  NSE closes for roughly 15 public
holidays a year on dates that do not repeat predictably, plus the occasional
special session (Muhurat trading on Diwali is a real, tradable session on a day
the exchange is otherwise shut).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Protocol, cast

import exchange_calendars as xcals
import pandas as pd

__all__ = ["ExchangeCalendarAdapter", "MarketCalendar"]


class MarketCalendar(Protocol):
    code: str

    def is_session(self, day: date) -> bool: ...
    def sessions_between(self, start: date, end: date) -> pd.DatetimeIndex: ...
    def session_close_utc(self, day: date) -> datetime: ...
    def is_open_at(self, moment: datetime) -> bool: ...


class ExchangeCalendarAdapter:
    """Adapter over the ``exchange_calendars`` package.

    Wrapped rather than used directly so the rest of the codebase depends on
    our protocol, and swapping the source later is one file.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        self._cal = xcals.get_calendar(code)

    def is_session(self, day: date) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(day)))

    def sessions_between(self, start: date, end: date) -> pd.DatetimeIndex:
        sessions = self._cal.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return pd.DatetimeIndex(sessions).tz_localize("UTC")

    def session_close_utc(self, day: date) -> datetime:
        """Bar-close labelling depends on this (Phase 0 §3.6)."""
        return cast(datetime, self._cal.session_close(pd.Timestamp(day)).to_pydatetime())

    def session_open_utc(self, day: date) -> datetime:
        return cast(datetime, self._cal.session_open(pd.Timestamp(day)).to_pydatetime())

    def is_open_at(self, moment: datetime) -> bool:
        return bool(self._cal.is_open_on_minute(pd.Timestamp(moment)))


def calendar_for(code: str) -> ExchangeCalendarAdapter:
    return ExchangeCalendarAdapter(code)

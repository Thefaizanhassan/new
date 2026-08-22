"""OHLCV bars.

Convention, fixed once (Phase 0 §3.6): a bar is labelled by its **close** time
and stored in UTC.  A daily NSE bar for 2026-08-21 is timestamped at that
session's close, so ``timestamp <= now`` is a sound test for "was this bar
knowable?" — which is what makes the look-ahead guard in StrategyContext work.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.core.clock import ensure_utc
from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe

__all__ = ["Bar", "BarValidationError"]


class BarValidationError(ValueError):
    """A bar failed internal consistency checks and must not enter the store."""


@dataclass(frozen=True, slots=True)
class Bar:
    instrument_id: InstrumentId
    timestamp: datetime
    timeframe: Timeframe
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        self.validate()

    def validate(self) -> None:
        """The OHLC sanity gate from Phase 0 §7.2.

        These are not paranoia: unapplied splits, bad ticks and provider
        glitches all show up here first, and a strategy fed an impossible bar
        produces a phantom signal that looks entirely plausible downstream.
        """
        where = f"{self.instrument_id} @ {self.timestamp:%Y-%m-%d %H:%M}"
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise BarValidationError(f"{where}: non-positive price")
        if self.high < max(self.open, self.close):
            raise BarValidationError(f"{where}: high {self.high} below open/close")
        if self.low > min(self.open, self.close):
            raise BarValidationError(f"{where}: low {self.low} above open/close")
        if self.high < self.low:
            raise BarValidationError(f"{where}: high {self.high} < low {self.low}")
        if self.volume < 0:
            raise BarValidationError(f"{where}: negative volume {self.volume}")

    @property
    def typical_price(self) -> Decimal:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def turnover(self) -> Decimal:
        """Approximate value traded — used for the liquidity/ADV risk check."""
        return self.typical_price * self.volume

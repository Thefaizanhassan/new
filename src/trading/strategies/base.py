"""The strategy interface.

The safety property lives in :class:`StrategyContext`: it is built per bar and
**physically does not contain future data**, so look-ahead bias is eliminated by
construction rather than by discipline (Phase 0 §10.2).  There is no rule to
forget and no review step to miss — the future is simply not reachable from
inside a strategy.

Strategies emit :class:`~trading.core.intent.Intent` — a *target*, not an action.
Declarative targets are idempotent, compose across strategies, and keep sizing
in exactly one place (Phase 0 §4.4a).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd

from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Intent
from trading.core.position import Position
from trading.core.types import Money

__all__ = ["LifecycleStatus", "Strategy", "StrategyContext", "StrategySpec"]


class LifecycleStatus(StrEnum):
    """Phase 0 §10.5. Transitions are gated on evidence, not on intent."""

    RESEARCH = "RESEARCH"
    BACKTESTING = "BACKTESTING"
    PROMISING = "PROMISING"
    VALIDATED = "VALIDATED"
    PAPER = "PAPER"
    LIVE_APPROVED = "LIVE_APPROVED"
    LIVE = "LIVE"
    PAUSED = "PAUSED"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class StrategySpec:
    """Versioned, content-hashed identity for a strategy."""

    id: str
    name: str
    version: str
    description: str
    universe: tuple[InstrumentId, ...]
    params: dict[str, Any] = field(default_factory=dict)
    lifecycle_status: LifecycleStatus = LifecycleStatus.RESEARCH
    author: str = "unknown"

    @property
    def content_hash(self) -> str:
        """Any change to parameters or universe produces a new version.

        A backtest result is keyed by this together with the dataset version,
        cost-model version and code git SHA, so a changed number is always
        attributable to exactly one input (Phase 0 §10.4).
        """
        payload = json.dumps(
            {
                "id": self.id,
                "version": self.version,
                "params": {k: str(v) for k, v in sorted(self.params.items())},
                "universe": sorted(str(i) for i in self.universe),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def manifest_entry(self) -> dict[str, str]:
        return {
            "strategy_id": self.id,
            "version": self.version,
            "content_hash": self.content_hash,
            "lifecycle_status": self.lifecycle_status,
            "params": json.dumps({k: str(v) for k, v in sorted(self.params.items())}),
        }


class StrategyContext:
    """Everything a strategy may know at ``now`` — and nothing more.

    Constructed by the engine with history already truncated.  ``history()``
    cannot return future data because this object never received any.
    """

    __slots__ = ("_equity", "_history", "_now", "_positions")

    def __init__(
        self,
        now: datetime,
        history: dict[InstrumentId, pd.DataFrame],
        positions: dict[InstrumentId, Position],
        equity: Money,
    ) -> None:
        self._now = now
        self._history = history
        self._positions = positions
        self._equity = equity

    @property
    def now(self) -> datetime:
        """The only source of time a strategy may use."""
        return self._now

    @property
    def equity(self) -> Money:
        return self._equity

    def history(self, instrument_id: InstrumentId, bars: int | None = None) -> pd.DataFrame:
        """Bars up to and including ``now``. Never beyond it."""
        frame = self._history.get(instrument_id)
        if frame is None:
            return pd.DataFrame()
        return frame if bars is None else frame.tail(bars)

    def last_close(self, instrument_id: InstrumentId) -> Decimal | None:
        frame = self.history(instrument_id, 1)
        if frame.empty:
            return None
        return Decimal(str(frame["close"].iloc[-1]))

    def position(self, instrument_id: InstrumentId) -> Position | None:
        position = self._positions.get(instrument_id)
        return position if position and not position.is_flat else None

    def has_position(self, instrument_id: InstrumentId) -> bool:
        return self.position(instrument_id) is not None

    def bars_available(self, instrument_id: InstrumentId) -> int:
        return len(self.history(instrument_id))


class Strategy(Protocol):
    spec: StrategySpec

    def warmup_bars(self) -> int:
        """Bars required before the first signal is trustworthy."""
        ...

    def on_bar(self, ctx: StrategyContext, instrument: Instrument) -> list[Intent]:
        """Return desired target exposures. An empty list means 'no change'."""
        ...

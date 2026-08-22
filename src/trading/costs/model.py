"""Transaction cost models.

Costs are a versioned, first-class object rather than a constant buried in the
engine.  Phase 0 §20.1 ranks "costs consuming the edge" as one of the three
risks that actually matter: a strategy with 0.3% gross edge and 0.35% round-trip
cost is a guaranteed loss that looks excellent in a naive backtest.

The interface deliberately takes far more context than a
``getcommission(size, price)`` signature would.  Indian delivery equity charges
a **flat DP fee per scrip per session, on the sell side only** — a charge that
cannot be computed from size and price alone, and that dominates at small
position sizes (₹15 on a ₹5,000 position is 0.3%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Protocol

from trading.core.fill import CostBreakdown
from trading.core.instrument import Instrument, InstrumentId
from trading.core.types import Side

__all__ = ["CostContext", "CostModel", "SessionCostState"]


@dataclass
class SessionCostState:
    """Per-session state a cost model needs but cannot derive from one trade.

    Owned by the engine and reset at each session boundary.  Exists because
    some real charges are session-scoped rather than trade-scoped.
    """

    session_date: date
    depository_charged: set[InstrumentId] = field(default_factory=set)

    def note_depository_charge(self, instrument_id: InstrumentId) -> None:
        self.depository_charged.add(instrument_id)

    def already_charged_depository(self, instrument_id: InstrumentId) -> bool:
        return instrument_id in self.depository_charged


@dataclass(frozen=True, slots=True)
class CostContext:
    """Everything a cost model may need to price one execution."""

    instrument: Instrument
    side: Side
    quantity: Decimal
    price: Decimal
    timestamp: datetime
    session: SessionCostState

    @property
    def turnover(self) -> Decimal:
        return self.instrument.contract_value(self.price, self.quantity)


class CostModel(Protocol):
    """A named, versioned cost model. The version is recorded in every run manifest."""

    model_id: str
    version: str

    def compute(self, ctx: CostContext) -> CostBreakdown:
        """Itemised charges for one execution."""
        ...

    def describe(self) -> dict[str, str]:
        """Human-readable assumptions, embedded in backtest reports."""
        ...

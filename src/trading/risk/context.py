"""Everything a risk rule may need, assembled by the caller.

Rules never fetch anything.  A rule that reaches out to a database or a broker
can hang, and a risk check that can hang is a risk check that will eventually
be skipped "just this once".  Everything arrives as data.

Fields typed ``| None`` mean *not known*, which is distinct from a value and is
treated as blocking by any rule that needs it — Phase 0 §13.2, fail closed.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from trading.core.instrument import InstrumentId
from trading.core.order import OrderRequest
from trading.core.position import Position
from trading.core.types import Money, TradingMode

__all__ = ["OrderRecord", "PortfolioView", "RiskContext"]


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """A recently submitted order, for rate limiting and duplicate detection."""

    client_order_id: str
    instrument_id: InstrumentId
    side: str
    quantity: Decimal
    submitted_at: dt.datetime
    strategy_id: str = "unknown"

    def fingerprint(self) -> tuple[str, str, str]:
        """What makes two orders 'the same' for duplicate detection."""
        return (str(self.instrument_id), self.side, str(self.quantity))


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """The portfolio as the risk engine sees it."""

    equity: Money
    cash: Money
    positions: Mapping[InstrumentId, Position] = field(default_factory=dict)
    marks: Mapping[InstrumentId, Decimal] = field(default_factory=dict)
    day_start_equity: Money | None = None
    peak_equity: Money | None = None
    strategy_allocations: Mapping[str, Money] = field(default_factory=dict)

    @property
    def open_positions(self) -> int:
        return sum(1 for p in self.positions.values() if not p.is_flat)

    def market_value(self, instrument_id: InstrumentId) -> Money:
        position = self.positions.get(instrument_id)
        mark = self.marks.get(instrument_id)
        if position is None or mark is None:
            return Money.zero(self.equity.currency)
        return position.market_value(mark)

    @property
    def gross_exposure(self) -> Money:
        """Longs plus the absolute value of shorts."""
        total = Money.zero(self.equity.currency)
        for instrument_id in self.positions:
            total = total + self.market_value(instrument_id).abs()
        return total

    @property
    def net_exposure(self) -> Money:
        """Longs minus shorts. A market-neutral book is 0 net, high gross."""
        total = Money.zero(self.equity.currency)
        for instrument_id in self.positions:
            total = total + self.market_value(instrument_id)
        return total

    def weights(self) -> dict[InstrumentId, Decimal]:
        """Signed position weights as a fraction of equity."""
        if self.equity.is_zero:
            return {}
        return {
            instrument_id: self.market_value(instrument_id).ratio_to(self.equity)
            for instrument_id, position in self.positions.items()
            if not position.is_flat
        }


@dataclass(frozen=True, slots=True)
class RiskContext:
    """One proposed trade, plus the state needed to judge it."""

    request: OrderRequest
    reference_price: Decimal
    portfolio: PortfolioView
    now: dt.datetime
    mode: TradingMode

    # ── market state ────────────────────────────────────────────────────────
    session_open: bool | None = None
    """None means the calendar was not consulted — blocks in live modes."""
    data_age: dt.timedelta | None = None
    average_daily_volume: Decimal | None = None
    returns: pd.DataFrame | None = None
    """Aligned daily returns per instrument, for correlation-adjusted exposure."""

    # ── recent activity ─────────────────────────────────────────────────────
    recent_orders: Sequence[OrderRecord] = ()
    day_trades_in_window: int = 0
    consecutive_losses: int = 0
    recent_error_count: int = 0

    @property
    def order_value(self) -> Decimal:
        return self.request.instrument.contract_value(self.reference_price, self.request.quantity)

    @property
    def order_weight(self) -> Decimal:
        equity = self.portfolio.equity.amount
        return self.order_value / equity if equity else Decimal(1)

    @property
    def is_live_like(self) -> bool:
        """PAPER and LIVE run against a real clock, so session and staleness matter."""
        return self.mode in (TradingMode.PAPER, TradingMode.LIVE)

    @property
    def signed_order_value(self) -> Decimal:
        """Notional with direction. Negative for a sell."""
        return self.order_value * Decimal(self.request.side.sign)

    @property
    def equity_amount(self) -> Decimal:
        """Equity, floored so exposure ratios never divide by zero."""
        return max(self.portfolio.equity.amount, Decimal("1e-9"))

    # ── projections ─────────────────────────────────────────────────────────
    # Exposure rules must judge the book *after* the order, accounting for
    # direction. Adding an order's value to the existing position regardless of
    # side means a sell that closes a long reads as doubling it — which blocks
    # the close and traps the position. A risk rule that prevents you exiting is
    # worse than no rule at all.

    def projected_position_value(self) -> Decimal:
        current = self.portfolio.market_value(self.request.instrument.id).amount
        return current + self.signed_order_value

    def projected_weights(self) -> dict[InstrumentId, Decimal]:
        if self.portfolio.equity.amount <= 0:
            return {}
        weights = dict(self.portfolio.weights())
        weights[self.request.instrument.id] = self.projected_position_value() / self.equity_amount
        return {k: v for k, v in weights.items() if v != 0}

    def projected_gross_exposure(self) -> Decimal:
        instrument_id = self.request.instrument.id
        total = abs(self.projected_position_value())
        for other in self.portfolio.positions:
            if other != instrument_id:
                total += abs(self.portfolio.market_value(other).amount)
        return total

    def projected_net_exposure(self) -> Decimal:
        instrument_id = self.request.instrument.id
        total = self.projected_position_value()
        for other in self.portfolio.positions:
            if other != instrument_id:
                total += self.portfolio.market_value(other).amount
        return total

    def projected_open_positions(self) -> int:
        instrument_id = self.request.instrument.id
        others = sum(
            1
            for other, position in self.portfolio.positions.items()
            if other != instrument_id and not position.is_flat
        )
        return others + (1 if self.projected_position_value() != 0 else 0)

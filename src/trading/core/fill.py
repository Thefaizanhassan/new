"""Executions, and what they cost."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.core.clock import ensure_utc
from trading.core.instrument import Instrument
from trading.core.types import Currency, Money, Side

__all__ = ["CostBreakdown", "Fill"]


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Every charge on a single execution, itemised.

    Itemising rather than storing one total is what lets the backtest report
    gross vs net equity curves (Phase 0 §11.4) and makes it obvious *which*
    charge is eating a strategy — at ₹4L the flat DP charge behaves very
    differently from percentage-based STT (addendum §2).
    """

    currency: Currency
    brokerage: Money
    exchange_fees: Money
    transaction_tax: Money
    stamp_duty: Money
    regulatory_fees: Money
    depository_fees: Money
    gst: Money

    @property
    def total(self) -> Money:
        parts = (
            self.brokerage,
            self.exchange_fees,
            self.transaction_tax,
            self.stamp_duty,
            self.regulatory_fees,
            self.depository_fees,
            self.gst,
        )
        total = Money.zero(parts[0].currency)
        for part in parts:
            total = total + part
        return total

    def items(self) -> dict[str, Money]:
        return {
            "brokerage": self.brokerage,
            "exchange_fees": self.exchange_fees,
            "transaction_tax": self.transaction_tax,
            "stamp_duty": self.stamp_duty,
            "regulatory_fees": self.regulatory_fees,
            "depository_fees": self.depository_fees,
            "gst": self.gst,
        }

    @classmethod
    def zero(cls, currency: Currency) -> CostBreakdown:
        z = Money.zero(currency)
        return cls(currency, z, z, z, z, z, z, z)


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. Several fills may belong to one order (partial fills)."""

    fill_id: str
    client_order_id: str
    instrument: Instrument
    side: Side
    quantity: Decimal
    price: Decimal
    costs: CostBreakdown
    timestamp: datetime
    strategy_id: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", ensure_utc(self.timestamp))
        if self.quantity <= 0:
            raise ValueError(f"Fill quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Fill price must be positive, got {self.price}")

    @classmethod
    def create(
        cls,
        *,
        client_order_id: str,
        instrument: Instrument,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        costs: CostBreakdown,
        timestamp: datetime,
        strategy_id: str = "unknown",
    ) -> Fill:
        return cls(
            fill_id=f"fill_{uuid.uuid4().hex[:16]}",
            client_order_id=client_order_id,
            instrument=instrument,
            side=side,
            quantity=quantity,
            price=price,
            costs=costs,
            timestamp=timestamp,
            strategy_id=strategy_id,
        )

    @property
    def gross_value(self) -> Money:
        """Notional traded, before costs."""
        return Money(
            self.instrument.contract_value(self.price, self.quantity), self.instrument.currency
        )

    @property
    def cash_delta(self) -> Money:
        """Effect on cash: a buy spends value plus costs, a sell receives value minus costs."""
        return -(self.gross_value * self.side.sign) - self.costs.total

    @property
    def signed_quantity(self) -> Decimal:
        return self.quantity * self.side.sign

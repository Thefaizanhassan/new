"""A minimal cost model.

Exists for two jobs: modelling a commission-free US broker, where the only real
cost is spread and slippage; and giving the backtrader cross-check something
both engines can express identically, so a disagreement between them is about
the engine rather than about tax arithmetic neither implements the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading.core.fill import CostBreakdown
from trading.core.types import Currency, Money
from trading.costs.model import CostContext

__all__ = ["FlatPercentCosts"]


@dataclass(frozen=True)
class FlatPercentCosts:
    """A single percentage of turnover, charged on both sides."""

    rate: Decimal = Decimal("0.001")
    currency: Currency = Currency.INR
    model_id: str = "flat_percent"
    version: str = "1.0.0"

    def compute(self, ctx: CostContext) -> CostBreakdown:
        zero = Money.zero(self.currency)
        return CostBreakdown(
            currency=self.currency,
            brokerage=Money(ctx.turnover * self.rate, self.currency),
            exchange_fees=zero,
            transaction_tax=zero,
            stamp_duty=zero,
            regulatory_fees=zero,
            depository_fees=zero,
            gst=zero,
        )

    def describe(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "rate": f"{self.rate:%} of turnover, both sides",
            "note": "no taxes or flat fees — for commission-only markets and cross-checks",
        }

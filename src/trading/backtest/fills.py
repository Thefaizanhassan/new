"""Fill models — where backtests lie.

Phase 0 §11.2.  Everything else in a backtest can be correct and the result
still be fiction, because the fill model decides what price you got and whether
you got it at all.

Four things this module refuses to pretend:

* **You cannot trade at a price the bar never printed.** A bar is a lossy
  summary; within it the price may have visited the high and the low in either
  order. Assuming favourable intra-bar ordering is the single most common way a
  backtest manufactures alpha.
* **You pay the spread.** Every round trip loses roughly the full spread before
  anything else happens.
* **Your own order moves the price.** Above a small share of a bar's volume,
  impact stops being negligible.
* **You cannot buy more than traded.** A volume cap forces partial fills, which
  is what stops a backtest "buying" ₹20 lakh of a stock that trades ₹5 lakh a
  day and reporting a spectacular return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from trading.core.order import Order, OrderType
from trading.core.types import Side

__all__ = [
    "FillModel",
    "FillOutcome",
    "FillRequest",
    "FixedBpsSlippage",
    "NoSlippage",
    "RealisticFillModel",
    "SlippageModel",
    "SpreadSlippage",
    "SquareRootImpact",
]


@dataclass(frozen=True, slots=True)
class FillRequest:
    """One order meeting one bar."""

    order: Order
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    previous_close: Decimal | None = None

    @property
    def side(self) -> Side:
        return self.order.request.side

    @property
    def quantity(self) -> Decimal:
        return self.order.request.quantity

    @property
    def gapped(self) -> bool:
        """Whether the bar opened outside the previous bar's range."""
        if self.previous_close is None:
            return False
        move = abs(self.open_price - self.previous_close) / self.previous_close
        return move > Decimal("0.10")


@dataclass(frozen=True, slots=True)
class FillOutcome:
    filled_quantity: Decimal
    fill_price: Decimal
    slippage_per_unit: Decimal = Decimal(0)
    reason: str = ""

    @property
    def filled(self) -> bool:
        return self.filled_quantity > 0

    @property
    def partial(self) -> bool:
        return self.filled_quantity > 0 and bool(self.reason)


# ── slippage ────────────────────────────────────────────────────────────────
class SlippageModel(Protocol):
    """Implementations are frozen dataclasses, so the id is read-only."""

    @property
    def model_id(self) -> str: ...

    def slippage(self, request: FillRequest, quantity: Decimal) -> Decimal:
        """Adverse price movement per unit. Always positive; direction is applied later."""
        ...

    def describe(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class NoSlippage:
    """For isolating other effects in a test. Never for a result you act on."""

    model_id: str = "none"

    def slippage(self, request: FillRequest, quantity: Decimal) -> Decimal:
        return Decimal(0)

    def describe(self) -> dict[str, str]:
        return {"model": self.model_id, "note": "no slippage — testing only"}


@dataclass(frozen=True)
class FixedBpsSlippage:
    """A flat haircut in basis points. Crude but honest, and easy to reason about."""

    bps: Decimal = Decimal("5")
    model_id: str = "fixed_bps"

    def slippage(self, request: FillRequest, quantity: Decimal) -> Decimal:
        return request.open_price * self.bps / Decimal(10_000)

    def describe(self) -> dict[str, str]:
        return {"model": self.model_id, "bps": str(self.bps)}


@dataclass(frozen=True)
class SpreadSlippage:
    """Half the bid-ask spread, which you pay on entry and again on exit.

    The default is a rough figure for a liquid large-cap. A mid-cap can be five
    to ten times wider, and for a strategy trading 200 times a year that
    difference alone can exceed the entire edge.
    """

    spread_bps: Decimal = Decimal("4")
    model_id: str = "half_spread"

    def slippage(self, request: FillRequest, quantity: Decimal) -> Decimal:
        return request.open_price * self.spread_bps / Decimal(20_000)

    def describe(self) -> dict[str, str]:
        return {
            "model": self.model_id,
            "spread_bps": str(self.spread_bps),
            "note": "half-spread per side, so a round trip pays the full spread",
        }


@dataclass(frozen=True)
class SquareRootImpact:
    """Market impact growing with the square root of participation.

    ``impact = coefficient x daily_volatility x sqrt(quantity / bar_volume)``

    The square-root law is the standard academic model: impact grows sub-linearly
    with size, so doubling an order costs about 1.4x rather than 2x. Combined
    with a half-spread it is the most defensible retail-scale assumption.
    """

    coefficient: Decimal = Decimal("0.5")
    daily_volatility: Decimal = Decimal("0.02")
    spread_bps: Decimal = Decimal("4")
    model_id: str = "sqrt_impact"

    def slippage(self, request: FillRequest, quantity: Decimal) -> Decimal:
        half_spread = request.open_price * self.spread_bps / Decimal(20_000)
        if request.volume <= 0:
            return half_spread
        participation = float(quantity / request.volume)
        impact_fraction = (
            float(self.coefficient) * float(self.daily_volatility) * math.sqrt(participation)
        )
        return half_spread + request.open_price * Decimal(str(impact_fraction))

    def describe(self) -> dict[str, str]:
        return {
            "model": self.model_id,
            "coefficient": str(self.coefficient),
            "daily_volatility": str(self.daily_volatility),
            "spread_bps": str(self.spread_bps),
        }


# ── the fill model ──────────────────────────────────────────────────────────
class FillModel(Protocol):
    """Implementations are frozen dataclasses, so the id is read-only."""

    @property
    def model_id(self) -> str: ...

    def fill(self, request: FillRequest) -> FillOutcome: ...

    def describe(self) -> dict[str, str]: ...


@dataclass(frozen=True)
class RealisticFillModel:
    """Market and limit orders against one bar, with the honest constraints.

    ``max_participation`` caps a fill at a share of the bar's volume. Anything
    above it becomes a partial fill, which is exactly what would happen — and
    what prevents a backtest from silently trading size the market never had.

    ``limit_fill_probability`` haircuts limit orders: a bar touching your price
    does not mean you were filled at it, because other orders were ahead of you
    in the queue. Without this haircut, limit-order backtests are systematically
    optimistic.
    """

    slippage_model: SlippageModel = SpreadSlippage()
    max_participation: Decimal = Decimal("0.05")
    limit_fill_probability: Decimal = Decimal("0.5")
    model_id: str = "realistic"

    def fill(self, request: FillRequest) -> FillOutcome:
        if request.volume <= 0:
            return FillOutcome(
                Decimal(0),
                Decimal(0),
                reason="no volume — the instrument did not trade in this bar",
            )

        capacity = request.volume * self.max_participation
        if request.order.request.order_type is OrderType.LIMIT:
            return self._limit_fill(request, capacity)
        return self._market_fill(request, capacity)

    def _market_fill(self, request: FillRequest, capacity: Decimal) -> FillOutcome:
        quantity = min(request.quantity, capacity)
        reason = (
            f"capped at {self.max_participation:%} of bar volume "
            f"({request.quantity} requested, {quantity} available)"
            if quantity < request.quantity
            else ""
        )
        slip = self.slippage_model.slippage(request, quantity)
        price = request.open_price + slip * Decimal(request.side.sign)
        # A fill cannot print outside the bar's range, whatever the model says.
        price = max(request.low_price, min(request.high_price, price))
        return FillOutcome(quantity, price, slip, reason)

    def _limit_fill(self, request: FillRequest, capacity: Decimal) -> FillOutcome:
        limit = request.order.request.limit_price
        assert limit is not None
        touched = (
            request.low_price <= limit if request.side is Side.BUY else request.high_price >= limit
        )
        if not touched:
            return FillOutcome(
                Decimal(0),
                Decimal(0),
                reason=f"limit {limit} not touched (bar {request.low_price}-{request.high_price})",
            )

        quantity = min(request.quantity, capacity) * self.limit_fill_probability
        quantity = quantity.quantize(Decimal(1))
        if quantity <= 0:
            return FillOutcome(Decimal(0), Decimal(0), reason="queue position — no fill")

        reason = "queue haircut: touching a price is not being filled at it"
        if quantity < request.quantity:
            reason += f"; {request.quantity} requested, {quantity} filled"
        return FillOutcome(quantity, limit, Decimal(0), reason)

    def describe(self) -> dict[str, str]:
        return {
            "fill_model": self.model_id,
            "max_participation": str(self.max_participation),
            "limit_fill_probability": str(self.limit_fill_probability),
            **{f"slippage_{k}": v for k, v in self.slippage_model.describe().items()},
        }

"""Orders, and the gate they must pass to exist."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Final, Self

from trading.core.clock import ensure_utc
from trading.core.instrument import Instrument
from trading.core.risk import RiskDecision, UnauthorizedOrderError
from trading.core.types import Side

__all__ = ["Order", "OrderRequest", "OrderState", "OrderType", "TimeInForce"]


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class OrderState(StrEnum):
    """Phase 0 §9.2.

    ``UNKNOWN`` is the important one: "we sent it and the connection died" is a
    real state, and treating it as either filled or not-filled is how automated
    systems double a position.  It blocks further orders on the instrument until
    resolved by querying the broker.
    """

    PENDING_NEW = "PENDING_NEW"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        )

    @property
    def blocks_instrument(self) -> bool:
        """Whether this state must stop new orders for the same instrument."""
        return self is OrderState.UNKNOWN


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """What the sizing layer wants to do. Not yet an order — risk has not run."""

    instrument: Instrument
    side: Side
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    strategy_id: str = "unknown"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError("A LIMIT order requires a limit_price")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValueError("A MARKET order must not carry a limit_price")

    @property
    def notional(self) -> Decimal | None:
        """Value at the limit price. ``None`` for market orders — no price is known yet."""
        if self.limit_price is None:
            return None
        return self.instrument.contract_value(self.limit_price, self.quantity)


class _OrderGuard:
    __slots__ = ()


_ORDER_GUARD: Final = _OrderGuard()


@dataclass(frozen=True, slots=True)
class Order:
    """A risk-approved instruction to trade.

    The only supported way to build one is :meth:`create`, which requires an
    approved :class:`RiskDecision`.  Direct construction raises
    :class:`UnauthorizedOrderError`.
    """

    client_order_id: str
    request: OrderRequest
    decision: RiskDecision
    created_at: datetime
    state: OrderState = OrderState.PENDING_NEW
    broker_order_id: str | None = None
    filled_quantity: Decimal = Decimal(0)
    _guard: _OrderGuard | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._guard is not _ORDER_GUARD:
            raise UnauthorizedOrderError(
                "Order must be created via Order.create() with an approved RiskDecision. "
                "This gate is the platform's core safety property — see Phase 0 §13.3."
            )

    @classmethod
    def create(cls, request: OrderRequest, decision: RiskDecision, at: datetime) -> Self:
        if not decision.approved:
            raise UnauthorizedOrderError(
                f"Refusing to create an order on a rejected risk decision "
                f"({decision.decision_id}): {decision.rejection_reason}"
            )
        return cls(
            # Client-generated and persisted before sending, so a timed-out
            # submission can be resolved by query rather than by resend
            # (Phase 0 §3.5). This is the anti-duplicate-order mechanism.
            client_order_id=f"coid_{uuid.uuid4().hex[:20]}",
            request=request,
            decision=decision,
            created_at=ensure_utc(at),
            _guard=_ORDER_GUARD,
        )

    def with_state(
        self,
        state: OrderState,
        *,
        filled_quantity: Decimal | None = None,
        broker_order_id: str | None = None,
    ) -> Order:
        """Return a new Order at a new state. Orders are immutable; history is events."""
        return Order(
            client_order_id=self.client_order_id,
            request=self.request,
            decision=self.decision,
            created_at=self.created_at,
            state=state,
            broker_order_id=broker_order_id or self.broker_order_id,
            filled_quantity=self.filled_quantity if filled_quantity is None else filled_quantity,
            _guard=_ORDER_GUARD,
        )

    @property
    def remaining_quantity(self) -> Decimal:
        return self.request.quantity - self.filled_quantity

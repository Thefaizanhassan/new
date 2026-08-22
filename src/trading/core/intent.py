"""Intents: what a strategy wants, expressed as a destination.

Phase 0 §4.4(a).  A strategy declares a *target* ("be 4% long RELIANCE"), never
an action ("buy 21 shares").  Declarative targets are idempotent, so a missed
message or a restart self-heals; they compose across strategies at the
portfolio layer; and they keep sizing in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import Any

from trading.core.instrument import InstrumentId

__all__ = ["Flat", "Intent", "Target", "TargetQty", "TargetWeight", "Urgency"]


class Urgency(StrEnum):
    """Maps to order type at the execution layer, not inside the strategy."""

    PATIENT = "PATIENT"
    NORMAL = "NORMAL"
    IMMEDIATE = "IMMEDIATE"


@dataclass(frozen=True, slots=True)
class TargetWeight:
    """Target as a fraction of portfolio equity. Negative means short."""

    weight: Decimal

    def __post_init__(self) -> None:
        if abs(self.weight) > 1:
            raise ValueError(
                f"Target weight {self.weight} exceeds 100% of equity. "
                f"Leverage must be requested explicitly, not implied by a weight."
            )


@dataclass(frozen=True, slots=True)
class TargetQty:
    """Target as an absolute signed quantity."""

    quantity: Decimal


@dataclass(frozen=True, slots=True)
class Flat:
    """Target no position."""


Target = TargetWeight | TargetQty | Flat


@dataclass(frozen=True, slots=True)
class Intent:
    instrument_id: InstrumentId
    target: Target
    strategy_id: str
    reason: str
    urgency: Urgency = Urgency.NORMAL
    confidence: Decimal | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def target_weight(self, equity_value: Decimal, price: Decimal, multiplier: Decimal) -> Decimal:
        """Normalise any target form to a portfolio weight."""
        match self.target:
            case Flat():
                return Decimal(0)
            case TargetWeight(weight=w):
                return w
            case TargetQty(quantity=q):
                if equity_value == 0:
                    return Decimal(0)
                return (q * price * multiplier) / equity_value

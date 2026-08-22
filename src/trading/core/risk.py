"""The risk decision token.

Phase 0 §4.4(b) and §13.3.  A convention that says "always call the risk
engine" gets forgotten during a 1am hotfix.  Instead, an :class:`Order` cannot
be constructed without an approved :class:`RiskDecision`, and a RiskDecision
cannot be constructed without the guard held in this module.

This is not cryptographic — Python has no private constructors.  The point is
that bypassing risk requires *deliberately editing the domain core*, which is
visible in a diff and in review, rather than being one forgotten call.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Self

from trading.core.clock import ensure_utc

__all__ = ["RiskDecision", "RiskRuleResult", "RiskVerdict", "UnauthorizedOrderError"]


class UnauthorizedOrderError(RuntimeError):
    """Something tried to build an Order without a valid approved RiskDecision."""


class _Guard:
    """Sentinel proving a value came from a sanctioned factory."""

    __slots__ = ()


_DECISION_GUARD: Final = _Guard()


@dataclass(frozen=True, slots=True)
class RiskRuleResult:
    """One rule's verdict, with the numbers that produced it.

    Storing ``observed`` and ``limit`` rather than just a boolean is what makes
    Phase 0 §16's decision chain readable months later, and what lets you tell
    a well-calibrated limit from one quietly strangling a strategy.
    """

    rule_id: str
    passed: bool
    observed: str
    limit: str
    detail: str = ""

    def __str__(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.rule_id}: {self.observed} vs limit {self.limit} {self.detail}".strip()
        )


class RiskVerdict:
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Proof that a proposed trade passed every risk rule.

    Construct only via :meth:`approve` or :meth:`reject`, which the risk engine
    calls.  Direct construction raises.
    """

    decision_id: str
    verdict: str
    issued_at: datetime
    issued_by: str
    results: tuple[RiskRuleResult, ...] = field(default_factory=tuple)
    _guard: _Guard | None = None

    def __post_init__(self) -> None:
        if self._guard is not _DECISION_GUARD:
            raise UnauthorizedOrderError(
                "RiskDecision must be created by the risk engine via "
                "RiskDecision.approve()/reject(), not constructed directly."
            )

    @classmethod
    def _issue(
        cls, verdict: str, issued_by: str, at: datetime, results: tuple[RiskRuleResult, ...]
    ) -> Self:
        return cls(
            decision_id=f"rd_{uuid.uuid4().hex[:16]}",
            verdict=verdict,
            issued_at=ensure_utc(at),
            issued_by=issued_by,
            results=results,
            _guard=_DECISION_GUARD,
        )

    @classmethod
    def approve(cls, issued_by: str, at: datetime, results: tuple[RiskRuleResult, ...]) -> Self:
        if any(not r.passed for r in results):
            raise ValueError("Cannot approve a decision containing failed rules")
        return cls._issue(RiskVerdict.APPROVED, issued_by, at, results)

    @classmethod
    def reject(cls, issued_by: str, at: datetime, results: tuple[RiskRuleResult, ...]) -> Self:
        if all(r.passed for r in results):
            raise ValueError("Cannot reject a decision in which every rule passed")
        return cls._issue(RiskVerdict.REJECTED, issued_by, at, results)

    @property
    def approved(self) -> bool:
        return self.verdict == RiskVerdict.APPROVED

    @property
    def failures(self) -> tuple[RiskRuleResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    @property
    def rejection_reason(self) -> str:
        return "; ".join(str(r) for r in self.failures) or "no failing rules"

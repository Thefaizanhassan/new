"""Lifecycle gates.

Phase 0 §10.5 said the lifecycle statuses should be **enforceable**, not labels
a strategy wears because someone typed them.  This is that enforcement.

Three principles, and the third is the one that keeps this honest.

1. **Fail closed.** A criterion that fails blocks promotion.
2. **Data tier caps ambition.** A strategy backtested on synthetic or
   prototype-tier data cannot be promoted past research, however good the
   numbers look.  Phase 2 made the tier travel with the bars precisely so this
   gate could read it.
3. **A check that does not exist yet blocks, rather than passes.**  Walk-forward
   validation arrives in Phase 6; until then the criterion reports
   ``UNAVAILABLE`` and refuses promotion.  A gate that silently passes because
   nobody implemented it is worse than no gate — it manufactures confidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from trading.data.provider import DataTier
from trading.strategies.base import LifecycleStatus

__all__ = [
    "CriterionStatus",
    "Evidence",
    "GateResult",
    "PromotionDecision",
    "evaluate_promotion",
]


class CriterionStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNAVAILABLE = "UNAVAILABLE"
    """The check is not implemented yet. Blocks promotion, deliberately."""

    @property
    def blocks(self) -> bool:
        return self is not CriterionStatus.PASS


@dataclass(frozen=True, slots=True)
class GateResult:
    criterion_id: str
    status: CriterionStatus
    observed: str
    required: str
    detail: str = ""

    def __str__(self) -> str:
        return (
            f"[{self.status}] {self.criterion_id}: {self.observed} "
            f"(required: {self.required}){' — ' + self.detail if self.detail else ''}"
        )


@dataclass(frozen=True, slots=True)
class Evidence:
    """What is actually known about a strategy.

    Fields typed ``| None`` mean "not yet assessed", which is distinct from
    "assessed and failed" and is treated as blocking rather than passing.
    """

    data_tier: DataTier
    backtest_completed: bool = False
    trades: int = 0
    net_expectancy: Decimal | None = None
    benchmark_compared: bool = False
    out_of_sample_passed: bool | None = None
    walk_forward_passed: bool | None = None
    parameter_plateau: bool | None = None
    monte_carlo_drawdown_ok: bool | None = None
    trial_count: int | None = None
    risk_limits_defined: bool = False
    kill_switch_drilled: bool = False
    reconciliation_clean: bool | None = None
    paper_trading_days: int = 0
    divergence_within_tolerance: bool | None = None
    human_signoff_at: datetime | None = None
    human_signoff_by: str = ""


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    strategy_id: str
    from_status: LifecycleStatus
    to_status: LifecycleStatus
    results: tuple[GateResult, ...] = field(default_factory=tuple)

    @property
    def approved(self) -> bool:
        return bool(self.results) and not any(r.status.blocks for r in self.results)

    @property
    def blockers(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if r.status.blocks)

    def summary(self) -> str:
        verdict = "APPROVED" if self.approved else "BLOCKED"
        return (
            f"{self.strategy_id}: {self.from_status} → {self.to_status} {verdict} "
            f"({len(self.blockers)} blocker(s) of {len(self.results)} criteria)"
        )


# ── individual criteria ─────────────────────────────────────────────────────
def _boolean(criterion: str, value: bool | None, required: str, detail: str = "") -> GateResult:
    if value is None:
        return GateResult(
            criterion,
            CriterionStatus.UNAVAILABLE,
            "not assessed",
            required,
            detail or "the check that would answer this is not implemented yet",
        )
    return GateResult(
        criterion,
        CriterionStatus.PASS if value else CriterionStatus.FAIL,
        str(value),
        required,
        detail,
    )


def _tier_allows(evidence: Evidence, to_status: LifecycleStatus) -> GateResult:
    """Data trust caps how far a strategy may go, regardless of its numbers."""
    research_only = (DataTier.SYNTHETIC, DataTier.PROTOTYPE)
    beyond_backtesting = to_status not in (
        LifecycleStatus.RESEARCH,
        LifecycleStatus.BACKTESTING,
    )
    if evidence.data_tier in research_only and beyond_backtesting:
        return GateResult(
            "GATE_000_data_tier",
            CriterionStatus.FAIL,
            str(evidence.data_tier),
            "PRODUCTION",
            "a strategy cannot be promoted past backtesting on data with no "
            "point-in-time guarantee, however good the results look",
        )
    return GateResult(
        "GATE_000_data_tier",
        CriterionStatus.PASS,
        str(evidence.data_tier),
        "PRODUCTION beyond backtesting",
    )


def _backtesting_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    return [tier]


def _promising_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    expectancy = evidence.net_expectancy
    return [
        tier,
        _boolean("GATE_001_backtest_complete", evidence.backtest_completed, "True"),
        GateResult(
            "GATE_002_net_expectancy",
            CriterionStatus.UNAVAILABLE
            if expectancy is None
            else (CriterionStatus.PASS if expectancy > 0 else CriterionStatus.FAIL),
            "not measured" if expectancy is None else f"{expectancy:.4f}",
            "> 0 after all costs",
            "gross expectancy is not evidence; costs decide" if expectancy is None else "",
        ),
        GateResult(
            "GATE_003_trade_count",
            CriterionStatus.PASS if evidence.trades >= 100 else CriterionStatus.FAIL,
            str(evidence.trades),
            ">= 100",
            "fewer than ~100 trades and the error bars swamp the estimate",
        ),
        _boolean(
            "GATE_004_benchmark_compared",
            evidence.benchmark_compared,
            "True",
            "beating buy-and-hold net of costs is the bar",
        ),
    ]


def _validated_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    return [
        tier,
        _boolean("GATE_010_out_of_sample", evidence.out_of_sample_passed, "True"),
        _boolean("GATE_011_walk_forward", evidence.walk_forward_passed, "True"),
        _boolean(
            "GATE_012_parameter_plateau",
            evidence.parameter_plateau,
            "True",
            "a spike rather than a plateau means the parameters fit noise",
        ),
        _boolean("GATE_013_monte_carlo_drawdown", evidence.monte_carlo_drawdown_ok, "True"),
        GateResult(
            "GATE_014_trial_count_recorded",
            CriterionStatus.PASS
            if evidence.trial_count is not None
            else CriterionStatus.UNAVAILABLE,
            "not recorded" if evidence.trial_count is None else str(evidence.trial_count),
            "recorded",
            "without the number of trials, the deflated Sharpe cannot be computed",
        ),
    ]


def _paper_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    return [
        tier,
        _boolean("GATE_020_risk_limits_defined", evidence.risk_limits_defined, "True"),
        _boolean(
            "GATE_021_kill_switch_drilled",
            evidence.kill_switch_drilled,
            "True",
            "an undrilled kill switch is an assumption, not a control",
        ),
        _boolean("GATE_022_reconciliation_clean", evidence.reconciliation_clean, "True"),
    ]


def _live_approved_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    return [
        tier,
        GateResult(
            "GATE_030_paper_days",
            CriterionStatus.PASS if evidence.paper_trading_days >= 60 else CriterionStatus.FAIL,
            str(evidence.paper_trading_days),
            ">= 60 trading days",
        ),
        _boolean(
            "GATE_031_divergence",
            evidence.divergence_within_tolerance,
            "True",
            "live results must track the backtest distribution",
        ),
        GateResult(
            "GATE_032_human_signoff",
            CriterionStatus.PASS if evidence.human_signoff_at else CriterionStatus.FAIL,
            evidence.human_signoff_by or "none",
            "an explicit, timestamped human sign-off",
        ),
    ]


def _live_criteria(evidence: Evidence, tier: GateResult) -> list[GateResult]:
    return [
        GateResult(
            "GATE_040_live_not_implemented",
            CriterionStatus.UNAVAILABLE,
            "Phase 12",
            "the live safety gates to exist",
            "live trading is not implemented; this status is unreachable by design",
        )
    ]


_CRITERIA: dict[LifecycleStatus, Callable[[Evidence, GateResult], list[GateResult]]] = {
    LifecycleStatus.BACKTESTING: _backtesting_criteria,
    LifecycleStatus.PROMISING: _promising_criteria,
    LifecycleStatus.VALIDATED: _validated_criteria,
    LifecycleStatus.PAPER: _paper_criteria,
    LifecycleStatus.LIVE_APPROVED: _live_approved_criteria,
    LifecycleStatus.LIVE: _live_criteria,
}


def _criteria_for(to_status: LifecycleStatus, evidence: Evidence) -> list[GateResult]:
    builder = _CRITERIA.get(to_status)
    if builder is None:
        # PAUSED and RETIRED — stopping never requires evidence.
        return [
            GateResult(
                "GATE_099_always_allowed",
                CriterionStatus.PASS,
                str(to_status),
                "no criteria",
                "stopping a strategy never requires evidence",
            )
        ]
    return builder(evidence, _tier_allows(evidence, to_status))


_ORDER: dict[LifecycleStatus, int] = {
    LifecycleStatus.RESEARCH: 0,
    LifecycleStatus.BACKTESTING: 1,
    LifecycleStatus.PROMISING: 2,
    LifecycleStatus.VALIDATED: 3,
    LifecycleStatus.PAPER: 4,
    LifecycleStatus.LIVE_APPROVED: 5,
    LifecycleStatus.LIVE: 6,
}


def evaluate_promotion(
    strategy_id: str,
    from_status: LifecycleStatus,
    to_status: LifecycleStatus,
    evidence: Evidence,
) -> PromotionDecision:
    """Decide whether a strategy may move to ``to_status``.

    Promotion must be one step at a time: skipping a stage is how a strategy
    reaches paper trading without ever having been validated.
    """
    if to_status in (LifecycleStatus.PAUSED, LifecycleStatus.RETIRED):
        return PromotionDecision(
            strategy_id, from_status, to_status, tuple(_criteria_for(to_status, evidence))
        )

    current = _ORDER.get(from_status)
    target = _ORDER.get(to_status)
    if current is None or target is None:
        return PromotionDecision(
            strategy_id,
            from_status,
            to_status,
            (
                GateResult(
                    "GATE_100_transition",
                    CriterionStatus.FAIL,
                    f"{from_status} → {to_status}",
                    "a known transition",
                ),
            ),
        )
    if target != current + 1:
        return PromotionDecision(
            strategy_id,
            from_status,
            to_status,
            (
                GateResult(
                    "GATE_100_transition",
                    CriterionStatus.FAIL,
                    f"{from_status} → {to_status}",
                    "exactly one step forward",
                    "skipping a stage is how a strategy reaches paper trading "
                    "without ever having been validated",
                ),
            ),
        )

    return PromotionDecision(
        strategy_id, from_status, to_status, tuple(_criteria_for(to_status, evidence))
    )

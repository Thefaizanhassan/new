"""The validation report: one object the lifecycle gate can read.

Phase 0 §10.5.  Until now the ``VALIDATED`` gate refused everything, because the
checks it asks about did not exist and a gate that passes for want of a check is
worse than no gate.  This is what finally answers it — and it answers by
*carrying the evidence*, not by asserting a verdict.

The structure matters.  Each criterion is derived from a specific artefact:

===========================================  ====================================
Gate                                          Answered by
===========================================  ====================================
``GATE_010_out_of_sample``                     pooled walk-forward test segments
``GATE_011_walk_forward``                      efficiency, consistency, stability
``GATE_012_parameter_plateau``                 the sensitivity surface's shape
``GATE_013_monte_carlo_drawdown``              resampled drawdown vs the risk limit
``GATE_014_trial_count_recorded``              the experiment ledger
``GATE_015_deflated_sharpe``                   pooled Sharpe, deflated by that count
===========================================  ====================================

A missing artefact yields ``None``, which the lifecycle gate reads as
``UNAVAILABLE`` and treats as blocking.  There is deliberately no way to hand it
a bare ``True``: every field here is computed from something that ran.

**The pooled out-of-sample record is scored by the same code as a backtest.**
:func:`trading.backtest.metrics.compute_metrics` reads the concatenated
walk-forward curve exactly as it reads a single run's curve.  A second,
subtly-different metric path for validation is how two numbers that should agree
end up differing by the thing you were trying to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from trading.backtest.metrics import PerformanceReport, compute_metrics
from trading.data.provider import DataTier
from trading.validation.montecarlo import MonteCarloResult
from trading.validation.purged_cv import PurgedCvResult
from trading.validation.sensitivity import SensitivitySurface
from trading.validation.walkforward import WalkForwardResult

__all__ = ["ValidationReport", "ValidationThresholds"]


@dataclass(frozen=True, slots=True)
class ValidationThresholds:
    """The bars for the gates this report answers, written down in one place."""

    min_deflated_sharpe: float = 0.95
    """Below roughly 0.95 the result is not distinguishable from the best of
    however many things were tried (López de Prado). This is the number the whole
    trial-counting apparatus exists to make honest."""
    min_oos_sharpe: float = 0.3
    """A low bar on purpose. It is here to exclude a positive return that came
    entirely from one lucky stretch, not to certify anything."""
    drawdown_percentile: float = 95.0
    """Which percentile of the resampled drawdown distribution has to fit inside
    the risk limit. The observed path is a sample of one and is not used."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything Phase 6 measured about one strategy, and what it implies."""

    strategy_id: str
    strategy_version: str
    instrument: str
    data_tier: DataTier
    walk_forward: WalkForwardResult
    trial_count: int
    """From the experiment ledger. A floor, not the truth — see
    :mod:`trading.validation.experiments`."""
    drawdown_limit: Decimal
    """The risk profile's max drawdown, which the resampled distribution is read
    against. Validation that does not know the limit cannot say whether the
    strategy fits inside it."""
    purged_cv: PurgedCvResult | None = None
    sensitivity: SensitivitySurface | None = None
    trade_monte_carlo: MonteCarloResult | None = None
    path_monte_carlo: MonteCarloResult | None = None
    thresholds: ValidationThresholds = ValidationThresholds()
    starting_capital: Decimal = Decimal(1)

    # ── the pooled out-of-sample record ─────────────────────────────────────
    @property
    def oos_metrics(self) -> PerformanceReport:
        """Metrics over the concatenated walk-forward test segments.

        ``trials`` is the ledger's count, so the deflated Sharpe here is deflated
        by what was actually tried rather than by 1.
        """
        curve = self.walk_forward.oos_equity_curve(float(self.starting_capital))
        if curve.empty or len(curve) < 3:
            return PerformanceReport({"error": "not enough out-of-sample observations"})
        return compute_metrics(
            curve,
            [],
            starting_capital=self.starting_capital,
            total_costs=Decimal(0),
            trials=max(self.trial_count, 1),
        )

    @property
    def oos_sharpe(self) -> float | None:
        value = self.oos_metrics.get("sharpe")
        return float(value) if value is not None else None

    @property
    def deflated_sharpe(self) -> float | None:
        value = self.oos_metrics.get("deflated_sharpe")
        return float(value) if value is not None else None

    @property
    def oos_net_return(self) -> float:
        return self.walk_forward.oos_net_return

    # ── the lifecycle criteria ──────────────────────────────────────────────
    @property
    def out_of_sample_passed(self) -> bool | None:
        """Is the pooled out-of-sample record profitable and not just lucky?

        Both halves are required.  A positive total return with a Sharpe near zero
        is one good stretch surrounded by noise, and promoting on it is promoting
        a coincidence.
        """
        sharpe = self.oos_sharpe
        if sharpe is None or not self.walk_forward.scored_folds:
            return None
        return self.oos_net_return > 0 and sharpe >= self.thresholds.min_oos_sharpe

    @property
    def walk_forward_passed(self) -> bool:
        return self.walk_forward.passed

    @property
    def parameter_plateau(self) -> bool | None:
        """``None`` when no sweep was run: not swept is not the same as flat."""
        return None if self.sensitivity is None else self.sensitivity.is_plateau

    @property
    def monte_carlo_drawdown_ok(self) -> bool | None:
        """Does the resampled drawdown distribution fit inside the risk limit?

        Read against *both* resamplings when both are present, and both must fit.
        Trade resampling asks how bad the ordering could have been; the block
        bootstrap asks how bad the path could have been.  A strategy that fails
        either one has a drawdown its risk limit does not cover.
        """
        results = [r for r in (self.trade_monte_carlo, self.path_monte_carlo) if r is not None]
        if not results:
            return None
        limit = float(self.drawdown_limit)
        return all(
            r.drawdown_percentile(self.thresholds.drawdown_percentile) <= limit for r in results
        )

    @property
    def deflated_sharpe_passed(self) -> bool | None:
        dsr = self.deflated_sharpe
        if dsr is None:
            return None
        return dsr >= self.thresholds.min_deflated_sharpe

    @property
    def worst_case_drawdown(self) -> float | None:
        """The deepest drawdown at the chosen percentile across both resamplings."""
        results = [r for r in (self.trade_monte_carlo, self.path_monte_carlo) if r is not None]
        if not results:
            return None
        return max(r.drawdown_percentile(self.thresholds.drawdown_percentile) for r in results)

    @property
    def recommended_drawdown_limit(self) -> float | None:
        """What the distribution suggests the halt should be, if not the profile's."""
        results = [r for r in (self.trade_monte_carlo, self.path_monte_carlo) if r is not None]
        if not results:
            return None
        return max(r.recommended_drawdown_limit() for r in results)

    def evidence_fields(self) -> dict[str, object]:
        """Exactly the :class:`trading.strategies.lifecycle.Evidence` fields this
        report can fill. Everything it cannot answer is absent, not defaulted."""
        return {
            "data_tier": self.data_tier,
            "out_of_sample_passed": self.out_of_sample_passed,
            "walk_forward_passed": self.walk_forward_passed,
            "parameter_plateau": self.parameter_plateau,
            "monte_carlo_drawdown_ok": self.monte_carlo_drawdown_ok,
            "trial_count": self.trial_count if self.trial_count > 0 else None,
            "deflated_sharpe": self.deflated_sharpe,
        }

    # ── presentation ────────────────────────────────────────────────────────
    @property
    def blockers(self) -> tuple[str, ...]:
        """Everything standing between this strategy and ``VALIDATED``."""
        reasons: list[str] = []
        if not self.walk_forward_passed:
            reasons.append(self.walk_forward.verdict())
        if self.out_of_sample_passed is False:
            reasons.append(
                f"pooled out-of-sample record fails: {self.oos_net_return:+.2%} return, "
                f"Sharpe {self.oos_sharpe if self.oos_sharpe is not None else float('nan'):.2f} "
                f"(need > 0 and >= {self.thresholds.min_oos_sharpe:.2f})"
            )
        elif self.out_of_sample_passed is None:
            reasons.append("no pooled out-of-sample record — walk-forward scored no folds")
        if self.parameter_plateau is None:
            reasons.append("no parameter sweep was run, so the surface's shape is unknown")
        elif self.parameter_plateau is False and self.sensitivity is not None:
            reasons.append(self.sensitivity.verdict())
        if self.monte_carlo_drawdown_ok is None:
            reasons.append("no Monte Carlo resampling was run")
        elif self.monte_carlo_drawdown_ok is False:
            worst = self.worst_case_drawdown or 0.0
            reasons.append(
                f"resampled p{self.thresholds.drawdown_percentile:.0f} drawdown "
                f"{worst:.2%} exceeds the {float(self.drawdown_limit):.2%} risk limit"
            )
        if self.trial_count <= 0:
            reasons.append("no trials recorded in the experiment ledger")
        if self.deflated_sharpe_passed is False:
            reasons.append(
                f"deflated Sharpe {self.deflated_sharpe or 0.0:.3f} below "
                f"{self.thresholds.min_deflated_sharpe:.2f} — the result is not "
                f"distinguishable from the best of {self.trial_count} trials"
            )
        if not self.data_tier.may_support_a_validated_strategy:
            reasons.append(
                f"data tier is {self.data_tier}; a validated strategy needs PRODUCTION "
                "data with point-in-time guarantees"
            )
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        return not self.blockers

    def summary_lines(self) -> list[tuple[str, str, str]]:
        """``(section, label, value)`` rows for the CLI table."""
        rows: list[tuple[str, str, str]] = [
            ("Scope", "strategy", f"{self.strategy_id} v{self.strategy_version}"),
            ("Scope", "instrument", self.instrument),
            ("Scope", "data tier", str(self.data_tier)),
        ]
        rows += [
            ("Walk-forward", label, value) for label, value in self.walk_forward.summary_lines()
        ]
        dsr, sharpe = self.deflated_sharpe, self.oos_sharpe
        rows += [
            ("Pooled OOS", "sharpe", f"{sharpe:.2f}" if sharpe is not None else "n/a"),
            ("Pooled OOS", "trials recorded", str(self.trial_count)),
            ("Pooled OOS", "deflated sharpe", f"{dsr:.3f}" if dsr is not None else "n/a"),
            (
                "Pooled OOS",
                "verdict",
                "PASSED" if self.out_of_sample_passed else "FAILED",
            ),
        ]
        if self.purged_cv is not None:
            rows += [("Purged CV", label, value) for label, value in self.purged_cv.summary_lines()]
        if self.sensitivity is not None:
            rows += [
                ("Sensitivity", label, value) for label, value in self.sensitivity.summary_lines()
            ]
        for name, result in (
            ("Monte Carlo (trades)", self.trade_monte_carlo),
            ("Monte Carlo (path)", self.path_monte_carlo),
        ):
            if result is not None:
                rows += [(name, label, value) for label, value in result.summary_lines()]
        recommended = self.recommended_drawdown_limit
        if recommended is not None:
            rows.append(
                (
                    "Risk",
                    "profile drawdown limit",
                    f"{float(self.drawdown_limit):.2%}",
                )
            )
            rows.append(("Risk", "distribution suggests", f"{recommended:.2%}"))
        rows.append(("Verdict", "validated", "YES" if self.passed else "NO"))
        return rows

    def verdict(self) -> str:
        if self.passed:
            return (
                f"{self.strategy_id} passes Phase 6 validation: "
                f"{self.oos_net_return:+.2%} pooled out-of-sample over "
                f"{len(self.walk_forward.oos_returns)} bars, deflated Sharpe "
                f"{self.deflated_sharpe or 0.0:.3f} against {self.trial_count} recorded trials"
            )
        return f"{self.strategy_id} is NOT validated:\n  - " + "\n  - ".join(self.blockers)

    def as_frame(self) -> pd.DataFrame:
        """The report as a frame, for writing alongside a run's artefacts."""
        return pd.DataFrame(self.summary_lines(), columns=["section", "metric", "value"])

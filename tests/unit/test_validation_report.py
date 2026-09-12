"""The validation report and its wiring into the lifecycle gate.

The property that matters most: there must be no path from "this check was not
run" to "this check passed". Every ``None`` here has to survive all the way to an
``UNAVAILABLE`` gate result.
"""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trading.data.provider import DataTier
from trading.strategies.base import LifecycleStatus as L
from trading.strategies.lifecycle import (
    CriterionStatus,
    evaluate_promotion,
    evidence_from_validation,
)
from trading.validation.harness import Objective, SegmentOutcome
from trading.validation.montecarlo import resample_trades
from trading.validation.report import ValidationReport, ValidationThresholds
from trading.validation.sensitivity import ParameterGrid, run_sensitivity
from trading.validation.splits import window_from
from trading.validation.walkforward import WalkForwardConfig, run_walk_forward

INDEX = pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=1600, tz="UTC"))
GRID = [{"fast": f} for f in (10, 20, 30)]
CONFIG = WalkForwardConfig(
    train_bars=252, test_bars=126, warmup_bars=0, objective=Objective("sharpe", min_trades=2)
)


def evaluator(daily: float, *, trades: int = 20, volatility: float = 0.008):
    """A stand-in backtest whose returns have real variance.

    Variance matters here rather than being incidental detail: a constant return
    series has zero standard deviation, so its Sharpe is undefined and its
    drawdown is exactly zero. Tests built on that would pass against a report that
    computed nothing at all.

    Seeded from the window's position so the same window always yields the same
    numbers — a validation test that changes answer between runs is useless.
    """

    def evaluate(params, window, measure_from):
        rng = np.random.default_rng(measure_from.start_index * 1_000 + measure_from.bars)
        values = rng.normal(daily, volatility, measure_from.bars)
        series = pd.Series(values, index=INDEX[measure_from.start_index : measure_from.stop_index])
        # Trade P&L that mixes wins and losses, so resampling produces a real
        # drawdown distribution rather than a monotonic line.
        pnl = tuple(float(x) for x in rng.normal(daily * 400_000 * 10, 4_000, trades))
        return SegmentOutcome(
            params=dict(params),
            window=measure_from,
            returns=series,
            trades=trades,
            trade_pnl=pnl,
            metrics={
                "sharpe": float(series.mean() / series.std(ddof=1) * np.sqrt(252)),
                "net_return": float((1 + series).prod() - 1),
            },
        )

    return evaluate


def build(
    *,
    daily=0.0004,
    tier=DataTier.PRODUCTION,
    with_sweep=True,
    with_mc=True,
    trials=25,
    limit="0.15",
) -> ValidationReport:
    wf = run_walk_forward(evaluator(daily), INDEX, GRID, CONFIG)
    sweep = None
    if with_sweep:
        sweep = run_sensitivity(
            evaluator(daily),
            ParameterGrid({"fast": [10, 20, 30]}),
            window_from(INDEX, 0, 252),
        )
    mc = None
    if with_mc and len(wf.oos_trade_pnl) >= 2:
        mc = resample_trades(wf.oos_trade_pnl, starting_capital=400_000, paths=200, seed=0)
    return ValidationReport(
        strategy_id="demo",
        strategy_version="1.0.0",
        instrument="NSE:RELIANCE",
        data_tier=tier,
        walk_forward=wf,
        trial_count=trials,
        drawdown_limit=Decimal(limit),
        sensitivity=sweep,
        trade_monte_carlo=mc,
        starting_capital=Decimal(400_000),
    )


def _gate(report: ValidationReport):
    """Put a report through the gate it is built to answer."""
    return evaluate_promotion("demo", L.PROMISING, L.VALIDATED, evidence_from_validation(report))


# ── the pooled record ───────────────────────────────────────────────────────
def test_pooled_out_of_sample_is_scored_by_the_shared_metric_code():
    report = build()
    assert report.oos_metrics.get("sharpe") is not None
    assert report.oos_metrics.get("trials") == 25, "the ledger count reaches the metric"
    assert report.oos_net_return > 0


def test_the_deflated_sharpe_uses_the_recorded_trial_count():
    """More trials must deflate the same return harder. That is the whole mechanism."""
    few = build(trials=1).deflated_sharpe
    many = build(trials=5_000).deflated_sharpe
    assert few is not None and many is not None
    assert many < few


def test_a_short_out_of_sample_record_reports_an_error_not_a_number():
    """Two out-of-sample bars is not a record. Say so; do not compute a ratio."""
    wf = run_walk_forward(
        evaluator(0.0004),
        INDEX[:300],
        GRID,
        WalkForwardConfig(
            train_bars=298, test_bars=1, warmup_bars=0, objective=Objective(min_trades=2)
        ),
    )
    report = ValidationReport(
        strategy_id="demo",
        strategy_version="1.0.0",
        instrument="NSE:RELIANCE",
        data_tier=DataTier.PRODUCTION,
        walk_forward=wf,
        trial_count=5,
        drawdown_limit=Decimal("0.15"),
    )
    assert report.oos_sharpe is None
    assert report.out_of_sample_passed is None


# ── the criteria ────────────────────────────────────────────────────────────
def test_a_positive_return_with_no_sharpe_does_not_pass_out_of_sample():
    """One good stretch surrounded by noise is not an edge."""
    report = build(daily=0.0004)
    assert report.oos_net_return > 0
    strict = replace(report, thresholds=ValidationThresholds(min_oos_sharpe=99.0))
    assert strict.out_of_sample_passed is False, "a positive return alone is not enough"


def test_no_sweep_means_the_plateau_question_is_unanswered_not_answered_yes():
    assert build(with_sweep=False).parameter_plateau is None
    assert build(with_sweep=True).parameter_plateau is not None


def test_no_monte_carlo_means_the_drawdown_question_is_unanswered():
    assert build(with_mc=False).monte_carlo_drawdown_ok is None


def test_a_drawdown_distribution_wider_than_the_limit_fails():
    generous = build(limit="0.99")
    tight = build(limit="0.0001")
    assert generous.monte_carlo_drawdown_ok is True
    assert tight.monte_carlo_drawdown_ok is False
    assert any("exceeds" in b for b in tight.blockers)


def test_prototype_data_blocks_however_good_the_numbers():
    report = build(tier=DataTier.PROTOTYPE, limit="0.99")
    assert not report.passed
    assert any("PRODUCTION" in b for b in report.blockers)


def test_a_recommendation_is_offered_when_the_limit_looks_too_tight():
    report = build(limit="0.0001")
    assert report.recommended_drawdown_limit is not None
    assert report.recommended_drawdown_limit > float(report.drawdown_limit)


# ── the gate ────────────────────────────────────────────────────────────────
def test_a_complete_report_can_reach_validated():
    report = build(limit="0.99", trials=3)
    assert report.passed, report.verdict()
    decision = _gate(report)
    assert decision.approved, [str(r) for r in decision.blockers]


def test_an_incomplete_report_produces_unavailable_gates_not_passes():
    report = build(with_sweep=False, with_mc=False, limit="0.99")
    decision = _gate(report)
    assert not decision.approved
    statuses = {r.criterion_id: r.status for r in decision.results}
    assert statuses["GATE_012_parameter_plateau"] is CriterionStatus.UNAVAILABLE
    assert statuses["GATE_013_monte_carlo_drawdown"] is CriterionStatus.UNAVAILABLE


def test_a_losing_strategy_is_blocked_with_specific_reasons():
    report = build(daily=-0.0004, limit="0.99")
    assert not report.passed
    assert not report.verdict().startswith("demo passes")
    assert len(report.blockers) >= 2
    for blocker in report.blockers:
        assert blocker, "every blocker must say something"


def test_summary_lines_are_renderable_triples():
    rows = build().summary_lines()
    assert all(len(row) == 3 and all(isinstance(c, str) for c in row) for row in rows)
    assert ("Verdict", "validated", "NO") in rows or ("Verdict", "validated", "YES") in rows
    frame = build().as_frame()
    assert list(frame.columns) == ["section", "metric", "value"]


def test_zero_trials_blocks_and_reports_it():
    report = build(trials=0, limit="0.99")
    assert any("ledger" in b for b in report.blockers)
    decision = _gate(report)
    statuses = {r.criterion_id: r.status for r in decision.results}
    assert statuses["GATE_014_trial_count_recorded"] is CriterionStatus.UNAVAILABLE


@pytest.mark.parametrize(
    "field", ["out_of_sample_passed", "walk_forward_passed", "parameter_plateau"]
)
def test_every_criterion_is_derived_not_settable(field):
    """A report must not accept a bare ``True`` for a check that did not run."""
    assert isinstance(getattr(ValidationReport, field), property)

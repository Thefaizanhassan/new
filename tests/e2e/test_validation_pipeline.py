"""Validation against the real backtest engine.

The unit tests use a stand-in evaluator so they can isolate the validation logic.
This file uses the actual engine — real bars, real Indian costs, the real risk
engine, the real fill model — because the failures that matter most in this layer
are integration failures: an off-by-one in the measurement window, a timezone
mismatch, warmup silently eating a test window.
"""

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from trading.core.instrument import Instrument, InstrumentId
from trading.core.types import Currency, InstrumentClass, Money
from trading.costs.india import IndiaDeliveryEquityCosts
from trading.data.fixture import FixtureProvider
from trading.data.provider import DataTier
from trading.strategies.config_strategy import load_strategy
from trading.strategies.lifecycle import (
    CriterionStatus,
    LifecycleStatus,
    evaluate_promotion,
    evidence_from_validation,
)
from trading.validation.engine_adapter import EngineEvaluator
from trading.validation.experiments import ExperimentLedger
from trading.validation.harness import Objective
from trading.validation.montecarlo import resample_trades
from trading.validation.report import ValidationReport
from trading.validation.sensitivity import ParameterGrid, run_sensitivity
from trading.validation.splits import window_from
from trading.validation.walkforward import WalkForwardConfig, run_walk_forward

CONFIG = "configs/strategies/sma_cross_param.yaml"
START, END = dt.date(2015, 1, 1), dt.date(2026, 8, 21)


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="RELIANCE",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


@pytest.fixture
def evaluator(instrument, tmp_path) -> EngineEvaluator:
    return EngineEvaluator(
        provider=FixtureProvider(),
        instrument=instrument,
        build_strategy=lambda p: load_strategy(CONFIG, overrides=p),
        cost_model=IndiaDeliveryEquityCosts(),
        starting_capital=Money(Decimal("400000"), Currency.INR),
        objective=Objective("sharpe", min_trades=2),
        ledger=ExperimentLedger(tmp_path / "experiments.sqlite"),
        ledger_kind="e2e",
    )


GRID = [
    {"fast": 20, "slow": 100, "warmup_bars": 101},
    {"fast": 50, "slow": 200, "warmup_bars": 201},
]


# ── the measurement window ──────────────────────────────────────────────────
def test_bars_are_fetched_once_and_reused(evaluator):
    """99 backtests must not mean 99 fetches — and must all see the same data."""
    index = evaluator.load(START, END)
    assert len(index) > 2_000
    before = evaluator.bars
    evaluator(GRID[0], window_from(index, 0, 400), window_from(index, 201, 400))
    assert evaluator.bars is before, "the held frame must not be refetched or mutated"


def test_only_the_measured_window_is_scored_not_the_warmup_prefix(evaluator):
    """The regression test for the split's whole purpose.

    The run window is 400 bars; the measured window is the last 199. If warmup
    bars leaked into the measurement, the return series would be 399 long and
    every fold's result would be diluted by a period the strategy spent warming
    up rather than trading.
    """
    index = evaluator.load(START, END)
    run = window_from(index, 0, 400)
    measured = window_from(index, 201, 400)
    outcome = evaluator(GRID[0], run, measured)

    assert len(outcome.returns) == measured.bars
    assert outcome.returns.index[0] == measured.start
    assert outcome.returns.index[-1] == measured.stop
    assert "warmed up over 201 earlier bars" in outcome.note


def test_the_first_measured_bar_contributes_its_own_return(evaluator):
    """Slicing at the window's first bar and taking pct_change drops that bar.

    A small error on a 126-bar window and a large one on a 20-bar window — and
    invisible either way, because the reported metrics all still look plausible.
    """
    index = evaluator.load(START, END)
    measured = window_from(index, 201, 260)
    outcome = evaluator(GRID[0], window_from(index, 0, 260), measured)
    assert len(outcome.returns) == measured.bars, (
        "expected one return per measured bar, including the first"
    )


def test_a_tz_aware_bar_index_is_handled_not_worked_around(evaluator):
    index = evaluator.load(START, END)
    assert index.tz is not None, "the fixture provider is UTC-aware; the rest assumes it"
    outcome = evaluator(GRID[0], window_from(index, 0, 400), window_from(index, 201, 400))
    assert outcome.returns.index.tz is not None


def test_only_fully_contained_round_trips_are_counted(evaluator):
    """A trade opened during warmup has P&L that partly accrued in-sample.

    Feeding it into the out-of-sample drawdown distribution would mix in-sample
    gains into an out-of-sample risk estimate.
    """
    index = evaluator.load(START, END)
    outcome = evaluator(GRID[1], window_from(index, 0, 1200), window_from(index, 900, 1200))
    assert "fully contained" in outcome.note
    assert outcome.trades == len(outcome.trade_pnl)


def test_halt_state_does_not_leak_between_evaluations(evaluator):
    """Each point gets a fresh risk engine, or the sweep's answer depends on order."""
    index = evaluator.load(START, END)
    window, measured = window_from(index, 0, 900), window_from(index, 201, 900)
    first = evaluator(GRID[0], window, measured)
    second = evaluator(GRID[0], window, measured)
    assert first.net_return == second.net_return
    assert first.trades == second.trades


# ── the full battery ────────────────────────────────────────────────────────
def test_walk_forward_runs_against_the_real_engine_and_records_its_trials(evaluator):
    index = evaluator.load(START, END)
    result = run_walk_forward(
        evaluator,
        index,
        GRID,
        WalkForwardConfig(
            train_bars=756,
            test_bars=252,
            warmup_bars=201,
            objective=Objective("sharpe", min_trades=2),
        ),
    )
    assert result.folds, "the history is long enough for several folds"
    assert result.trials == evaluator.evaluations
    assert evaluator.ledger is not None
    assert evaluator.ledger.trial_count("sma_cross_param") == len(GRID)
    for fold in result.scored_folds:
        assert fold.out_of_sample is not None
        assert fold.out_of_sample.trades >= 1
        assert not fold.split.contaminated


def test_the_reference_strategy_does_not_pass_validation_on_synthetic_data(evaluator):
    """The honest end-to-end assertion.

    SMA crossover on a synthetic random walk has no edge to find, and the data is
    SYNTHETIC tier besides. A pipeline that passed it would be broken, so this
    test asserts the failure and names the reasons.
    """
    index = evaluator.load(START, END)
    wf = run_walk_forward(
        evaluator,
        index,
        GRID,
        WalkForwardConfig(
            train_bars=756,
            test_bars=252,
            warmup_bars=201,
            objective=Objective("sharpe", min_trades=2),
        ),
    )
    sweep = run_sensitivity(
        evaluator,
        ParameterGrid({"fast": [20, 50]}),
        window_from(index, 0, 900),
        measure_from=window_from(index, 201, 900),
        objective=Objective("sharpe", min_trades=2),
    )
    mc = (
        resample_trades(wf.oos_trade_pnl, starting_capital=400_000, paths=200, seed=0)
        if len(wf.oos_trade_pnl) >= 2
        else None
    )
    report = ValidationReport(
        strategy_id="sma_cross_param",
        strategy_version="1.0.0",
        instrument="NSE:RELIANCE",
        data_tier=evaluator.data_tier,
        walk_forward=wf,
        trial_count=evaluator.ledger.trial_count("sma_cross_param") if evaluator.ledger else 0,
        drawdown_limit=Decimal("0.15"),
        sensitivity=sweep,
        trade_monte_carlo=mc,
        starting_capital=Decimal("400000"),
    )

    assert not report.passed
    assert evaluator.data_tier is DataTier.SYNTHETIC
    assert any("SYNTHETIC" in b for b in report.blockers)

    decision = evaluate_promotion(
        "sma_cross_param",
        LifecycleStatus.PROMISING,
        LifecycleStatus.VALIDATED,
        evidence_from_validation(report),
    )
    assert not decision.approved
    # Every validation criterion is now *answered* — the point of Phase 6. Before
    # it, all of these came back UNAVAILABLE regardless of what was measured.
    answered = {
        r.criterion_id for r in decision.results if r.status is not CriterionStatus.UNAVAILABLE
    }
    assert {
        "GATE_010_out_of_sample",
        "GATE_011_walk_forward",
        "GATE_012_parameter_plateau",
        "GATE_013_monte_carlo_drawdown",
        "GATE_014_trial_count_recorded",
        "GATE_015_deflated_sharpe",
    } <= answered


def test_a_provider_returning_nothing_is_refused(instrument):
    """Silently validating over zero bars would produce a report full of neutral
    values, which reads as "nothing was wrong" rather than "nothing was checked".

    Uses a stub rather than the fixture provider, which generates bars for any
    range it is asked for and so cannot exercise this path.
    """

    class Empty:
        info = FixtureProvider().info

        def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
            return pd.DataFrame()

    bare = EngineEvaluator(
        provider=Empty(),
        instrument=instrument,
        build_strategy=lambda p: load_strategy(CONFIG, overrides=p),
        cost_model=IndiaDeliveryEquityCosts(),
        starting_capital=Money(Decimal("400000"), Currency.INR),
    )
    with pytest.raises(ValueError, match="no bars"):
        bare.load(START, END)


def test_evaluating_before_loading_is_an_error_not_an_empty_answer(instrument, tmp_path):
    bare = EngineEvaluator(
        provider=FixtureProvider(),
        instrument=instrument,
        build_strategy=lambda p: load_strategy(CONFIG, overrides=p),
        cost_model=IndiaDeliveryEquityCosts(),
        starting_capital=Money(Decimal("400000"), Currency.INR),
    )
    with pytest.raises(RuntimeError, match="call load"):
        _ = bare.bars


def test_the_ledger_records_the_data_tier_so_the_gate_can_read_it(evaluator):
    index = evaluator.load(START, END)
    evaluator(GRID[0], window_from(index, 0, 400), window_from(index, 201, 400))
    assert evaluator.ledger is not None
    assert evaluator.ledger.tiers_used("sma_cross_param") == {"SYNTHETIC"}


def test_a_run_result_is_reproducible_bar_for_bar(evaluator):
    """Two identical evaluations must produce identical return series.

    Reproducibility sits second in the decision hierarchy, and a validation layer
    whose answer drifts between runs cannot support a promotion decision.
    """
    index = evaluator.load(START, END)
    window, measured = window_from(index, 0, 1200), window_from(index, 201, 1200)
    first = evaluator(GRID[1], window, measured)
    second = evaluator(GRID[1], window, measured)
    pd.testing.assert_series_equal(first.returns, second.returns)
    assert first.trade_pnl == second.trade_pnl

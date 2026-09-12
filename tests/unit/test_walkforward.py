"""Walk-forward analysis.

The important tests here are the ones that prove a *bad* strategy fails. A
walk-forward implementation that reports a pass for a strategy fitted to noise is
worse than no walk-forward at all, because it comes with a certificate.
"""

import numpy as np
import pandas as pd
import pytest

from trading.validation.harness import Objective, SegmentOutcome
from trading.validation.splits import Window, window_from
from trading.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardThresholds,
    aggregate_outcomes,
    run_walk_forward,
)

INDEX = pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=1200, tz="UTC"))


def make_evaluator(daily_return, *, trades=10, sharpe=None):
    """An evaluator whose behaviour is a pure function of the parameters.

    ``daily_return(params, window) -> float`` keeps the tests about the
    walk-forward logic rather than about any strategy.
    """

    def evaluate(params, window: Window, measure_from: Window) -> SegmentOutcome:
        r = daily_return(params, measure_from)
        series = pd.Series(
            np.full(measure_from.bars, r),
            index=INDEX[measure_from.start_index : measure_from.stop_index],
        )
        computed = r / 0.01 * np.sqrt(252) if sharpe is None else sharpe(params, measure_from)
        return SegmentOutcome(
            params=dict(params),
            window=measure_from,
            returns=series,
            trades=trades if not callable(trades) else trades(params, measure_from),
            trade_pnl=tuple([r * 1000] * 10),
            metrics={"sharpe": computed, "net_return": float((1 + series).prod() - 1)},
        )

    return evaluate


GRID = [{"fast": f} for f in (10, 20, 30)]
CONFIG = WalkForwardConfig(
    train_bars=252, test_bars=126, warmup_bars=0, objective=Objective("sharpe", min_trades=2)
)
assert (CONFIG.train_bars, CONFIG.test_bars) == (252, 126), "the length discriminator assumes these"


# ── the honest cases ────────────────────────────────────────────────────────
def test_a_genuine_edge_passes():
    """Same positive return everywhere: efficiency 1, full consistency, stable params."""
    result = run_walk_forward(make_evaluator(lambda p, w: 0.0004), INDEX, GRID, CONFIG)
    assert result.passed, result.verdict()
    assert result.efficiency == pytest.approx(1.0)
    assert result.consistency == 1.0
    assert result.parameter_stability == 1.0
    assert result.oos_net_return > 0


TRAIN_BARS, TEST_BARS = 252, 126


def _is_test(window) -> bool:
    """Is the evaluator being asked for a forward test or for a tuning pass?

    Keyed on window *length*, not on start position: with ``step == test_bars``
    and ``train_bars`` a multiple of the step, a single position is both the start
    of one fold's test window and the start of a later fold's training window, so
    positions cannot tell the two apart.
    """
    return window.bars == TEST_BARS


def test_noise_fitted_in_sample_fails_out_of_sample():
    """Profitable only on training windows, losing on test windows.

    This is the exact shape of an overfitted strategy, and the efficiency ratio
    must come back negative rather than merely small.
    """

    def daily(params, window):
        return -0.0005 if _is_test(window) else 0.0008

    result = run_walk_forward(make_evaluator(daily), INDEX, GRID, CONFIG)
    assert not result.passed
    assert result.efficiency < 0
    assert "efficiency" in result.verdict()


# ── the guards ──────────────────────────────────────────────────────────────
def test_a_fold_with_no_complete_round_trip_is_skipped_not_scored():
    """The regression test for the most dangerous bug in this module.

    A test window run with a warmup prefix can carry a position in from the
    tuning period. Its mark-to-market movement is real, but it is the tail of an
    in-sample decision. Counting it would credit the fold for something the
    strategy did while it was being fitted — free out-of-sample return.
    """

    def trades(params, window):
        # Plenty of trades while tuning, no completed round trip while testing.
        return 0 if _is_test(window) else 10

    result = run_walk_forward(
        make_evaluator(lambda p, w: 0.001, trades=trades), INDEX, GRID, CONFIG
    )
    assert result.scored_folds == ()
    assert not result.passed
    assert all("no complete round trip" in f.skip_reason for f in result.skipped_folds)
    assert result.oos_returns.empty, "a skipped fold must contribute no return"


def test_a_fold_where_nothing_reached_the_trade_floor_is_skipped():
    """No candidate qualifying is a real answer, not a reason to pick the first one."""
    result = run_walk_forward(make_evaluator(lambda p, w: 0.001, trades=1), INDEX, GRID, CONFIG)
    assert result.scored_folds == ()
    assert all("reached 2 trades" in f.skip_reason for f in result.skipped_folds)


def test_objective_rejects_a_lucky_low_trade_candidate():
    """A two-trade candidate with a spectacular ratio must not win the fold."""

    def daily(params, window):
        return 0.01 if params["fast"] == 10 else 0.0002

    def trades(params, window):
        return 2 if params["fast"] == 10 else 20

    result = run_walk_forward(
        make_evaluator(daily, trades=trades),
        INDEX,
        GRID,
        WalkForwardConfig(
            train_bars=252,
            test_bars=126,
            warmup_bars=0,
            objective=Objective("sharpe", min_trades=5),
        ),
    )
    assert all(f.chosen and f.chosen["fast"] != 10 for f in result.scored_folds)


def test_pooled_out_of_sample_series_never_double_counts_a_bar():
    """Overlapping test windows must not compound the same day twice."""
    config = WalkForwardConfig(
        train_bars=252,
        test_bars=126,
        step_bars=63,
        warmup_bars=0,
        objective=Objective("sharpe", min_trades=2),
    )
    result = run_walk_forward(make_evaluator(lambda p, w: 0.0004), INDEX, GRID, config)
    returns = result.oos_returns
    assert returns.index.is_unique
    assert returns.index.is_monotonic_increasing


def test_equity_curve_is_shaped_for_the_shared_metric_code():
    result = run_walk_forward(make_evaluator(lambda p, w: 0.0004), INDEX, GRID, CONFIG)
    curve = result.oos_equity_curve(400_000)
    assert list(curve.columns) == ["equity"]
    assert curve.index.name == "timestamp"
    assert curve["equity"].iloc[-1] > 400_000


def test_parameter_instability_is_detected():
    """Every fold choosing a different optimum means there is no optimum."""
    picks = iter([10, 20, 30, 10, 20, 30, 10, 20])
    chosen = {}

    def sharpe(params, window):
        key = window.stop_index
        if key not in chosen:
            chosen[key] = next(picks, 10)
        return 2.0 if params["fast"] == chosen[key] else 0.1

    result = run_walk_forward(
        make_evaluator(lambda p, w: 0.0004, sharpe=sharpe), INDEX, GRID, CONFIG
    )
    assert result.parameter_stability < 0.5
    assert any("parameter stability" in f for f in result.failures)


def test_pooled_trade_floor_blocks_a_thin_record():
    """Six profitable trades across eight folds is not evidence, however positive."""
    result = run_walk_forward(
        make_evaluator(lambda p, w: 0.0004, trades=2),
        INDEX,
        GRID,
        CONFIG,
        WalkForwardThresholds(min_oos_trades=30),
    )
    assert result.oos_trades < 30
    assert any("out-of-sample trades" in f for f in result.failures)


def test_efficiency_is_zero_when_in_sample_lost_money():
    """A negative denominator would flip the sign and dress a failure as a success."""
    result = run_walk_forward(make_evaluator(lambda p, w: -0.0005), INDEX, GRID, CONFIG)
    assert result.efficiency == 0.0
    assert result.losing_in_sample_folds == len(result.scored_folds)


def test_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="at least one parameter set"):
        run_walk_forward(make_evaluator(lambda p, w: 0.0), INDEX, [], CONFIG)


# ── aggregation across non-contiguous chunks ────────────────────────────────
def test_aggregate_weights_chunks_by_bar_count():
    """A 30-bar chunk must not count as much as a 600-bar one."""
    small = SegmentOutcome(
        params={},
        window=window_from(INDEX, 0, 30),
        returns=pd.Series([0.0] * 30, index=INDEX[:30]),
        trades=5,
        trade_pnl=(),
        metrics={"sharpe": 10.0},
    )
    large = SegmentOutcome(
        params={},
        window=window_from(INDEX, 30, 630),
        returns=pd.Series([0.0] * 600, index=INDEX[30:630]),
        trades=5,
        trade_pnl=(),
        metrics={"sharpe": 0.0},
    )
    combined = aggregate_outcomes([small, large])
    assert combined["sharpe"] == pytest.approx(10.0 * 30 / 630)
    assert combined["bars"] == 630


def test_aggregate_of_nothing_is_empty_not_zero():
    assert aggregate_outcomes([]) == {}

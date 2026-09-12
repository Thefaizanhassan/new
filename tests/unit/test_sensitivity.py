"""Parameter sensitivity: plateau versus spike.

Two properties are load-bearing. A genuinely broad surface must read as a
plateau, and a surface where one setting works and its immediate neighbours lose
money must read as a spike. Getting the second one wrong turns the cheapest
overfitting detector in the system into a rubber stamp.
"""

import math

import numpy as np
import pandas as pd
import pytest

from trading.validation.harness import Objective, SegmentOutcome
from trading.validation.sensitivity import ParameterGrid, run_sensitivity
from trading.validation.splits import window_from

INDEX = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=300, tz="UTC"))
WINDOW = window_from(INDEX, 0, 300)


def evaluator(score_for, *, trades=30):
    def evaluate(params, window, measure_from):
        score = score_for(params)
        daily = score / 252 / 16
        series = pd.Series(np.full(299, daily), index=INDEX[1:])
        count = trades(params) if callable(trades) else trades
        return SegmentOutcome(
            params=dict(params),
            window=measure_from,
            returns=series,
            trades=count,
            trade_pnl=(score,) * max(count, 1),
            metrics={"sharpe": score, "net_return": float((1 + series).prod() - 1)},
        )

    return evaluate


# ── shape detection ─────────────────────────────────────────────────────────
def test_a_broad_hill_reads_as_a_plateau():
    grid = ParameterGrid({"fast": [20, 30, 40, 50, 60, 70, 80]})
    surface = run_sensitivity(evaluator(lambda p: 1.2 - abs(p["fast"] - 50) * 0.004), grid, WINDOW)
    assert surface.is_plateau
    assert not surface.is_spike
    assert surface.plateau_ratio > 0.9
    assert surface.positive_fraction == 1.0


def test_one_working_setting_reads_as_a_spike():
    grid = ParameterGrid({"fast": [20, 30, 40, 50, 60, 70, 80]})
    surface = run_sensitivity(evaluator(lambda p: 2.0 if p["fast"] == 50 else -0.3), grid, WINDOW)
    assert surface.is_spike
    assert surface.plateau_ratio < 0
    assert "spike" in surface.verdict()


def test_the_peak_is_refused_when_the_surface_is_spiky():
    """``robust_best`` must not hand back the spike. It is the whole point."""
    grid = ParameterGrid({"fast": [20, 30, 40, 50, 60, 70, 80]})
    surface = run_sensitivity(evaluator(lambda p: 2.0 if p["fast"] == 50 else 0.1), grid, WINDOW)
    assert surface.best is not None and surface.best.params["fast"] == 50
    assert surface.robust_best is not None and surface.robust_best.params["fast"] != 50


def test_the_middle_of_a_plateau_is_preferred_to_its_edge():
    """A flat shelf with a slightly higher edge: deploy the middle, not the edge."""
    scores = {20: 0.5, 30: 1.00, 40: 1.00, 50: 1.02, 60: 1.00, 70: 1.00, 80: 0.5}
    grid = ParameterGrid({"fast": sorted(scores)})
    surface = run_sensitivity(evaluator(lambda p: scores[p["fast"]]), grid, WINDOW)
    assert surface.is_plateau
    assert surface.robust_best is not None
    assert surface.robust_best.params["fast"] in (40, 50, 60)


def test_a_broadly_losing_grid_fails_even_where_the_peak_is_smooth():
    """Smoothness is not enough: most of the grid has to actually make money.

    The shape used here — flat and losing across most of the range, rising to a
    smooth shelf at one edge — is a real and instructive one. It passes the
    neighbourhood test and still should not be deployed, both because most of the
    range loses money and because a peak at the edge of a sweep means the true
    optimum may lie outside the grid entirely.
    """
    scores = {20: -0.5, 30: -0.5, 40: -0.5, 50: -0.5, 60: 0.90, 70: 0.95, 80: 1.0}
    grid = ParameterGrid({"fast": sorted(scores)})
    surface = run_sensitivity(evaluator(lambda p: scores[p["fast"]]), grid, WINDOW)
    assert surface.plateau_ratio > 0.9, "the peak's neighbourhood is smooth"
    assert surface.positive_fraction < 0.5
    assert not surface.is_plateau, "but most of the grid loses money"
    assert any("profitable" in f for f in surface.failures)


# ── neighbourhood mechanics ─────────────────────────────────────────────────
def test_neighbours_are_axis_aligned_not_diagonal():
    """A diagonal neighbour differs in two parameters, which blurs the question."""
    grid = ParameterGrid({"fast": [10, 20, 30], "slow": [100, 200, 300]})
    middle = grid.neighbours({"fast": 20, "slow": 200})
    assert len(middle) == 4
    assert {"fast": 10, "slow": 100} not in middle


def test_corner_points_have_fewer_neighbours():
    grid = ParameterGrid({"fast": [10, 20, 30], "slow": [100, 200, 300]})
    assert len(grid.neighbours({"fast": 10, "slow": 100})) == 2


def test_neighbourhood_score_excludes_the_point_itself():
    """A spike must not be able to prop up its own neighbourhood."""
    grid = ParameterGrid({"fast": [20, 30, 40]})
    surface = run_sensitivity(evaluator(lambda p: 9.0 if p["fast"] == 30 else 0.0), grid, WINDOW)
    assert surface.neighbourhood_score({"fast": 30}) == pytest.approx(0.0)


def test_unscoreable_points_count_against_the_profitable_fraction():
    """A grid where most settings never trade has not been swept, only skipped."""
    grid = ParameterGrid({"fast": [10, 20, 30, 40]})
    surface = run_sensitivity(
        evaluator(lambda p: 1.0, trades=lambda p: 30 if p["fast"] == 10 else 0),
        grid,
        WINDOW,
        objective=Objective("sharpe", min_trades=5),
    )
    assert len(surface.scoreable) == 1
    assert surface.positive_fraction == 1.0, "returns were positive at every point"
    assert not math.isfinite(surface.neighbourhood_score({"fast": 10}))


# ── grid validation ─────────────────────────────────────────────────────────
def test_an_unordered_numeric_axis_is_rejected():
    """Neighbourhood is adjacency in the listed order, so order must be real."""
    with pytest.raises(ValueError, match="ascending order"):
        ParameterGrid({"fast": [10, 200, 50]})


def test_duplicate_and_empty_axes_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ParameterGrid({"fast": [10, 10, 20]})
    with pytest.raises(ValueError, match="no values"):
        ParameterGrid({"fast": []})
    with pytest.raises(ValueError, match="at least one axis"):
        ParameterGrid({})


def test_grid_size_and_points_agree():
    grid = ParameterGrid({"fast": [10, 20, 30], "slow": [100, 200]})
    assert grid.size == 6
    assert len(grid.points()) == 6
    assert len({tuple(sorted(p.items())) for p in grid.points()}) == 6


def test_axis_profile_marginalises_the_other_axes():
    grid = ParameterGrid({"fast": [10, 20], "slow": [100, 200]})
    surface = run_sensitivity(evaluator(lambda p: p["fast"] / 10), grid, WINDOW)
    profile = dict(surface.axis_profile("fast"))
    assert profile[10] == pytest.approx(1.0)
    assert profile[20] == pytest.approx(2.0)
    assert dict(surface.axis_profile("slow"))[100] == pytest.approx(1.5)


def test_an_empty_surface_reports_no_shape_rather_than_a_plateau():
    grid = ParameterGrid({"fast": [10, 20]})
    surface = run_sensitivity(
        evaluator(lambda p: 1.0, trades=0), grid, WINDOW, objective=Objective(min_trades=5)
    )
    assert surface.best is None
    assert not surface.is_plateau
    assert not surface.is_spike, "no data is not a spike either"
    assert surface.failures == ("no grid point traded enough to be scored",)

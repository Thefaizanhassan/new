"""Purged cross-validation.

Two things need testing: that the purge and embargo actually remove the bars they
claim to, and that a result from here is never presented as evidence the strategy
could have been traded. The second is a documentation property, so the test
asserts on the wording — it is the only guard against a future edit quietly
upgrading the claim.
"""

import numpy as np
import pandas as pd
import pytest

from trading.validation.harness import Objective, SegmentOutcome
from trading.validation.purged_cv import PurgedCvConfig, run_purged_cv

INDEX = pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=1000, tz="UTC"))
GRID = [{"fast": f} for f in (10, 20, 30)]


def make_evaluator(score_for, *, trades=10, seen=None):
    def evaluate(params, window, measure_from):
        if seen is not None:
            seen.append((dict(params), window.start_index, window.stop_index))
        score = score_for(params, measure_from)
        series = pd.Series(
            np.full(measure_from.bars, score / 252 / 16),
            index=INDEX[measure_from.start_index : measure_from.stop_index],
        )
        return SegmentOutcome(
            params=dict(params),
            window=measure_from,
            returns=series,
            trades=trades,
            trade_pnl=(score * 100,) * 10,
            metrics={"sharpe": score, "net_return": float((1 + series).prod() - 1)},
        )

    return evaluate


CONFIG = PurgedCvConfig(
    n_splits=5,
    purge_bars=20,
    embargo_bars=10,
    warmup_bars=0,
    objective=Objective("sharpe", min_trades=2),
)


def test_every_fold_is_tested_including_the_earliest_history():
    """The reason to run this at all: walk-forward never tests its first training window."""
    result = run_purged_cv(make_evaluator(lambda p, w: 1.0), INDEX, GRID, CONFIG)
    assert len(result.scored_folds) == 5
    assert result.folds[0].split.test.start_index == 0


def test_training_never_touches_the_purged_or_embargoed_bars():
    seen: list[tuple[dict, int, int]] = []
    run_purged_cv(make_evaluator(lambda p, w: 1.0, seen=seen), INDEX, GRID, CONFIG)

    for split in run_purged_cv(make_evaluator(lambda p, w: 1.0), INDEX, GRID, CONFIG).folds:
        train = {i for w in split.split.train for i in w.positions}
        forbidden = set(range(split.split.test.start_index - 20, split.split.test.stop_index + 10))
        assert not train & forbidden


def test_middle_folds_record_their_mid_sample_restart():
    """Training on a split sample restarts position state. That approximation is
    recorded on the result so it cannot be forgotten."""
    result = run_purged_cv(make_evaluator(lambda p, w: 1.0), INDEX, GRID, CONFIG)
    assert result.folds[0].chunk_restarts == 1, "an edge fold trains on one chunk"
    assert result.folds[2].chunk_restarts == 2
    assert result.chunk_restarts == 3, "three middle folds, one extra chunk each"


def test_a_strategy_that_works_in_one_regime_only_is_caught():
    """High mean with high dispersion is the signature the mean alone hides."""

    def score(params, window):
        return 4.0 if window.start_index < 200 else -0.4

    result = run_purged_cv(make_evaluator(score), INDEX, GRID, CONFIG)
    assert not result.passed
    assert result.score_dispersion > 1.0
    assert result.positive_fraction < 0.6
    assert result.worst_fold is not None


def test_a_consistently_positive_strategy_passes():
    result = run_purged_cv(make_evaluator(lambda p, w: 1.2), INDEX, GRID, CONFIG)
    assert result.passed, result.verdict()
    assert result.positive_fraction == 1.0
    assert result.score_dispersion == pytest.approx(0.0)


def test_the_verdict_refuses_to_claim_tradeability():
    """A purged-CV number is about generalisation across regimes, never about what
    could have been traded. Only walk-forward is that, and the wording has to say
    so — a future edit that drops this sentence should fail a test.
    """
    result = run_purged_cv(make_evaluator(lambda p, w: 1.2), INDEX, GRID, CONFIG)
    assert "not evidence of tradeability" in result.verdict()


def test_folds_with_nothing_scoreable_are_skipped_with_a_reason():
    result = run_purged_cv(make_evaluator(lambda p, w: 1.0, trades=0), INDEX, GRID, CONFIG)
    assert result.scored_folds == ()
    assert not result.passed
    assert all("purged training sample" in f.skip_reason for f in result.folds)


def test_trades_are_pooled_across_non_contiguous_training_chunks():
    """The training sample is the union of its chunks, so its trade count is the total.

    One trade per chunk clears a floor of two on a two-chunk fold and not on a
    one-chunk edge fold — which is the correct reading, and worth pinning because
    it is surprising.
    """
    result = run_purged_cv(make_evaluator(lambda p, w: 1.0, trades=1), INDEX, GRID, CONFIG)
    assert [f.scored for f in result.folds] == [False, True, True, True, False]
    assert "purged training sample" in result.folds[0].skip_reason


def test_a_test_fold_with_no_completed_round_trip_is_skipped():
    """Same guard as walk-forward: a carried position is not an out-of-sample decision."""

    def trades(window):
        return 0 if window.bars <= 200 else 10

    def evaluate(params, window, measure_from):
        count = trades(measure_from)
        series = pd.Series(
            np.full(measure_from.bars, 0.0001),
            index=INDEX[measure_from.start_index : measure_from.stop_index],
        )
        return SegmentOutcome(
            params=dict(params),
            window=measure_from,
            returns=series,
            trades=count,
            trade_pnl=(1.0,) * count,
            metrics={"sharpe": 1.0, "net_return": 0.02},
        )

    result = run_purged_cv(evaluate, INDEX, GRID, CONFIG)
    assert result.scored_folds == ()
    assert all("no complete round trip" in f.skip_reason for f in result.folds)


def test_an_empty_grid_is_rejected():
    with pytest.raises(ValueError, match="at least one parameter set"):
        run_purged_cv(make_evaluator(lambda p, w: 1.0), INDEX, [], CONFIG)


def test_config_describes_itself_for_the_report():
    assert "purge 20 bars" in CONFIG.describe()
    assert "embargo 10 bars" in CONFIG.describe()

"""Monte Carlo resampling.

The property this file most needs to establish is that the block bootstrap
actually preserves volatility clustering. A day-by-day shuffle destroys it and
systematically *understates* drawdown, which would hand back a comfortable number
that is an artefact of the method rather than a fact about the strategy.
"""

import numpy as np
import pandas as pd
import pytest

from trading.validation.montecarlo import block_bootstrap_returns, resample_trades


def clustered_returns(n=1000, seed=0) -> pd.Series:
    """Returns with volatility clustering — calm stretches and stressed stretches.

    Built explicitly rather than drawn i.i.d., because on i.i.d. data there is no
    clustering for a block bootstrap to preserve and the test would pass for the
    wrong reason.
    """
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([0.004, 0.030], size=n // 50 + 1), 50)[:n]
    drift = np.where(regime > 0.01, -0.0015, 0.0010)
    return pd.Series(
        rng.normal(drift, regime), index=pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    )


# ── trade resampling ────────────────────────────────────────────────────────
def test_the_observed_path_is_one_draw_from_a_much_wider_distribution():
    """The whole point: the historical drawdown is a sample of one."""
    rng = np.random.default_rng(3)
    pnl = np.concatenate([rng.normal(1800, 6000, 80), [-45000]])
    result = resample_trades(pnl, starting_capital=400_000, paths=2000, seed=1)
    assert result.p95_drawdown > result.observed_max_drawdown
    assert result.p99_drawdown >= result.p95_drawdown >= result.median_drawdown
    assert result.worst_drawdown >= result.p99_drawdown


def test_resampling_is_reproducible_from_its_seed():
    pnl = [500.0, -200.0, 1200.0, -900.0, 300.0, -100.0]
    a = resample_trades(pnl, starting_capital=100_000, paths=200, seed=7)
    b = resample_trades(pnl, starting_capital=100_000, paths=200, seed=7)
    c = resample_trades(pnl, starting_capital=100_000, paths=200, seed=8)
    assert a.drawdowns == b.drawdowns
    assert a.drawdowns != c.drawdowns


def test_a_strategy_that_only_wins_has_no_drawdown():
    result = resample_trades([100.0, 200.0, 150.0], starting_capital=10_000, paths=50, seed=0)
    assert result.worst_drawdown == 0.0
    assert result.probability_of_loss == 0.0
    assert result.probability_of_ruin == 0.0


def test_ruin_is_a_drawdown_an_operator_would_stop_at_not_zero_equity():
    pnl = [-3000.0] * 5 + [3200.0] * 5
    result = resample_trades(pnl, starting_capital=100_000, paths=1000, seed=0, ruin_threshold=0.10)
    assert 0.0 < result.probability_of_ruin < 1.0
    # The threshold is named in the row's label, so a reader cannot mistake which
    # level the probability refers to.
    rows = dict(result.summary_lines())
    assert "probability of >= 10% drawdown" in rows
    assert rows["probability of >= 10% drawdown"] == f"{result.probability_of_ruin:.0%}"


def test_a_recommended_limit_sits_above_the_p95():
    """A halt set at the observed drawdown trips on ordinary luck and trains its
    operator to ignore it."""
    rng = np.random.default_rng(11)
    result = resample_trades(rng.normal(0, 5000, 60), starting_capital=200_000, paths=500, seed=0)
    assert result.recommended_drawdown_limit() > result.p95_drawdown


def test_one_trade_cannot_make_a_distribution():
    with pytest.raises(ValueError, match="at least 2 trades"):
        resample_trades([100.0], starting_capital=10_000)


def test_invalid_capital_and_paths_are_rejected():
    with pytest.raises(ValueError, match="starting_capital"):
        resample_trades([1.0, 2.0], starting_capital=0)
    with pytest.raises(ValueError, match="paths"):
        resample_trades([1.0, 2.0], starting_capital=100, paths=0)


# ── block bootstrap ─────────────────────────────────────────────────────────
def test_blocks_preserve_clustering_that_a_daily_shuffle_destroys():
    """The regression test for the method itself.

    Resampling one day at a time breaks up the runs of bad days that create
    drawdowns, so it reports a shallower distribution than the data supports.
    Blocks keep those runs intact. The 20-bar estimate must come out deeper.
    """
    returns = clustered_returns(1000, seed=5)
    daily = block_bootstrap_returns(returns, block_bars=1, paths=800, seed=0)
    blocked = block_bootstrap_returns(returns, block_bars=40, paths=800, seed=0)
    assert blocked.p95_drawdown > daily.p95_drawdown
    assert blocked.worst_drawdown > daily.worst_drawdown


def test_block_bootstrap_compounds_rather_than_adding():
    """A path of constant positive returns must compound, or the method is wrong."""
    flat = pd.Series([0.001] * 200, index=pd.bdate_range("2020-01-01", periods=200, tz="UTC"))
    result = block_bootstrap_returns(flat, block_bars=20, paths=10, seed=0)
    expected = 1.001**200 - 1
    assert result.observed_net_return == pytest.approx(expected, rel=1e-9)
    assert all(r == pytest.approx(expected, rel=1e-9) for r in result.net_returns)


def test_block_bootstrap_paths_are_the_same_length_as_the_original():
    returns = clustered_returns(500, seed=1)
    result = block_bootstrap_returns(returns, block_bars=30, paths=5, seed=0)
    assert f"{len(returns)} observed bars" in result.note


def test_block_bootstrap_is_reproducible_from_its_seed():
    returns = clustered_returns(400, seed=2)
    a = block_bootstrap_returns(returns, block_bars=20, paths=100, seed=4)
    b = block_bootstrap_returns(returns, block_bars=20, paths=100, seed=4)
    assert a.drawdowns == b.drawdowns


def test_too_few_bars_for_the_block_size_is_rejected():
    short = pd.Series([0.001] * 30, index=pd.bdate_range("2020-01-01", periods=30, tz="UTC"))
    with pytest.raises(ValueError, match="need at least 40 returns"):
        block_bootstrap_returns(short, block_bars=20)


def test_nan_returns_are_dropped_not_propagated():
    values = [0.01, float("nan"), -0.02] * 40
    returns = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=120, tz="UTC"))
    result = block_bootstrap_returns(returns, block_bars=10, paths=50, seed=0)
    assert all(not np.isnan(d) for d in result.drawdowns)


# ── reading a result against a limit ────────────────────────────────────────
def test_verdict_names_the_limit_it_was_read_against():
    rng = np.random.default_rng(9)
    result = resample_trades(rng.normal(0, 8000, 50), starting_capital=200_000, paths=500, seed=0)
    assert "EXCEEDS" in result.verdict(0.01)
    assert "inside" in result.verdict(0.99)


def test_observed_percentile_says_whether_history_was_kind():
    """Near the 5th percentile means the one historical path was unusually lucky,
    and the observed drawdown badly understates what to prepare for."""
    pnl = [1000.0] * 40 + [-20000.0]
    result = resample_trades(pnl, starting_capital=100_000, paths=1000, seed=0)
    assert 0.0 <= result.observed_percentile <= 1.0
    assert result.observed_percentile < 0.5

"""The leakage canary, including the leak it cannot catch."""

import numpy as np
import pandas as pd
import pytest

from trading.backtest.canary import CanaryResult, run_leakage_canary, shuffle_bars
from trading.data.validation import validate_ohlcv


@pytest.fixture
def bars() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 400
    close = 1000 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.bdate_range("2024-01-01", periods=n, tz="UTC"),
    )


# ── the shuffle ────────────────────────────────────────────────────────────
def test_shuffled_bars_still_pass_the_validation_gate(bars):
    """A canary producing invalid bars would be testing the validator."""
    report = validate_ohlcv(shuffle_bars(bars, seed=1), symbol="shuffled")
    assert report.is_usable, [str(i) for i in report.errors]


def test_shuffling_destroys_structure_but_keeps_the_distribution(bars):
    shuffled = shuffle_bars(bars, seed=1)
    original_returns = bars["close"].pct_change().dropna()
    shuffled_returns = shuffled["close"].pct_change().dropna()

    assert shuffled_returns.std() == pytest.approx(original_returns.std(), rel=0.25)
    # Autocorrelation is what a legitimate strategy exploits, and it should be gone.
    assert abs(shuffled_returns.autocorr(lag=1)) < 0.15


def test_shuffling_is_deterministic_for_a_seed(bars):
    assert shuffle_bars(bars, seed=7).equals(shuffle_bars(bars, seed=7))
    assert not shuffle_bars(bars, seed=7).equals(shuffle_bars(bars, seed=8))


def test_shuffling_needs_enough_bars(bars):
    with pytest.raises(ValueError, match="at least 3 bars"):
        shuffle_bars(bars.iloc[:2])


# ── detection ──────────────────────────────────────────────────────────────
def test_a_strategy_that_profits_on_structureless_data_is_flagged(bars):
    """The signature of a leak: it does not need market structure to make money."""
    result = run_leakage_canary(lambda frame: 0.5, bars, trials=6)
    assert result.leaking
    assert "LEAK SUSPECTED" in result.summary()


def test_a_strategy_that_loses_on_structureless_data_is_not_flagged(bars):
    calls = {"n": 0}

    def run(frame: pd.DataFrame) -> float:
        calls["n"] += 1
        return 0.20 if calls["n"] == 1 else -0.05  # profits only on the real data

    result = run_leakage_canary(run, bars, trials=6)
    assert not result.leaking
    assert result.percentile_of_real == 1.0


def test_an_extreme_real_result_is_surfaced_separately(bars):
    """Not a leak test — a prompt to look. It is the signal that catches the
    leak the profitability test misses."""
    calls = {"n": 0}
    noise = np.random.default_rng(2).normal(-0.05, 0.03, 20)

    def run(frame: pd.DataFrame) -> float:
        calls["n"] += 1
        return 25.0 if calls["n"] == 1 else float(noise[calls["n"]])

    result = run_leakage_canary(run, bars, trials=6)
    assert not result.leaking
    assert result.implausible_magnitude
    assert "WORTH A LOOK" in result.summary()


def test_the_magnitude_check_survives_a_degenerate_noise_distribution(bars):
    """Every shuffle landing identically must not silently disable the check."""
    calls = {"n": 0}

    def run(frame: pd.DataFrame) -> float:
        calls["n"] += 1
        return 25.0 if calls["n"] == 1 else -0.05

    assert run_leakage_canary(run, bars, trials=6).implausible_magnitude


def test_a_modest_edge_is_neither_flagged_nor_suspicious(bars):
    calls = {"n": 0}

    def run(frame: pd.DataFrame) -> float:
        calls["n"] += 1
        return 0.08 if calls["n"] == 1 else float(np.random.default_rng(calls["n"]).normal(0, 0.05))

    result = run_leakage_canary(run, bars, trials=10)
    assert not result.leaking
    assert "no leak detected" in result.summary()


def test_no_trials_means_no_verdict():
    result = CanaryResult(real_return=1.0, shuffled_returns=(), trials=0)
    assert not result.leaking
    assert not result.implausible_magnitude

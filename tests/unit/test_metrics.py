"""Performance metrics, checked against hand-computed values."""

import math
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trading.backtest.metrics import (
    _normal_cdf,
    _normal_ppf,
    compute_metrics,
    deflated_sharpe_ratio,
    extract_trades,
)
from trading.core.fill import CostBreakdown, Fill
from trading.core.instrument import Instrument, InstrumentId
from trading.core.types import Currency, InstrumentClass, Money, Side

NOW = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


def make_fill(instrument, side, quantity, price, day=0, cost="0") -> Fill:
    zero = Money.zero(Currency.INR)
    costs = CostBreakdown(
        currency=Currency.INR,
        brokerage=Money.inr(cost),
        exchange_fees=zero,
        transaction_tax=zero,
        stamp_duty=zero,
        regulatory_fees=zero,
        depository_fees=zero,
        gst=zero,
    )
    return Fill.create(
        client_order_id="c",
        instrument=instrument,
        side=side,
        quantity=Decimal(quantity),
        price=Decimal(price),
        costs=costs,
        timestamp=NOW + timedelta(days=day),
    )


# ── the normal distribution helpers ────────────────────────────────────────
@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.975, 1.959964), (0.95, 1.644854), (0.5, 0.0), (0.01, -2.326348)],
)
def test_inverse_normal_matches_known_values(p, expected):
    assert _normal_ppf(p) == pytest.approx(expected, abs=1e-5)


def test_normal_cdf_and_ppf_are_inverses():
    for p in (0.05, 0.3, 0.7, 0.99):
        assert _normal_cdf(_normal_ppf(p)) == pytest.approx(p, abs=1e-9)


def test_ppf_rejects_values_outside_the_open_interval():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        _normal_ppf(0.0)


# ── trade extraction ───────────────────────────────────────────────────────
def test_a_simple_round_trip_is_one_trade(instrument):
    fills = [
        make_fill(instrument, Side.BUY, "100", "1000", day=0),
        make_fill(instrument, Side.SELL, "100", "1100", day=10),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 1
    assert trades[0].gross_pnl == Decimal("10000")
    assert trades[0].direction == "LONG"
    assert trades[0].holding_days == pytest.approx(10.0)


def test_a_short_round_trip_profits_when_the_price_falls(instrument):
    fills = [
        make_fill(instrument, Side.SELL, "100", "1000", day=0),
        make_fill(instrument, Side.BUY, "100", "900", day=5),
    ]
    trades = extract_trades(fills)
    assert trades[0].direction == "SHORT"
    assert trades[0].gross_pnl == Decimal("10000")


def test_partial_closes_produce_separate_trades(instrument):
    fills = [
        make_fill(instrument, Side.BUY, "100", "1000", day=0),
        make_fill(instrument, Side.SELL, "40", "1100", day=5),
        make_fill(instrument, Side.SELL, "60", "1200", day=10),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 2
    assert sum(t.gross_pnl for t in trades) == Decimal("40") * 100 + Decimal("60") * 200


def test_lots_are_matched_first_in_first_out(instrument):
    fills = [
        make_fill(instrument, Side.BUY, "100", "1000", day=0),
        make_fill(instrument, Side.BUY, "100", "1200", day=1),
        make_fill(instrument, Side.SELL, "100", "1100", day=5),
    ]
    trades = extract_trades(fills)
    assert len(trades) == 1
    assert trades[0].entry_price == Decimal("1000")  # the first lot, not the second
    assert trades[0].gross_pnl == Decimal("10000")


def test_costs_are_charged_to_the_trade_that_incurred_them(instrument):
    fills = [
        make_fill(instrument, Side.BUY, "100", "1000", day=0, cost="100"),
        make_fill(instrument, Side.SELL, "100", "1100", day=5, cost="110"),
    ]
    trade = extract_trades(fills)[0]
    assert trade.costs == Decimal("210")
    assert trade.net_pnl == Decimal("9790")


def test_a_trade_profitable_only_before_costs_counts_as_a_loss(instrument):
    fills = [
        make_fill(instrument, Side.BUY, "100", "1000", day=0, cost="300"),
        make_fill(instrument, Side.SELL, "100", "1005", day=5, cost="300"),
    ]
    trade = extract_trades(fills)[0]
    assert trade.gross_pnl > 0
    assert not trade.is_win


def test_an_unclosed_position_produces_no_trade(instrument):
    assert extract_trades([make_fill(instrument, Side.BUY, "100", "1000")]) == []


# ── the deflated Sharpe ratio ──────────────────────────────────────────────
def test_more_trials_deflate_the_same_result():
    """The central anti-overfitting property: testing 500 things and reporting
    the best is not the same as finding one good thing."""
    rng = np.random.default_rng(11)
    returns = pd.Series(rng.normal(0.0006, 0.01, 1000))

    single, _ = deflated_sharpe_ratio(returns, trials=1)
    many, expected_max = deflated_sharpe_ratio(returns, trials=500)

    assert single > 0.95
    assert many < single
    assert expected_max > 0, "noise alone should reach a positive Sharpe over 500 trials"


def test_noise_alone_reaches_a_higher_sharpe_with_more_trials():
    returns = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, 800))
    _, ten = deflated_sharpe_ratio(returns, trials=10)
    _, thousand = deflated_sharpe_ratio(returns, trials=1000)
    assert thousand > ten


def test_a_flat_series_has_no_deflated_sharpe():
    assert deflated_sharpe_ratio(pd.Series([0.0] * 100)) == (0.0, 0.0)


def test_too_few_observations_returns_zero():
    assert deflated_sharpe_ratio(pd.Series([0.01, 0.02])) == (0.0, 0.0)


# ── the full metric set ────────────────────────────────────────────────────
@pytest.fixture
def equity_curve() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=252, tz="UTC")
    equity = 100_000 * np.cumprod(1 + np.random.default_rng(7).normal(0.0004, 0.008, 252))
    return pd.DataFrame(
        {
            "equity": equity,
            "position_value": equity * 0.5,
            "cumulative_costs": np.linspace(0, 500, 252),
        },
        index=index,
    )


def test_metrics_cover_returns_risk_and_trades(equity_curve, instrument):
    trades = extract_trades(
        [
            make_fill(instrument, Side.BUY, "100", "1000", day=0, cost="50"),
            make_fill(instrument, Side.SELL, "100", "1100", day=5, cost="55"),
        ]
    )
    report = compute_metrics(
        equity_curve, trades, starting_capital=Decimal("100000"), total_costs=Decimal("500")
    )
    for key in (
        "net_return",
        "gross_return",
        "cagr",
        "volatility",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "deflated_sharpe",
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy",
        "turnover",
    ):
        assert key in report.values, f"missing {key}"


def test_gross_return_always_exceeds_net(equity_curve):
    report = compute_metrics(
        equity_curve, [], starting_capital=Decimal("100000"), total_costs=Decimal("500")
    )
    assert report["gross_return"] > report["net_return"]
    assert report["cost_drag"] == pytest.approx(0.005)


def test_drawdown_is_measured_from_the_running_peak():
    index = pd.date_range("2024-01-01", periods=5, tz="UTC")
    curve = pd.DataFrame(
        {"equity": [100.0, 120.0, 90.0, 95.0, 130.0], "cumulative_costs": [0.0] * 5},
        index=index,
    )
    report = compute_metrics(curve, [], starting_capital=Decimal("100"), total_costs=Decimal("0"))
    assert report["max_drawdown"] == pytest.approx(0.25)  # 120 -> 90


def test_sharpe_is_zero_for_a_flat_curve():
    index = pd.date_range("2024-01-01", periods=50, tz="UTC")
    curve = pd.DataFrame({"equity": [100.0] * 50, "cumulative_costs": [0.0] * 50}, index=index)
    report = compute_metrics(curve, [], starting_capital=Decimal("100"), total_costs=Decimal("0"))
    assert report["sharpe"] == 0.0
    assert report["max_drawdown"] == 0.0


def test_profit_factor_is_infinite_when_nothing_lost(instrument):
    trades = extract_trades(
        [
            make_fill(instrument, Side.BUY, "10", "100", day=0),
            make_fill(instrument, Side.SELL, "10", "110", day=1),
        ]
    )
    index = pd.date_range("2024-01-01", periods=5, tz="UTC")
    curve = pd.DataFrame(
        {"equity": [100.0, 101.0, 102.0, 103.0, 104.0], "cumulative_costs": [0.0] * 5},
        index=index,
    )
    report = compute_metrics(
        curve, trades, starting_capital=Decimal("100"), total_costs=Decimal("0")
    )
    assert math.isinf(report["profit_factor"])


def test_benchmark_comparison_is_undefined_rather_than_nan_when_flat():
    """A strategy that never traded has no variance to correlate."""
    index = pd.date_range("2024-01-01", periods=50, tz="UTC")
    curve = pd.DataFrame({"equity": [100.0] * 50, "cumulative_costs": [0.0] * 50}, index=index)
    benchmark = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 50), index=index)
    report = compute_metrics(
        curve,
        [],
        starting_capital=Decimal("100"),
        total_costs=Decimal("0"),
        benchmark=benchmark,
    )
    assert "undefined" in report["benchmark_comparison"]
    assert "benchmark_return" in report.values


def test_too_short_a_curve_reports_an_error_rather_than_guessing():
    index = pd.date_range("2024-01-01", periods=1, tz="UTC")
    curve = pd.DataFrame({"equity": [100.0], "cumulative_costs": [0.0]}, index=index)
    report = compute_metrics(curve, [], starting_capital=Decimal("100"), total_costs=Decimal("0"))
    assert "error" in report.values

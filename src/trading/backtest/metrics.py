"""Performance metrics.

Phase 0 §11.4.  Two principles shape what is here and how it is reported.

**Gross and net are always both reported.**  The gap between them is the cost
drag, and it is the number that kills most strategies. A report showing only one
of them is hiding the interesting half.

**A single number is never enough.**  Sharpe alone is trivially gamed by
strategies that make small consistent gains and rare catastrophic losses. Every
ratio here is reported alongside the distribution shape that would expose that.

The Deflated Sharpe Ratio is the one metric worth reading before any other: it
answers "given how many strategies I tried, how surprised should I be by this
one?", which is the question that separates a discovery from a lucky draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from trading.core.fill import Fill

__all__ = [
    "PerformanceReport",
    "Trade",
    "compute_metrics",
    "deflated_sharpe_ratio",
    "extract_trades",
]

TRADING_DAYS = 252
_EULER_MASCHERONI = 0.5772156649015329


# ── round trips ─────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Trade:
    """One completed round trip, matched FIFO from the fill sequence."""

    instrument_id: str
    strategy_id: str
    direction: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    gross_pnl: Decimal
    costs: Decimal

    @property
    def net_pnl(self) -> Decimal:
        return self.gross_pnl - self.costs

    @property
    def is_win(self) -> bool:
        """Judged net. A trade that is profitable only before costs is a loss."""
        return self.net_pnl > 0

    @property
    def holding_days(self) -> float:
        return float((self.exit_time - self.entry_time).total_seconds()) / 86_400

    @property
    def return_pct(self) -> Decimal:
        notional = self.entry_price * self.quantity
        return self.net_pnl / notional if notional else Decimal(0)


def extract_trades(fills: list[Fill]) -> list[Trade]:
    """Match fills into completed round trips, FIFO.

    Costs are attributed to the trade that incurred them: the entry's cost is
    apportioned by the share of the position being closed, and the exit's cost
    is charged in full to the closing trade.
    """
    open_lots: dict[str, list[dict[str, Any]]] = {}
    trades: list[Trade] = []

    for fill in fills:
        key = str(fill.instrument.id)
        lots = open_lots.setdefault(key, [])
        signed = fill.signed_quantity
        cost_per_unit = fill.costs.total.amount / fill.quantity if fill.quantity else Decimal(0)

        while lots and (lots[0]["signed"] > 0) != (signed > 0) and signed != 0:
            lot = lots[0]
            closed = min(abs(signed), abs(lot["signed"]))
            direction = Decimal(1) if lot["signed"] > 0 else Decimal(-1)
            gross = (fill.price - lot["price"]) * closed * direction
            entry_cost = lot["cost_per_unit"] * closed
            exit_cost = cost_per_unit * closed

            trades.append(
                Trade(
                    instrument_id=key,
                    strategy_id=lot["strategy_id"],
                    direction="LONG" if lot["signed"] > 0 else "SHORT",
                    quantity=closed,
                    entry_price=lot["price"],
                    exit_price=fill.price,
                    entry_time=pd.Timestamp(lot["time"]),
                    exit_time=pd.Timestamp(fill.timestamp),
                    gross_pnl=gross,
                    costs=entry_cost + exit_cost,
                )
            )

            remaining = abs(lot["signed"]) - closed
            if remaining > 0:
                lot["signed"] = remaining * direction
            else:
                lots.pop(0)
            signed += closed * (Decimal(1) if signed < 0 else Decimal(-1))

        if signed != 0:
            lots.append(
                {
                    "signed": signed,
                    "price": fill.price,
                    "time": fill.timestamp,
                    "cost_per_unit": cost_per_unit,
                    "strategy_id": fill.strategy_id,
                }
            )

    return trades


# ── the deflated Sharpe ratio ───────────────────────────────────────────────
def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9 accurate)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def deflated_sharpe_ratio(
    returns: pd.Series,
    *,
    trials: int = 1,
    trial_sharpe_variance: float | None = None,
) -> tuple[float, float]:
    """Probability the observed Sharpe reflects skill rather than selection.

    Returns ``(dsr, expected_max_sharpe_from_noise)``.

    Test 500 strategy variants and report the best, and its Sharpe estimates the
    *maximum of 500 draws*, not that strategy's edge (Phase 0 §3.7). The DSR
    corrects for exactly that, plus the non-normality of returns — a strategy
    with fat left tails needs a much higher raw Sharpe to be believable.

    A DSR below ~0.95 means the result is not distinguishable from the best of
    however many things you tried.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3 or clean.std(ddof=1) == 0:
        return 0.0, 0.0

    observed = float(clean.mean() / clean.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    skew = float(clean.skew())
    kurtosis = float(clean.kurtosis()) + 3.0  # pandas reports excess

    # Expected maximum Sharpe from `trials` independent draws of pure noise.
    variance = trial_sharpe_variance if trial_sharpe_variance is not None else 1.0 / n
    if trials <= 1:
        expected_max = 0.0
    else:
        expected_max = (
            math.sqrt(variance)
            * (
                (1 - _EULER_MASCHERONI) * _normal_ppf(1 - 1.0 / trials)
                + _EULER_MASCHERONI * _normal_ppf(1 - 1.0 / (trials * math.e))
            )
            * math.sqrt(TRADING_DAYS)
        )

    daily_observed = observed / math.sqrt(TRADING_DAYS)
    daily_expected = expected_max / math.sqrt(TRADING_DAYS)
    denominator = 1 - skew * daily_observed + (kurtosis - 1) / 4 * daily_observed**2
    if denominator <= 0:
        return 0.0, expected_max

    statistic = (daily_observed - daily_expected) * math.sqrt(n - 1) / math.sqrt(denominator)
    return _normal_cdf(statistic), expected_max


# ── the report ──────────────────────────────────────────────────────────────
@dataclass
class PerformanceReport:
    """Every metric, computed once. Gross and net are always both present."""

    values: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def section(self, prefix: str) -> dict[str, Any]:
        return {k: v for k, v in self.values.items() if k.startswith(prefix)}

    @property
    def headline(self) -> dict[str, Any]:
        keys = (
            "net_return",
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
            "cost_drag",
        )
        return {k: self.values.get(k) for k in keys}


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (peak - equity) / peak.replace(0, np.nan)


def _return_metrics(
    equity: pd.Series, starting_capital: Decimal, total_costs: Decimal
) -> dict[str, Any]:
    start, end = float(starting_capital), float(equity.iloc[-1])
    days = max((equity.index[-1] - equity.index[0]).days, 1)
    years = days / 365.25
    return {
        "net_return": end / start - 1,
        "gross_return": (end + float(total_costs)) / start - 1,
        "cost_drag": float(total_costs) / start,
        "cagr": (end / start) ** (1 / years) - 1 if start > 0 and years > 0 else 0.0,
        "years": years,
    }


def _risk_metrics(equity: pd.Series, returns: pd.Series) -> dict[str, Any]:
    daily_std = float(returns.std(ddof=1))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0

    drawdowns = _drawdown_series(equity)
    positive = drawdowns[drawdowns > 0]
    underwater = (drawdowns > 0).astype(int)
    runs = underwater.groupby((underwater != underwater.shift()).cumsum()).sum()

    return {
        "volatility": daily_std * math.sqrt(TRADING_DAYS),
        "downside_volatility": downside_std * math.sqrt(TRADING_DAYS),
        "max_drawdown": float(drawdowns.max()) if len(drawdowns) else 0.0,
        "average_drawdown": float(positive.mean()) if len(positive) else 0.0,
        "longest_drawdown_days": int(runs.max()) if len(runs) else 0,
        "_daily_std": daily_std,
        "_downside_std": downside_std,
    }


def _trade_metrics(trades: list[Trade]) -> dict[str, Any]:
    if not trades:
        return dict.fromkeys(
            (
                "win_rate",
                "loss_rate",
                "gross_profit",
                "gross_loss",
                "profit_factor",
                "expectancy",
                "average_win",
                "average_loss",
                "largest_win",
                "largest_loss",
                "average_holding_days",
            ),
            0.0,
        )

    pnls = [float(t.net_pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_loss = abs(sum(losses))
    return {
        "win_rate": len(wins) / len(pnls),
        "loss_rate": len(losses) / len(pnls),
        "gross_profit": sum(wins),
        "gross_loss": gross_loss,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss else math.inf,
        "expectancy": sum(pnls) / len(pnls),
        "average_win": sum(wins) / len(wins) if wins else 0.0,
        "average_loss": sum(losses) / len(losses) if losses else 0.0,
        "largest_win": max(pnls),
        "largest_loss": min(pnls),
        "average_holding_days": sum(t.holding_days for t in trades) / len(trades),
    }


def _benchmark_metrics(
    returns: pd.Series, benchmark: pd.Series, net_return: float
) -> dict[str, Any]:
    aligned = pd.concat([returns, benchmark.astype(float)], axis=1, join="inner").dropna()
    if len(aligned) <= 2:
        return {}
    strategy_returns, bench_returns = aligned.iloc[:, 0], aligned.iloc[:, 1]

    # A strategy that never traded has a flat equity curve and therefore no
    # variance. Correlating it divides by zero and yields a silent NaN, so
    # report the comparison as undefined instead of pretending to a number.
    if float(strategy_returns.std(ddof=1)) == 0 or float(bench_returns.std(ddof=1)) == 0:
        bench_total = float((1 + bench_returns).prod() - 1)
        return {
            "benchmark_return": bench_total,
            "excess_over_benchmark": net_return - bench_total,
            "benchmark_comparison": "undefined — the strategy never traded",
        }
    variance = float(bench_returns.var(ddof=1))
    beta = float(strategy_returns.cov(bench_returns) / variance) if variance else 0.0
    bench_total = float((1 + bench_returns).prod() - 1)
    return {
        "beta": beta,
        "alpha": (float(strategy_returns.mean()) - beta * float(bench_returns.mean()))
        * TRADING_DAYS,
        "correlation_to_benchmark": float(strategy_returns.corr(bench_returns)),
        "tracking_error": float((strategy_returns - bench_returns).std(ddof=1))
        * math.sqrt(TRADING_DAYS),
        "benchmark_return": bench_total,
        "excess_over_benchmark": net_return - bench_total,
    }


def compute_metrics(
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    *,
    starting_capital: Decimal,
    total_costs: Decimal,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    trials: int = 1,
) -> PerformanceReport:
    """Compute the full metric set from an equity curve and its trades."""
    equity = equity_curve["equity"].astype(float)
    if len(equity) < 2:
        return PerformanceReport({"error": "not enough observations"})

    returns = equity.pct_change().dropna()
    values: dict[str, Any] = {"observations": len(returns)}
    values.update(_return_metrics(equity, starting_capital, total_costs))
    values.update(_risk_metrics(equity, returns))

    # Risk-adjusted ratios, from the pieces the risk section already computed.
    daily_std = values.pop("_daily_std")
    downside_std = values.pop("_downside_std")
    excess = float(returns.mean()) - risk_free_rate / TRADING_DAYS
    values["sharpe"] = excess / daily_std * math.sqrt(TRADING_DAYS) if daily_std else 0.0
    values["sortino"] = excess / downside_std * math.sqrt(TRADING_DAYS) if downside_std else 0.0
    values["calmar"] = (
        values["cagr"] / values["max_drawdown"] if values["max_drawdown"] > 0 else 0.0
    )

    dsr, expected_max = deflated_sharpe_ratio(returns, trials=trials)
    values["deflated_sharpe"] = dsr
    values["expected_max_sharpe_from_noise"] = expected_max
    values["trials"] = trials

    values["trades"] = len(trades)
    values.update(_trade_metrics(trades))

    if "position_value" in equity_curve.columns:
        exposure = (equity_curve["position_value"].abs() / equity_curve["equity"]).astype(float)
        values["average_exposure"] = float(exposure.mean())
        values["time_in_market"] = float((exposure > 0.001).mean())
    turnover_value = sum(float(t.entry_price * t.quantity) for t in trades)
    values["turnover"] = turnover_value / float(starting_capital) if starting_capital else 0.0

    if benchmark is not None and len(benchmark) > 2:
        values.update(_benchmark_metrics(returns, benchmark, values["net_return"]))

    return PerformanceReport(values)

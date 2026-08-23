"""Backtest reporting.

The most educational output this platform produces is the **gross and net
equity curves on the same axes** (Phase 0 §11.4). The gap between them is the
cost drag, and for most strategies it is the whole story.

Everything here returns data structures rather than pictures. Charts belong to
the dashboard in Phase 8; a report you can assert on in a test is more useful
than one you can only look at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from trading.backtest.metrics import PerformanceReport, Trade

__all__ = ["BacktestReport", "build_report"]


@dataclass
class BacktestReport:
    """Everything a human needs to judge a backtest, as data."""

    metrics: PerformanceReport
    equity: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown: pd.Series
    trade_distribution: pd.DataFrame
    rolling_sharpe: pd.Series
    manifest: dict[str, str]

    def summary_lines(self) -> list[tuple[str, str, str]]:
        """``(section, label, value)`` triples, ready for a table."""
        m = self.metrics

        def pct(key: str) -> str:
            return f"{m.get(key, 0):.2%}"

        def num(key: str, places: int = 2) -> str:
            return f"{m.get(key, 0):.{places}f}"

        rows: list[tuple[str, str, str]] = [
            ("Returns", "Net return", pct("net_return")),
            ("Returns", "Gross return", pct("gross_return")),
            ("Returns", "Cost drag", pct("cost_drag")),
            ("Returns", "CAGR", pct("cagr")),
            ("Risk", "Volatility", pct("volatility")),
            ("Risk", "Max drawdown", pct("max_drawdown")),
            ("Risk", "Average drawdown", pct("average_drawdown")),
            ("Risk", "Longest drawdown", f"{m.get('longest_drawdown_days', 0)} days"),
            ("Risk-adjusted", "Sharpe", num("sharpe")),
            ("Risk-adjusted", "Sortino", num("sortino")),
            ("Risk-adjusted", "Calmar", num("calmar")),
            ("Risk-adjusted", "Deflated Sharpe", num("deflated_sharpe", 4)),
            ("Risk-adjusted", "Trials claimed", str(m.get("trials", 1))),
            ("Trades", "Count", str(m.get("trades", 0))),
            ("Trades", "Win rate", pct("win_rate")),
            ("Trades", "Profit factor", num("profit_factor")),
            ("Trades", "Expectancy", num("expectancy")),
            ("Trades", "Average win", num("average_win")),
            ("Trades", "Average loss", num("average_loss")),
            ("Trades", "Avg holding (days)", num("average_holding_days", 1)),
            ("Portfolio", "Average exposure", pct("average_exposure")),
            ("Portfolio", "Time in market", pct("time_in_market")),
            ("Portfolio", "Turnover", num("turnover")),
        ]
        if "beta" in m.values:
            rows += [
                ("Benchmark", "Benchmark return", pct("benchmark_return")),
                ("Benchmark", "Excess over benchmark", pct("excess_over_benchmark")),
                ("Benchmark", "Beta", num("beta")),
                ("Benchmark", "Alpha (annualised)", pct("alpha")),
                ("Benchmark", "Correlation", num("correlation_to_benchmark")),
            ]
        elif "benchmark_comparison" in m.values:
            rows.append(("Benchmark", "Comparison", str(m.get("benchmark_comparison"))))
        return rows

    def verdict(self) -> list[str]:
        """Automated overfitting and viability flags (Phase 0 §12.3).

        These are prompts to look harder, not conclusions. Every one of them has
        an innocent explanation; several together rarely do.
        """
        m = self.metrics
        flags: list[str] = []

        if m.get("sharpe", 0) > 2.0:
            flags.append(
                f"Sharpe {m['sharpe']:.2f} on retail daily data — suspect a bug or "
                f"look-ahead before believing it"
            )
        if 0 < m.get("trades", 0) < 100:
            flags.append(f"Only {m['trades']} trades — the error bars swamp the estimate")
        if m.get("deflated_sharpe", 1) < 0.95:
            flags.append(
                f"Deflated Sharpe {m.get('deflated_sharpe', 0):.3f} — not distinguishable "
                f"from the best of {m.get('trials', 1)} trials"
            )
        if m.get("net_return", 0) < m.get("benchmark_return", float("-inf")):
            flags.append(
                "Underperforms buy-and-hold of the same instrument, net of costs — "
                "the bar every strategy has to clear"
            )
        if m.get("cost_drag", 0) > abs(m.get("gross_return", 0)) * 0.5:
            flags.append(
                f"Costs consumed {m['cost_drag']:.2%} against a gross return of "
                f"{m.get('gross_return', 0):.2%} — turnover is eating the edge"
            )
        if m.get("max_drawdown", 0) > 0.30:
            flags.append(
                f"Max drawdown {m['max_drawdown']:.1%} — ask honestly whether you would "
                f"have stayed in at the bottom"
            )
        return flags


def build_report(
    metrics: PerformanceReport,
    equity_curve: pd.DataFrame,
    trades: list[Trade],
    manifest: dict[str, str],
    *,
    rolling_window: int = 126,
) -> BacktestReport:
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()

    peak = equity.cummax()
    drawdown = (peak - equity) / peak.replace(0, float("nan"))

    monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_frame = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "return": monthly.to_numpy()}
    )
    monthly_pivot = (
        monthly_frame.pivot(index="year", columns="month", values="return")
        if not monthly_frame.empty
        else pd.DataFrame()
    )

    distribution = pd.DataFrame(
        [
            {
                "instrument": t.instrument_id,
                "direction": t.direction,
                "entry": t.entry_time,
                "exit": t.exit_time,
                "quantity": float(t.quantity),
                "gross_pnl": float(t.gross_pnl),
                "costs": float(t.costs),
                "net_pnl": float(t.net_pnl),
                "return_pct": float(t.return_pct),
                "holding_days": t.holding_days,
                "win": t.is_win,
            }
            for t in trades
        ]
    )

    # Rolling Sharpe shows *when* a strategy worked. A good average hiding one
    # spectacular quarter and three flat years is not a strategy.
    if len(returns) > rolling_window:
        rolling = returns.rolling(rolling_window)
        rolling_sharpe = (rolling.mean() / rolling.std(ddof=1)) * (252**0.5)
    else:
        rolling_sharpe = pd.Series(dtype=float)

    gross_equity = equity + equity_curve["cumulative_costs"].astype(float)
    curve = pd.DataFrame(
        {
            "equity_net": equity,
            "equity_gross": gross_equity,
            "cumulative_costs": equity_curve["cumulative_costs"].astype(float),
            "drawdown": drawdown,
        }
    )

    return BacktestReport(
        metrics=metrics,
        equity=curve,
        monthly_returns=monthly_pivot,
        drawdown=drawdown,
        trade_distribution=distribution,
        rolling_sharpe=rolling_sharpe.dropna(),
        manifest=manifest,
    )


def report_to_dict(report: BacktestReport) -> dict[str, Any]:
    """Flatten for JSON persistence and comparison across runs."""
    return {
        "metrics": report.metrics.values,
        "manifest": report.manifest,
        "flags": report.verdict(),
        "monthly_returns": report.monthly_returns.to_dict(),
    }

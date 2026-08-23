"""Continuous risk monitors and the halt state.

Pre-trade rules judge one proposed order.  Monitors watch the **portfolio** and
can stop trading entirely — which is a different job, because the thing that
ruins an account is rarely one bad order. It is fifteen reasonable-looking ones
during a day that was going wrong.

Halt levels follow Phase 0 §13.4, and the deliberate omission matters:

* ``SOFT``  — stop opening; keep managing what is already held
* ``HARD``  — cancel resting orders, stop everything, **do not liquidate**
* ``PANIC`` — flatten everything, manual confirmation only

Auto-liquidation on a hard halt is excluded on purpose.  Force-closing every
position during the chaos that triggered the halt means selling into the worst
liquidity of the day, and is frequently worse than holding. Halt automatically;
liquidate by human decision.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import Protocol

from trading.core.types import Money
from trading.risk.context import PortfolioView
from trading.risk.profiles import RiskLimits

__all__ = [
    "HaltLevel",
    "HaltState",
    "MonitorReading",
    "MonitorResult",
    "RiskMonitorSet",
]


class HaltLevel(IntEnum):
    """Ordered so the most severe reading wins when several fire at once."""

    NONE = 0
    SOFT = 1
    HARD = 2
    PANIC = 3

    def __str__(self) -> str:
        """IntEnum stringifies to its number; a log saying "2" helps nobody."""
        return self.name

    @property
    def blocks_new_positions(self) -> bool:
        return self >= HaltLevel.SOFT

    @property
    def blocks_everything(self) -> bool:
        return self >= HaltLevel.HARD


@dataclass(frozen=True, slots=True)
class MonitorReading:
    """Everything a monitor needs, gathered once per evaluation."""

    portfolio: PortfolioView
    now: dt.datetime
    consecutive_losses: int = 0
    recent_error_count: int = 0
    data_age: dt.timedelta | None = None
    is_live_like: bool = False


@dataclass(frozen=True, slots=True)
class MonitorResult:
    monitor_id: str
    level: HaltLevel
    observed: str
    limit: str
    detail: str = ""

    @property
    def triggered(self) -> bool:
        return self.level is not HaltLevel.NONE

    def __str__(self) -> str:
        mark = "OK" if not self.triggered else str(self.level)
        return f"[{mark}] {self.monitor_id}: {self.observed} (limit {self.limit})" + (
            f" — {self.detail}" if self.detail else ""
        )


class RiskMonitor(Protocol):
    """Implementations are frozen dataclasses, so the id is read-only."""

    @property
    def monitor_id(self) -> str: ...

    def check(self, reading: MonitorReading) -> MonitorResult: ...


def _ok(monitor_id: str, observed: object, limit: object, detail: str = "") -> MonitorResult:
    return MonitorResult(monitor_id, HaltLevel.NONE, str(observed), str(limit), detail)


@dataclass(frozen=True)
class DailyLossMonitor:
    """The day's loss, marked to market.

    Deliberately includes unrealised P&L. A realised-only daily loss limit is a
    hole you can drive a portfolio through: hold every loser open and the limit
    never fires while the account bleeds (Phase 0 §13.1).
    """

    limits: RiskLimits
    monitor_id: str = "MON_001_daily_loss"

    def check(self, reading: MonitorReading) -> MonitorResult:
        start = reading.portfolio.day_start_equity
        if start is None or start.is_zero:
            return _ok(
                self.monitor_id,
                "unknown",
                self.limits.max_daily_loss,
                "no day-start equity recorded",
            )
        change = (reading.portfolio.equity - start).ratio_to(start)
        breached = change < -self.limits.max_daily_loss
        return MonitorResult(
            self.monitor_id,
            HaltLevel.HARD if breached else HaltLevel.NONE,
            f"{change:.4f}",
            f"-{self.limits.max_daily_loss}",
            "mark-to-market, including unrealised" if breached else "",
        )


@dataclass(frozen=True)
class DrawdownMonitor:
    """Distance below the equity peak — the number that decides runnability."""

    limits: RiskLimits
    monitor_id: str = "MON_002_drawdown"

    def check(self, reading: MonitorReading) -> MonitorResult:
        peak = reading.portfolio.peak_equity
        if peak is None or peak.is_zero:
            return _ok(
                self.monitor_id, "unknown", self.limits.max_drawdown, "no peak equity recorded"
            )
        drawdown = (peak - reading.portfolio.equity).ratio_to(peak)
        breached = drawdown > self.limits.max_drawdown
        return MonitorResult(
            self.monitor_id,
            HaltLevel.HARD if breached else HaltLevel.NONE,
            f"{drawdown:.4f}",
            str(self.limits.max_drawdown),
        )


@dataclass(frozen=True)
class ConsecutiveLossMonitor:
    """A run of losses is weak evidence of a broken strategy and strong evidence
    that something changed. Soft-halts rather than hard: keep managing what is
    open, stop adding to it."""

    limits: RiskLimits
    monitor_id: str = "MON_003_consecutive_losses"

    def check(self, reading: MonitorReading) -> MonitorResult:
        breached = reading.consecutive_losses >= self.limits.max_consecutive_losses
        return MonitorResult(
            self.monitor_id,
            HaltLevel.SOFT if breached else HaltLevel.NONE,
            str(reading.consecutive_losses),
            str(self.limits.max_consecutive_losses),
        )


@dataclass(frozen=True)
class ErrorRateMonitor:
    """Repeated failures mean the system's picture of the world is unreliable.

    Continuing to trade on an unreliable picture is how a bug becomes a loss.
    """

    limits: RiskLimits
    monitor_id: str = "MON_004_error_rate"

    def check(self, reading: MonitorReading) -> MonitorResult:
        breached = reading.recent_error_count >= self.limits.max_error_rate
        return MonitorResult(
            self.monitor_id,
            HaltLevel.HARD if breached else HaltLevel.NONE,
            str(reading.recent_error_count),
            str(self.limits.max_error_rate),
        )


@dataclass(frozen=True)
class DataFeedMonitor:
    limits: RiskLimits
    monitor_id: str = "MON_005_data_feed"

    def check(self, reading: MonitorReading) -> MonitorResult:
        if not reading.is_live_like:
            return _ok(self.monitor_id, "n/a", "n/a", "not enforced in backtest")
        if reading.data_age is None:
            return MonitorResult(
                self.monitor_id,
                HaltLevel.SOFT,
                "unknown",
                f"{self.limits.max_data_age_seconds}s",
                "data age not measured; failing closed",
            )
        age = int(reading.data_age.total_seconds())
        breached = age > self.limits.max_data_age_seconds
        return MonitorResult(
            self.monitor_id,
            HaltLevel.HARD if breached else HaltLevel.NONE,
            f"{age}s",
            f"{self.limits.max_data_age_seconds}s",
        )


@dataclass
class HaltState:
    """The current halt, and why.

    A halt never clears itself. Whatever tripped it needs a human to look, and
    an automatic resume would just re-enter the situation that caused it.
    """

    level: HaltLevel = HaltLevel.NONE
    reasons: list[MonitorResult] = field(default_factory=list)
    engaged_at: dt.datetime | None = None

    @property
    def engaged(self) -> bool:
        return self.level is not HaltLevel.NONE

    def apply(self, results: list[MonitorResult], now: dt.datetime) -> HaltState:
        """Escalate to the most severe reading. Never de-escalates."""
        triggered = [r for r in results if r.triggered]
        if not triggered:
            return self
        worst = max(r.level for r in triggered)
        if worst <= self.level:
            return self
        return HaltState(
            level=worst,
            reasons=[r for r in triggered if r.level == worst],
            engaged_at=self.engaged_at or now,
        )

    def release(self) -> HaltState:
        """Clear the halt. Deliberately a separate, explicit action."""
        return HaltState()

    def summary(self) -> str:
        if not self.engaged:
            return "NORMAL"
        why = "; ".join(f"{r.monitor_id} ({r.observed})" for r in self.reasons)
        return f"{self.level} since {self.engaged_at:%Y-%m-%d %H:%M} — {why}"


class RiskMonitorSet:
    """Runs every monitor and maintains the halt state."""

    def __init__(self, limits: RiskLimits, monitors: list[RiskMonitor] | None = None) -> None:
        self.limits = limits
        self.monitors: list[RiskMonitor] = monitors or [
            DailyLossMonitor(limits),
            DrawdownMonitor(limits),
            ConsecutiveLossMonitor(limits),
            ErrorRateMonitor(limits),
            DataFeedMonitor(limits),
        ]
        self.halt = HaltState()

    def check(self, reading: MonitorReading) -> list[MonitorResult]:
        """Evaluate every monitor. A monitor that raises escalates, never passes."""
        results: list[MonitorResult] = []
        for monitor in self.monitors:
            try:
                results.append(monitor.check(reading))
            except Exception as exc:  # a monitor that cannot answer is not a pass
                results.append(
                    MonitorResult(
                        getattr(monitor, "monitor_id", "MON_000_unknown"),
                        HaltLevel.HARD,
                        f"{type(exc).__name__}: {exc}",
                        "no exception",
                        "monitor failed to evaluate; failing closed",
                    )
                )
        self.halt = self.halt.apply(results, reading.now)
        return results

    @staticmethod
    def track_peak(peak: Money | None, equity: Money) -> Money:
        return equity if peak is None or equity > peak else peak

    @staticmethod
    def daily_change(portfolio: PortfolioView) -> Decimal:
        start = portfolio.day_start_equity
        if start is None or start.is_zero:
            return Decimal(0)
        return (portfolio.equity - start).ratio_to(start)

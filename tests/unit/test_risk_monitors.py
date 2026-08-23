"""Continuous monitors and the halt state."""

import datetime as dt
from decimal import Decimal

import pytest

from trading.config.compliance import INDIA_SEBI
from trading.core.instrument import Instrument, InstrumentId
from trading.core.order import OrderRequest
from trading.core.types import Currency, InstrumentClass, Money, Side, TradingMode
from trading.risk.context import PortfolioView, RiskContext
from trading.risk.engine import RiskEngine
from trading.risk.monitors import (
    ConsecutiveLossMonitor,
    DailyLossMonitor,
    DataFeedMonitor,
    DrawdownMonitor,
    ErrorRateMonitor,
    HaltLevel,
    HaltState,
    MonitorReading,
    RiskMonitorSet,
)
from trading.risk.profiles import RiskLimits

NOW = dt.datetime(2026, 8, 21, 10, 0, tzinfo=dt.UTC)


def portfolio(equity: str, *, day_start: str = "400000", peak: str = "400000") -> PortfolioView:
    return PortfolioView(
        equity=Money.inr(equity),
        cash=Money.inr(equity),
        day_start_equity=Money.inr(day_start),
        peak_equity=Money.inr(peak),
    )


def reading(equity: str, **kwargs) -> MonitorReading:
    return MonitorReading(
        portfolio=portfolio(
            equity,
            day_start=kwargs.pop("day_start", "400000"),
            peak=kwargs.pop("peak", "400000"),
        ),
        now=NOW,
        **kwargs,
    )


# ── individual monitors ────────────────────────────────────────────────────
def test_daily_loss_uses_mark_to_market_not_just_realised():
    """A realised-only limit is a hole you can drive a portfolio through:
    hold every loser open and it never fires while the account bleeds."""
    limits = RiskLimits(max_daily_loss=Decimal("0.02"))
    assert not DailyLossMonitor(limits).check(reading("396000")).triggered  # −1%
    breach = DailyLossMonitor(limits).check(reading("386000"))  # −3.5%
    assert breach.level is HaltLevel.HARD
    assert "unrealised" in breach.detail


def test_daily_loss_is_unknown_rather_than_passing_without_a_baseline():
    view = PortfolioView(equity=Money.inr("100"), cash=Money.inr("100"))
    result = DailyLossMonitor(RiskLimits()).check(MonitorReading(portfolio=view, now=NOW))
    assert not result.triggered
    assert "no day-start equity" in result.detail


def test_drawdown_measures_distance_below_the_peak():
    limits = RiskLimits(max_drawdown=Decimal("0.10"))
    assert not DrawdownMonitor(limits).check(reading("380000", peak="400000")).triggered
    assert DrawdownMonitor(limits).check(reading("340000", peak="400000")).level is HaltLevel.HARD


def test_consecutive_losses_soft_halt_rather_than_hard():
    """Keep managing what is open; stop adding to it."""
    limits = RiskLimits(max_consecutive_losses=5)
    result = ConsecutiveLossMonitor(limits).check(reading("400000", consecutive_losses=6))
    assert result.level is HaltLevel.SOFT
    assert result.level.blocks_new_positions
    assert not result.level.blocks_everything


def test_repeated_errors_hard_halt():
    limits = RiskLimits(max_error_rate=3)
    result = ErrorRateMonitor(limits).check(reading("400000", recent_error_count=5))
    assert result.level is HaltLevel.HARD


def test_data_feed_is_not_monitored_in_a_backtest():
    assert not DataFeedMonitor(RiskLimits()).check(reading("400000")).triggered


def test_an_unmeasured_data_age_in_live_mode_soft_halts():
    result = DataFeedMonitor(RiskLimits()).check(
        reading("400000", is_live_like=True, data_age=None)
    )
    assert result.level is HaltLevel.SOFT
    assert "failing closed" in result.detail


# ── halt state ─────────────────────────────────────────────────────────────
def test_a_halt_escalates_to_the_most_severe_reading():
    limits = RiskLimits(max_consecutive_losses=1, max_error_rate=1)
    monitors = RiskMonitorSet(limits)
    monitors.check(reading("400000", consecutive_losses=5))
    assert monitors.halt.level is HaltLevel.SOFT
    monitors.check(reading("400000", recent_error_count=5))
    assert monitors.halt.level is HaltLevel.HARD


def test_a_halt_never_de_escalates_on_its_own():
    """Whatever tripped it needs a human to look."""
    limits = RiskLimits(max_error_rate=1)
    monitors = RiskMonitorSet(limits)
    monitors.check(reading("400000", recent_error_count=5))
    assert monitors.halt.level is HaltLevel.HARD
    for _ in range(5):
        monitors.check(reading("400000"))  # everything healthy again
    assert monitors.halt.level is HaltLevel.HARD


def test_releasing_a_halt_is_explicit_and_attributed():
    engine = RiskEngine(RiskLimits(max_error_rate=1), INDIA_SEBI)
    engine.check_monitors(reading("400000", recent_error_count=5))
    assert engine.halted
    engine.release_halt(acknowledged_by="tester")
    assert not engine.halted


def test_a_monitor_that_raises_escalates_rather_than_passes():
    class Exploding:
        monitor_id = "MON_TEST_explodes"

        def check(self, reading):
            raise RuntimeError("state store unreachable")

    monitors = RiskMonitorSet(RiskLimits(), monitors=[Exploding()])
    results = monitors.check(reading("400000"))
    assert results[0].level is HaltLevel.HARD
    assert "failing closed" in results[0].detail


def test_halt_levels_stringify_to_their_names():
    """IntEnum stringifies to a number; a log saying '2' helps nobody."""
    assert str(HaltLevel.HARD) == "HARD"
    assert str(HaltLevel.SOFT) == "SOFT"


def test_an_untriggered_halt_summarises_as_normal():
    assert HaltState().summary() == "NORMAL"


# ── the halt outranks every rule ───────────────────────────────────────────
def test_an_engaged_halt_rejects_orders_that_would_otherwise_pass(tmp_path):
    instrument = Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )
    engine = RiskEngine(RiskLimits(), INDIA_SEBI, kill_switch_path=tmp_path / "none")
    ctx = RiskContext(
        request=OrderRequest(instrument=instrument, side=Side.BUY, quantity=Decimal("50")),
        reference_price=Decimal("1400"),
        portfolio=portfolio("400000"),
        now=NOW,
        mode=TradingMode.BACKTEST,
    )
    assert engine.evaluate(ctx).approved

    engine.engage_halt(HaltLevel.HARD, "manual drill", NOW)
    decision = engine.evaluate(ctx)
    assert not decision.approved
    assert any(r.rule_id == "RISK_000_halt_engaged" for r in decision.failures)


def test_the_halt_summary_names_what_tripped_it():
    monitors = RiskMonitorSet(RiskLimits(max_drawdown=Decimal("0.01")))
    monitors.check(reading("300000", peak="400000"))
    assert "MON_002_drawdown" in monitors.halt.summary()


@pytest.mark.parametrize(
    ("level", "blocks_new", "blocks_all"),
    [
        (HaltLevel.NONE, False, False),
        (HaltLevel.SOFT, True, False),
        (HaltLevel.HARD, True, True),
        (HaltLevel.PANIC, True, True),
    ],
)
def test_halt_levels_gate_the_right_things(level, blocks_new, blocks_all):
    assert level.blocks_new_positions is blocks_new
    assert level.blocks_everything is blocks_all

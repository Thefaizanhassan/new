"""Cross-validation against backtrader.

The Phase 0 technology evaluation kept backtrader out of production — it has no
Indian broker, and its cost model cannot express a per-scrip-per-session fee —
but gave it a job it is genuinely good at: **an independent implementation to
check ours against.**

An independent engine disagreeing is the single most effective check on a
backtester. It turns backtrader from a dependency into a test asset: used,
respected, and not load-bearing.

For the comparison to mean anything, both engines must be given the same
problem. That means no slippage, a flat percentage commission, a fixed share
count rather than target weights, and the same fill convention — signal on bar
N's close, fill at bar N+1's open, which is backtrader's default too.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from trading.backtest.fills import NoSlippage, RealisticFillModel
from trading.config.compliance import INDIA_SEBI
from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Flat, Intent, TargetQty
from trading.core.types import Currency, InstrumentClass, Money, TradingMode
from trading.costs.simple import FlatPercentCosts
from trading.data.provider import DataTier, ProviderInfo
from trading.engine.runner import WalkingSkeletonRunner
from trading.features import indicators as ind
from trading.risk.engine import RiskEngine
from trading.risk.profiles import RiskLimits
from trading.strategies.base import StrategySpec

STARTING_CASH = 1_000_000.0
SHARES = 100
FAST, SLOW = 10, 30
COMMISSION = 0.001


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    """A deterministic series with enough trend changes to force several trades."""
    rng = np.random.default_rng(20260823)
    n = 800
    # Several full trend cycles, so the crossover fires repeatedly. A comparison
    # over one or two trades would not exercise much of either engine.
    cycles = 0.22 * np.sin(np.linspace(0, 7 * np.pi, n))
    drift = np.linspace(0, 0.15, n)
    close = 1000 * np.exp(cycles + drift + np.cumsum(rng.normal(0, 0.003, n)))
    index = pd.bdate_range("2022-01-03", periods=n, tz="UTC")
    open_ = np.concatenate([[close[0]], close[:-1]])
    # High and low must bracket both open and close, or the validation gate
    # quarantines the fixture — as it should.
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
            "volume": np.full(n, 5_000_000.0),
        },
        index=index,
    )


# ── our engine ──────────────────────────────────────────────────────────────
class FixedQtySmaCross:
    """A fixed share count, so sizing cannot explain any disagreement."""

    def __init__(self, instrument_id: InstrumentId) -> None:
        self.spec = StrategySpec(
            id="crosscheck_sma",
            name="Crosscheck SMA",
            version="1.0.0",
            description="fixed-quantity SMA cross",
            universe=(instrument_id,),
            params={"fast": FAST, "slow": SLOW, "shares": SHARES},
        )

    def warmup_bars(self) -> int:
        return SLOW + 1

    def on_bar(self, ctx, instrument):  # type: ignore[no-untyped-def]
        history = ctx.history(instrument.id)
        if len(history) < self.warmup_bars():
            return []
        fast = ind.compute("SMA", history, timeperiod=FAST)["value"]
        slow = ind.compute("SMA", history, timeperiod=SLOW)["value"]
        if fast.isna().iloc[-1] or slow.isna().iloc[-1]:
            return []

        uptrend = float(fast.iloc[-1]) > float(slow.iloc[-1])
        holding = ctx.has_position(instrument.id)
        if uptrend and not holding:
            return [Intent(instrument.id, TargetQty(Decimal(SHARES)), self.spec.id, "cross up")]
        if not uptrend and holding:
            return [Intent(instrument.id, Flat(), self.spec.id, "cross down")]
        return []


def run_our_engine(bars: pd.DataFrame) -> tuple[float, int]:
    instrument = Instrument(
        id=InstrumentId("NSE", "XCHECK"),
        name="Crosscheck",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
        tick_size=Decimal("0.0001"),
    )

    class Provider:
        info = ProviderInfo("crosscheck", DataTier.SYNTHETIC, supports_as_of=False)

        def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
            return bars

    result = WalkingSkeletonRunner(
        provider=Provider(),
        instrument=instrument,
        strategy=FixedQtySmaCross(instrument.id),
        cost_model=FlatPercentCosts(rate=Decimal(str(COMMISSION))),
        risk_engine=RiskEngine(
            RiskLimits(max_position_weight=Decimal("1"), max_gross_exposure=Decimal("1")),
            INDIA_SEBI,
            kill_switch_path=Path("/nonexistent/KILL"),
        ),
        compliance=INDIA_SEBI,
        starting_capital=Money.inr(str(STARTING_CASH)),
        mode=TradingMode.BACKTEST,
        # No slippage: backtrader has no equivalent, and an unmatched assumption
        # would show up as an engine disagreement that isn't one.
        fill_model=RealisticFillModel(slippage_model=NoSlippage(), max_participation=Decimal("1")),
    ).run(dt.date(2022, 1, 1), dt.date(2026, 12, 31))

    return float(result.final_equity.amount), len(result.fills)


# ── backtrader ──────────────────────────────────────────────────────────────
class BtSmaCross(bt.Strategy):  # type: ignore[misc]
    def __init__(self) -> None:
        self.fast = bt.indicators.SMA(period=FAST)
        self.slow = bt.indicators.SMA(period=SLOW)
        self.fills = 0

    def notify_order(self, order: object) -> None:
        if getattr(order, "status", None) == bt.Order.Completed:
            self.fills += 1

    def next(self) -> None:
        if len(self) < SLOW + 1:
            return
        uptrend = self.fast[0] > self.slow[0]
        if uptrend and not self.position:
            self.buy(size=SHARES)
        elif not uptrend and self.position:
            self.close()


def run_backtrader(bars: pd.DataFrame) -> tuple[float, int]:
    frame = bars.copy()
    frame.index = frame.index.tz_convert(None)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=frame))
    cerebro.broker.setcash(STARTING_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.addstrategy(BtSmaCross)
    strategies = cerebro.run()
    return float(cerebro.broker.getvalue()), int(strategies[0].fills)


# ── the comparison ──────────────────────────────────────────────────────────
def test_both_engines_agree_on_the_same_strategy(bars):
    """If these diverge, one of the two engines has a bug — and it is worth
    finding out which before trusting either with a promotion decision."""
    ours, our_fills = run_our_engine(bars)
    theirs, their_fills = run_backtrader(bars)

    assert our_fills == their_fills, (
        f"trade counts differ: ours {our_fills}, backtrader {their_fills} — "
        f"the engines are not solving the same problem"
    )

    difference = abs(ours - theirs) / STARTING_CASH
    assert difference < 0.005, (
        f"final equity differs by {difference:.4%}: ours {ours:,.2f}, backtrader {theirs:,.2f}"
    )


def test_both_engines_trade_at_all(bars):
    """A comparison of two engines that both did nothing proves nothing."""
    _, our_fills = run_our_engine(bars)
    assert our_fills >= 8, "the fixture should force several round trips"

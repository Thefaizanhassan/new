"""Reference strategies.

Deliberately simple and well understood.  Their job in the MVP is to exercise
the pipeline and act as regression tests, not to make money — and buy-and-hold
is the benchmark every other strategy has to beat *after costs and taxes*,
which most do not (Phase 0 §1).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Flat, Intent, TargetWeight
from trading.features import indicators as ind
from trading.strategies.base import StrategyContext, StrategySpec

__all__ = ["BuyAndHold", "SmaCross"]


@dataclass
class BuyAndHold:
    """Buy once, hold forever. The benchmark.

    Not a trivial case to skip: it is the honest baseline, it has the lowest
    possible cost drag, and any strategy that cannot beat it net of costs is
    not worth running.
    """

    instrument_id: InstrumentId
    target_weight: Decimal = Decimal("0.95")
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        self.spec = StrategySpec(
            id="buy_and_hold",
            name="Buy and Hold",
            version=self.version,
            description="Take a single long position and never trade again.",
            universe=(self.instrument_id,),
            params={"target_weight": self.target_weight},
            author="reference",
        )

    def warmup_bars(self) -> int:
        return 1

    def on_bar(self, ctx: StrategyContext, instrument: Instrument) -> list[Intent]:
        if ctx.has_position(instrument.id):
            return []
        return [
            Intent(
                instrument_id=instrument.id,
                target=TargetWeight(self.target_weight),
                strategy_id=self.spec.id,
                reason="initial entry — benchmark holds indefinitely",
                evidence={"bars_seen": ctx.bars_available(instrument.id)},
            )
        ]


@dataclass
class SmaCross:
    """Classic trend following: long while the fast SMA is above the slow SMA.

    Included because its failure modes are instructive. It whipsaws in ranging
    markets, and at Indian delivery costs (~0.25–0.35% per round trip) each
    false crossing is a real, measurable loss — which makes it a good test of
    whether the cost model is being taken seriously.
    """

    instrument_id: InstrumentId
    fast: int = 50
    slow: int = 200
    target_weight: Decimal = Decimal("0.95")
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if self.fast >= self.slow:
            raise ValueError(f"fast period ({self.fast}) must be below slow ({self.slow})")
        self.spec = StrategySpec(
            id="sma_cross",
            name="SMA Crossover",
            version=self.version,
            description="Long while SMA(fast) > SMA(slow); flat otherwise.",
            universe=(self.instrument_id,),
            params={"fast": self.fast, "slow": self.slow, "target_weight": self.target_weight},
            author="reference",
        )

    def warmup_bars(self) -> int:
        return self.slow + 1

    def on_bar(self, ctx: StrategyContext, instrument: Instrument) -> list[Intent]:
        history = ctx.history(instrument.id)
        if len(history) < self.warmup_bars():
            return []

        fast = ind.compute("SMA", history, timeperiod=self.fast)["value"]
        slow = ind.compute("SMA", history, timeperiod=self.slow)["value"]
        if fast.isna().iloc[-1] or slow.isna().iloc[-1]:
            return []

        fast_now, slow_now = float(fast.iloc[-1]), float(slow.iloc[-1])
        in_uptrend = fast_now > slow_now
        holding = ctx.has_position(instrument.id)

        evidence = {
            "sma_fast": round(fast_now, 2),
            "sma_slow": round(slow_now, 2),
            "spread_pct": round((fast_now / slow_now - 1) * 100, 3),
        }

        if in_uptrend and not holding:
            return [
                Intent(
                    instrument_id=instrument.id,
                    target=TargetWeight(self.target_weight),
                    strategy_id=self.spec.id,
                    reason=f"SMA({self.fast}) crossed above SMA({self.slow})",
                    evidence=evidence,
                )
            ]
        if not in_uptrend and holding:
            return [
                Intent(
                    instrument_id=instrument.id,
                    target=Flat(),
                    strategy_id=self.spec.id,
                    reason=f"SMA({self.fast}) fell below SMA({self.slow})",
                    evidence=evidence,
                )
            ]
        return []

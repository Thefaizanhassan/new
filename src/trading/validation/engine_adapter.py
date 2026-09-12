"""Turning the backtest engine into something validation can call repeatedly.

Every other module in this package takes a :class:`SegmentEvaluator` and never
imports the engine.  This is the one place that knows both sides, and it exists
so that a walk-forward run and a unit test exercise identical validation logic.

Three decisions here are load-bearing, and each would silently corrupt every
out-of-sample number in the system if it were made the lazy way.

**Bars are fetched once, then sliced.**  A ten-point grid over nine folds is 99
backtests.  Re-fetching the same history 99 times would make a sweep unusable and
— worse, on a provider that revises — could hand different runs different data.
One fetch, one dataset version, sliced by position.

**Returns are measured from the bar before the window opens.**  The return *of*
the first measured bar is the change from the previous bar's equity.  Slicing the
equity curve at the window's first bar and taking ``pct_change`` silently drops
it, so every fold would lose its first day — which is a small error on a
126-bar window and a large one on a 20-bar window.

**Only fully-contained round trips are counted as trades.**  A test window is run
with a warmup prefix, so a position can be carried in from bars that were used
for tuning.  The equity curve legitimately includes that position's subsequent
movement — a live operator switching parameters holds whatever the previous run
left them.  But a *trade* opened during warmup has P&L that partly accrued
in-sample, and feeding that into the Monte Carlo drawdown distribution would mix
in-sample gains into an out-of-sample risk estimate.  So the trade list is
restricted to round trips that both opened and closed inside the measured window.

A consequence worth knowing rather than discovering: a strategy holding for six
months will report zero complete trades on a three-month test window, and the fold
will be skipped for want of trades.  That is the correct answer — the window is
shorter than the strategy's holding period and cannot evaluate it — and the skip
reason says so.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pandas as pd

from trading.backtest.fills import RealisticFillModel
from trading.backtest.metrics import Trade, compute_metrics
from trading.config.compliance import ComplianceProfile, profile_for
from trading.core.instrument import Instrument
from trading.core.types import Money, TradingMode
from trading.costs.model import CostModel
from trading.data.provider import DataTier, HistoricalDataProvider, ProviderInfo
from trading.engine.runner import RunResult, WalkingSkeletonRunner
from trading.risk.engine import RiskEngine
from trading.risk.profiles import RiskLimits
from trading.strategies.base import Strategy
from trading.validation.experiments import ExperimentLedger, ExperimentRecord
from trading.validation.harness import Objective, ParameterSet, SegmentOutcome
from trading.validation.splits import Window

__all__ = ["EngineEvaluator", "FrozenFrameProvider"]


@dataclass(frozen=True, slots=True)
class FrozenFrameProvider:
    """Serves one pre-fetched frame, so a sweep reads the same bars every time.

    The engine calls ``get_bars`` with a date range; this ignores it and returns
    the slice it was constructed with.  That is only safe because the caller
    already sliced by position — and positions are what the splitters reason
    about, so the window the engine runs is exactly the window that was split.
    """

    frame: pd.DataFrame
    info: ProviderInfo
    dataset_version_value: str = "unversioned"

    def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
        return self.frame

    def dataset_version(self, *args: object, **kwargs: object) -> str:
        return self.dataset_version_value


StrategyFactory = Callable[[ParameterSet], Strategy]
"""Builds a strategy from one parameter set. For a YAML strategy this is
``lambda p: load_strategy(path, overrides=p)``."""


@dataclass
class EngineEvaluator:
    """A :class:`SegmentEvaluator` backed by the real backtest engine.

    Callable, so it can be handed straight to :func:`run_walk_forward`,
    :func:`run_purged_cv` or :func:`run_sensitivity`.
    """

    provider: HistoricalDataProvider
    instrument: Instrument
    build_strategy: StrategyFactory
    cost_model: CostModel
    starting_capital: Money
    risk_limits: RiskLimits = field(default_factory=RiskLimits)
    compliance: ComplianceProfile | None = None
    fill_model: RealisticFillModel | None = None
    objective: Objective = field(default_factory=Objective)
    ledger: ExperimentLedger | None = None
    ledger_kind: str = "validation"
    _bars: pd.DataFrame = field(default_factory=pd.DataFrame, init=False, repr=False)
    _dataset_version: str = field(default="unversioned", init=False, repr=False)
    evaluations: int = field(default=0, init=False)

    def load(self, start: dt.date, end: dt.date) -> pd.DatetimeIndex:
        """Fetch the full history once. Returns the bar index the splitters use."""
        bars = self.provider.get_bars(self.instrument.id, start, end)
        if bars.empty:
            raise ValueError(
                f"no bars for {self.instrument.id} between {start} and {end} — "
                "validation cannot proceed without data"
            )
        self._bars = bars
        versioner = getattr(self.provider, "dataset_version", None)
        if callable(versioner):
            self._dataset_version = str(versioner(self.instrument.id, start, end))
        index = bars.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"provider returned a {type(index).__name__}, need a DatetimeIndex")
        return index

    @property
    def bars(self) -> pd.DataFrame:
        if self._bars.empty:
            raise RuntimeError("call load() before evaluating — no bars are held")
        return self._bars

    @property
    def data_tier(self) -> DataTier:
        return self.provider.info.tier

    def __call__(
        self, params: ParameterSet, window: Window, measure_from: Window
    ) -> SegmentOutcome:
        frame = self.bars.iloc[window.start_index : window.stop_index]
        strategy = self.build_strategy(params)
        compliance = self.compliance or profile_for(self.instrument.market)

        result = _run(
            frame=frame,
            provider_info=self.provider.info,
            dataset_version=self._dataset_version,
            instrument=self.instrument,
            strategy=strategy,
            cost_model=self.cost_model,
            risk_limits=self.risk_limits,
            compliance=compliance,
            starting_capital=self.starting_capital,
            fill_model=self.fill_model,
        )
        self.evaluations += 1

        outcome = _measure(
            result_curve=result.equity_curve,
            trades=result.trades,
            params=params,
            measure_from=measure_from,
            full_window=window,
            starting_capital=self.starting_capital.amount,
            total_costs=result.total_costs.amount,
            benchmark=frame["close"].pct_change().dropna(),
        )
        if self.ledger is not None:
            self._record(outcome, measure_from, strategy)
        return outcome

    def _record(self, outcome: SegmentOutcome, window: Window, strategy: Strategy) -> None:
        assert self.ledger is not None
        self.ledger.record(
            ExperimentRecord(
                kind=self.ledger_kind,
                strategy_id=strategy.spec.id,
                strategy_version=strategy.spec.version,
                instrument=str(self.instrument.id),
                start_date=window.start.date().isoformat(),
                end_date=window.stop.date().isoformat(),
                params=dict(outcome.params),
                data_tier=str(self.data_tier),
                dataset_version=self._dataset_version,
                objective_name=self.objective.metric,
                objective_value=_finite_or_none(self.objective.score(outcome)),
                metrics=dict(outcome.metrics),
                manifest={"content_hash": strategy.spec.content_hash},
                note=outcome.note,
            )
        )


def _finite_or_none(value: float) -> float | None:
    """``-inf`` is not a score, it is a refusal; the ledger stores that as NULL."""
    return value if pd.notna(value) and abs(value) != float("inf") else None


def _run(
    *,
    frame: pd.DataFrame,
    provider_info: ProviderInfo,
    dataset_version: str,
    instrument: Instrument,
    strategy: Strategy,
    cost_model: CostModel,
    risk_limits: RiskLimits,
    compliance: ComplianceProfile,
    starting_capital: Money,
    fill_model: RealisticFillModel | None,
) -> RunResult:
    """One backtest over one frame, with the engine assembled fresh.

    Fresh every time on purpose: a :class:`RiskEngine` carries halt state, and a
    halt leaking from one grid point into the next would make the sweep's results
    depend on evaluation order.
    """
    return WalkingSkeletonRunner(
        provider=FrozenFrameProvider(frame, provider_info, dataset_version),
        instrument=instrument,
        strategy=strategy,
        cost_model=cost_model,
        risk_engine=RiskEngine(risk_limits, compliance, kill_switch_path=Path("KILL")),
        compliance=compliance,
        starting_capital=starting_capital,
        mode=TradingMode.BACKTEST,
        fill_model=fill_model or RealisticFillModel(),
    ).run(frame.index[0].date(), frame.index[-1].date())


def _measure(
    *,
    result_curve: pd.DataFrame,
    trades: list[Trade],
    params: ParameterSet,
    measure_from: Window,
    full_window: Window,
    starting_capital: Decimal,
    total_costs: Decimal,
    benchmark: pd.Series,
) -> SegmentOutcome:
    """Reduce a run to the measured window's returns, trades and metrics."""
    equity = result_curve["equity"].astype(float)
    start, stop = measure_from.start, measure_from.stop

    # One bar earlier so the first measured bar's own return is included.
    position = equity.index.searchsorted(start)
    base = max(int(position) - 1, 0)
    measured = equity.iloc[base:][lambda s: s.index <= stop]
    returns = measured.pct_change().dropna()

    contained = [
        t
        for t in trades
        if pd.Timestamp(t.entry_time) >= start and pd.Timestamp(t.exit_time) <= stop
    ]
    warmup_bars = measure_from.start_index - full_window.start_index

    curve = pd.DataFrame({"equity": measured.to_numpy()}, index=measured.index)
    curve.index.name = "timestamp"
    report = compute_metrics(
        curve,
        contained,
        starting_capital=Decimal(str(measured.iloc[0])) if len(measured) else starting_capital,
        total_costs=sum((Decimal(str(t.costs)) for t in contained), Decimal(0)),
        benchmark=benchmark[benchmark.index >= start] if not benchmark.empty else None,
    )

    return SegmentOutcome(
        params=dict(params),
        window=measure_from,
        returns=returns,
        trades=len(contained),
        trade_pnl=tuple(float(t.net_pnl) for t in contained),
        metrics={k: v for k, v in report.values.items() if isinstance(v, (int, float))},
        note=(
            f"measured {len(returns)} bars of {measure_from.label()}"
            + (f", warmed up over {warmup_bars} earlier bars" if warmup_bars else "")
            + f"; {len(contained)} of {len(trades)} round trips fully contained"
        ),
    )


def evaluator_metrics(outcome: SegmentOutcome) -> Mapping[str, float]:
    """The subset of metrics worth printing for one segment."""
    keys = ("net_return", "sharpe", "max_drawdown", "profit_factor", "win_rate", "cost_drag")
    return {k: v for k in keys if (v := outcome.metric(k)) is not None}

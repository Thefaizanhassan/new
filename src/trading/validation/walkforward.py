"""Walk-forward analysis.

Phase 0 §11.5.  This is the only validation in the system that simulates what an
operator would actually have experienced, and the distinction matters enough to
spell out.

A single backtest tunes parameters on the whole history and reports the result on
the same history.  That number is not a prediction; it is a description of how
well the tuning worked.  Walk-forward instead repeats the honest procedure:

    choose parameters using bars 1..250, trade bars 251..310 with them,
    then choose again using bars 61..310, trade bars 311..370, and so on.

Each test segment is traded with parameters chosen *before* it began.  Stitching
those segments together produces the equity curve a disciplined operator
re-tuning on that schedule would have had.  **Only that stitched curve is
quotable.**  The in-sample numbers exist to be compared against, not reported.

**Walk-forward efficiency** is the ratio between out-of-sample and in-sample
performance.  A strategy earning a Sharpe of 2.0 in-sample and 0.2
out-of-sample has an efficiency of 0.1: the tuning found noise.  Around 0.5 or
better is the conventional threshold for "the edge survived the transition",
and even that is a weak claim — it says the strategy is not *purely* fitted, not
that it is profitable.

**Parameter stability** is the quieter and often more damning signal.  If every
fold picks a wildly different "optimal" setting, there is no stable optimum to
find, and whichever value the final full-history fit lands on is arbitrary.
A strategy can post an acceptable efficiency and still fail this.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading.validation.harness import (
    Objective,
    ParameterSet,
    ParameterValue,
    SegmentEvaluator,
    SegmentOutcome,
)
from trading.validation.splits import Split, walk_forward_splits

__all__ = [
    "WalkForwardConfig",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardThresholds",
    "run_walk_forward",
]


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """How to cut the history.

    ``train_bars`` and ``test_bars`` are a judgement call with a real trade-off.
    Long training windows give a stabler parameter estimate but assume the market
    has not changed; short ones adapt faster but tune on less evidence.  Few long
    test windows give fewer, more reliable folds; many short ones give a longer
    out-of-sample record made of noisier pieces.  There is no correct answer, so
    there is no hidden default — and whatever is chosen lands in the report.
    """

    train_bars: int = 504
    test_bars: int = 126
    step_bars: int | None = None
    warmup_bars: int = 200
    anchored: bool = False
    objective: Objective = field(default_factory=Objective)

    def describe(self) -> str:
        style = "anchored (training window grows)" if self.anchored else "rolling"
        return (
            f"{style}, train {self.train_bars} bars / test {self.test_bars} bars, "
            f"step {self.step_bars or self.test_bars}, warmup {self.warmup_bars}, "
            f"objective: {self.objective.describe()}"
        )


@dataclass(frozen=True, slots=True)
class WalkForwardThresholds:
    """The bar a strategy has to clear, written down before the numbers arrive.

    Deciding what counts as a pass *after* seeing the result is how every
    threshold ends up exactly below whatever was measured.  These are defaults,
    they are arguable, and they are meant to be argued with explicitly rather
    than adjusted quietly.
    """

    min_efficiency: float = 0.5
    min_consistency: float = 0.5
    min_oos_net_return: float = 0.0
    min_parameter_stability: float = 0.5
    min_scored_folds: int = 3
    min_oos_trades: int = 30
    """Pooled across folds, not per fold. Per-fold trade counts are tiny for any
    slow strategy, and gating each fold at a meaningful number would discard real
    folds; what matters is whether the *concatenated* record has enough trades to
    say anything. 30 is modest — the lifecycle gate for PROMISING wants 100."""


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One optimise-then-test cycle."""

    split: Split
    chosen: ParameterSet | None
    in_sample: SegmentOutcome | None
    out_of_sample: SegmentOutcome | None
    candidates_evaluated: int
    skip_reason: str = ""

    @property
    def scored(self) -> bool:
        return self.out_of_sample is not None and self.in_sample is not None

    def label(self) -> str:
        if not self.scored:
            return f"fold {self.split.fold}: SKIPPED — {self.skip_reason}"
        assert self.in_sample is not None and self.out_of_sample is not None
        return (
            f"fold {self.split.fold}: {dict(self.chosen or {})} "
            f"IS {self.in_sample.net_return:+.2%} → OOS {self.out_of_sample.net_return:+.2%} "
            f"({self.out_of_sample.trades} trades over {self.split.test.label()})"
        )


def _ratio(out_of_sample: float, in_sample: float) -> float:
    """Out-of-sample over in-sample, guarding the cases that produce nonsense.

    A negative in-sample denominator would flip the sign and make a failure look
    like a success, so anything that is not a positive in-sample result reports
    efficiency as zero — which is the honest reading: nothing was carried
    forward.

    A *negative* result is left as it is, and means something specific and worth
    seeing: the training windows found a positive objective and the test windows
    delivered the opposite sign. That is a stronger failure than an efficiency of
    zero, not a number to clamp away.
    """
    if in_sample <= 0 or not np.isfinite(in_sample) or not np.isfinite(out_of_sample):
        return 0.0
    return out_of_sample / in_sample


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """What survived the transition from tuning to trading."""

    folds: tuple[WalkForwardFold, ...]
    config: WalkForwardConfig
    thresholds: WalkForwardThresholds
    trials: int
    """Total parameter evaluations performed — the real denominator for the
    deflated Sharpe ratio, not a number typed in by hand."""

    @property
    def scored_folds(self) -> tuple[WalkForwardFold, ...]:
        return tuple(f for f in self.folds if f.scored)

    @property
    def skipped_folds(self) -> tuple[WalkForwardFold, ...]:
        return tuple(f for f in self.folds if not f.scored)

    @property
    def oos_returns(self) -> pd.Series:
        """The concatenated out-of-sample return series — the quotable result.

        Test windows never overlap by default, so concatenation does not
        double-count any bar.  A custom ``step_bars`` smaller than ``test_bars``
        would overlap, and duplicated timestamps are dropped here rather than
        silently compounding the same day twice.
        """
        pieces = [f.out_of_sample.returns for f in self.scored_folds if f.out_of_sample]
        if not pieces:
            return pd.Series(dtype=float)
        joined = pd.concat(pieces).sort_index()
        return joined[~joined.index.duplicated(keep="first")].astype(float)

    def oos_equity_curve(self, starting_capital: float = 1.0) -> pd.DataFrame:
        """Compound the out-of-sample returns into an equity curve.

        Shaped so :func:`trading.backtest.metrics.compute_metrics` can read it,
        which means the out-of-sample record is scored by exactly the same code
        as a plain backtest — no second, subtly different metric path.
        """
        returns = self.oos_returns
        if returns.empty:
            return pd.DataFrame(columns=["equity"], index=pd.DatetimeIndex([], name="timestamp"))
        equity = starting_capital * (1.0 + returns).cumprod()
        frame = pd.DataFrame({"equity": equity.astype(float)})
        frame.index.name = "timestamp"
        return frame

    @property
    def oos_net_return(self) -> float:
        returns = self.oos_returns
        return float((1.0 + returns).prod() - 1.0) if not returns.empty else 0.0

    @property
    def oos_trades(self) -> int:
        return sum(f.out_of_sample.trades for f in self.scored_folds if f.out_of_sample)

    @property
    def oos_trade_pnl(self) -> tuple[float, ...]:
        """Every out-of-sample trade's net P&L, for Monte Carlo resampling."""
        pnl: list[float] = []
        for fold in self.scored_folds:
            if fold.out_of_sample:
                pnl.extend(fold.out_of_sample.trade_pnl)
        return tuple(pnl)

    @property
    def losing_in_sample_folds(self) -> int:
        """Folds whose *best* training candidate still lost money.

        Not a failure on its own — walk-forward across a bad year is informative —
        but a run where most folds had nothing good to pick from is optimising
        over noise, and the efficiency ratio alone will not say so.
        """
        return sum(1 for f in self.scored_folds if f.in_sample and f.in_sample.net_return <= 0)

    @property
    def efficiency(self) -> float:
        """Mean out-of-sample objective over mean in-sample objective."""
        scored = self.scored_folds
        if not scored:
            return 0.0
        objective = self.config.objective
        in_sample = [objective.score(f.in_sample) for f in scored if f.in_sample]
        out_sample = [objective.score(f.out_of_sample) for f in scored if f.out_of_sample]
        finite_in = [v for v in in_sample if np.isfinite(v)]
        finite_out = [v for v in out_sample if np.isfinite(v)]
        if not finite_in or not finite_out:
            return 0.0
        return _ratio(float(np.mean(finite_out)), float(np.mean(finite_in)))

    @property
    def consistency(self) -> float:
        """Fraction of out-of-sample segments that made money.

        A strategy whose whole return comes from one fold is one regime's luck,
        however good the total looks.
        """
        scored = self.scored_folds
        if not scored:
            return 0.0
        winners = sum(1 for f in scored if f.out_of_sample and f.out_of_sample.net_return > 0)
        return winners / len(scored)

    @property
    def parameter_stability(self) -> float:
        """How often the folds agreed on the best parameters.

        1.0 means every fold chose the same setting; 1/n means they never agreed.
        Low stability says the optimum is not a property of the market.
        """
        choices = [tuple(sorted((f.chosen or {}).items())) for f in self.scored_folds]
        if not choices:
            return 0.0
        counts: dict[tuple[tuple[str, ParameterValue], ...], int] = {}
        for choice in choices:
            counts[choice] = counts.get(choice, 0) + 1
        return max(counts.values()) / len(choices)

    @property
    def modal_parameters(self) -> ParameterSet:
        """The setting the folds chose most often — the one worth deploying."""
        counts: dict[tuple[tuple[str, ParameterValue], ...], int] = {}
        for fold in self.scored_folds:
            key = tuple(sorted((fold.chosen or {}).items()))
            counts[key] = counts.get(key, 0) + 1
        if not counts:
            return {}
        return dict(max(counts.items(), key=lambda pair: pair[1])[0])

    @property
    def failures(self) -> tuple[str, ...]:
        """Every threshold this result missed, named."""
        t = self.thresholds
        checks = (
            (
                len(self.scored_folds) >= t.min_scored_folds,
                f"only {len(self.scored_folds)} scoreable folds (need {t.min_scored_folds})",
            ),
            (
                self.oos_net_return > t.min_oos_net_return,
                f"out-of-sample return {self.oos_net_return:+.2%} "
                f"(need > {t.min_oos_net_return:+.2%})",
            ),
            (
                self.efficiency >= t.min_efficiency,
                f"walk-forward efficiency {self.efficiency:.2f} (need >= {t.min_efficiency:.2f})",
            ),
            (
                self.consistency >= t.min_consistency,
                f"consistency {self.consistency:.0%} (need >= {t.min_consistency:.0%})",
            ),
            (
                self.parameter_stability >= t.min_parameter_stability,
                f"parameter stability {self.parameter_stability:.0%} "
                f"(need >= {t.min_parameter_stability:.0%})",
            ),
            (
                self.oos_trades >= t.min_oos_trades,
                f"only {self.oos_trades} out-of-sample trades across all folds "
                f"(need >= {t.min_oos_trades})",
            ),
        )
        return tuple(message for ok, message in checks if not ok)

    @property
    def passed(self) -> bool:
        """Every threshold cleared. Absence of evidence is a failure, not a pass."""
        return bool(self.scored_folds) and not self.failures

    def summary_lines(self) -> list[tuple[str, str]]:
        return [
            ("folds scored", f"{len(self.scored_folds)} of {len(self.folds)}"),
            ("out-of-sample return", f"{self.oos_net_return:+.2%}"),
            ("out-of-sample bars", str(len(self.oos_returns))),
            ("out-of-sample trades", str(self.oos_trades)),
            ("walk-forward efficiency", f"{self.efficiency:.2f}"),
            ("folds with losing in-sample", f"{self.losing_in_sample_folds}"),
            ("consistency", f"{self.consistency:.0%}"),
            ("parameter stability", f"{self.parameter_stability:.0%}"),
            ("modal parameters", str(dict(self.modal_parameters)) or "none"),
            ("parameter evaluations", str(self.trials)),
            ("verdict", "PASSED" if self.passed else "FAILED"),
        ]

    def verdict(self) -> str:
        if self.passed:
            return (
                f"walk-forward PASSED: {self.oos_net_return:+.2%} out of sample over "
                f"{len(self.oos_returns)} bars, efficiency {self.efficiency:.2f}"
            )
        return "walk-forward FAILED: " + "; ".join(self.failures)


def run_walk_forward(
    evaluate: SegmentEvaluator,
    index: pd.DatetimeIndex,
    grid: Sequence[ParameterSet],
    config: WalkForwardConfig | None = None,
    thresholds: WalkForwardThresholds | None = None,
) -> WalkForwardResult:
    """Optimise on each training window, test on the window that follows it.

    ``grid`` may be a single-element sequence, in which case nothing is optimised
    and this becomes plain out-of-sample testing — still worth running, because
    it measures how the fixed parameters behaved across regimes.
    """
    config = config or WalkForwardConfig()
    thresholds = thresholds or WalkForwardThresholds()
    if not grid:
        raise ValueError("grid must contain at least one parameter set")

    splits = walk_forward_splits(
        index,
        train_bars=config.train_bars,
        test_bars=config.test_bars,
        step_bars=config.step_bars,
        warmup_bars=config.warmup_bars,
        anchored=config.anchored,
    )

    folds: list[WalkForwardFold] = []
    trials = 0
    for split in splits:
        train_window = split.train[0]
        candidates: list[SegmentOutcome] = []
        for params in grid:
            candidates.append(evaluate(params, train_window, train_window))
            trials += 1

        best = config.objective.best(candidates)
        if best is None:
            folds.append(
                WalkForwardFold(
                    split=split,
                    chosen=None,
                    in_sample=None,
                    out_of_sample=None,
                    candidates_evaluated=len(candidates),
                    skip_reason=(
                        f"no parameter set reached {config.objective.min_trades} trades "
                        f"in training window {train_window.label()}"
                    ),
                )
            )
            continue

        out_of_sample = evaluate(best.params, split.run, split.test)
        trials += 1
        if out_of_sample.trades < 1:
            # The equity curve may still have moved — a position carried in from
            # the warmup prefix keeps marking to market — but that movement is
            # the tail of an in-sample decision, not an out-of-sample one.
            # Scoring it would credit the fold for something the strategy did
            # while it was still being tuned.
            folds.append(
                WalkForwardFold(
                    split=split,
                    chosen=best.params,
                    in_sample=best,
                    out_of_sample=None,
                    candidates_evaluated=len(candidates),
                    skip_reason=(
                        "no complete round trip inside the test window "
                        f"{split.test.label()} — the window is shorter than this "
                        "strategy's holding period, so it would measure a carried "
                        "position rather than a decision"
                    ),
                )
            )
            continue

        folds.append(
            WalkForwardFold(
                split=split,
                chosen=best.params,
                in_sample=best,
                out_of_sample=out_of_sample,
                candidates_evaluated=len(candidates),
            )
        )

    return WalkForwardResult(
        folds=tuple(folds), config=config, thresholds=thresholds, trials=trials
    )


def aggregate_outcomes(outcomes: Sequence[SegmentOutcome]) -> Mapping[str, float]:
    """Bar-weighted mean of a metric across non-contiguous segments.

    Purged k-fold trains on a split sample, so its objective has to be combined
    across two chunks.  Weighting by bar count rather than taking a plain mean
    stops a 30-bar chunk from counting as much as a 600-bar one.
    """
    total_bars = sum(o.bars for o in outcomes)
    if not outcomes or total_bars == 0:
        return {}
    keys = {k for o in outcomes for k in o.metrics}
    combined: dict[str, float] = {}
    for key in keys:
        weighted = [(o.metric(key) or 0.0) * o.bars for o in outcomes if o.metric(key) is not None]
        weight = sum(o.bars for o in outcomes if o.metric(key) is not None)
        if weight:
            combined[key] = sum(weighted) / weight
    combined["bars"] = float(total_bars)
    return combined

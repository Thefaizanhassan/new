"""Purged, embargoed cross-validation.

Phase 0 §11.5.  A second opinion on walk-forward, answering a different question
and carrying a different caveat.  Both need stating, because this is the check
most likely to be quoted as though it meant more than it does.

**What it answers.**  Walk-forward can never test its own first training window —
the earliest years of history are only ever used for tuning.  If those years hold
the one regime the strategy cannot survive, walk-forward will not find out.
Purged k-fold uses *every* fold as a test fold in turn, so every regime in the
sample gets tested.  Agreement across folds is evidence the parameters describe
something durable; a single catastrophic fold is a regime the strategy does not
handle, and it is worth knowing about before capital meets it.

**What it does not answer.**  For any fold but the first, the training data
includes bars that come *after* the test fold.  That is intrinsic to the method,
not a bug to fix — but it means a number from here is **not** evidence the
strategy could have been traded.  Only walk-forward is.  Reporting a purged-CV
Sharpe as expected performance would be using tomorrow's data to justify today's
trade, which is the exact failure this whole layer exists to prevent.

**Purge and embargo.**  A trade opened a week before a test window closes inside
it, so those two windows share an outcome and are not independent.  Purging drops
the bars immediately before each test fold from training; the embargo drops the
bars immediately after, because volatility clusters and the days following a
stressed window still carry its regime.  Set ``purge_bars`` to at least the
longest holding period the strategy permits.

**The honest limitation of applying this to a backtest at all.**  Purged k-fold
was designed for a feature/label matrix, where dropping rows is exact.  A
backtest is a *path*: positions carry between bars.  Training on a
non-contiguous sample therefore means running the backtest separately over each
contiguous chunk and combining the objectives, which restarts position state at
each chunk boundary.  That approximation is recorded on every result as
``chunk_restarts`` so it cannot be forgotten.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trading.validation.harness import (
    Objective,
    ParameterSet,
    SegmentEvaluator,
    SegmentOutcome,
)
from trading.validation.splits import Split, purged_kfold_splits
from trading.validation.walkforward import aggregate_outcomes

__all__ = [
    "PurgedCvConfig",
    "PurgedCvFold",
    "PurgedCvResult",
    "run_purged_cv",
]


@dataclass(frozen=True, slots=True)
class PurgedCvConfig:
    n_splits: int = 5
    purge_bars: int = 10
    embargo_bars: int = 5
    warmup_bars: int = 200
    objective: Objective = field(default_factory=Objective)

    def describe(self) -> str:
        return (
            f"{self.n_splits} folds, purge {self.purge_bars} bars, "
            f"embargo {self.embargo_bars} bars, warmup {self.warmup_bars}, "
            f"objective: {self.objective.describe()}"
        )


@dataclass(frozen=True, slots=True)
class PurgedCvFold:
    split: Split
    chosen: ParameterSet | None
    test: SegmentOutcome | None
    train_bars: int
    chunk_restarts: int
    """How many contiguous training chunks this fold was optimised over. Above 1
    means position state was restarted mid-sample — see the module docstring."""
    skip_reason: str = ""

    @property
    def scored(self) -> bool:
        return self.test is not None

    def label(self) -> str:
        if self.test is None:
            return f"fold {self.split.fold}: SKIPPED — {self.skip_reason}"
        return (
            f"fold {self.split.fold}: {dict(self.chosen or {})} → "
            f"{self.test.net_return:+.2%} over {self.split.test.label()} "
            f"({self.test.trades} trades)"
        )


@dataclass(frozen=True, slots=True)
class PurgedCvResult:
    folds: tuple[PurgedCvFold, ...]
    config: PurgedCvConfig
    trials: int
    min_positive_folds_fraction: float = 0.6

    @property
    def scored_folds(self) -> tuple[PurgedCvFold, ...]:
        return tuple(f for f in self.folds if f.scored)

    @property
    def fold_returns(self) -> tuple[float, ...]:
        return tuple(f.test.net_return for f in self.scored_folds if f.test)

    @property
    def fold_scores(self) -> tuple[float, ...]:
        scores = [self.config.objective.score(f.test) for f in self.scored_folds if f.test]
        return tuple(s for s in scores if np.isfinite(s))

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.fold_scores)) if self.fold_scores else 0.0

    @property
    def score_dispersion(self) -> float:
        """Standard deviation of the objective across folds.

        High dispersion with a decent mean is the signature of a strategy that
        works in one regime and not others.  The mean alone hides it.
        """
        if len(self.fold_scores) < 2:
            return 0.0
        return float(np.std(self.fold_scores, ddof=1))

    @property
    def positive_fraction(self) -> float:
        returns = self.fold_returns
        if not returns:
            return 0.0
        return sum(1 for r in returns if r > 0) / len(returns)

    @property
    def worst_fold(self) -> PurgedCvFold | None:
        scored = [f for f in self.scored_folds if f.test]
        if not scored:
            return None
        return min(scored, key=lambda f: f.test.net_return if f.test else 0.0)

    @property
    def chunk_restarts(self) -> int:
        """Total mid-sample position restarts across all folds. Provenance, not a metric."""
        return sum(f.chunk_restarts - 1 for f in self.folds if f.chunk_restarts > 1)

    @property
    def failures(self) -> tuple[str, ...]:
        checks = (
            (bool(self.scored_folds), "no fold produced a scoreable result"),
            (
                self.positive_fraction >= self.min_positive_folds_fraction,
                f"only {self.positive_fraction:.0%} of folds were profitable "
                f"(need >= {self.min_positive_folds_fraction:.0%})",
            ),
            (self.mean_score > 0, f"mean objective across folds {self.mean_score:.2f} (need > 0)"),
        )
        return tuple(message for ok, message in checks if not ok)

    @property
    def passed(self) -> bool:
        return bool(self.scored_folds) and not self.failures

    def summary_lines(self) -> list[tuple[str, str]]:
        worst = self.worst_fold
        return [
            ("folds scored", f"{len(self.scored_folds)} of {len(self.folds)}"),
            ("mean objective", f"{self.mean_score:.3f}"),
            ("objective dispersion", f"{self.score_dispersion:.3f}"),
            ("profitable folds", f"{self.positive_fraction:.0%}"),
            (
                "worst fold",
                f"{worst.test.net_return:+.2%} over {worst.split.test.label()}"
                if worst and worst.test
                else "n/a",
            ),
            ("parameter evaluations", str(self.trials)),
            ("mid-sample restarts", str(self.chunk_restarts)),
            ("verdict", "PASSED" if self.passed else "FAILED"),
        ]

    def verdict(self) -> str:
        if self.passed:
            return (
                f"purged CV PASSED: {self.positive_fraction:.0%} of folds profitable, "
                f"mean objective {self.mean_score:.2f} — generalisation across regimes, "
                "not evidence of tradeability"
            )
        return "purged CV FAILED: " + "; ".join(self.failures)


def run_purged_cv(
    evaluate: SegmentEvaluator,
    index: pd.DatetimeIndex,
    grid: Sequence[ParameterSet],
    config: PurgedCvConfig | None = None,
) -> PurgedCvResult:
    """Hold out each fold in turn, optimising on the purged remainder."""
    config = config or PurgedCvConfig()
    if not grid:
        raise ValueError("grid must contain at least one parameter set")

    splits = purged_kfold_splits(
        index,
        n_splits=config.n_splits,
        purge_bars=config.purge_bars,
        embargo_bars=config.embargo_bars,
        warmup_bars=config.warmup_bars,
    )

    folds: list[PurgedCvFold] = []
    trials = 0
    for split in splits:
        scored: list[tuple[float, ParameterSet]] = []
        for params in grid:
            chunks = [evaluate(params, chunk, chunk) for chunk in split.train]
            trials += len(chunks)
            combined = aggregate_outcomes(chunks)
            merged = SegmentOutcome(
                params=dict(params),
                window=split.train[0],
                returns=pd.concat([c.returns for c in chunks]).sort_index(),
                trades=sum(c.trades for c in chunks),
                trade_pnl=tuple(p for c in chunks for p in c.trade_pnl),
                metrics=combined,
                note=f"aggregated over {len(chunks)} contiguous training chunk(s)",
            )
            score = config.objective.score(merged)
            if np.isfinite(score):
                scored.append((score, dict(params)))

        if not scored:
            folds.append(
                PurgedCvFold(
                    split=split,
                    chosen=None,
                    test=None,
                    train_bars=split.train_bars,
                    chunk_restarts=len(split.train),
                    skip_reason=(
                        f"no parameter set reached {config.objective.min_trades} trades "
                        f"on the purged training sample ({split.train_bars} bars)"
                    ),
                )
            )
            continue

        best_params = max(scored, key=lambda pair: pair[0])[1]
        test = evaluate(best_params, split.run, split.test)
        trials += 1
        if test.trades < 1:
            # Same guard as walk-forward, for the same reason: the test window is
            # run with a warmup prefix, so a fold with no completed round trip is
            # measuring a position carried in from outside it rather than a
            # decision made inside it.
            folds.append(
                PurgedCvFold(
                    split=split,
                    chosen=best_params,
                    test=None,
                    train_bars=split.train_bars,
                    chunk_restarts=len(split.train),
                    skip_reason=(
                        "no complete round trip inside the test fold "
                        f"{split.test.label()} — it would measure a carried position "
                        "rather than a decision"
                    ),
                )
            )
            continue

        folds.append(
            PurgedCvFold(
                split=split,
                chosen=best_params,
                test=test,
                train_bars=split.train_bars,
                chunk_restarts=len(split.train),
            )
        )

    return PurgedCvResult(folds=tuple(folds), config=config, trials=trials)

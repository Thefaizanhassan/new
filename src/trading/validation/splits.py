"""Time-series splits for validation.

Phase 0 §11.5.  Every split in here exists to answer one question: *would this
strategy have worked on data it was not tuned on?*  Getting that wrong is the
single easiest way to produce a backtest that looks excellent and loses money,
so the mechanics are worth stating precisely.

**Why ordinary cross-validation is wrong for this.**  Standard k-fold shuffles
rows and trains on a random 80%.  On a price series that means training on
2025 to predict 2023 — the model is allowed to know the future.  It will score
beautifully and tell you nothing.  Both splitters here are ordered: training
bars and test bars never interleave.

**Warmup is part of the problem, not a detail.**  A strategy needing a 200-bar
moving average does nothing for the first 200 bars it sees.  Hand it a six-month
test window and half the window is dead — the measured result is then mostly an
artefact of the split, not of the strategy.  Every test window here therefore
carries a ``run`` window extended backwards by the warmup length: the engine
*reads* those earlier bars to warm its indicators, and only the bars from
``test.start`` onward are measured.  Re-reading earlier bars to warm up an
average is not look-ahead — it is exactly what a live system does every morning.

**Purging and embargo.**  Two adjacent bars are not independent observations.  A
trade opened three days before a test window closes inside it, so the two
windows share information; and volatility clusters, so the bars immediately
after a test window still carry its regime.  Purging drops the bars just before
a test window from training, and the embargo drops the bars just after.  Without
them, "out of sample" overlaps in-sample by the length of a trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Split",
    "Window",
    "purged_kfold_splits",
    "walk_forward_splits",
    "window_from",
]


@dataclass(frozen=True, slots=True)
class Window:
    """A contiguous span of bars, identified both by position and by date.

    Positions are what the splitters reason about, because overlap between two
    windows is then exactly checkable rather than a guess about weekends and
    exchange holidays.  Dates are what the backtest engine takes.  Carrying both
    is the only way to be sure the thing that ran is the thing that was split.
    """

    start_index: int
    stop_index: int
    """Exclusive, like a Python slice."""
    start: pd.Timestamp
    stop: pd.Timestamp
    """Inclusive — the timestamp of the last bar in the window."""

    @property
    def bars(self) -> int:
        return self.stop_index - self.start_index

    @property
    def positions(self) -> range:
        return range(self.start_index, self.stop_index)

    def label(self) -> str:
        return f"{self.start.date()}..{self.stop.date()} ({self.bars} bars)"

    def overlaps(self, other: Window) -> bool:
        return self.start_index < other.stop_index and other.start_index < self.stop_index


def window_from(index: pd.DatetimeIndex, start_index: int, stop_index: int) -> Window:
    """Build a window from positions, resolving the dates from the bar index."""
    if not 0 <= start_index < stop_index <= len(index):
        raise ValueError(f"invalid window [{start_index}, {stop_index}) over {len(index)} bars")
    return Window(
        start_index=start_index,
        stop_index=stop_index,
        start=pd.Timestamp(index[start_index]),
        stop=pd.Timestamp(index[stop_index - 1]),
    )


@dataclass(frozen=True, slots=True)
class Split:
    """One training/test division.

    ``train`` is a tuple because purged k-fold training data is genuinely
    non-contiguous: holding out a middle fold leaves a chunk before it and a
    chunk after it.  Walk-forward always yields a single training chunk, and
    keeps the same shape so both splitters feed the same code.

    ``run`` is the window the engine is actually given for the test: ``test``
    extended backwards far enough to warm the strategy's indicators.  Measure
    ``test``; run ``run``.
    """

    fold: int
    train: tuple[Window, ...]
    test: Window
    run: Window
    purged_bars: int = 0
    embargo_bars: int = 0

    @property
    def train_bars(self) -> int:
        return sum(w.bars for w in self.train)

    @property
    def contaminated(self) -> bool:
        """True if any training bar falls inside the measured test window.

        This must never be true.  Both splitters assert it before returning, so
        a future change to the index arithmetic fails loudly instead of quietly
        inflating every out-of-sample number in the system.
        """
        return any(w.overlaps(self.test) for w in self.train)

    def label(self) -> str:
        train = " + ".join(w.label() for w in self.train)
        return f"fold {self.fold}: train [{train}] → test [{self.test.label()}]"


def _with_warmup(index: pd.DatetimeIndex, test: Window, warmup_bars: int) -> Window:
    """Extend a test window backwards so indicators are warm at ``test.start``."""
    return window_from(index, max(0, test.start_index - max(warmup_bars, 0)), test.stop_index)


def walk_forward_splits(
    index: pd.DatetimeIndex,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    warmup_bars: int = 0,
    anchored: bool = False,
) -> tuple[Split, ...]:
    """Rolling optimise-then-test windows, in chronological order.

    This is the deployment-realistic validation: at each step the parameters are
    chosen using only bars that had already happened, then applied forward to
    bars that had not.  Concatenating the test segments produces the equity curve
    a disciplined operator re-tuning on that schedule would actually have had —
    and that concatenated curve is the *only* result worth quoting.

    ``anchored=True`` grows the training window from a fixed origin instead of
    rolling it; use it when you believe older data still applies, and roll when
    you believe the market regime has changed.  Neither is obviously right, which
    is why it is a parameter rather than a default buried in the code.

    ``step_bars`` defaults to ``test_bars``, giving non-overlapping test windows.
    A smaller step re-tests bars that a previous fold already tested, which
    inflates the apparent sample size — the concatenated curve would double-count
    those bars — so it is opt-in.
    """
    if train_bars < 1 or test_bars < 1:
        raise ValueError("train_bars and test_bars must both be at least 1")
    step = step_bars if step_bars is not None else test_bars
    if step < 1:
        raise ValueError("step_bars must be at least 1")
    if len(index) < train_bars + test_bars:
        raise ValueError(
            f"need at least {train_bars + test_bars} bars for a "
            f"{train_bars}/{test_bars} split, have {len(index)}"
        )

    splits: list[Split] = []
    fold = 0
    train_start = 0
    train_stop = train_bars
    while train_stop + test_bars <= len(index):
        test = window_from(index, train_stop, train_stop + test_bars)
        train = window_from(index, train_start if not anchored else 0, train_stop)
        splits.append(
            Split(
                fold=fold,
                train=(train,),
                test=test,
                run=_with_warmup(index, test, warmup_bars),
            )
        )
        fold += 1
        train_stop += step
        train_start += step

    _assert_clean(splits)
    return tuple(splits)


def purged_kfold_splits(
    index: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    purge_bars: int = 10,
    embargo_bars: int = 5,
    warmup_bars: int = 0,
) -> tuple[Split, ...]:
    """K contiguous test folds, with the bars around each one removed from training.

    **What this measures, and what it does not.**  Unlike walk-forward, the
    training data for a middle fold includes bars *after* the test fold.  That is
    deliberate and it is not a bug: purged k-fold asks "do these parameters
    generalise across regimes?", which is a question about the parameter set, not
    a simulation of deployment.  It is a useful second opinion precisely because
    it uses every fold as a test fold — walk-forward can never test its own first
    training window.

    It is *not* evidence that the strategy could have been traded.  Only
    walk-forward is, and a result quoted from here as though it were live
    performance is a lie of omission.  ``purge_bars`` should be at least the
    longest holding period the strategy allows, or a trade opened before the test
    window and closed inside it links the two.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars cannot be negative")
    total = len(index)
    fold_size = total // n_splits
    if fold_size < 1:
        raise ValueError(f"{total} bars cannot be divided into {n_splits} folds")

    splits: list[Split] = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_stop = total if fold == n_splits - 1 else test_start + fold_size
        test = window_from(index, test_start, test_stop)

        train: list[Window] = []
        left_stop = test_start - purge_bars
        if left_stop > 0:
            train.append(window_from(index, 0, left_stop))
        right_start = test_stop + embargo_bars
        if right_start < total:
            train.append(window_from(index, right_start, total))
        if not train:
            raise ValueError(
                f"fold {fold} has no training data left after purging {purge_bars} "
                f"and embargoing {embargo_bars} bars; reduce them or use fewer folds"
            )

        splits.append(
            Split(
                fold=fold,
                train=tuple(train),
                test=test,
                run=_with_warmup(index, test, warmup_bars),
                purged_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        )

    _assert_clean(splits)
    return tuple(splits)


def _assert_clean(splits: list[Split]) -> None:
    """Fail loudly if any split leaks training bars into its own test window."""
    for split in splits:
        if split.contaminated:
            raise AssertionError(
                f"split arithmetic produced overlapping train/test windows: {split.label()}"
            )

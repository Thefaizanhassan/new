"""Time-series splits.

The assertions that matter here are the ones about contamination and warmup.
An off-by-one in this file would inflate every out-of-sample number in the
system, and it would do so invisibly — the reported metrics would all still look
plausible.
"""

import itertools

import pandas as pd
import pytest

from trading.validation.splits import (
    purged_kfold_splits,
    walk_forward_splits,
    window_from,
)


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=1000, tz="UTC"))


# ── windows ─────────────────────────────────────────────────────────────────
def test_window_carries_both_positions_and_dates(index):
    w = window_from(index, 10, 20)
    assert w.bars == 10
    assert w.start == index[10]
    assert w.stop == index[19], "stop is the last bar, not one past it"


def test_window_rejects_impossible_ranges(index):
    for bad in ((-1, 10), (10, 10), (5, 2), (0, len(index) + 1)):
        with pytest.raises(ValueError, match="invalid window"):
            window_from(index, *bad)


def test_overlap_is_exact_by_position(index):
    a, b, c = window_from(index, 0, 50), window_from(index, 50, 100), window_from(index, 49, 60)
    assert not a.overlaps(b), "adjacent windows must not count as overlapping"
    assert a.overlaps(c)
    assert c.overlaps(a), "overlap is symmetric"


# ── walk-forward ────────────────────────────────────────────────────────────
def test_walk_forward_never_trains_on_its_own_test_window(index):
    for split in walk_forward_splits(index, train_bars=250, test_bars=60):
        assert not split.contaminated
        assert split.train[0].stop_index == split.test.start_index


def test_walk_forward_test_windows_do_not_overlap_by_default(index):
    splits = walk_forward_splits(index, train_bars=250, test_bars=60)
    for earlier, later in itertools.pairwise(splits):
        assert earlier.test.stop_index == later.test.start_index


def test_walk_forward_warmup_extends_backwards_only(index):
    """The run window reaches back for warmup; the measured window does not move.

    If warmup shifted the measured window the fold would silently score bars the
    previous fold already scored.
    """
    split = walk_forward_splits(index, train_bars=250, test_bars=60, warmup_bars=200)[0]
    assert split.run.start_index == split.test.start_index - 200
    assert split.run.stop_index == split.test.stop_index
    assert split.test.bars == 60


def test_walk_forward_warmup_clamps_at_the_start_of_history(index):
    """Asking for more warmup than exists must clamp, not produce a negative index."""
    split = walk_forward_splits(index, train_bars=50, test_bars=60, warmup_bars=500)[0]
    assert split.run.start_index == 0


def test_anchored_grows_the_training_window(index):
    rolling = walk_forward_splits(index, train_bars=250, test_bars=60)
    anchored = walk_forward_splits(index, train_bars=250, test_bars=60, anchored=True)
    assert all(s.train[0].start_index == 0 for s in anchored)
    assert anchored[-1].train_bars > rolling[-1].train_bars
    assert rolling[-1].train_bars == 250, "a rolling window keeps its length"


def test_walk_forward_rejects_impossible_configurations(index):
    with pytest.raises(ValueError, match="at least 1"):
        walk_forward_splits(index, train_bars=0, test_bars=10)
    with pytest.raises(ValueError, match="need at least"):
        walk_forward_splits(index, train_bars=900, test_bars=200)


def test_smaller_step_overlaps_and_is_opt_in(index):
    """A step below test_bars re-tests bars. Allowed, but it must be asked for."""
    default = walk_forward_splits(index, train_bars=250, test_bars=60)
    stepped = walk_forward_splits(index, train_bars=250, test_bars=60, step_bars=30)
    assert len(stepped) > len(default)
    assert stepped[0].test.overlaps(stepped[1].test)


# ── purged k-fold ───────────────────────────────────────────────────────────
def test_purged_kfold_covers_every_bar_exactly_once(index):
    splits = purged_kfold_splits(index, n_splits=5)
    covered = [i for s in splits for i in s.test.positions]
    assert sorted(covered) == list(range(len(index)))


def test_purge_and_embargo_remove_bars_from_training(index):
    """The bars either side of the test fold must be absent from training.

    Without this, a trade opened before the test window and closed inside it
    links the two samples, and the fold is not out-of-sample by the length of a
    trade.
    """
    split = purged_kfold_splits(index, n_splits=5, purge_bars=10, embargo_bars=7)[2]
    train_positions = {i for w in split.train for i in w.positions}
    purged = set(range(split.test.start_index - 10, split.test.start_index))
    embargoed = set(range(split.test.stop_index, split.test.stop_index + 7))
    assert not train_positions & purged
    assert not train_positions & embargoed
    assert not split.contaminated


def test_middle_folds_train_on_two_chunks_and_edges_on_one(index):
    splits = purged_kfold_splits(index, n_splits=5, purge_bars=10, embargo_bars=5)
    assert len(splits[0].train) == 1, "the first fold has nothing before it"
    assert len(splits[-1].train) == 1, "the last fold has nothing after it"
    assert len(splits[2].train) == 2


def test_purged_kfold_rejects_a_configuration_that_leaves_no_training_data():
    small = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10, tz="UTC"))
    with pytest.raises(ValueError, match="no training data"):
        purged_kfold_splits(small, n_splits=2, purge_bars=6, embargo_bars=6)


def test_purged_kfold_rejects_too_few_folds(index):
    with pytest.raises(ValueError, match="at least 2"):
        purged_kfold_splits(index, n_splits=1)

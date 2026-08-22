"""Bitemporal Parquet store."""

import datetime as dt

import pandas as pd
import pytest

from trading.core.instrument import InstrumentId
from trading.data.fixture import FixtureProvider
from trading.data.store import ParquetBarStore

RELIANCE = InstrumentId("NSE", "RELIANCE")
JULY = dt.datetime(2024, 7, 1, tzinfo=dt.UTC)
AUGUST = dt.datetime(2024, 8, 1, tzinfo=dt.UTC)


@pytest.fixture
def store(tmp_path) -> ParquetBarStore:
    return ParquetBarStore(tmp_path)


@pytest.fixture
def bars() -> pd.DataFrame:
    return FixtureProvider().get_bars(RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 6, 30))


def test_write_then_read_round_trips(store, bars):
    receipt = store.write(RELIANCE, bars, provider="fixture")
    assert receipt.rows == len(bars)
    back = store.read(RELIANCE)
    assert len(back) == len(bars)
    assert back["close"].round(6).equals(bars["close"].round(6))


def test_reading_an_unknown_instrument_returns_an_empty_frame(store):
    assert store.read(InstrumentId("NSE", "NOTHING")).empty


def test_as_of_returns_the_view_that_was_knowable_then(store, bars):
    """The central bitemporal property. Without it a backtest uses the future."""
    store.write(RELIANCE, bars, provider="fixture", ingested_at=JULY)
    restated = bars.copy()
    restated[["open", "high", "low", "close"]] /= 4  # a 4:1 split restatement
    store.write(RELIANCE, restated, provider="fixture", ingested_at=AUGUST)

    latest = store.read(RELIANCE)
    as_of_july = store.read(RELIANCE, as_of=dt.datetime(2024, 7, 15, tzinfo=dt.UTC))

    assert len(latest) == len(as_of_july) == len(bars)
    assert latest["close"].iloc[0] == pytest.approx(bars["close"].iloc[0] / 4)
    assert as_of_july["close"].iloc[0] == pytest.approx(bars["close"].iloc[0])


def test_as_of_before_any_ingestion_returns_nothing(store, bars):
    store.write(RELIANCE, bars, provider="fixture", ingested_at=JULY)
    assert store.read(RELIANCE, as_of=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)).empty


def test_a_revision_never_destroys_the_earlier_one(store, bars):
    store.write(RELIANCE, bars, provider="fixture", ingested_at=JULY)
    store.write(RELIANCE, bars * 2, provider="fixture", ingested_at=AUGUST)
    coverage = store.coverage(RELIANCE)
    assert coverage.revisions == 2
    assert coverage.rows == len(bars)  # rows counts distinct timestamps, not files


def test_two_writes_for_one_year_on_one_day_merge_rather_than_clobber(store):
    """Regression: a chunked backfill writes several times per year per day.
    Replacing the file instead of merging silently lost every chunk but the last."""
    provider = FixtureProvider()
    first = provider.get_bars(RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 3, 31))
    second = provider.get_bars(RELIANCE, dt.date(2024, 4, 1), dt.date(2024, 6, 30))
    stamp = dt.datetime(2024, 7, 1, 12, tzinfo=dt.UTC)

    store.write(RELIANCE, first, provider="fixture", ingested_at=stamp)
    store.write(RELIANCE, second, provider="fixture", ingested_at=stamp)

    assert len(store.read(RELIANCE)) == len(first) + len(second)


def test_date_bounds_filter_the_read(store, bars):
    store.write(RELIANCE, bars, provider="fixture")
    window = store.read(RELIANCE, dt.date(2024, 3, 1), dt.date(2024, 3, 31))
    assert 0 < len(window) < len(bars)
    assert window.index.min() >= pd.Timestamp("2024-03-01", tz="UTC")


def test_coverage_separates_restatements_from_ingestion_moments(store, bars):
    """A chunked backfill makes many ingestions but restates nothing."""
    halves = [bars.iloc[:60], bars.iloc[60:]]
    for half in halves:
        store.write(RELIANCE, half, provider="fixture")
    coverage = store.coverage(RELIANCE)
    assert coverage.revisions == 1, "no bar was restated"
    assert coverage.rows == len(bars)


def test_dataset_version_changes_with_the_data_and_with_as_of(store, bars):
    store.write(RELIANCE, bars, provider="fixture", ingested_at=JULY)
    original = store.dataset_version(RELIANCE)
    store.write(RELIANCE, bars * 2, provider="fixture", ingested_at=AUGUST)

    assert store.dataset_version(RELIANCE) != original
    assert store.dataset_version(RELIANCE, as_of=JULY) == original


def test_dataset_version_is_stable_for_identical_data(store, bars):
    store.write(RELIANCE, bars, provider="fixture")
    assert store.dataset_version(RELIANCE) == store.dataset_version(RELIANCE)


def test_tier_is_stored_so_trust_travels_with_the_data(store, bars):
    store.write(RELIANCE, bars, provider="fixture", tier="SYNTHETIC")
    assert store.stored_tier(RELIANCE) == "SYNTHETIC"


def test_mixed_tiers_resolve_to_the_least_trustworthy(store, bars):
    store.write(RELIANCE, bars.iloc[:60], provider="a", tier="PRODUCTION")
    store.write(RELIANCE, bars.iloc[60:], provider="b", tier="PROTOTYPE")
    assert store.stored_tier(RELIANCE) == "PROTOTYPE"


def test_quarantine_preserves_bad_data_with_its_reason(store, bars):
    path = store.quarantine(RELIANCE, bars, provider="fixture", reason="negative price")
    assert path.exists()
    saved = pd.read_parquet(path)
    assert saved["quarantine_reason"].iloc[0] == "negative price"
    assert store.read(RELIANCE).empty, "quarantined data must not enter the main store"


def test_catalog_lists_what_is_held(store, bars):
    store.write(RELIANCE, bars, provider="fixture", tier="SYNTHETIC")
    catalog = store.catalog()
    assert len(catalog) == 1
    assert catalog.iloc[0]["instrument_id"] == "NSE:RELIANCE"
    assert catalog.iloc[0]["rows"] == len(bars)


def test_empty_store_returns_an_empty_catalog(store):
    assert store.catalog().empty


def test_naive_timestamps_are_refused(store, bars):
    naive = bars.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-naive"):
        store.write(RELIANCE, naive, provider="fixture")


def test_incomplete_bars_are_refused(store, bars):
    with pytest.raises(ValueError, match="missing"):
        store.write(RELIANCE, bars.drop(columns=["volume"]), provider="fixture")

"""Incremental backfill and the store-backed provider."""

import datetime as dt
from decimal import Decimal

import pytest

from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe
from trading.data.backfill import BackfillService
from trading.data.corporate_actions import (
    ActionType,
    AdjustmentMode,
    CorporateAction,
    CorporateActionStore,
)
from trading.data.fixture import FixtureProvider
from trading.data.provider import DataTier, ProviderInfo
from trading.data.store import ParquetBarStore
from trading.data.store_provider import StoreBackedProvider

RELIANCE = InstrumentId("NSE", "RELIANCE")
START, END = dt.date(2023, 1, 1), dt.date(2024, 12, 31)


@pytest.fixture
def store(tmp_path) -> ParquetBarStore:
    return ParquetBarStore(tmp_path)


@pytest.fixture
def service(store) -> BackfillService:
    return BackfillService(FixtureProvider(), store, chunk_days=180)


# ── backfill ───────────────────────────────────────────────────────────────
def test_chunked_backfill_stores_exactly_what_a_single_fetch_returns(service, store):
    report = service.backfill(RELIANCE, START, END)
    direct = FixtureProvider().get_bars(RELIANCE, START, END)
    stored = store.read(RELIANCE)

    assert report.ok
    assert len(stored) == len(direct)
    assert stored["close"].round(6).equals(direct["close"].round(6))
    assert len(report.chunks) > 1, "the range should have been split"


def test_rerunning_a_completed_backfill_writes_nothing(service):
    service.backfill(RELIANCE, START, END)
    again = service.backfill(RELIANCE, START, END)
    assert again.rows_written == 0
    assert all(c.status in ("already_present", "no_sessions") for c in again.chunks)


def test_backfill_resumes_from_what_is_already_stored(service, store):
    service.backfill(RELIANCE, START, dt.date(2023, 12, 31))
    partial = len(store.read(RELIANCE))

    extended = service.backfill(RELIANCE, START, END)
    assert extended.rows_written > 0
    assert extended.skipped, "the already-fetched year should be skipped"
    assert len(store.read(RELIANCE)) > partial


def test_force_refetches_and_keeps_the_earlier_revision(service, store):
    service.backfill(RELIANCE, START, dt.date(2023, 6, 30))
    forced = service.backfill(RELIANCE, START, dt.date(2023, 6, 30), force=True)
    assert forced.rows_written > 0
    assert not forced.skipped


def test_a_chunk_that_fails_validation_is_quarantined_not_stored(store):
    class BadProvider:
        info = ProviderInfo("bad", DataTier.SYNTHETIC, supports_as_of=False)

        def get_bars(self, instrument_id, start, end, timeframe=Timeframe.DAY_1, *, as_of=None):
            bars = FixtureProvider().get_bars(instrument_id, start, end)
            bars.loc[bars.index[3], "close"] = -1  # impossible price
            return bars

    report = BackfillService(BadProvider(), store, chunk_days=3650).backfill(
        RELIANCE, START, dt.date(2023, 3, 31)
    )
    assert not report.ok
    assert report.quarantined
    assert "impossible_price" in report.quarantined[0].detail
    assert store.read(RELIANCE).empty, "bad data must never reach the main store"
    assert list(store.quarantine_root.rglob("*.parquet")), "it must be kept for inspection"


def test_a_failing_provider_does_not_lose_the_other_chunks(store):
    calls = {"n": 0}

    class FlakyProvider:
        info = ProviderInfo("flaky", DataTier.SYNTHETIC, supports_as_of=False)

        def get_bars(self, instrument_id, start, end, timeframe=Timeframe.DAY_1, *, as_of=None):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ConnectionError("simulated timeout")
            return FixtureProvider().get_bars(instrument_id, start, end)

    report = BackfillService(FlakyProvider(), store, chunk_days=180).backfill(RELIANCE, START, END)
    assert any(c.status == "error" for c in report.chunks)
    assert report.rows_written > 0, "one bad chunk must not lose the rest"


def test_a_range_with_no_sessions_is_recorded_rather_than_fetched(service):
    report = service.backfill(RELIANCE, dt.date(2024, 1, 27), dt.date(2024, 1, 28))  # a weekend
    assert all(c.status == "no_sessions" for c in report.chunks)
    assert report.rows_written == 0


def test_the_provider_tier_is_recorded_with_the_bars(service, store):
    service.backfill(RELIANCE, START, dt.date(2023, 6, 30))
    assert store.stored_tier(RELIANCE) == "SYNTHETIC"


# ── store-backed provider ──────────────────────────────────────────────────
def test_store_backed_provider_supports_as_of_unlike_the_network_providers(service, store):
    service.backfill(RELIANCE, START, END)
    provider = StoreBackedProvider(store)
    assert provider.info.supports_as_of
    assert len(provider.get_bars(RELIANCE, START, END)) > 0


def test_store_backed_provider_applies_corporate_actions_on_read(service, store, tmp_path):
    service.backfill(RELIANCE, START, END)
    actions = CorporateActionStore(tmp_path)
    actions.save(
        RELIANCE,
        [CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 1, 2), ratio=Decimal(4))],
    )
    plain = StoreBackedProvider(store).get_bars(RELIANCE, START, END)
    adjusted = StoreBackedProvider(
        store, action_store=actions, adjustment=AdjustmentMode.SPLIT_ONLY
    ).get_bars(RELIANCE, START, END)

    assert adjusted["close"].iloc[0] == pytest.approx(plain["close"].iloc[0] / 4)
    assert adjusted["close"].iloc[-1] == pytest.approx(plain["close"].iloc[-1])


def test_an_action_that_had_not_happened_yet_is_not_applied(service, store, tmp_path):
    """At simulated time T, a split with a later ex-date has not occurred."""
    service.backfill(RELIANCE, START, END)
    actions = CorporateActionStore(tmp_path)
    actions.save(
        RELIANCE,
        [CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 6, 1), ratio=Decimal(4))],
    )
    provider = StoreBackedProvider(
        store, action_store=actions, adjustment=AdjustmentMode.SPLIT_ONLY
    )
    with_hindsight = provider.get_bars(RELIANCE, START, dt.date(2024, 3, 1))
    at_the_time = provider.get_bars(
        RELIANCE, START, dt.date(2024, 3, 1), as_of=dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    )
    assert not at_the_time.empty
    assert at_the_time["close"].iloc[0] == pytest.approx(with_hindsight["close"].iloc[0] * 4)


def test_revision_as_of_is_separate_from_the_simulation_clock(service, store):
    """Backfilling today means every bar has today's ingested_at, so asking what
    you knew in 2024 correctly returns nothing — while the simulation clock
    still works. Conflating the two silently emptied every backtest."""
    service.backfill(RELIANCE, START, END)
    provider = StoreBackedProvider(store)

    assert not provider.get_bars(
        RELIANCE, START, END, as_of=dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    ).empty
    assert provider.get_bars(
        RELIANCE, START, END, revision_as_of=dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    ).empty


def test_storage_does_not_upgrade_trust(service, store):
    service.backfill(RELIANCE, START, dt.date(2023, 6, 30))
    assert StoreBackedProvider(store).tier_for(RELIANCE) is DataTier.SYNTHETIC
    assert not StoreBackedProvider(store).tier_for(RELIANCE).may_support_a_validated_strategy


def test_intraday_is_refused_until_the_store_supports_it(store):
    with pytest.raises(NotImplementedError, match="daily bars only"):
        StoreBackedProvider(store).get_bars(RELIANCE, START, END, Timeframe.MIN_15)

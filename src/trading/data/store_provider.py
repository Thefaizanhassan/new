"""Reads bars from the store, applying corporate actions on the way out.

This is what the engine consumes.  It is a :class:`HistoricalDataProvider` like
any other, so nothing above the data layer knows whether bars came from disk or
from a network call.

**Two different "as of" questions, deliberately kept apart.**  A first draft
used one parameter for both and produced empty results, which is what surfaced
the distinction:

``as_of`` — *the simulation instant.*  At simulated time T, only corporate
actions with an ex-date at or before T had happened, so only those may be
applied.  This is always correct and always available, because it depends on
the action's ex-date rather than on when you happened to ingest anything.

``revision_as_of`` — *the knowledge cutoff for data revisions.*  Which version
of a bar you had.  This only becomes meaningful once you have been ingesting
continuously for a while: if you backfilled twenty years of history this
morning, every bar has today's ``ingested_at``, and asking what you knew in
2024 correctly returns nothing.

Conflating them silently returns an empty frame for any backtest run on
freshly-backfilled data, which looks like a bug in the strategy.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe
from trading.data.corporate_actions import (
    AdjustmentMode,
    CorporateActionStore,
    adjust_bars,
)
from trading.data.provider import DataTier, ProviderInfo
from trading.data.store import ParquetBarStore

__all__ = ["StoreBackedProvider"]


class StoreBackedProvider:
    def __init__(
        self,
        store: ParquetBarStore,
        *,
        action_store: CorporateActionStore | None = None,
        adjustment: AdjustmentMode = AdjustmentMode.SPLIT_ONLY,
        source_provider: str | None = None,
        source_tier: DataTier | None = None,
    ) -> None:
        self.store = store
        self.action_store = action_store
        self.adjustment = adjustment
        self.source_provider = source_provider
        self.info = ProviderInfo(
            provider_id=f"store[{source_provider or 'any'}, {adjustment}]",
            # The store cannot launder its source: a strategy backtested on
            # yfinance data stays PROTOTYPE-tier no matter how it was stored.
            # The tier is read back from what is actually on disk rather than
            # asserted by the caller.
            tier=source_tier or DataTier.PROTOTYPE,
            supports_as_of=True,
            caveats=(
                f"Prices adjusted with mode {adjustment}.",
                "Underlying source tier is preserved — storage does not upgrade trust.",
            ),
        )

    def tier_for(self, instrument_id: InstrumentId) -> DataTier:
        """The tier actually on disk for this instrument."""
        stored = self.store.stored_tier(instrument_id, self.source_provider)
        return DataTier(stored) if stored else self.info.tier

    def get_bars(
        self,
        instrument_id: InstrumentId,
        start: dt.date,
        end: dt.date,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        as_of: dt.datetime | None = None,
        revision_as_of: dt.datetime | None = None,
    ) -> pd.DataFrame:
        if timeframe is not Timeframe.DAY_1:
            raise NotImplementedError("The store holds daily bars only until Phase 13")

        bars = self.store.read_ohlcv(
            instrument_id, start, end, as_of=revision_as_of, provider=self.source_provider
        )
        if bars.empty or self.action_store is None:
            return bars

        actions = self.action_store.load(instrument_id)
        if as_of is not None:
            # An action that had not happened yet cannot be applied.
            actions = [a for a in actions if pd.Timestamp(a.ex_date, tz="UTC") <= as_of]
        return adjust_bars(bars, actions, self.adjustment)

    def dataset_version(
        self,
        instrument_id: InstrumentId,
        start: dt.date | None = None,
        end: dt.date | None = None,
        *,
        revision_as_of: dt.datetime | None = None,
    ) -> str:
        return self.store.dataset_version(
            instrument_id, start, end, as_of=revision_as_of, provider=self.source_provider
        )

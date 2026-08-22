"""Historical data provider protocol.

Two ideas encoded here.

**Tiering.** Not all data deserves equal trust.  A provider declares its tier,
and the tier travels with the data into the run manifest, so a backtest can
never quietly claim production-grade provenance it does not have.

**``as_of`` is not optional.** A backtest declares the date it is pretending to
be and receives only what was knowable then (Phase 0 §3.3).  Phase 2 stores
``ingested_at`` alongside every bar so this can be honoured properly; until
then the parameter is carried and recorded, and providers that cannot support
it say so rather than silently ignoring it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol

import pandas as pd

from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe

__all__ = ["DataTier", "HistoricalDataProvider", "ProviderInfo"]


class DataTier(StrEnum):
    """How much this source may be trusted."""

    SYNTHETIC = "SYNTHETIC"
    """Generated. Deterministic and free, and never evidence about a market."""

    PROTOTYPE = "PROTOTYPE"
    """Real data, no guarantees. No SLA, silent revisions, no point-in-time
    correctness. Fine for building a pipeline; **not fine for a result you
    would act on.** yfinance is here."""

    PRODUCTION = "PRODUCTION"
    """A paid, contracted source with defined revision behaviour."""

    @property
    def may_support_a_validated_strategy(self) -> bool:
        return self is DataTier.PRODUCTION


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    provider_id: str
    tier: DataTier
    supports_as_of: bool
    caveats: tuple[str, ...] = ()

    def manifest_entry(self) -> dict[str, str]:
        return {
            "provider_id": self.provider_id,
            "tier": self.tier,
            "supports_as_of": str(self.supports_as_of),
            "caveats": " | ".join(self.caveats),
        }


class HistoricalDataProvider(Protocol):
    info: ProviderInfo

    def get_bars(
        self,
        instrument_id: InstrumentId,
        start: date,
        end: date,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        """Bars in the canonical schema — UTC index, bar-close labelled."""
        ...

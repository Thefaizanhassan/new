"""Deterministic synthetic bars — SYNTHETIC tier.

Exists so the walking skeleton, the test suite and CI run offline, fast and
identically every time.  Real providers are network-dependent and therefore
flaky; a test suite that needs the internet is a test suite that fails for
reasons unrelated to your code.

The series is generated from a seed, aligned to a real exchange calendar, and
labelled SYNTHETIC so nothing downstream can mistake it for market evidence.
"""

from __future__ import annotations

import datetime as dt
import zlib

import numpy as np
import pandas as pd

from trading.calendars.base import calendar_for
from trading.core.instrument import EXCHANGES, InstrumentId
from trading.core.types import Timeframe
from trading.data.provider import DataTier, ProviderInfo

__all__ = ["FixtureProvider"]


class FixtureProvider:
    """Generates a plausible-looking but entirely fictional daily series."""

    #: The series is always generated from this date forward and then sliced,
    #: so any sub-range is a slice of one series. Without this, fetching
    #: 2023-2024 in two chunks would produce different values than fetching it
    #: in one — which would make chunked backfill untestable and would not
    #: resemble a real provider, where history is history.
    EPOCH = dt.date(2015, 1, 1)

    def __init__(
        self,
        *,
        seed: int = 42,
        start_price: float = 1400.0,
        annual_drift: float = 0.08,
        annual_volatility: float = 0.24,
    ) -> None:
        self.seed = seed
        self.start_price = start_price
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.info = ProviderInfo(
            provider_id=f"fixture(seed={seed})",
            tier=DataTier.SYNTHETIC,
            supports_as_of=True,
            caveats=(
                "Generated data. Says nothing about any real market.",
                "Use for pipeline verification and tests only.",
            ),
        )

    def get_bars(
        self,
        instrument_id: InstrumentId,
        start: dt.date,
        end: dt.date,
        timeframe: Timeframe = Timeframe.DAY_1,
        *,
        as_of: dt.datetime | None = None,
    ) -> pd.DataFrame:
        if timeframe is not Timeframe.DAY_1:
            raise ValueError("The fixture provider generates daily bars only")

        calendar = calendar_for(EXCHANGES[instrument_id.exchange].calendar_code)
        # Generate from the epoch so a bar's value depends only on its date,
        # never on the window it was requested in.
        sessions = calendar.sessions_between(min(self.EPOCH, start), end)
        if len(sessions) == 0:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )

        # Seeded per instrument so two symbols differ but each is reproducible.
        # crc32, not hash(): Python randomises string hashing per process
        # (PYTHONHASHSEED), so hash() would give different "deterministic" data
        # on every run — which is precisely the reproducibility failure this
        # project exists to prevent.
        symbol_seed = zlib.crc32(str(instrument_id).encode()) % 10_000
        rng = np.random.default_rng(self.seed + symbol_seed)
        n = len(sessions)
        daily_drift = self.annual_drift / 252
        daily_vol = self.annual_volatility / np.sqrt(252)

        returns = rng.normal(daily_drift, daily_vol, n)
        close = self.start_price * np.exp(np.cumsum(returns))
        gap = rng.normal(0, daily_vol / 3, n)
        open_ = np.concatenate([[self.start_price], close[:-1]]) * np.exp(gap)
        spread = np.abs(rng.normal(0, daily_vol / 2, n)) * close
        high = np.maximum(open_, close) + spread
        low = np.minimum(open_, close) - spread
        volume = rng.lognormal(np.log(5_000_000), 0.4, n).round()

        index = pd.DatetimeIndex(sessions).normalize() + pd.Timedelta(hours=10)
        index.name = "timestamp"
        frame = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        )

        frame = frame[frame.index >= pd.Timestamp(start, tz="UTC")]
        if as_of is not None:
            frame = frame[frame.index <= pd.Timestamp(as_of)]
        return frame

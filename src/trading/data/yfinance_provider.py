"""yfinance adapter — PROTOTYPE tier.

Chosen deliberately to get real NSE data into the pipeline at zero cost, with
its limitations recorded rather than forgotten:

* **No SLA and no support.** It scrapes an undocumented endpoint that can change
  or rate-limit without notice.
* **Silent revisions.** Adjusted prices are recomputed on every request, so the
  "close of 2019-06-03" you get today may differ from last month's. This breaks
  reproducibility, which is why raw (``auto_adjust=False``) prices are requested
  and adjustment factors kept separate.
* **No point-in-time guarantee**, so ``as_of`` cannot be honoured.
* **Terms-of-service ambiguity** around programmatic use.

The tier makes this visible downstream: a strategy cannot be promoted past
research on PROTOTYPE data.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe
from trading.data.provider import DataTier, ProviderInfo
from trading.data.schema import normalize_ohlcv

__all__ = ["YFinanceProvider", "to_yahoo_symbol"]

# Yahoo's suffix for each exchange we support.
_SUFFIX = {"NSE": ".NS", "BSE": ".BO", "NASDAQ": "", "NYSE": ""}

_INTERVAL = {
    Timeframe.DAY_1: "1d",
    Timeframe.HOUR_1: "1h",
    Timeframe.MIN_15: "15m",
    Timeframe.MIN_5: "5m",
    Timeframe.MIN_1: "1m",
}


class UnsupportedExchangeError(ValueError):
    pass


def to_yahoo_symbol(instrument_id: InstrumentId) -> str:
    """``NSE:RELIANCE`` -> ``RELIANCE.NS``."""
    try:
        return f"{instrument_id.symbol}{_SUFFIX[instrument_id.exchange]}"
    except KeyError as exc:
        raise UnsupportedExchangeError(
            f"No Yahoo suffix known for exchange {instrument_id.exchange!r}"
        ) from exc


class YFinanceProvider:
    """Fetches raw (unadjusted) bars and normalises them to the canonical schema."""

    def __init__(self, downloader: Any = None) -> None:
        """``downloader`` is injected so tests never touch the network."""
        self._downloader = downloader
        self.info = ProviderInfo(
            provider_id="yfinance",
            tier=DataTier.PROTOTYPE,
            supports_as_of=False,
            caveats=(
                "No SLA; undocumented endpoint that may change without notice.",
                "Adjusted prices are recomputed per request, so history is not stable.",
                "No point-in-time guarantee — as_of cannot be honoured.",
                "Terms-of-service ambiguity around programmatic use.",
                "Prototype only: do not promote a strategy validated on this data.",
            ),
        )

    def _download(self, symbol: str, start: dt.date, end: dt.date, interval: str) -> pd.DataFrame:
        if self._downloader is not None:
            return self._downloader(symbol, start=start, end=end, interval=interval)
        # Imported lazily so the fixture provider and the test suite never
        # require yfinance to be installed or the network to be reachable.
        import yfinance as yf  # noqa: PLC0415

        return yf.download(
            symbol,
            start=start.isoformat(),
            # yfinance treats `end` as exclusive; add a day so the caller's
            # inclusive range means what they wrote.
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval=interval,
            # Raw prices only. Adjusted series are derived on read from a
            # separate factor table (Phase 0 §7.2), never stored as truth.
            auto_adjust=False,
            progress=False,
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
        if as_of is not None:
            raise NotImplementedError(
                "yfinance cannot answer point-in-time queries: it returns today's "
                "view of history, including retroactive adjustments. Use a "
                "PRODUCTION-tier provider for as_of queries."
            )
        if timeframe not in _INTERVAL:
            raise ValueError(f"Unsupported timeframe {timeframe}")

        raw = self._download(to_yahoo_symbol(instrument_id), start, end, _INTERVAL[timeframe])
        if raw is None or raw.empty:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )

        df = raw.copy()
        # yfinance returns a MultiIndex column layout for some request shapes.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()

        # Daily NSE bars arrive dated, not timestamped. Label them at the
        # session close so `timestamp <= now` remains a sound knowability test.
        source_tz = (
            "Asia/Kolkata" if instrument_id.exchange in ("NSE", "BSE") else "America/New_York"
        )
        normalized = normalize_ohlcv(df, tz=source_tz)

        if timeframe is Timeframe.DAY_1:
            normalized.index = normalized.index.normalize() + pd.Timedelta(hours=10)
            normalized.index.name = "timestamp"

        return normalized

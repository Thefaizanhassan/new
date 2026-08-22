"""Canonical market-data schema and normalisation.

Providers disagree about column names, casing, timezones and column order.
Everything downstream depends on one shape, so normalisation happens once, at
the edge, and nothing past this point sees a provider's quirks.

Canonical form (your Technology Standards §1):
    index   : UTC DatetimeIndex, named 'timestamp', bar-CLOSE labelled, sorted
    columns : open, high, low, close, volume  [, adj_close, bid, ask, spread]
"""

from __future__ import annotations

from typing import Final

import pandas as pd

__all__ = ["OHLCV_COLUMNS", "OPTIONAL_COLUMNS", "SchemaError", "normalize_ohlcv"]

OHLCV_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS: Final[tuple[str, ...]] = ("adj_close", "bid", "ask", "spread")

# Provider spellings we accept and fold into the canonical names.
_ALIASES: Final[dict[str, str]] = {
    "date": "timestamp", "datetime": "timestamp", "time": "timestamp", "ts": "timestamp",
    "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
    "vol": "volume", "qty": "volume", "adjclose": "adj_close",
    "adj close": "adj_close", "adjusted_close": "adj_close",
}


class SchemaError(ValueError):
    """Incoming data cannot be coerced to the canonical schema."""


def normalize_ohlcv(frame: pd.DataFrame, *, tz: str = "UTC") -> pd.DataFrame:
    """Coerce a provider frame to the canonical schema.

    Deliberately strict: a missing required column is an error rather than a
    silently-filled NaN column, because a silently-absent volume column turns
    every liquidity check into a no-op.
    """
    df = frame.copy()
    df.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError(
            "No 'timestamp' column and the index is not a DatetimeIndex. "
            f"Columns seen: {list(df.columns)}"
        )

    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    if df.index.isna().any():
        raise SchemaError(f"{int(df.index.isna().sum())} timestamps could not be parsed")

    # Timezone discipline (Phase 0 §3.6): naive input is localised to the
    # declared source zone, then everything is stored in UTC.
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    df.index = df.index.tz_convert("UTC")
    df.index.name = "timestamp"

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}. Got: {list(df.columns)}")

    keep = list(OHLCV_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep]
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_index()

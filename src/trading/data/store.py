"""Bitemporal Parquet bar store, queried through DuckDB.

Two ideas do the work here.

**Bitemporality.** Market data is not immutable. Prices get revised, and adjusted
history is rewritten by every new corporate action — the "close of 2019-06-03"
you read today is not the number you read last year.  Every row therefore
carries both the time it *refers to* (``timestamp``) and the time we *learned
it* (``ingested_at``).  A backtest declares an ``as_of`` and sees only what was
knowable then (Phase 0 §3.3).  Without this, a backtest silently uses knowledge
from the future and no amount of careful strategy code can save it.

**Append-only across ingestion dates.** A write never touches a file laid down
on an earlier date, so an earlier backtest stays reproducible even after the
provider changes its mind.  Within a single ingestion date the file accumulates
— a chunked backfill makes many writes for the same year on the same day, and
they must merge rather than clobber one another.  Storage is cheap;
un-reproducible results, and silently lost chunks, are not.

Layout::

    data/bars/{provider}/{exchange}/{symbol}/{year}/{ingest_date}.parquet

Parquet because it is columnar and typically 5–10x smaller than CSV; DuckDB
because it queries those files in place with no import step and no server.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from glob import glob
from pathlib import Path

import duckdb
import pandas as pd

from trading.core.instrument import InstrumentId
from trading.data.schema import OHLCV_COLUMNS

__all__ = ["Coverage", "ParquetBarStore", "WriteReceipt"]

_META_COLUMNS = ("ingested_at", "provider", "instrument_id", "tier")


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    """What a write actually did. Returned rather than logged so callers can assert on it."""

    instrument_id: InstrumentId
    provider: str
    rows: int
    ingested_at: dt.datetime
    files: tuple[Path, ...]
    first_timestamp: dt.datetime | None
    last_timestamp: dt.datetime | None

    def __str__(self) -> str:
        span = (
            f"{self.first_timestamp:%Y-%m-%d} to {self.last_timestamp:%Y-%m-%d}"
            if self.first_timestamp and self.last_timestamp
            else "empty"
        )
        return f"{self.instrument_id} via {self.provider}: {self.rows} rows ({span})"


@dataclass(frozen=True, slots=True)
class Coverage:
    """What the store holds for one instrument."""

    instrument_id: InstrumentId
    provider: str
    rows: int
    first_timestamp: dt.datetime
    last_timestamp: dt.datetime
    revisions: int
    """Most versions any single timestamp has. 1 means nothing was ever restated."""
    ingestions: int
    """Distinct ingestion moments — a chunked backfill produces several."""
    last_ingested_at: dt.datetime


class ParquetBarStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.bars_root = self.root / "bars"
        self.quarantine_root = self.root / "quarantine"
        self.bars_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)

    # ── paths ───────────────────────────────────────────────────────────────
    def _symbol_dir(self, instrument_id: InstrumentId, provider: str) -> Path:
        return self.bars_root / provider / instrument_id.exchange / instrument_id.symbol

    def _glob(self, instrument_id: InstrumentId, provider: str | None) -> str:
        provider_part = provider if provider else "*"
        return str(
            self.bars_root
            / provider_part
            / instrument_id.exchange
            / instrument_id.symbol
            / "*"
            / "*.parquet"
        )

    def _has_files(self, instrument_id: InstrumentId, provider: str | None = None) -> bool:
        return bool(glob(self._glob(instrument_id, provider)))

    # ── write ───────────────────────────────────────────────────────────────
    def write(
        self,
        instrument_id: InstrumentId,
        bars: pd.DataFrame,
        *,
        provider: str,
        tier: str = "PROTOTYPE",
        ingested_at: dt.datetime | None = None,
    ) -> WriteReceipt:
        """Append a revision. Never overwrites an earlier one.

        Writes for the same year on the same ingestion date **merge** into one
        file rather than replacing it: a chunked backfill makes several such
        writes, and replacing would silently discard every chunk but the last.
        Rows are deduplicated on timestamp with the newest write winning, so a
        retried backfill stays idempotent.
        """
        ingested_at = ingested_at or dt.datetime.now(dt.UTC)
        if bars.empty:
            return WriteReceipt(instrument_id, provider, 0, ingested_at, (), None, None)

        missing = [c for c in OHLCV_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(f"Cannot store bars missing {missing}")
        if bars.index.tz is None:
            raise ValueError("Refusing to store timezone-naive timestamps")

        frame = bars.copy()
        frame["ingested_at"] = pd.Timestamp(ingested_at).tz_convert("UTC")
        frame["provider"] = provider
        frame["instrument_id"] = str(instrument_id)
        # Trust travels with the data. A strategy backtested on prototype-tier
        # bars stays prototype-tier no matter how many times it is re-read.
        frame["tier"] = tier

        written: list[Path] = []
        ingest_date = ingested_at.date().isoformat()
        for year, chunk in frame.groupby(frame.index.year):
            directory = self._symbol_dir(instrument_id, provider) / str(year)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{ingest_date}.parquet"
            merged = chunk
            if path.exists():
                previous = pd.read_parquet(path)
                merged = pd.concat([previous, chunk])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.to_parquet(path, index=True)
            written.append(path)

        return WriteReceipt(
            instrument_id=instrument_id,
            provider=provider,
            rows=len(frame),
            ingested_at=ingested_at,
            files=tuple(written),
            first_timestamp=frame.index.min().to_pydatetime(),
            last_timestamp=frame.index.max().to_pydatetime(),
        )

    def quarantine(
        self,
        instrument_id: InstrumentId,
        bars: pd.DataFrame,
        *,
        provider: str,
        reason: str,
    ) -> Path:
        """Set bad data aside instead of dropping it.

        A silent drop creates a gap, and a gap makes a strategy skip a session
        it should have traded. Quarantined data stays inspectable.
        """
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
        directory = self.quarantine_root / provider / instrument_id.exchange
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{instrument_id.symbol}_{stamp}.parquet"

        frame = bars.copy()
        frame["quarantine_reason"] = reason
        frame["quarantined_at"] = pd.Timestamp(dt.datetime.now(dt.UTC))
        frame.to_parquet(path, index=True)
        return path

    # ── read ────────────────────────────────────────────────────────────────
    def read(
        self,
        instrument_id: InstrumentId,
        start: dt.date | None = None,
        end: dt.date | None = None,
        *,
        as_of: dt.datetime | None = None,
        provider: str | None = None,
    ) -> pd.DataFrame:
        """Return the view of history that was knowable at ``as_of``.

        When several revisions exist for one timestamp, the most recent one
        ingested at or before ``as_of`` wins. Omitting ``as_of`` means "latest
        known", which is right for live trading and wrong for a backtest.
        """
        if not self._has_files(instrument_id, provider):
            return _empty_frame()

        conditions = ["1 = 1"]
        params: list[object] = []
        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(dt.datetime.combine(start, dt.time.min, tzinfo=dt.UTC))
        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(dt.datetime.combine(end, dt.time.max, tzinfo=dt.UTC))
        if as_of is not None:
            conditions.append("ingested_at <= ?")
            params.append(as_of)

        query = f"""
            SELECT {", ".join(OHLCV_COLUMNS)}, timestamp, ingested_at, provider, tier
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY timestamp ORDER BY ingested_at DESC
                ) AS revision_rank
                FROM read_parquet(?, union_by_name = true)
                WHERE {" AND ".join(conditions)}
            )
            WHERE revision_rank = 1
            ORDER BY timestamp
        """
        with duckdb.connect() as conn:
            frame = conn.execute(query, [self._glob(instrument_id, provider), *params]).df()

        if frame.empty:
            return _empty_frame()

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.set_index("timestamp").sort_index()

    def read_ohlcv(self, *args: object, **kwargs: object) -> pd.DataFrame:
        """:meth:`read` without the provenance columns, for feeding indicators."""
        frame = self.read(*args, **kwargs)  # type: ignore[arg-type]
        return frame[list(OHLCV_COLUMNS)] if not frame.empty else frame

    # ── introspection ───────────────────────────────────────────────────────
    def coverage(self, instrument_id: InstrumentId, provider: str | None = None) -> Coverage | None:
        if not self._has_files(instrument_id, provider):
            return None
        query = """
            SELECT
                any_value(provider)         AS provider,
                count(DISTINCT timestamp)   AS rows,
                min(timestamp)              AS first_timestamp,
                max(timestamp)              AS last_timestamp,
                max(versions)               AS revisions,
                count(DISTINCT ingested_at) AS ingestions,
                max(ingested_at)            AS last_ingested_at
            FROM (
                SELECT *, count(*) OVER (PARTITION BY timestamp) AS versions
                FROM read_parquet(?, union_by_name = true)
            )
        """
        with duckdb.connect() as conn:
            row = conn.execute(query, [self._glob(instrument_id, provider)]).fetchone()
        if row is None or row[1] == 0:
            return None
        return Coverage(
            instrument_id=instrument_id,
            provider=str(row[0]),
            rows=int(row[1]),
            first_timestamp=pd.Timestamp(row[2]).to_pydatetime(),
            last_timestamp=pd.Timestamp(row[3]).to_pydatetime(),
            revisions=int(row[4]),
            ingestions=int(row[5]),
            last_ingested_at=pd.Timestamp(row[6]).to_pydatetime(),
        )

    def catalog(self) -> pd.DataFrame:
        """Everything the store holds. The answer to 'what data do I actually have?'"""
        pattern = str(self.bars_root / "*" / "*" / "*" / "*" / "*.parquet")
        if not glob(pattern):
            return pd.DataFrame(
                columns=[
                    "instrument_id",
                    "provider",
                    "rows",
                    "first",
                    "last",
                    "revisions",
                    "tier",
                ]
            )
        query = """
            SELECT
                instrument_id,
                provider,
                count(DISTINCT timestamp)   AS rows,
                min(timestamp)              AS first,
                max(timestamp)              AS last,
                max(versions)               AS revisions,
                any_value(tier)             AS tier
            FROM (
                SELECT *, count(*) OVER (PARTITION BY instrument_id, provider, timestamp)
                    AS versions
                FROM read_parquet(?, union_by_name = true)
            )
            GROUP BY instrument_id, provider
            ORDER BY instrument_id, provider
        """
        with duckdb.connect() as conn:
            return conn.execute(query, [pattern]).df()

    def stored_tier(self, instrument_id: InstrumentId, provider: str | None = None) -> str | None:
        """The least trustworthy tier present. Mixing sources takes the lower one."""
        if not self._has_files(instrument_id, provider):
            return None
        query = """
            SELECT DISTINCT tier FROM read_parquet(?, union_by_name = true)
        """
        with duckdb.connect() as conn:
            tiers = {
                str(r[0])
                for r in conn.execute(query, [self._glob(instrument_id, provider)]).fetchall()
            }
        if not tiers:
            return None
        order = {"SYNTHETIC": 0, "PROTOTYPE": 1, "PRODUCTION": 2}
        return min(tiers, key=lambda t: order.get(t, 0))

    def dataset_version(
        self,
        instrument_id: InstrumentId,
        start: dt.date | None = None,
        end: dt.date | None = None,
        *,
        as_of: dt.datetime | None = None,
        provider: str | None = None,
    ) -> str:
        """A content hash of exactly the data a run will see.

        Goes into the run manifest. If a backtest result changes, comparing this
        against the previous run's says whether the data moved or the code did
        (Phase 0 §16.3).
        """
        frame = self.read(instrument_id, start, end, as_of=as_of, provider=provider)
        if frame.empty:
            return "empty"
        payload = frame[list(OHLCV_COLUMNS)].to_csv().encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[*OHLCV_COLUMNS, *_META_COLUMNS],
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )

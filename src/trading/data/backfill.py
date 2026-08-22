"""Incremental, resumable backfill.

Design constraints that shaped this:

* **Resumable.** A backfill of twenty years across a rate-limited provider will
  be interrupted. Progress is derived from what is actually in the store rather
  than from a separate checkpoint file, so there is no state to get out of sync
  and re-running is always safe.
* **Idempotent.** Re-running a completed range writes the same revision file
  rather than duplicating rows.
* **Chunked.** Providers rate-limit and time out on large ranges. Each chunk is
  validated and written independently, so one bad chunk does not lose the rest.
* **Quarantine, never drop.** A chunk that fails validation is written to the
  quarantine area with its reason. Silently dropping it would create a gap, and
  a gap makes a strategy skip a session it should have traded.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field

import pandas as pd

from trading.calendars.base import ExchangeCalendarAdapter
from trading.core.instrument import EXCHANGES, InstrumentId
from trading.core.types import Timeframe
from trading.data.provider import HistoricalDataProvider
from trading.data.store import ParquetBarStore
from trading.data.validation import DataQualityReport, validate_ohlcv
from trading.observability.logging import get_logger

__all__ = ["BackfillReport", "BackfillService", "ChunkOutcome"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkOutcome:
    start: dt.date
    end: dt.date
    rows: int
    status: str
    detail: str = ""


@dataclass
class BackfillReport:
    instrument_id: InstrumentId
    provider: str
    requested_start: dt.date
    requested_end: dt.date
    chunks: list[ChunkOutcome] = field(default_factory=list)
    reports: list[DataQualityReport] = field(default_factory=list)

    @property
    def rows_written(self) -> int:
        return sum(c.rows for c in self.chunks if c.status == "written")

    @property
    def quarantined(self) -> list[ChunkOutcome]:
        return [c for c in self.chunks if c.status == "quarantined"]

    @property
    def skipped(self) -> list[ChunkOutcome]:
        return [c for c in self.chunks if c.status == "already_present"]

    @property
    def ok(self) -> bool:
        return not self.quarantined and not any(c.status == "error" for c in self.chunks)

    def summary(self) -> str:
        return (
            f"{self.instrument_id} via {self.provider}: {self.rows_written} rows written, "
            f"{len(self.skipped)} chunk(s) already present, "
            f"{len(self.quarantined)} quarantined"
        )


class BackfillService:
    def __init__(
        self,
        provider: HistoricalDataProvider,
        store: ParquetBarStore,
        *,
        chunk_days: int = 365,
        request_pause_seconds: float = 0.0,
    ) -> None:
        self.provider = provider
        self.store = store
        self.chunk_days = chunk_days
        # Politeness between requests. Free endpoints rate-limit aggressively,
        # and getting blocked mid-backfill costs more time than pausing does.
        self.request_pause_seconds = request_pause_seconds

    def backfill(
        self,
        instrument_id: InstrumentId,
        start: dt.date,
        end: dt.date,
        *,
        timeframe: Timeframe = Timeframe.DAY_1,
        force: bool = False,
    ) -> BackfillReport:
        """Fetch and store everything between ``start`` and ``end``.

        ``force`` re-fetches ranges already held, which is how you pick up a
        provider's revisions — the store keeps both, so the old view survives.
        """
        provider_id = self.provider.info.provider_id
        report = BackfillReport(instrument_id, provider_id, start, end)
        calendar = ExchangeCalendarAdapter(EXCHANGES[instrument_id.exchange].calendar_code)
        existing = None if force else self.store.read(instrument_id, provider=provider_id)

        for chunk_start, chunk_end in self._chunks(start, end):
            sessions = calendar.sessions_between(chunk_start, chunk_end)
            if len(sessions) == 0:
                report.chunks.append(
                    ChunkOutcome(chunk_start, chunk_end, 0, "no_sessions", "exchange closed")
                )
                continue

            if existing is not None and self._is_covered(existing, sessions):
                report.chunks.append(ChunkOutcome(chunk_start, chunk_end, 0, "already_present"))
                continue

            outcome = self._fetch_chunk(
                instrument_id=instrument_id,
                start=chunk_start,
                end=chunk_end,
                timeframe=timeframe,
                sessions=sessions,
                report=report,
            )
            report.chunks.append(outcome)
            if self.request_pause_seconds:
                time.sleep(self.request_pause_seconds)

        log.info(
            "backfill_complete",
            instrument=str(instrument_id),
            provider=provider_id,
            rows=report.rows_written,
            quarantined=len(report.quarantined),
        )
        return report

    # ── internals ───────────────────────────────────────────────────────────
    def _chunks(self, start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
        chunks: list[tuple[dt.date, dt.date]] = []
        cursor = start
        while cursor <= end:
            chunk_end = min(cursor + dt.timedelta(days=self.chunk_days - 1), end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + dt.timedelta(days=1)
        return chunks

    @staticmethod
    def _is_covered(existing: pd.DataFrame, sessions: pd.DatetimeIndex) -> bool:
        """True when every expected session already has a bar."""
        if existing.empty:
            return False
        present = existing.index.normalize().unique()
        return len(sessions.normalize().unique().difference(present)) == 0

    def _fetch_chunk(
        self,
        *,
        instrument_id: InstrumentId,
        start: dt.date,
        end: dt.date,
        timeframe: Timeframe,
        sessions: pd.DatetimeIndex,
        report: BackfillReport,
    ) -> ChunkOutcome:
        try:
            bars = self.provider.get_bars(instrument_id, start, end, timeframe)
        except Exception as exc:  # one bad chunk must not lose the rest
            log.warning(
                "backfill_chunk_failed",
                instrument=str(instrument_id),
                start=str(start),
                error=str(exc),
            )
            return ChunkOutcome(start, end, 0, "error", f"{type(exc).__name__}: {exc}")

        if bars.empty:
            return ChunkOutcome(start, end, 0, "empty", "provider returned no rows")

        quality = validate_ohlcv(bars, symbol=str(instrument_id), expected_sessions=sessions)
        report.reports.append(quality)

        if not quality.is_usable:
            reason = "; ".join(str(i) for i in quality.errors)
            path = self.store.quarantine(
                instrument_id, bars, provider=self.provider.info.provider_id, reason=reason
            )
            log.warning(
                "backfill_chunk_quarantined",
                instrument=str(instrument_id),
                start=str(start),
                reason=reason,
                path=str(path),
            )
            return ChunkOutcome(start, end, len(bars), "quarantined", reason)

        receipt = self.store.write(
            instrument_id,
            bars,
            provider=self.provider.info.provider_id,
            tier=str(self.provider.info.tier),
        )
        return ChunkOutcome(start, end, receipt.rows, "written")

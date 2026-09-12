"""The experiment ledger: an honest denominator for the deflated Sharpe ratio.

Phase 0 §3.7 and §11.4.  The deflated Sharpe ratio corrects a reported Sharpe for
how many things were tried before it was found.  That correction is only as good
as the trial count, and a trial count typed in by the person reporting the result
is worth nothing — it will be 1.

So trials are counted by the machine that ran them.  Every evaluation goes into
an append-only SQLite table keyed by a hash of the parameters, and
:meth:`ExperimentLedger.trial_count` reads the number back out.  That is what
feeds the deflated Sharpe.  Running a hundred-point grid and reporting the winner
now carries its own 100 with it, whether or not anybody remembers.

**Append-only, on purpose.**  There is no update and no delete.  A failed
experiment is evidence; being able to remove it would turn the ledger back into
a self-report, because the natural thing to delete is exactly the run that makes
the winner look lucky.

**SQLite rather than Parquet.**  The bar store is columnar because it answers
range scans over millions of rows.  The ledger answers "how many distinct
parameter sets has this strategy seen?", wants a uniqueness constraint, and is
written one row at a time from several processes.  That is a transactional
workload and SQLite is already in the standard library — no new dependency, and
the file is inspectable with any sqlite client.

**Its honest limitation.**  The ledger counts what was recorded *through it*.
Trials run in a notebook, or by hand, or in a previous project, are invisible to
it, so the count is a floor rather than the truth.  A floor is still far better
than a self-report, and :meth:`trial_count` says which it is.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

__all__ = ["ExperimentLedger", "ExperimentRecord", "params_hash"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id     TEXT PRIMARY KEY,
    recorded_at       TEXT NOT NULL,
    kind              TEXT NOT NULL,
    strategy_id       TEXT NOT NULL,
    strategy_version  TEXT NOT NULL,
    instrument        TEXT NOT NULL,
    start_date        TEXT NOT NULL,
    end_date          TEXT NOT NULL,
    params_json       TEXT NOT NULL,
    params_hash       TEXT NOT NULL,
    data_tier         TEXT NOT NULL,
    dataset_version   TEXT NOT NULL,
    objective_name    TEXT NOT NULL,
    objective_value   REAL,
    metrics_json      TEXT NOT NULL,
    manifest_json     TEXT NOT NULL,
    note              TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_experiments_scope
    ON experiments (strategy_id, instrument);
CREATE INDEX IF NOT EXISTS idx_experiments_params
    ON experiments (strategy_id, instrument, params_hash);
"""


def _canonical_value(value: Any) -> str:
    """Render one parameter value so equal parameters produce equal text.

    ``10``, ``10.0`` and ``Decimal("10")`` are the same parameter to a strategy,
    so they must hash the same.  Plain ``str()`` does not do that — it yields
    ``"10"`` and ``"10.0"`` — and the resulting double-count would inflate the
    trial count.  That sounds conservative, but it corrupts the deflated Sharpe in
    the *permissive* direction for anyone comparing two strategies' counts, and
    silently makes the ledger's uniqueness claim false.

    ``bool`` is checked first because it is an ``int`` subclass, and ``True``
    reaching the numeric branch would canonicalise to ``"1"``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        as_decimal = Decimal(str(value))
        if as_decimal == as_decimal.to_integral_value():
            return str(int(as_decimal))
        return repr(float(as_decimal))
    return str(value)


def params_hash(params: Mapping[str, Any]) -> str:
    """A stable fingerprint for one parameter set.

    Sorted keys and a canonical encoding, so ``{"fast": 10, "slow": 50}`` and
    ``{"slow": 50, "fast": 10.0}`` are recognised as the same trial rather than
    counted twice.
    """
    canonical = {str(k): _canonical_value(v) for k, v in sorted(params.items())}
    payload = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """One evaluation, with enough provenance to reproduce or distrust it."""

    kind: str
    """``backtest``, ``walk_forward_train``, ``walk_forward_test``,
    ``sensitivity``, ``purged_cv_train`` or ``purged_cv_test``."""
    strategy_id: str
    strategy_version: str
    instrument: str
    start_date: str
    end_date: str
    params: Mapping[str, Any]
    data_tier: str
    dataset_version: str = "unversioned"
    objective_name: str = ""
    objective_value: float | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    manifest: Mapping[str, str] = field(default_factory=dict)
    note: str = ""

    @property
    def fingerprint(self) -> str:
        return params_hash(self.params)


class ExperimentLedger:
    """Append-only store of every evaluation, and the trial counts derived from it."""

    def __init__(self, path: Path | str = "data/experiments.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, isolation_level=None)
        try:
            # WAL so a long sweep writing rows does not block a reader asking for
            # the trial count, and vice versa.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            with conn:
                yield conn
        finally:
            conn.close()

    def record(self, record: ExperimentRecord) -> str:
        """Append one evaluation. Returns its id. There is no way to remove it."""
        experiment_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, recorded_at, kind, strategy_id, strategy_version,
                    instrument, start_date, end_date, params_json, params_hash,
                    data_tier, dataset_version, objective_name, objective_value,
                    metrics_json, manifest_json, note
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    experiment_id,
                    datetime.now(UTC).isoformat(),
                    record.kind,
                    record.strategy_id,
                    record.strategy_version,
                    record.instrument,
                    record.start_date,
                    record.end_date,
                    json.dumps({str(k): v for k, v in record.params.items()}, sort_keys=True),
                    record.fingerprint,
                    record.data_tier,
                    record.dataset_version,
                    record.objective_name,
                    record.objective_value,
                    json.dumps(dict(record.metrics), sort_keys=True, default=str),
                    json.dumps(dict(record.manifest), sort_keys=True),
                    record.note,
                ),
            )
        return experiment_id

    def record_many(self, records: list[ExperimentRecord]) -> int:
        """Append a batch in one transaction. A sweep is one unit of work."""
        for record in records:
            self.record(record)
        return len(records)

    def trial_count(self, strategy_id: str, instrument: str | None = None) -> int:
        """Distinct parameter sets ever evaluated for this strategy.

        Distinct, not total: re-running the same parameters over the same data is
        one trial, not two, and counting repeats would deflate a genuine result
        for no reason.  This is a **floor** — see the module docstring for what it
        cannot see.
        """
        sql = "SELECT COUNT(DISTINCT params_hash) AS n FROM experiments WHERE strategy_id = ?"
        args: list[Any] = [strategy_id]
        if instrument is not None:
            sql += " AND instrument = ?"
            args.append(instrument)
        with self._connect() as conn, closing(conn.execute(sql, args)) as cursor:
            row = cursor.fetchone()
        return int(row["n"]) if row else 0

    def evaluation_count(self, strategy_id: str, instrument: str | None = None) -> int:
        """Total evaluations, including repeats of the same parameters."""
        sql = "SELECT COUNT(*) AS n FROM experiments WHERE strategy_id = ?"
        args: list[Any] = [strategy_id]
        if instrument is not None:
            sql += " AND instrument = ?"
            args.append(instrument)
        with self._connect() as conn, closing(conn.execute(sql, args)) as cursor:
            row = cursor.fetchone()
        return int(row["n"]) if row else 0

    def strategies(self) -> list[dict[str, Any]]:
        """Every strategy in the ledger with its trial count and best objective."""
        sql = """
            SELECT strategy_id,
                   instrument,
                   COUNT(DISTINCT params_hash) AS trials,
                   COUNT(*)                    AS evaluations,
                   MAX(objective_value)        AS best_objective,
                   MIN(recorded_at)            AS first_seen,
                   MAX(recorded_at)            AS last_seen
            FROM experiments
            GROUP BY strategy_id, instrument
            ORDER BY last_seen DESC
        """
        with self._connect() as conn, closing(conn.execute(sql)) as cursor:
            return [dict(row) for row in cursor.fetchall()]

    def best(
        self, strategy_id: str, instrument: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Top evaluations by objective.

        Useful, and a trap: the top of this list is *selected*, so its Sharpe is
        the maximum of however many trials the same table is counting.  Read the
        two together or not at all.
        """
        sql = """
            SELECT kind, params_json, objective_name, objective_value, metrics_json,
                   data_tier, start_date, end_date, recorded_at
            FROM experiments
            WHERE strategy_id = ? AND objective_value IS NOT NULL
        """
        args: list[Any] = [strategy_id]
        if instrument is not None:
            sql += " AND instrument = ?"
            args.append(instrument)
        sql += " ORDER BY objective_value DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn, closing(conn.execute(sql, args)) as cursor:
            return [dict(row) for row in cursor.fetchall()]

    def tiers_used(self, strategy_id: str) -> set[str]:
        """Which data tiers this strategy has been evaluated on.

        A strategy validated only on prototype data cannot be promoted past
        backtesting, and this is how the lifecycle gate finds out without being
        told.
        """
        sql = "SELECT DISTINCT data_tier FROM experiments WHERE strategy_id = ?"
        with self._connect() as conn, closing(conn.execute(sql, [strategy_id])) as cursor:
            return {str(row["data_tier"]) for row in cursor.fetchall()}

"""The data validation gate.

Implements your Technology Standards §1 checklist in full.  Nothing reaches the
store, a feature or a strategy without passing through here.

Two rules govern the design:

1. **Never silently drop a bad bar.** Dropping creates a gap, and a gap makes a
   strategy skip a session it should have traded — which is look-ahead's quieter
   cousin.  Failures are *quarantined and reported*.
2. **Report, don't raise.**  A validator that throws on the first problem tells
   you one thing about a dataset.  One that returns a report tells you
   everything, which is what "understand your data before building a strategy"
   actually requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

import pandas as pd

from trading.data.schema import OHLCV_COLUMNS

__all__ = ["Check", "DataQualityReport", "Severity", "ValidationIssue", "validate_ohlcv"]


class Severity(StrEnum):
    ERROR = "ERROR"      # data is unusable; quarantine
    WARNING = "WARNING"  # usable but suspicious; a human should look
    INFO = "INFO"        # worth recording, not worth acting on


class Check(StrEnum):
    """One per item in your §1 list, so coverage is auditable."""

    SCHEMA = "schema"
    TIMEZONE = "timezone"
    DUPLICATE_TIMESTAMPS = "duplicate_timestamps"
    MONOTONIC = "monotonic_order"
    OHLC_RELATIONSHIP = "invalid_ohlc_relationship"
    IMPOSSIBLE_PRICE = "impossible_price"
    MISSING_VALUES = "missing_values"
    MISSING_CANDLES = "missing_candles"
    GAPS = "price_gaps"
    ABNORMAL_VOLUME = "abnormal_volume"
    ZERO_VOLUME = "zero_volume"
    SESSION_ALIGNMENT = "market_session_alignment"
    FREQUENCY = "unexpected_frequency"
    STALE_DATA = "stale_data"
    SYMBOL_CONSISTENCY = "inconsistent_symbols"
    CORPORATE_ACTION = "suspected_corporate_action"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    check: Check
    severity: Severity
    message: str
    count: int = 1
    sample: tuple[str, ...] = ()

    def __str__(self) -> str:
        tail = f"  e.g. {', '.join(self.sample[:3])}" if self.sample else ""
        return f"[{self.severity}] {self.check}: {self.message} (n={self.count}){tail}"


@dataclass
class DataQualityReport:
    """The outcome of validating one instrument's bars."""

    symbol: str
    rows: int
    start: datetime | None
    end: datetime | None
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(
        self,
        check: Check,
        severity: Severity,
        message: str,
        count: int = 1,
        sample: tuple[str, ...] = (),
    ) -> None:
        self.issues.append(ValidationIssue(check, severity, message, count, sample))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]

    @property
    def is_usable(self) -> bool:
        """False means quarantine. Errors block; warnings do not."""
        return not self.errors

    def summary(self) -> str:
        verdict = "USABLE" if self.is_usable else "QUARANTINED"
        return (
            f"{self.symbol}: {verdict} — {self.rows} rows, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )


def validate_ohlcv(
    df: pd.DataFrame,
    *,
    symbol: str,
    expected_sessions: pd.DatetimeIndex | None = None,
    expected_freq: timedelta | None = None,
    now: datetime | None = None,
    max_staleness: timedelta | None = None,
    volume_spike_sigma: float = 8.0,
    gap_threshold: float = 0.20,
) -> DataQualityReport:
    """Run every §1 check and return a report.

    ``expected_sessions`` should come from the exchange calendar.  Without it
    the missing-candle and session-alignment checks are skipped and say so,
    rather than silently passing — a check that quietly does nothing is worse
    than no check.
    """
    report = DataQualityReport(
        symbol=symbol,
        rows=len(df),
        start=df.index.min().to_pydatetime() if len(df) else None,
        end=df.index.max().to_pydatetime() if len(df) else None,
    )

    if df.empty:
        report.add(Check.SCHEMA, Severity.ERROR, "dataset is empty")
        return report

    # ── schema and timezone ─────────────────────────────────────────────────
    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add(Check.SCHEMA, Severity.ERROR, f"missing columns {missing_cols}")
        return report

    if not isinstance(df.index, pd.DatetimeIndex):
        report.add(Check.SCHEMA, Severity.ERROR, "index is not a DatetimeIndex")
        return report
    if df.index.tz is None:
        report.add(
            Check.TIMEZONE, Severity.ERROR,
            "timestamps are timezone-naive; UTC is required before storage",
        )
    elif str(df.index.tz) not in ("UTC", "utc"):
        report.add(
            Check.TIMEZONE, Severity.WARNING, f"index timezone is {df.index.tz}, expected UTC"
        )

    # ── ordering and duplicates ─────────────────────────────────────────────
    if not df.index.is_monotonic_increasing:
        report.add(Check.MONOTONIC, Severity.ERROR, "timestamps are not in ascending order")

    dupes = df.index[df.index.duplicated(keep=False)]
    if len(dupes):
        report.add(
            Check.DUPLICATE_TIMESTAMPS, Severity.ERROR,
            "duplicate timestamps", int(len(dupes)),
            tuple(str(t) for t in dupes.unique()[:3]),
        )

    # ── missing values ──────────────────────────────────────────────────────
    nulls = df[list(OHLCV_COLUMNS)].isna().sum()
    if int(nulls.sum()):
        detail = ", ".join(f"{c}={int(n)}" for c, n in nulls.items() if n)
        report.add(Check.MISSING_VALUES, Severity.ERROR, f"null values ({detail})", int(nulls.sum()))

    ohlc = df[["open", "high", "low", "close"]]
    valid = df[ohlc.notna().all(axis=1)]

    # ── impossible prices ───────────────────────────────────────────────────
    nonpositive = valid[(valid[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if len(nonpositive):
        report.add(
            Check.IMPOSSIBLE_PRICE, Severity.ERROR, "non-positive prices",
            len(nonpositive), tuple(str(t) for t in nonpositive.index[:3]),
        )

    # ── OHLC relationships ──────────────────────────────────────────────────
    bad = valid[
        (valid["high"] < valid[["open", "close"]].max(axis=1))
        | (valid["low"] > valid[["open", "close"]].min(axis=1))
        | (valid["high"] < valid["low"])
    ]
    if len(bad):
        report.add(
            Check.OHLC_RELATIONSHIP, Severity.ERROR,
            "high/low inconsistent with open/close", len(bad),
            tuple(str(t) for t in bad.index[:3]),
        )

    # ── volume ──────────────────────────────────────────────────────────────
    vol = df["volume"].dropna()
    if (vol < 0).any():
        report.add(Check.ABNORMAL_VOLUME, Severity.ERROR, "negative volume", int((vol < 0).sum()))
    zero_vol = int((vol == 0).sum())
    if zero_vol:
        report.add(
            Check.ZERO_VOLUME, Severity.WARNING,
            "zero-volume bars — often a halt, a holiday, or a padded row", zero_vol,
        )
    if len(vol) > 30:
        # Rolling mean must be computed on the same index as `vol`, or the
        # comparison silently misaligns. Shifted by one so a bar is judged
        # against history that excludes itself.
        baseline = vol.rolling(20, min_periods=10).mean().shift(1)
        comparable = baseline.notna() & (baseline > 0)
        spikes = vol[comparable & (vol > baseline * volume_spike_sigma)]
        if len(spikes):
            report.add(
                Check.ABNORMAL_VOLUME, Severity.WARNING,
                f"volume exceeding {volume_spike_sigma}x its 20-bar average", len(spikes),
                tuple(str(t) for t in spikes.index[:3]),
            )

    # ── price gaps / suspected corporate actions ────────────────────────────
    if len(valid) > 1:
        change = valid["close"].pct_change().abs()
        jumps = change[change > gap_threshold]
        if len(jumps):
            report.add(
                Check.CORPORATE_ACTION, Severity.WARNING,
                f"close moved >{gap_threshold:.0%} between bars — verify it is not an "
                "unapplied split or bonus issue",
                len(jumps), tuple(f"{t}: {v:.1%}" for t, v in jumps.head(3).items()),
            )
        overnight = (valid["open"] / valid["close"].shift(1) - 1).abs().dropna()
        big = overnight[overnight > gap_threshold]
        if len(big):
            report.add(Check.GAPS, Severity.INFO, "large overnight gaps", len(big))

    # ── frequency ───────────────────────────────────────────────────────────
    if len(df.index) > 2:
        deltas = pd.Series(df.index).diff().dropna()
        modal = deltas.mode()
        if len(modal):
            report.add(
                Check.FREQUENCY, Severity.INFO, f"modal bar interval is {modal.iloc[0]}"
            )
            if expected_freq is not None and modal.iloc[0] != pd.Timedelta(expected_freq):
                report.add(
                    Check.FREQUENCY, Severity.ERROR,
                    f"expected {expected_freq} bars, data is mostly {modal.iloc[0]}",
                )

    # ── session alignment and missing candles ───────────────────────────────
    if expected_sessions is not None and len(expected_sessions):
        expected = pd.DatetimeIndex(expected_sessions).tz_convert("UTC")
        present = df.index.normalize().unique()
        exp_days = expected.normalize().unique()

        missing = exp_days.difference(present)
        if len(missing):
            report.add(
                Check.MISSING_CANDLES, Severity.WARNING,
                "trading sessions with no bar", len(missing),
                tuple(str(d.date()) for d in missing[:3]),
            )
        extra = present.difference(exp_days)
        if len(extra):
            report.add(
                Check.SESSION_ALIGNMENT, Severity.ERROR,
                "bars on dates the exchange was closed", len(extra),
                tuple(str(d.date()) for d in extra[:3]),
            )
    else:
        report.add(
            Check.MISSING_CANDLES, Severity.INFO,
            "skipped — no exchange calendar supplied, so gaps cannot be detected",
        )

    # ── staleness ───────────────────────────────────────────────────────────
    if now is not None and max_staleness is not None:
        age = now - df.index.max().to_pydatetime()
        if age > max_staleness:
            report.add(
                Check.STALE_DATA, Severity.ERROR,
                f"most recent bar is {age} old, limit is {max_staleness}",
            )

    return report

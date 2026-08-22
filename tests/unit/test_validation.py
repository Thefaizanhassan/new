"""The data validation gate — Technology Standards §1."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading.data.schema import SchemaError, normalize_ohlcv
from trading.data.validation import Check, Severity, validate_ohlcv


@pytest.fixture
def clean_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-01", periods=60, tz="Asia/Kolkata")
    base = np.linspace(1400, 1500, 60)
    return pd.DataFrame(
        {
            "Date": idx,
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": 1_000_000.0,
        }
    )


def _checks(report) -> set[Check]:
    return {i.check for i in report.issues if i.severity is not Severity.INFO}


def test_normalizer_folds_provider_column_spellings(clean_frame):
    df = normalize_ohlcv(clean_frame, tz="Asia/Kolkata")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing


def test_normalizer_rejects_missing_required_columns(clean_frame):
    with pytest.raises(SchemaError, match="Missing required column"):
        normalize_ohlcv(clean_frame.drop(columns=["Volume"]), tz="Asia/Kolkata")


def test_clean_data_is_usable(clean_frame):
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="NSE:RELIANCE")
    assert report.is_usable
    assert not report.errors


def test_invalid_ohlc_relationship_is_an_error(clean_frame):
    clean_frame.loc[5, "High"] = clean_frame.loc[5, "Low"] - 10
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="X")
    assert Check.OHLC_RELATIONSHIP in _checks(report)
    assert not report.is_usable


def test_impossible_price_is_an_error(clean_frame):
    clean_frame.loc[10, "Close"] = -5
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="X")
    assert Check.IMPOSSIBLE_PRICE in _checks(report)


def test_duplicate_timestamps_are_an_error(clean_frame):
    dup = pd.concat([clean_frame, clean_frame.iloc[[3]]], ignore_index=True)
    report = validate_ohlcv(normalize_ohlcv(dup, tz="Asia/Kolkata"), symbol="X")
    assert Check.DUPLICATE_TIMESTAMPS in _checks(report)


def test_zero_and_abnormal_volume_are_warnings_not_errors(clean_frame):
    clean_frame.loc[15, "Volume"] = 0
    clean_frame.loc[40, "Volume"] = 900_000_000
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="X")
    checks = _checks(report)
    assert Check.ZERO_VOLUME in checks
    assert Check.ABNORMAL_VOLUME in checks
    assert report.is_usable, "suspicious volume should flag, not quarantine"


def test_unapplied_split_is_flagged_as_a_suspected_corporate_action(clean_frame):
    clean_frame.loc[30:, ["Open", "High", "Low", "Close"]] /= 4
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="X")
    assert Check.CORPORATE_ACTION in _checks(report)


def test_stale_data_is_an_error_when_a_limit_is_given(clean_frame):
    report = validate_ohlcv(
        normalize_ohlcv(clean_frame, tz="Asia/Kolkata"),
        symbol="X",
        now=datetime(2026, 8, 22, tzinfo=UTC),
        max_staleness=timedelta(days=5),
    )
    assert Check.STALE_DATA in _checks(report)


def test_missing_sessions_are_detected_against_the_calendar(clean_frame):
    df = normalize_ohlcv(clean_frame, tz="Asia/Kolkata")
    sessions = df.index.normalize().unique()
    dropped = df.drop(df.index[10:14])
    report = validate_ohlcv(dropped, symbol="X", expected_sessions=sessions)
    assert Check.MISSING_CANDLES in _checks(report)


def test_bars_on_non_session_dates_are_an_error(clean_frame):
    df = normalize_ohlcv(clean_frame, tz="Asia/Kolkata")
    report = validate_ohlcv(df, symbol="X", expected_sessions=df.index.normalize().unique()[:-5])
    assert Check.SESSION_ALIGNMENT in _checks(report)


def test_missing_candle_check_announces_when_it_is_skipped(clean_frame):
    report = validate_ohlcv(normalize_ohlcv(clean_frame, tz="Asia/Kolkata"), symbol="X")
    skipped = [i for i in report.issues if i.check is Check.MISSING_CANDLES]
    assert skipped and "skipped" in skipped[0].message, "a silent no-op check is worse than none"


def test_empty_dataset_is_an_error():
    report = validate_ohlcv(
        pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC")
        ),
        symbol="X",
    )
    assert not report.is_usable

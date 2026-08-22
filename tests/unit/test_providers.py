"""Data providers, including the yfinance adapter (tested without the network)."""

import datetime as dt
import subprocess
import sys
import textwrap

import pandas as pd
import pytest

from trading.calendars.base import calendar_for
from trading.core.instrument import InstrumentId
from trading.core.types import Timeframe
from trading.data.fixture import FixtureProvider
from trading.data.provider import DataTier
from trading.data.validation import validate_ohlcv
from trading.data.yfinance_provider import (
    UnsupportedExchangeError,
    YFinanceProvider,
    to_yahoo_symbol,
)

RELIANCE = InstrumentId("NSE", "RELIANCE")


# ── fixture provider ───────────────────────────────────────────────────────
def test_fixture_data_passes_the_full_validation_gate():
    start, end = dt.date(2024, 1, 1), dt.date(2025, 12, 31)
    bars = FixtureProvider().get_bars(RELIANCE, start, end)
    sessions = calendar_for("XBOM").sessions_between(start, end)
    report = validate_ohlcv(bars, symbol=str(RELIANCE), expected_sessions=sessions)
    assert report.is_usable
    assert not report.errors and not report.warnings


def test_fixture_is_deterministic_across_separate_processes():
    """Regression: the seed was derived from ``hash()``, which Python randomises
    per process, so identical runs produced different data. Reproducibility is
    the whole point of this project, so this is asserted across interpreters."""
    script = textwrap.dedent("""
        import datetime as dt
        import hashlib

        from trading.core.instrument import InstrumentId
        from trading.data.fixture import FixtureProvider

        bars = FixtureProvider().get_bars(
            InstrumentId("NSE", "RELIANCE"), dt.date(2024, 1, 1), dt.date(2024, 3, 1)
        )
        print(hashlib.sha256(bars.to_csv().encode()).hexdigest())
    """)
    digests = {
        subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(3)
    }
    assert len(digests) == 1, f"fixture data differs between processes: {digests}"


def test_fixture_honours_as_of_by_truncating_history():
    bars = FixtureProvider().get_bars(
        RELIANCE,
        dt.date(2024, 1, 1),
        dt.date(2024, 12, 31),
        as_of=dt.datetime(2024, 6, 30, tzinfo=dt.UTC),
    )
    assert bars.index.max() <= pd.Timestamp("2024-06-30", tz="UTC")


def test_fixture_declares_itself_synthetic():
    info = FixtureProvider().info
    assert info.tier is DataTier.SYNTHETIC
    assert not info.tier.may_support_a_validated_strategy


def test_different_instruments_get_different_series():
    a = FixtureProvider().get_bars(RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 3, 1))
    b = FixtureProvider().get_bars(
        InstrumentId("NSE", "INFY"), dt.date(2024, 1, 1), dt.date(2024, 3, 1)
    )
    assert not a["close"].equals(b["close"])


# ── yfinance adapter ───────────────────────────────────────────────────────
def test_yahoo_symbol_mapping():
    assert to_yahoo_symbol(RELIANCE) == "RELIANCE.NS"
    assert to_yahoo_symbol(InstrumentId("BSE", "RELIANCE")) == "RELIANCE.BO"
    assert to_yahoo_symbol(InstrumentId("NASDAQ", "AAPL")) == "AAPL"


def test_unknown_exchange_is_rejected():
    with pytest.raises(UnsupportedExchangeError):
        to_yahoo_symbol(InstrumentId("LSE", "VOD"))


def test_yfinance_refuses_as_of_rather_than_silently_ignoring_it():
    """It cannot answer point-in-time queries, so it must say so."""
    provider = YFinanceProvider(downloader=lambda *a, **k: pd.DataFrame())
    assert not provider.info.supports_as_of
    with pytest.raises(NotImplementedError, match="point-in-time"):
        provider.get_bars(
            RELIANCE,
            dt.date(2024, 1, 1),
            dt.date(2024, 2, 1),
            as_of=dt.datetime(2024, 1, 15, tzinfo=dt.UTC),
        )


def test_yfinance_is_prototype_tier_and_carries_its_caveats():
    info = YFinanceProvider().info
    assert info.tier is DataTier.PROTOTYPE
    assert not info.tier.may_support_a_validated_strategy
    assert any("SLA" in c for c in info.caveats)
    assert any("Prototype only" in c for c in info.caveats)


def test_yfinance_normalizes_a_realistic_response():
    """Yahoo returns IST-dated bars with title-case columns; daily bars must end
    up UTC and labelled at the NSE close (10:00 UTC = 15:30 IST)."""
    index = pd.DatetimeIndex(
        [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 3)], name="Date"
    )
    raw = pd.DataFrame(
        {
            "Open": [1400.0, 1410.0, 1405.0],
            "High": [1420.0, 1415.0, 1430.0],
            "Low": [1395.0, 1400.0, 1402.0],
            "Close": [1410.0, 1405.0, 1425.0],
            "Adj Close": [1410.0, 1405.0, 1425.0],
            "Volume": [5e6, 4e6, 6e6],
        },
        index=index,
    )
    provider = YFinanceProvider(downloader=lambda *a, **k: raw)
    bars = provider.get_bars(RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    assert list(bars.columns)[:5] == ["open", "high", "low", "close", "volume"]
    assert str(bars.index.tz) == "UTC"
    assert bars.index[0].hour == 10  # NSE close, in UTC
    assert validate_ohlcv(bars, symbol=str(RELIANCE)).is_usable


def test_yfinance_handles_a_multiindex_column_response():
    """yfinance returns MultiIndex columns for some request shapes."""
    index = pd.DatetimeIndex([dt.date(2024, 1, 1), dt.date(2024, 1, 2)], name="Date")
    raw = pd.DataFrame(
        [[1400.0, 1420.0, 1395.0, 1410.0, 5e6], [1410.0, 1415.0, 1400.0, 1405.0, 4e6]],
        index=index,
        columns=pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["RELIANCE.NS"]]
        ),
    )
    bars = YFinanceProvider(downloader=lambda *a, **k: raw).get_bars(
        RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert len(bars) == 2
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_yfinance_returns_an_empty_canonical_frame_when_there_is_no_data():
    bars = YFinanceProvider(downloader=lambda *a, **k: pd.DataFrame()).get_bars(
        RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )
    assert bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_intraday_timeframes_map_to_yahoo_intervals():
    seen = {}

    def downloader(symbol, start, end, interval):
        seen["interval"] = interval
        return pd.DataFrame()

    YFinanceProvider(downloader=downloader).get_bars(
        RELIANCE, dt.date(2024, 1, 1), dt.date(2024, 1, 2), Timeframe.MIN_15
    )
    assert seen["interval"] == "15m"

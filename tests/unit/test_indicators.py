"""TA-Lib indicator layer.

Values are checked against hand-computed references, not against TA-Lib itself —
a test that asserts TA-Lib equals TA-Lib proves nothing.
"""

import numpy as np
import pandas as pd
import pytest

from trading.features import indicators as ind


@pytest.fixture
def frame() -> pd.DataFrame:
    close = np.arange(1.0, 101.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.full(100, 1000.0),
        },
        index=pd.date_range("2024-01-01", periods=100, tz="UTC"),
    )


def test_sma_matches_a_hand_computed_average(frame):
    result = ind.compute("SMA", frame, timeperiod=10)["value"]
    # closes are 1..100, so the 10-bar SMA at index 9 is mean(1..10) = 5.5
    assert result.iloc[9] == pytest.approx(5.5)
    assert result.iloc[-1] == pytest.approx(95.5)


def test_warmup_period_is_nan_and_is_not_silently_dropped(frame):
    result = ind.compute("SMA", frame, timeperiod=10)["value"]
    assert result.iloc[:9].isna().all()
    assert len(result) == len(frame), "output must stay aligned to the input index"


def test_rsi_is_100_for_a_monotonically_rising_series(frame):
    """With no down-closes there are no losses, so RSI pins at its maximum."""
    assert ind.compute("RSI", frame)["value"].iloc[-1] == pytest.approx(100.0)


def test_atr_of_a_constant_range_equals_that_range(frame):
    # high - low is exactly 2 on every bar
    assert ind.compute("ATR", frame)["value"].iloc[-1] == pytest.approx(2.0, abs=0.01)


def test_multi_output_indicators_return_named_columns(frame):
    assert list(ind.compute("MACD", frame).columns) == ["macd", "signal", "hist"]
    assert list(ind.compute("BBANDS", frame).columns) == ["upper", "middle", "lower"]


def test_bollinger_middle_band_is_the_moving_average(frame):
    bands = ind.compute("BBANDS", frame, timeperiod=20)
    sma = ind.compute("SMA", frame, timeperiod=20)["value"]
    assert bands["middle"].iloc[-1] == pytest.approx(sma.iloc[-1])
    assert bands["upper"].iloc[-1] > bands["lower"].iloc[-1]


def test_missing_required_column_is_an_error(frame):
    with pytest.raises(KeyError, match="requires column"):
        ind.compute("ATR", frame.drop(columns=["high"]))


def test_unknown_indicator_lists_what_is_available(frame):
    with pytest.raises(KeyError, match="Known:"):
        ind.compute("SUPERTREND", frame)


def test_every_indicator_documents_the_required_fields():
    """Technology Standards §4: name, purpose, input, parameters, interpretation,
    limitations — for every indicator, not just the ones we happen to explain."""
    for key, spec in ind.INDICATORS.items():
        doc = spec.document()
        for field in ("Purpose", "Input data", "Parameters", "Interpretation", "Limitations"):
            assert field in doc, f"{key} is missing {field}"
        assert len(spec.limitations) > 40, f"{key} has a token limitations note"
        assert spec.inputs, f"{key} declares no inputs"


def test_smoothed_indicators_get_a_longer_warmup_than_their_period():
    """An EMA or RSI is defined after `period` bars but has not converged."""
    assert ind.warmup_for("RSI") > 14
    assert ind.warmup_for("EMA", timeperiod=20) > 20
    assert ind.warmup_for("SMA", timeperiod=20) == 20

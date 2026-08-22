"""Technical indicators, via TA-Lib.

Every indicator carries the documentation your Technology Standards §4 requires
— purpose, inputs, parameters, timeframe, interpretation and limitations —
because an indicator whose limitations are undocumented gets misread, and a
misread indicator becomes a strategy that "works" for reasons nobody can state.

Two boundaries this module enforces:

* **TA-Lib operates on float64 and that is correct here.** Indicators are
  signals, not money. Nothing in this module returns a value that reaches the
  ledger; conversion to ``Decimal`` happens at the strategy boundary.
* **TA-Lib does not protect you from look-ahead bias.** It will happily compute
  an indicator over an array containing future values. The guard is
  ``StrategyContext``, which never holds future bars in the first place.

Warm-up periods return ``NaN`` and are surfaced explicitly rather than dropped:
dropping creates a gap, and a gap makes a strategy skip a session it should
have traded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import talib

__all__ = ["INDICATORS", "IndicatorSpec", "compute", "describe", "warmup_for"]


@dataclass(frozen=True, slots=True)
class IndicatorSpec:
    name: str
    purpose: str
    inputs: tuple[str, ...]
    parameters: dict[str, Any]
    interpretation: str
    limitations: str
    outputs: tuple[str, ...] = ("value",)
    warmup_multiplier: int = 1

    def warmup(self, **params: Any) -> int:
        """Bars needed before the first trustworthy value.

        Deliberately conservative for the smoothed indicators: an EMA or RSI is
        mathematically defined after `period` bars but has not converged, and a
        strategy acting on an unconverged value is trading an artefact of where
        the series happened to start.
        """
        period = int(params.get("timeperiod", self.parameters.get("timeperiod", 1)))
        slow = int(params.get("slowperiod", self.parameters.get("slowperiod", 0)))
        return max(period, slow) * self.warmup_multiplier

    def document(self) -> str:
        lines = [
            f"{self.name}",
            f"  Purpose        : {self.purpose}",
            f"  Input data     : {', '.join(self.inputs)}",
            f"  Parameters     : {self.parameters or 'none'}",
            f"  Outputs        : {', '.join(self.outputs)}",
            f"  Interpretation : {self.interpretation}",
            f"  Limitations    : {self.limitations}",
        ]
        return "\n".join(lines)


INDICATORS: dict[str, IndicatorSpec] = {
    "SMA": IndicatorSpec(
        name="SMA — Simple Moving Average",
        purpose="Smooths price to expose the underlying trend by averaging the last N closes.",
        inputs=("close",),
        parameters={"timeperiod": 20},
        interpretation=(
            "Price above a rising SMA is conventionally read as an uptrend. Crossings of a "
            "fast and slow SMA are the classic trend-following entry and exit."
        ),
        limitations=(
            "Lags price by roughly half the period — it confirms a trend rather than "
            "predicting one. In a sideways market it produces frequent false crossings, "
            "each of which costs a full round trip in fees."
        ),
    ),
    "EMA": IndicatorSpec(
        name="EMA — Exponential Moving Average",
        purpose="A moving average weighting recent prices more heavily, so it turns sooner.",
        inputs=("close",),
        parameters={"timeperiod": 20},
        interpretation="Reacts faster than an SMA of the same period.",
        limitations=(
            "Faster reaction means more whipsaws. Never fully forgets old data, so the "
            "first values after warm-up are biased by wherever the series began."
        ),
        warmup_multiplier=3,
    ),
    "RSI": IndicatorSpec(
        name="RSI — Relative Strength Index",
        purpose="Measures recent momentum on a 0–100 scale from average gains versus losses.",
        inputs=("close",),
        parameters={"timeperiod": 14},
        interpretation=(
            "Above 70 is conventionally 'overbought', below 30 'oversold'. Those thresholds "
            "are convention, not law, and were chosen for 1970s commodity markets."
        ),
        limitations=(
            "In a strong trend RSI stays overbought or oversold for weeks — selling every "
            "RSI>70 in a bull market is a reliable way to lose money. It is a momentum "
            "measure, not a reversal signal."
        ),
        warmup_multiplier=4,
    ),
    "ATR": IndicatorSpec(
        name="ATR — Average True Range",
        purpose=("Measures typical bar-to-bar movement in price units, including overnight gaps."),
        inputs=("high", "low", "close"),
        parameters={"timeperiod": 14},
        interpretation=(
            "Used for volatility-scaled position sizing and for stop distances, so a stop "
            "is placed beyond normal noise rather than at an arbitrary percentage."
        ),
        limitations=(
            "Says nothing about direction. Expressed in price units, so it is not "
            "comparable across instruments without dividing by price."
        ),
        warmup_multiplier=3,
    ),
    "MACD": IndicatorSpec(
        name="MACD — Moving Average Convergence Divergence",
        purpose="Trend and momentum from the gap between a fast and a slow EMA.",
        inputs=("close",),
        parameters={"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        interpretation="The MACD line crossing its signal line is the conventional trigger.",
        limitations=(
            "Two lagging averages compounded, so it lags more than either. The 12/26/9 "
            "defaults come from 1970s daily equity data and carry no special authority."
        ),
        outputs=("macd", "signal", "hist"),
        warmup_multiplier=3,
    ),
    "BBANDS": IndicatorSpec(
        name="Bollinger Bands",
        purpose="Bands at N standard deviations around a moving average.",
        inputs=("close",),
        parameters={"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
        interpretation=(
            "Band width shows volatility. Touching a band is often read as a mean-reversion "
            "signal — with the caveat below."
        ),
        limitations=(
            "Assumes roughly normal returns, which markets are not: prices touch the bands "
            "far more often than the maths implies. In a trend, price can ride a band for "
            "weeks while every reversion trade loses."
        ),
        outputs=("upper", "middle", "lower"),
    ),
    "ADX": IndicatorSpec(
        name="ADX — Average Directional Index",
        purpose="Measures trend strength, 0–100, regardless of direction.",
        inputs=("high", "low", "close"),
        parameters={"timeperiod": 14},
        interpretation=(
            "Above ~25 conventionally indicates a trend worth following; below ~20 suggests "
            "a range where trend strategies bleed. A useful regime filter."
        ),
        limitations=(
            "Heavily smoothed and therefore slow — it confirms a trend well after it began."
        ),
        warmup_multiplier=4,
    ),
    "OBV": IndicatorSpec(
        name="OBV — On Balance Volume",
        purpose="Cumulative volume, added on up-closes and subtracted on down-closes.",
        inputs=("close", "volume"),
        parameters={},
        interpretation="Divergence between OBV and price is read as weakening conviction.",
        limitations=(
            "An unbounded cumulative sum, so its absolute level is meaningless and only "
            "its slope carries information. Assumes a bar's entire volume shares the "
            "direction of its close, which is a crude approximation."
        ),
    ),
}


def describe(name: str | None = None) -> str:
    """Human-readable documentation for one indicator or all of them."""
    if name:
        return INDICATORS[name.upper()].document()
    return "\n\n".join(spec.document() for spec in INDICATORS.values())


def warmup_for(name: str, **params: Any) -> int:
    return INDICATORS[name.upper()].warmup(**params)


def compute(name: str, frame: pd.DataFrame, **params: Any) -> pd.DataFrame:
    """Compute an indicator over a canonical OHLCV frame.

    Returns a DataFrame aligned to ``frame.index``, with ``NaN`` through the
    warm-up period. Callers must handle those explicitly.
    """
    key = name.upper()
    if key not in INDICATORS:
        raise KeyError(f"Unknown indicator {name!r}. Known: {sorted(INDICATORS)}")
    spec = INDICATORS[key]

    missing = [c for c in spec.inputs if c not in frame.columns]
    if missing:
        raise KeyError(f"{key} requires column(s) {missing}")

    args = [np.asarray(frame[c], dtype="float64") for c in spec.inputs]
    kwargs = {**spec.parameters, **params}
    result = getattr(talib, key)(*args, **kwargs)

    if isinstance(result, tuple):
        return pd.DataFrame(dict(zip(spec.outputs, result, strict=True)), index=frame.index)
    return pd.DataFrame({spec.outputs[0]: result}, index=frame.index)

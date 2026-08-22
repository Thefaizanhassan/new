"""Corporate actions and price adjustment.

When a company splits 4:1 the share price quarters overnight.  A raw price
series shows a −75% crash, and a strategy fires a signal on a non-event.
Adjusted prices restate history as if the split had always applied.

The rule this module exists to enforce (Phase 0 §2.4):

    **Raw prices are stored immutably. Adjusted series are computed on read,
    never written back.**

Two reasons that matters:

* You need *both*. Signals and returns need adjusted prices so history is
  continuous; order placement and position accounting need raw prices, because
  you buy real shares at real prices. Storing only one is a bug you find months
  later.
* Adjusted history is rewritten by every new corporate action. If it were the
  stored truth, every past backtest would silently change. Keeping raw
  immutable and deriving adjusted on read means an old run stays reproducible.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

import pandas as pd

from trading.core.instrument import InstrumentId

__all__ = [
    "ActionType",
    "AdjustmentMode",
    "CorporateAction",
    "CorporateActionStore",
    "adjust_bars",
    "split_factors",
]


class ActionType(StrEnum):
    SPLIT = "SPLIT"
    """Share count multiplies, price divides. Economically neutral."""

    BONUS = "BONUS"
    """Free shares issued pro rata. Mechanically identical to a split — a 1:1
    bonus doubles the share count, exactly like a 2:1 split. Common in India."""

    DIVIDEND = "DIVIDEND"
    """Cash paid to holders. The price drops by roughly the dividend on the
    ex-date, which is a real return to a holder but looks like a loss in a
    price-only series."""


class AdjustmentMode(StrEnum):
    RAW = "RAW"
    """Untouched. What actually traded. Use for order pricing and accounting."""

    SPLIT_ONLY = "SPLIT_ONLY"
    """Splits and bonuses applied. Continuous price history without treating
    dividends as returns. The sane default for technical signals."""

    TOTAL_RETURN = "TOTAL_RETURN"
    """Splits and dividends applied. Use when measuring what a holder actually
    earned — otherwise a high-dividend stock looks like it went nowhere."""


@dataclass(frozen=True, slots=True)
class CorporateAction:
    instrument_id: InstrumentId
    action_type: ActionType
    ex_date: dt.date
    ratio: Decimal = Decimal(1)
    """For SPLIT/BONUS: shares after per share before. A 4:1 split is 4;
    a 1:1 bonus is 2."""
    amount: Decimal = Decimal(0)
    """For DIVIDEND: cash per share, in the instrument's currency."""
    source: str = "unknown"

    def __post_init__(self) -> None:
        if self.action_type in (ActionType.SPLIT, ActionType.BONUS):
            if self.ratio <= 0:
                raise ValueError(f"{self.action_type} ratio must be positive, got {self.ratio}")
            if self.ratio == 1:
                raise ValueError(f"{self.action_type} with ratio 1 is not an action")
        elif self.action_type is ActionType.DIVIDEND and self.amount <= 0:
            raise ValueError(f"DIVIDEND amount must be positive, got {self.amount}")

    def __str__(self) -> str:
        if self.action_type is ActionType.DIVIDEND:
            return f"{self.ex_date} DIVIDEND {self.amount}/share"
        return f"{self.ex_date} {self.action_type} {self.ratio}:1"


def split_factors(index: pd.DatetimeIndex, actions: list[CorporateAction]) -> pd.Series:
    """Cumulative split factor for each bar.

    A bar's factor is the product of every split/bonus ratio taking effect
    *after* it.  Dividing raw prices by this factor puts the whole series on
    today's share basis.

    Bars on or after the last split have a factor of 1 and are unchanged, which
    is the property that makes adjusted and raw agree at the right-hand edge.
    """
    factors = pd.Series(1.0, index=index)
    for action in actions:
        if action.action_type not in (ActionType.SPLIT, ActionType.BONUS):
            continue
        ex = pd.Timestamp(action.ex_date, tz="UTC")
        factors.loc[index < ex] *= float(action.ratio)
    return factors


def _dividend_factors(
    index: pd.DatetimeIndex, closes: pd.Series, actions: list[CorporateAction]
) -> pd.Series:
    """Cumulative dividend factor for each bar.

    For an ex-date with dividend ``A`` and prior close ``C``, the factor is
    ``1 − A/C``. Bars before that date are multiplied by it, which folds the
    dividend back into the price series as if reinvested.
    """
    factors = pd.Series(1.0, index=index)
    for action in actions:
        if action.action_type is not ActionType.DIVIDEND:
            continue
        ex = pd.Timestamp(action.ex_date, tz="UTC")
        prior = closes.loc[index < ex]
        if prior.empty:
            continue
        prior_close = float(prior.iloc[-1])
        if prior_close <= 0:
            continue
        ratio = 1.0 - float(action.amount) / prior_close
        if ratio <= 0:
            # A dividend at or above the prior close is almost certainly a data
            # error rather than a real distribution. Skip rather than produce a
            # negative price series.
            continue
        factors.loc[index < ex] *= ratio
    return factors


def adjust_bars(
    bars: pd.DataFrame,
    actions: list[CorporateAction],
    mode: AdjustmentMode = AdjustmentMode.SPLIT_ONLY,
) -> pd.DataFrame:
    """Return an adjusted copy of ``bars``. The input is never modified."""
    if mode is AdjustmentMode.RAW or not actions or bars.empty:
        return bars.copy()

    index = bars.index
    adjusted = bars.copy()
    price_columns = [c for c in ("open", "high", "low", "close") if c in adjusted.columns]

    splits = split_factors(index, actions)
    if (splits != 1.0).any():
        for column in price_columns:
            adjusted[column] = adjusted[column] / splits
        if "volume" in adjusted.columns:
            # Share count moved the other way, so volume scales up.
            adjusted["volume"] = adjusted["volume"] * splits

    if mode is AdjustmentMode.TOTAL_RETURN:
        dividends = _dividend_factors(index, bars["close"], actions)
        if (dividends != 1.0).any():
            for column in price_columns:
                adjusted[column] = adjusted[column] * dividends

    return adjusted


class CorporateActionStore:
    """Actions on disk, one Parquet file per instrument."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root) / "corporate_actions"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, instrument_id: InstrumentId) -> Path:
        directory = self.root / instrument_id.exchange
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{instrument_id.symbol}.parquet"

    def save(self, instrument_id: InstrumentId, actions: list[CorporateAction]) -> Path:
        path = self._path(instrument_id)
        frame = pd.DataFrame(
            [
                {
                    "instrument_id": str(a.instrument_id),
                    "action_type": str(a.action_type),
                    "ex_date": a.ex_date,
                    "ratio": float(a.ratio),
                    "amount": float(a.amount),
                    "source": a.source,
                }
                for a in actions
            ]
        )
        frame.to_parquet(path, index=False)
        return path

    def load(self, instrument_id: InstrumentId) -> list[CorporateAction]:
        path = self._path(instrument_id)
        if not path.exists():
            return []
        frame = pd.read_parquet(path)
        actions = [
            CorporateAction(
                instrument_id=InstrumentId.parse(row["instrument_id"]),
                action_type=ActionType(row["action_type"]),
                ex_date=pd.Timestamp(row["ex_date"]).date(),
                ratio=Decimal(str(row["ratio"])),
                amount=Decimal(str(row["amount"])),
                source=str(row["source"]),
            )
            for _, row in frame.iterrows()
        ]
        return sorted(actions, key=lambda a: a.ex_date)

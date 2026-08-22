"""Corporate actions and price adjustment."""

import datetime as dt
from decimal import Decimal

import pandas as pd
import pytest

from trading.core.instrument import InstrumentId
from trading.data.corporate_actions import (
    ActionType,
    AdjustmentMode,
    CorporateAction,
    CorporateActionStore,
    adjust_bars,
)

RELIANCE = InstrumentId("NSE", "RELIANCE")


@pytest.fixture
def bars() -> pd.DataFrame:
    """Steady at 2000, then a raw 4:1 split drops it to 500 on the sixth bar."""
    index = pd.date_range("2024-01-01", periods=10, tz="UTC")
    return pd.DataFrame(
        {
            "open": [2000.0] * 5 + [500.0] * 5,
            "high": [2010.0] * 5 + [502.5] * 5,
            "low": [1990.0] * 5 + [497.5] * 5,
            "close": [2000.0] * 5 + [500.0] * 5,
            "volume": [1000.0] * 5 + [4000.0] * 5,
        },
        index=index,
    )


@pytest.fixture
def split() -> CorporateAction:
    return CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 1, 6), ratio=Decimal(4))


def test_split_adjustment_removes_the_phantom_crash(bars, split):
    raw_move = bars["close"].pct_change().iloc[5]
    adjusted = adjust_bars(bars, [split], AdjustmentMode.SPLIT_ONLY)
    assert raw_move == pytest.approx(-0.75)
    assert adjusted["close"].pct_change().iloc[5] == pytest.approx(0.0)
    assert adjusted["close"].nunique() == 1


def test_split_adjustment_scales_volume_the_other_way(bars, split):
    adjusted = adjust_bars(bars, [split], AdjustmentMode.SPLIT_ONLY)
    assert adjusted["volume"].nunique() == 1


def test_the_right_hand_edge_is_never_adjusted(bars, split):
    """Bars after the last action are already on today's basis."""
    adjusted = adjust_bars(bars, [split], AdjustmentMode.SPLIT_ONLY)
    assert adjusted["close"].iloc[-1] == bars["close"].iloc[-1]


def test_adjustment_never_mutates_the_input(bars, split):
    before = bars["close"].copy()
    adjust_bars(bars, [split], AdjustmentMode.SPLIT_ONLY)
    assert bars["close"].equals(before), "raw prices must remain immutable"


def test_raw_mode_is_a_no_op(bars, split):
    assert adjust_bars(bars, [split], AdjustmentMode.RAW)["close"].equals(bars["close"])


def test_no_actions_is_a_no_op(bars):
    assert adjust_bars(bars, [], AdjustmentMode.SPLIT_ONLY)["close"].equals(bars["close"])


def test_bonus_issue_adjusts_like_a_split(bars):
    bonus = CorporateAction(RELIANCE, ActionType.BONUS, dt.date(2024, 1, 6), ratio=Decimal(4))
    assert adjust_bars(bars, [bonus], AdjustmentMode.SPLIT_ONLY)["close"].equals(
        adjust_bars(
            bars,
            [CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 1, 6), ratio=Decimal(4))],
            AdjustmentMode.SPLIT_ONLY,
        )["close"]
    )


def test_dividend_adjustment_scales_prior_prices_by_the_yield(bars):
    dividend = CorporateAction(
        RELIANCE, ActionType.DIVIDEND, dt.date(2024, 1, 6), amount=Decimal(20)
    )
    adjusted = adjust_bars(bars, [dividend], AdjustmentMode.TOTAL_RETURN)
    # ₹20 against a prior close of ₹2000 is a 1% factor
    assert adjusted["close"].iloc[0] == pytest.approx(1980.0)
    assert adjusted["close"].iloc[-1] == bars["close"].iloc[-1]


def test_split_only_mode_ignores_dividends(bars):
    dividend = CorporateAction(
        RELIANCE, ActionType.DIVIDEND, dt.date(2024, 1, 6), amount=Decimal(20)
    )
    assert adjust_bars(bars, [dividend], AdjustmentMode.SPLIT_ONLY)["close"].equals(bars["close"])


def test_multiple_splits_compound(bars):
    actions = [
        CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 1, 4), ratio=Decimal(2)),
        CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 1, 8), ratio=Decimal(5)),
    ]
    adjusted = adjust_bars(bars, actions, AdjustmentMode.SPLIT_ONLY)
    assert adjusted["close"].iloc[0] == pytest.approx(2000.0 / 10)


def test_an_absurd_dividend_is_skipped_rather_than_producing_negative_prices(bars):
    absurd = CorporateAction(
        RELIANCE, ActionType.DIVIDEND, dt.date(2024, 1, 6), amount=Decimal(5000)
    )
    adjusted = adjust_bars(bars, [absurd], AdjustmentMode.TOTAL_RETURN)
    assert (adjusted["close"] > 0).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"action_type": ActionType.SPLIT, "ratio": Decimal(0)}, "positive"),
        ({"action_type": ActionType.SPLIT, "ratio": Decimal(1)}, "not an action"),
        ({"action_type": ActionType.DIVIDEND, "amount": Decimal(0)}, "positive"),
    ],
)
def test_nonsensical_actions_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CorporateAction(RELIANCE, ex_date=dt.date(2024, 1, 1), **kwargs)


def test_action_store_round_trips_and_sorts_by_ex_date(tmp_path):
    store = CorporateActionStore(tmp_path)
    actions = [
        CorporateAction(RELIANCE, ActionType.SPLIT, dt.date(2024, 6, 1), ratio=Decimal(2)),
        CorporateAction(RELIANCE, ActionType.DIVIDEND, dt.date(2024, 1, 1), amount=Decimal("10.5")),
    ]
    store.save(RELIANCE, actions)
    loaded = store.load(RELIANCE)

    assert [a.ex_date for a in loaded] == [dt.date(2024, 1, 1), dt.date(2024, 6, 1)]
    assert loaded[0].amount == Decimal("10.5")
    assert loaded[1].ratio == Decimal(2)


def test_loading_an_instrument_with_no_actions_returns_empty(tmp_path):
    assert CorporateActionStore(tmp_path).load(RELIANCE) == []

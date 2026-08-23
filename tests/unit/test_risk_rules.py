"""Pre-trade risk rules.

The regression tests here matter as much as the happy paths: a risk rule that
blocks you from *closing* a position is more dangerous than no rule at all,
because it traps you in the trade.
"""

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trading.config.compliance import INDIA_SEBI, US_RETAIL
from trading.core.fill import CostBreakdown, Fill
from trading.core.instrument import Instrument, InstrumentId
from trading.core.order import OrderRequest, OrderType
from trading.core.position import Position
from trading.core.types import Currency, InstrumentClass, Money, Side, TradingMode
from trading.risk.context import OrderRecord, PortfolioView, RiskContext
from trading.risk.engine import RiskEngine
from trading.risk.profiles import RiskLimits, load_risk_profile
from trading.risk.rules import (
    BuyingPowerRule,
    CorrelatedExposureRule,
    DataStalenessRule,
    DuplicateOrderRule,
    KillSwitchRule,
    LimitPriceSanityRule,
    LiquidityRule,
    MarketSessionRule,
    MaxOpenPositionsRule,
    MaxPositionWeightRule,
    OrderRateRule,
    PatternDayTraderRule,
    SymbolUniverseRule,
)

NOW = dt.datetime(2026, 8, 21, 10, 0, tzinfo=dt.UTC)
RELIANCE = InstrumentId("NSE", "RELIANCE")
TCS = InstrumentId("NSE", "TCS")


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=RELIANCE,
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


def make_position(instrument: Instrument, quantity: str, price: str) -> Position:
    return Position(instrument=instrument).apply(
        Fill.create(
            client_order_id="c",
            instrument=instrument,
            side=Side.BUY,
            quantity=Decimal(quantity),
            price=Decimal(price),
            costs=CostBreakdown.zero(Currency.INR),
            timestamp=NOW,
        )
    )


def context(
    instrument: Instrument,
    *,
    side: Side = Side.BUY,
    quantity: str = "50",
    price: str = "1400",
    equity: str = "400000",
    cash: str = "400000",
    positions=None,
    marks=None,
    **kwargs,
) -> RiskContext:
    portfolio = PortfolioView(
        equity=Money.inr(equity),
        cash=Money.inr(cash),
        positions=positions or {},
        marks=marks or {},
        day_start_equity=Money.inr(equity),
        peak_equity=Money.inr(equity),
    )
    return RiskContext(
        request=OrderRequest(
            instrument=instrument,
            side=side,
            quantity=Decimal(quantity),
            **{
                k: v for k, v in kwargs.items() if k in ("order_type", "limit_price", "strategy_id")
            },
        ),
        reference_price=Decimal(price),
        portfolio=portfolio,
        now=NOW,
        mode=kwargs.get("mode", TradingMode.BACKTEST),
        **{
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "session_open",
                "data_age",
                "average_daily_volume",
                "returns",
                "recent_orders",
                "day_trades_in_window",
            )
        },
    )


# ── the regression that matters most ───────────────────────────────────────
def test_closing_a_position_is_never_blocked_by_the_position_limit(instrument):
    """A rule that stops you exiting is worse than no rule.

    Regression: exposure rules added the order's value to the existing position
    regardless of side, so a sell that closed a long read as doubling it — and
    every exit was rejected, trapping the position.
    """
    position = make_position(instrument, "100", "1400")
    ctx = context(
        instrument,
        side=Side.SELL,
        quantity="100",
        price="1500",
        positions={RELIANCE: position},
        marks={RELIANCE: Decimal("1500")},
    )
    # The position is 150000/400000 = 37.5% of equity, well over the 25% cap...
    assert abs(ctx.portfolio.market_value(RELIANCE).amount) / 400000 > Decimal("0.25")
    # ...but closing it projects to zero, so it must pass.
    assert ctx.projected_position_value() == 0
    assert MaxPositionWeightRule(RiskLimits()).evaluate(ctx).passed


def test_adding_to_an_oversized_position_is_still_blocked(instrument):
    position = make_position(instrument, "100", "1400")
    ctx = context(
        instrument,
        side=Side.BUY,
        quantity="50",
        price="1500",
        positions={RELIANCE: position},
        marks={RELIANCE: Decimal("1500")},
    )
    assert not MaxPositionWeightRule(RiskLimits()).evaluate(ctx).passed


def test_closing_reduces_the_open_position_count(instrument):
    position = make_position(instrument, "100", "1400")
    ctx = context(
        instrument,
        side=Side.SELL,
        quantity="100",
        price="1400",
        positions={RELIANCE: position},
        marks={RELIANCE: Decimal("1400")},
    )
    assert ctx.projected_open_positions() == 0


# ── individual rules ───────────────────────────────────────────────────────
def test_kill_switch_blocks_when_the_sentinel_exists(instrument, tmp_path):
    sentinel = tmp_path / "KILL"
    ctx = context(instrument)
    assert KillSwitchRule(sentinel).evaluate(ctx).passed
    sentinel.write_text("halt")
    assert not KillSwitchRule(sentinel).evaluate(ctx).passed


def test_market_session_fails_closed_when_the_calendar_was_not_consulted(instrument):
    live = context(instrument, mode=TradingMode.PAPER, session_open=None)
    assert not MarketSessionRule().evaluate(live).passed
    assert "failing closed" in MarketSessionRule().evaluate(live).detail


def test_market_session_is_not_enforced_in_a_backtest(instrument):
    assert MarketSessionRule().evaluate(context(instrument, session_open=None)).passed


def test_stale_data_blocks_in_live_modes(instrument):
    limits = RiskLimits(max_data_age_seconds=60)
    fresh = context(instrument, mode=TradingMode.PAPER, data_age=dt.timedelta(seconds=10))
    stale = context(instrument, mode=TradingMode.PAPER, data_age=dt.timedelta(seconds=600))
    assert DataStalenessRule(limits).evaluate(fresh).passed
    assert not DataStalenessRule(limits).evaluate(stale).passed


def test_unmeasured_data_age_fails_closed(instrument):
    ctx = context(instrument, mode=TradingMode.PAPER, data_age=None)
    assert not DataStalenessRule(RiskLimits()).evaluate(ctx).passed


def test_a_limit_price_far_from_the_market_is_rejected(instrument):
    sane = context(instrument, order_type=OrderType.LIMIT, limit_price=Decimal("1420"))
    fat_finger = context(instrument, order_type=OrderType.LIMIT, limit_price=Decimal("14000"))
    assert LimitPriceSanityRule(RiskLimits()).evaluate(sane).passed
    assert not LimitPriceSanityRule(RiskLimits()).evaluate(fat_finger).passed


def test_market_orders_skip_the_limit_price_check(instrument):
    assert LimitPriceSanityRule(RiskLimits()).evaluate(context(instrument)).passed


def test_buying_power_blocks_a_buy_but_never_a_sell(instrument):
    poor = context(instrument, cash="10000")
    assert not BuyingPowerRule().evaluate(poor).passed
    selling = context(instrument, side=Side.SELL, cash="10000")
    assert BuyingPowerRule().evaluate(selling).passed


def test_liquidity_blocks_an_order_that_would_move_the_price(instrument):
    limits = RiskLimits(max_adv_participation=Decimal("0.01"))
    small = context(instrument, quantity="50", average_daily_volume=Decimal("1000000"))
    huge = context(instrument, quantity="50000", average_daily_volume=Decimal("1000000"))
    assert LiquidityRule(limits).evaluate(small).passed
    assert not LiquidityRule(limits).evaluate(huge).passed


def test_liquidity_is_not_enforced_without_volume_data(instrument):
    result = LiquidityRule(RiskLimits()).evaluate(context(instrument, average_daily_volume=None))
    assert result.passed
    assert "unavailable" in result.detail


def test_a_denied_symbol_is_rejected(instrument):
    limits = RiskLimits(denied_symbols=("NSE:RELIANCE",))
    assert not SymbolUniverseRule(limits).evaluate(context(instrument)).passed


def test_an_allowlist_is_exhaustive(instrument):
    limits = RiskLimits(allowed_symbols=("NSE:TCS",))
    result = SymbolUniverseRule(limits).evaluate(context(instrument))
    assert not result.passed
    assert "exhaustive" in result.detail


def test_duplicate_orders_within_the_window_are_caught(instrument):
    recent = (OrderRecord("coid_1", RELIANCE, "BUY", Decimal("50"), NOW - dt.timedelta(seconds=5)),)
    ctx = context(instrument, recent_orders=recent)
    assert not DuplicateOrderRule().evaluate(ctx).passed


def test_an_older_identical_order_is_not_a_duplicate(instrument):
    recent = (OrderRecord("coid_1", RELIANCE, "BUY", Decimal("50"), NOW - dt.timedelta(hours=3)),)
    assert DuplicateOrderRule().evaluate(context(instrument, recent_orders=recent)).passed


def test_max_open_positions_counts_the_projected_book(instrument):
    limits = RiskLimits(max_open_positions=1)
    other = Instrument(
        id=TCS, name="TCS", instrument_class=InstrumentClass.EQUITY, currency=Currency.INR
    )
    ctx = context(
        instrument,
        positions={TCS: make_position(other, "10", "3000")},
        marks={TCS: Decimal("3000")},
    )
    assert not MaxOpenPositionsRule(limits).evaluate(ctx).passed


# ── correlation ────────────────────────────────────────────────────────────
def test_correlated_positions_consume_more_of_the_exposure_budget(instrument):
    """Two 20% positions in names that move together are one 40% bet."""
    limits = RiskLimits(max_correlated_exposure=Decimal("0.30"))
    other = Instrument(
        id=TCS, name="TCS", instrument_class=InstrumentClass.EQUITY, currency=Currency.INR
    )
    rng = np.random.default_rng(3)
    shared = rng.normal(0, 0.01, 200)
    correlated = pd.DataFrame({str(RELIANCE): shared, str(TCS): shared})
    independent = pd.DataFrame({str(RELIANCE): shared, str(TCS): rng.normal(0, 0.01, 200)})
    kwargs = {
        "positions": {TCS: make_position(other, "27", "3000")},
        "marks": {TCS: Decimal("3000")},
        "quantity": "57",
    }
    assert (
        not CorrelatedExposureRule(limits)
        .evaluate(context(instrument, returns=correlated, **kwargs))
        .passed
    )
    assert (
        CorrelatedExposureRule(limits)
        .evaluate(context(instrument, returns=independent, **kwargs))
        .passed
    )


def test_without_return_history_positions_are_assumed_correlated(instrument):
    """Assuming unmeasured diversification is how a concentrated book passes."""
    result = CorrelatedExposureRule(RiskLimits()).evaluate(context(instrument, returns=None))
    assert "assumed fully correlated" in result.detail


# ── compliance as risk ─────────────────────────────────────────────────────
def test_sebi_order_rate_blocks_above_ten_per_second(instrument):
    burst = tuple(
        OrderRecord(f"c{i}", RELIANCE, "BUY", Decimal("1"), NOW - dt.timedelta(milliseconds=100))
        for i in range(10)
    )
    assert not OrderRateRule(INDIA_SEBI).evaluate(context(instrument, recent_orders=burst)).passed
    assert OrderRateRule(INDIA_SEBI).evaluate(context(instrument, recent_orders=burst[:3])).passed


def test_orders_outside_the_one_second_window_do_not_count(instrument):
    old = tuple(
        OrderRecord(f"c{i}", RELIANCE, "BUY", Decimal("1"), NOW - dt.timedelta(seconds=5))
        for i in range(20)
    )
    assert OrderRateRule(INDIA_SEBI).evaluate(context(instrument, recent_orders=old)).passed


def test_pdt_blocks_a_small_us_account_after_three_day_trades(instrument):
    below = context(instrument, equity="10000", day_trades_in_window=3)
    assert not PatternDayTraderRule(US_RETAIL).evaluate(below).passed
    above = context(instrument, equity="30000", day_trades_in_window=10)
    assert PatternDayTraderRule(US_RETAIL).evaluate(above).passed


def test_pdt_does_not_apply_in_india(instrument):
    ctx = context(instrument, equity="10000", day_trades_in_window=50)
    assert PatternDayTraderRule(INDIA_SEBI).evaluate(ctx).passed


# ── engine behaviour ───────────────────────────────────────────────────────
def test_a_rule_that_raises_becomes_a_rejection_not_a_skip(instrument):
    class Exploding:
        rule_id = "RISK_TEST_explodes"

        def evaluate(self, ctx):
            raise RuntimeError("database unreachable")

    engine = RiskEngine(RiskLimits(), INDIA_SEBI, rules=[Exploding()])
    decision = engine.evaluate(context(instrument))
    assert not decision.approved
    assert "failing closed" in decision.rejection_reason


def test_every_rule_records_its_observed_value_and_limit(instrument, tmp_path):
    engine = RiskEngine(RiskLimits(), INDIA_SEBI, kill_switch_path=tmp_path / "none")
    decision = engine.evaluate(context(instrument))
    assert decision.approved
    assert len(decision.results) >= 20
    for result in decision.results:
        assert result.rule_id and result.observed and result.limit


def test_profiles_load_from_yaml_and_reach_the_rules(instrument, tmp_path):
    limits = load_risk_profile("configs/risk/conservative.yaml")
    engine = RiskEngine(limits, INDIA_SEBI, kill_switch_path=tmp_path / "none")
    # 50 shares at 1400 is 17.5% of equity — over the conservative 10% cap.
    assert not engine.evaluate(context(instrument)).approved


def test_unknown_keys_in_a_risk_profile_are_errors(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("profile_id: x\nmax_positon_weight: '0.2'\n")
    with pytest.raises(ValueError, match=r"max_positon_weight|Extra inputs"):
        load_risk_profile(path)

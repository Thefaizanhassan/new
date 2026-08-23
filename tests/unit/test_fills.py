"""Fill and slippage models — where backtests lie."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.backtest.fills import (
    FillRequest,
    FixedBpsSlippage,
    NoSlippage,
    RealisticFillModel,
    SpreadSlippage,
    SquareRootImpact,
)
from trading.core.instrument import Instrument, InstrumentId
from trading.core.order import Order, OrderRequest, OrderType
from trading.core.risk import RiskDecision, RiskRuleResult
from trading.core.types import Currency, InstrumentClass, Side

NOW = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


def make_order(instrument, quantity="1000", side=Side.BUY, order_type=OrderType.MARKET, limit=None):
    decision = RiskDecision.approve("t", NOW, (RiskRuleResult("r", True, "1", "2"),))
    return Order.create(
        OrderRequest(
            instrument=instrument,
            side=side,
            quantity=Decimal(quantity),
            order_type=order_type,
            limit_price=limit,
        ),
        decision,
        NOW,
    )


def request(order, volume="1000000", **kwargs) -> FillRequest:
    defaults = {
        "open_price": Decimal("1400"),
        "high_price": Decimal("1420"),
        "low_price": Decimal("1390"),
        "close_price": Decimal("1410"),
    }
    return FillRequest(order=order, volume=Decimal(volume), **{**defaults, **kwargs})


# ── slippage ───────────────────────────────────────────────────────────────
def test_no_slippage_is_exactly_zero(instrument):
    assert NoSlippage().slippage(request(make_order(instrument)), Decimal("1000")) == 0


def test_fixed_bps_is_a_flat_fraction_of_price(instrument):
    slip = FixedBpsSlippage(Decimal("10")).slippage(request(make_order(instrument)), Decimal(1))
    assert slip == Decimal("1400") * Decimal("10") / Decimal(10_000)


def test_half_spread_is_charged_per_side(instrument):
    """A round trip pays the full spread — half on the way in, half on the way out."""
    model = SpreadSlippage(Decimal("20"))
    per_side = model.slippage(request(make_order(instrument)), Decimal(1))
    assert per_side == Decimal("1400") * Decimal("20") / Decimal(20_000)


def test_impact_grows_with_the_square_root_of_size(instrument):
    """Doubling an order costs about 1.4x more, not 2x — the standard model."""
    model = SquareRootImpact()
    small = model.slippage(request(make_order(instrument)), Decimal("1000"))
    large = model.slippage(request(make_order(instrument)), Decimal("100000"))
    assert large > small
    # 100x the size gives ~10x the impact component, not 100x
    assert large < small * 20


def test_impact_falls_back_to_the_spread_when_there_is_no_volume(instrument):
    model = SquareRootImpact()
    assert model.slippage(request(make_order(instrument), volume="0"), Decimal(1)) > 0


# ── direction ──────────────────────────────────────────────────────────────
def test_slippage_always_works_against_you(instrument):
    """A buy fills above the open, a sell below it. That is what makes it slippage."""
    model = RealisticFillModel()
    buy = model.fill(request(make_order(instrument, side=Side.BUY)))
    sell = model.fill(request(make_order(instrument, side=Side.SELL)))
    assert buy.fill_price > Decimal("1400")
    assert sell.fill_price < Decimal("1400")


def test_a_fill_can_never_print_outside_the_bar_range(instrument):
    """A bar is a lossy summary; you cannot trade at a price it never printed."""
    model = RealisticFillModel(slippage_model=FixedBpsSlippage(Decimal("100000")))
    outcome = model.fill(request(make_order(instrument)))
    assert Decimal("1390") <= outcome.fill_price <= Decimal("1420")


# ── volume ─────────────────────────────────────────────────────────────────
def test_a_large_order_is_capped_at_a_share_of_bar_volume(instrument):
    """Prevents a backtest buying size the market never had."""
    model = RealisticFillModel(max_participation=Decimal("0.05"))
    outcome = model.fill(request(make_order(instrument, "100000"), volume="1000000"))
    assert outcome.filled_quantity == Decimal("50000")
    assert outcome.partial
    assert "capped" in outcome.reason


def test_a_small_order_fills_completely(instrument):
    outcome = RealisticFillModel().fill(request(make_order(instrument, "1000")))
    assert outcome.filled_quantity == Decimal("1000")
    assert not outcome.partial


def test_a_bar_with_no_volume_produces_no_fill(instrument):
    outcome = RealisticFillModel().fill(request(make_order(instrument), volume="0"))
    assert not outcome.filled
    assert "did not trade" in outcome.reason


# ── limit orders ───────────────────────────────────────────────────────────
def test_a_limit_the_bar_never_touched_does_not_fill(instrument):
    order = make_order(instrument, "100", order_type=OrderType.LIMIT, limit=Decimal("1300"))
    outcome = RealisticFillModel().fill(request(order))
    assert not outcome.filled
    assert "not touched" in outcome.reason


def test_touching_a_limit_price_is_not_being_filled_at_it(instrument):
    """Someone was ahead of you in the queue. Without this haircut, limit-order
    backtests are systematically optimistic."""
    order = make_order(instrument, "100", order_type=OrderType.LIMIT, limit=Decimal("1395"))
    outcome = RealisticFillModel(limit_fill_probability=Decimal("0.5")).fill(request(order))
    assert outcome.filled
    assert outcome.filled_quantity == Decimal("50")
    assert outcome.fill_price == Decimal("1395")
    assert "queue" in outcome.reason


def test_a_sell_limit_fills_when_the_high_reaches_it(instrument):
    order = make_order(
        instrument, "100", side=Side.SELL, order_type=OrderType.LIMIT, limit=Decimal("1415")
    )
    assert RealisticFillModel().fill(request(order)).filled


def test_the_model_describes_its_assumptions_for_the_manifest():
    described = RealisticFillModel().describe()
    assert described["fill_model"] == "realistic"
    assert "max_participation" in described
    assert any(key.startswith("slippage_") for key in described)

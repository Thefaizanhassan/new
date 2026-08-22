"""Position accounting — the four cases, including the flip that naive code gets wrong."""

from decimal import Decimal

from trading.core.fill import CostBreakdown
from trading.core.position import Position
from trading.core.types import Currency, Money, Side


def test_open_from_flat(reliance, make_fill):
    p = Position(instrument=reliance).apply(make_fill(Side.BUY, "10", "100"))
    assert p.quantity == Decimal(10)
    assert p.average_price == Decimal(100)
    assert p.realized.is_zero


def test_adding_uses_weighted_average_cost(reliance, make_fill):
    p = Position(instrument=reliance)
    p = p.apply(make_fill(Side.BUY, "10", "100")).apply(make_fill(Side.BUY, "10", "110"))
    assert p.quantity == Decimal(20)
    assert p.average_price == Decimal(105)


def test_partial_close_realizes_only_the_closed_portion(reliance, make_fill):
    p = Position(instrument=reliance)
    p = p.apply(make_fill(Side.BUY, "20", "105")).apply(make_fill(Side.SELL, "5", "120"))
    assert p.quantity == Decimal(15)
    assert p.average_price == Decimal(105)  # basis unchanged by a close
    assert p.realized.amount == Decimal(75)  # 5 * (120 - 105)


def test_flip_realizes_whole_old_position_then_reopens_at_fill_price(reliance, make_fill):
    """The case naive implementations get wrong: a sell larger than the long."""
    p = Position(instrument=reliance)
    p = p.apply(make_fill(Side.BUY, "15", "105")).apply(make_fill(Side.SELL, "20", "90"))
    assert p.quantity == Decimal(-5)  # now short 5
    assert p.average_price == Decimal(90)  # new basis is the fill price
    assert p.realized.amount == Decimal(-225)  # 15 * (90 - 105), not 20 *


def test_short_position_unrealized_pnl_has_the_right_sign(reliance, make_fill):
    p = Position(instrument=reliance).apply(make_fill(Side.SELL, "10", "100"))
    assert p.is_short
    assert p.unrealized_pnl(Decimal(90)).amount == Decimal(100)  # price fell -> profit
    assert p.unrealized_pnl(Decimal(110)).amount == Decimal(-100)


def test_costs_accumulate_and_net_realized_subtracts_them(reliance, make_fill, free_costs):
    fee = CostBreakdown(
        currency=Currency.INR,
        brokerage=Money.inr(20),
        exchange_fees=Money.zero(Currency.INR),
        transaction_tax=Money.zero(Currency.INR),
        stamp_duty=Money.zero(Currency.INR),
        regulatory_fees=Money.zero(Currency.INR),
        depository_fees=Money.zero(Currency.INR),
        gst=Money.zero(Currency.INR),
    )
    p = Position(instrument=reliance)
    p = p.apply(make_fill(Side.BUY, "10", "100", fee)).apply(make_fill(Side.SELL, "10", "110", fee))
    assert p.realized.amount == Decimal(100)  # gross
    assert p.costs.amount == Decimal(40)
    assert p.net_realized_pnl.amount == Decimal(60)  # the honest number

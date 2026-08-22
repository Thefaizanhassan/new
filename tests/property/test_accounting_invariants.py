"""Property-based accounting invariants.

Phase 0 §5.1: "for *any* sequence of fills, the books must balance" is exactly
the kind of statement example-based tests under-check and Hypothesis excels at.
"""

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st

from trading.core.fill import CostBreakdown, Fill
from trading.core.instrument import Instrument, InstrumentId
from trading.core.position import Position
from trading.core.types import Currency, InstrumentClass, Money, Side

INSTRUMENT = Instrument(
    id=InstrumentId("NSE", "RELIANCE"),
    name="Reliance",
    instrument_class=InstrumentClass.EQUITY,
    currency=Currency.INR,
)
NOW = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
FREE = CostBreakdown.zero(Currency.INR)

trade = st.tuples(
    st.sampled_from([Side.BUY, Side.SELL]),
    st.integers(min_value=1, max_value=500),
    st.integers(min_value=1, max_value=5000),
)


def _fill(side: Side, qty: int, price: int) -> Fill:
    return Fill.create(
        client_order_id="c",
        instrument=INSTRUMENT,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        costs=FREE,
        timestamp=NOW,
    )


@given(trades=st.lists(trade, min_size=1, max_size=25))
@settings(max_examples=250, deadline=None)
def test_quantity_always_equals_the_sum_of_signed_fills(trades):
    position = Position(instrument=INSTRUMENT)
    expected = Decimal(0)
    for side, qty, price in trades:
        position = position.apply(_fill(side, qty, price))
        expected += Decimal(qty) * side.sign
    assert position.quantity == expected


@given(trades=st.lists(trade, min_size=1, max_size=25))
@settings(max_examples=250, deadline=None)
def test_a_round_trip_back_to_flat_realizes_exactly_the_net_cash_flow(trades):
    """If a position starts and ends flat, gross realised P&L must equal the
    net cash received across all fills. This is the books-balance invariant."""
    position = Position(instrument=INSTRUMENT)
    cash = Decimal(0)
    for side, qty, price in trades:
        f = _fill(side, qty, price)
        position = position.apply(f)
        cash += -f.signed_quantity * f.price

    if not position.is_flat:  # flatten so the invariant applies
        closing_side = Side.SELL if position.is_long else Side.BUY
        last_price = Decimal(trades[-1][2])
        f = _fill(closing_side, int(abs(position.quantity)), int(last_price))
        position = position.apply(f)
        cash += -f.signed_quantity * f.price

    assert position.is_flat
    assert position.realized.amount == cash


@given(trades=st.lists(trade, min_size=1, max_size=25))
@settings(max_examples=250, deadline=None)
def test_average_price_is_never_negative_and_is_zero_only_when_flat(trades):
    position = Position(instrument=INSTRUMENT)
    for side, qty, price in trades:
        position = position.apply(_fill(side, qty, price))
        assert position.average_price >= 0
        if position.average_price == 0:
            assert position.is_flat


@given(
    trades=st.lists(trade, min_size=1, max_size=15),
    fee=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=150, deadline=None)
def test_costs_are_monotonic_and_net_never_exceeds_gross(trades, fee):
    charge = CostBreakdown(
        currency=Currency.INR,
        brokerage=Money.inr(fee),
        exchange_fees=Money.zero(Currency.INR),
        transaction_tax=Money.zero(Currency.INR),
        stamp_duty=Money.zero(Currency.INR),
        regulatory_fees=Money.zero(Currency.INR),
        depository_fees=Money.zero(Currency.INR),
        gst=Money.zero(Currency.INR),
    )
    position = Position(instrument=INSTRUMENT)
    previous = Decimal(0)
    for side, qty, price in trades:
        f = Fill.create(
            client_order_id="c",
            instrument=INSTRUMENT,
            side=side,
            quantity=Decimal(qty),
            price=Decimal(price),
            costs=charge,
            timestamp=NOW,
        )
        position = position.apply(f)
        assert position.costs.amount >= previous
        previous = position.costs.amount
    assert position.net_realized_pnl <= position.realized

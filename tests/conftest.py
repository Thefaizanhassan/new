from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from trading.core.fill import CostBreakdown, Fill
from trading.core.instrument import Instrument, InstrumentId
from trading.core.types import Currency, InstrumentClass, Side
from trading.costs.model import SessionCostState


@pytest.fixture
def reliance() -> Instrument:
    return Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="Reliance Industries",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


@pytest.fixture
def now() -> datetime:
    return datetime(2026, 8, 21, 10, 0, tzinfo=UTC)


@pytest.fixture
def free_costs() -> CostBreakdown:
    return CostBreakdown.zero(Currency.INR)


@pytest.fixture
def session() -> SessionCostState:
    return SessionCostState(session_date=date(2026, 8, 21))


@pytest.fixture
def make_fill(reliance, now, free_costs):
    def _make(side: Side, qty: str, price: str, costs: CostBreakdown | None = None) -> Fill:
        return Fill.create(
            client_order_id="coid_test",
            instrument=reliance,
            side=side,
            quantity=Decimal(qty),
            price=Decimal(price),
            costs=costs or free_costs,
            timestamp=now,
        )

    return _make

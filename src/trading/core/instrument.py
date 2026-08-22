"""Canonical instrument identity.

A ticker string is not an identity: ``RELIANCE`` trades on both NSE and BSE, and
``INFY`` on NSE is not the ``INFY`` ADR on NYSE.  Phase 0 addendum §4 — running
two markets makes this mandatory rather than tidy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from trading.core.types import Currency, InstrumentClass, Market

__all__ = ["Exchange", "Instrument", "InstrumentId"]


@dataclass(frozen=True, slots=True)
class Exchange:
    """A venue. ``calendar_code`` is the ISO MIC used by ``exchange_calendars``."""

    code: str
    market: Market
    calendar_code: str
    currency: Currency

    def __str__(self) -> str:
        return self.code


# `exchange_calendars` has no XNSE calendar — verified 2026-08-22, it ships 102
# calendars and the Indian one is XBOM. NSE and BSE keep the same hours
# (09:15-15:30 IST = 03:45-10:00 UTC) and the same holiday list, so XBOM is the
# correct calendar for both. Revisit if the two ever diverge on a special session.
NSE = Exchange(code="NSE", market=Market.INDIA, calendar_code="XBOM", currency=Currency.INR)
BSE = Exchange(code="BSE", market=Market.INDIA, calendar_code="XBOM", currency=Currency.INR)
NASDAQ = Exchange(code="NASDAQ", market=Market.US, calendar_code="XNAS", currency=Currency.USD)
NYSE = Exchange(code="NYSE", market=Market.US, calendar_code="XNYS", currency=Currency.USD)

EXCHANGES: dict[str, Exchange] = {e.code: e for e in (NSE, BSE, NASDAQ, NYSE)}


@dataclass(frozen=True, slots=True, order=True)
class InstrumentId:
    """Globally unique, stable, and safe to use as a dict key or filename."""

    exchange: str
    symbol: str

    def __str__(self) -> str:
        return f"{self.exchange}:{self.symbol}"

    @classmethod
    def parse(cls, value: str) -> InstrumentId:
        exchange, _, symbol = value.partition(":")
        if not symbol:
            raise ValueError(f"Instrument id must be 'EXCHANGE:SYMBOL', got {value!r}")
        return cls(exchange=exchange.upper(), symbol=symbol.upper())

    @property
    def filename(self) -> str:
        return f"{self.exchange}_{self.symbol}"


@dataclass(frozen=True, slots=True)
class Instrument:
    """A tradable instrument and everything the engine needs to price it.

    The derivative fields (``multiplier``, ``expiry``, ``underlying``,
    ``strike``) are unused by equities but present from Phase 1 so that the
    position ledger, fill model and cost models never have to be rewritten when
    options arrive in Phase 14.
    """

    id: InstrumentId
    name: str
    instrument_class: InstrumentClass
    currency: Currency
    tick_size: Decimal = Decimal("0.05")
    lot_size: Decimal = Decimal("1")
    multiplier: Decimal = Decimal("1")
    fractionable: bool = False
    shortable: bool = False
    expiry: date | None = None
    underlying: InstrumentId | None = None
    strike: Decimal | None = None

    def __post_init__(self) -> None:
        if self.instrument_class.is_derivative and self.expiry is None:
            raise ValueError(f"{self.id}: a derivative must carry an expiry")
        if self.multiplier <= 0:
            raise ValueError(f"{self.id}: multiplier must be positive")
        if self.tick_size <= 0 or self.lot_size <= 0:
            raise ValueError(f"{self.id}: tick_size and lot_size must be positive")

    @property
    def exchange(self) -> Exchange:
        return EXCHANGES[self.id.exchange]

    @property
    def market(self) -> Market:
        return self.exchange.market

    def contract_value(self, price: Decimal, quantity: Decimal) -> Decimal:
        """Notional value. For equities this is price*qty; derivatives scale by multiplier."""
        return price * quantity * self.multiplier

    def round_to_tick(self, price: Decimal) -> Decimal:
        """Snap a price to a valid tick. Exchanges reject prices that are not on a tick."""
        return (price / self.tick_size).quantize(Decimal("1")) * self.tick_size

    def round_to_lot(self, quantity: Decimal) -> Decimal:
        """Snap a quantity down to a tradable size (0 if below one lot)."""
        if self.fractionable and self.lot_size == 1:
            return quantity
        lots = (abs(quantity) / self.lot_size).to_integral_value(rounding="ROUND_DOWN")
        return lots * self.lot_size * (1 if quantity >= 0 else -1)

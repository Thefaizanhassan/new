"""Primitive value types shared by the whole domain.

Money is ``Decimal``-based and currency-aware by construction.  Phase 0 §5.1:
floats cannot represent 0.1 exactly, and float drift accumulated over thousands
of fills is exactly how a position ledger stops reconciling against a broker.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Final, Self

__all__ = [
    "Currency",
    "InstrumentClass",
    "Market",
    "Money",
    "Price",
    "Quantity",
    "Side",
    "Timeframe",
    "TradingMode",
]


class Currency(StrEnum):
    """Currencies the platform can hold. Extend deliberately, never implicitly."""

    INR = "INR"
    USD = "USD"

    @property
    def minor_units(self) -> int:
        """Decimal places used when an amount is actually settled."""
        return 2

    @property
    def symbol(self) -> str:
        return {Currency.INR: "₹", Currency.USD: "$"}[self]


class Market(StrEnum):
    """A market the platform can be configured for.

    A market bundles a regulator, a set of exchanges, a calendar, a cost model
    and a compliance profile — see :mod:`trading.config.compliance`.
    """

    INDIA = "INDIA"
    US = "US"

    @property
    def default_currency(self) -> Currency:
        return {Market.INDIA: Currency.INR, Market.US: Currency.USD}[self]


class InstrumentClass(StrEnum):
    """What kind of thing is being traded.

    Only ``EQUITY`` and ``ETF`` are implemented in Phase 1.  ``OPTION`` and
    ``FUTURE`` exist here from day one so that the position and fill models
    carry ``multiplier``/``expiry``/``underlying`` now — Phase 0 addendum §3:
    retrofitting options into a model that assumes shares is genuinely painful,
    while designing the seam up front is nearly free.
    """

    EQUITY = "EQUITY"
    ETF = "ETF"
    OPTION = "OPTION"
    FUTURE = "FUTURE"

    @property
    def is_derivative(self) -> bool:
        return self in (InstrumentClass.OPTION, InstrumentClass.FUTURE)


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """+1 for BUY, -1 for SELL. Use this instead of scattering ternaries."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class TradingMode(StrEnum):
    """Phase 0 §15.2. Default is RESEARCH; LIVE requires deliberate escalation."""

    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"

    @property
    def places_real_orders(self) -> bool:
        return self is TradingMode.LIVE


class Timeframe(StrEnum):
    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"

    @property
    def seconds(self) -> int:
        return {
            Timeframe.MIN_1: 60,
            Timeframe.MIN_5: 300,
            Timeframe.MIN_15: 900,
            Timeframe.HOUR_1: 3600,
            Timeframe.DAY_1: 86_400,
        }[self]

    @property
    def is_intraday(self) -> bool:
        return self is not Timeframe.DAY_1


# Quantity and Price are Decimal aliases rather than wrapper types: they are
# dimensionless enough that a wrapper adds friction without adding safety.
# Money is different — mixing currencies silently is a real failure mode.
Quantity = Decimal
Price = Decimal

_ZERO: Final = Decimal("0")


class CurrencyMismatchError(ValueError):
    """Raised when two Money values in different currencies are combined.

    This is deliberately an error rather than an implicit conversion.  An
    implicit rate would silently bake an unrecorded FX assumption into the
    ledger; conversion must go through :class:`trading.core.fx.FxConverter`,
    which records the rate it used.
    """

    def __init__(self, left: Currency, right: Currency) -> None:
        super().__init__(
            f"Cannot combine {left} and {right} directly. "
            f"Convert explicitly via FxConverter so the rate is recorded."
        )


class Money:
    """An exact amount in a specific currency.

    Immutable. Arithmetic between different currencies raises rather than
    guessing a rate.  Full ``Decimal`` precision is retained internally;
    :meth:`settled` rounds to the currency's minor units at the point where an
    amount actually changes hands.
    """

    __slots__ = ("_amount", "_currency")

    def __init__(self, amount: Decimal | int | str, currency: Currency) -> None:
        self._amount: Final[Decimal] = Decimal(amount)
        self._currency: Final[Currency] = currency

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def zero(cls, currency: Currency) -> Self:
        return cls(_ZERO, currency)

    @classmethod
    def inr(cls, amount: Decimal | int | str) -> Money:
        return Money(amount, Currency.INR)

    @classmethod
    def usd(cls, amount: Decimal | int | str) -> Money:
        return Money(amount, Currency.USD)

    # ── accessors ───────────────────────────────────────────────────────────
    @property
    def amount(self) -> Decimal:
        return self._amount

    @property
    def currency(self) -> Currency:
        return self._currency

    @property
    def is_zero(self) -> bool:
        return self._amount == _ZERO

    @property
    def is_negative(self) -> bool:
        return self._amount < _ZERO

    def settled(self) -> Money:
        """Round to the currency's minor units, as at an actual settlement."""
        exp = Decimal(1).scaleb(-self._currency.minor_units)
        return Money(self._amount.quantize(exp, rounding=ROUND_HALF_UP), self._currency)

    def abs(self) -> Money:
        return Money(abs(self._amount), self._currency)

    # ── arithmetic ──────────────────────────────────────────────────────────
    def _check(self, other: Money) -> None:
        if self._currency is not other._currency:
            raise CurrencyMismatchError(self._currency, other._currency)

    def __add__(self, other: Money) -> Money:
        self._check(other)
        return Money(self._amount + other._amount, self._currency)

    def __sub__(self, other: Money) -> Money:
        self._check(other)
        return Money(self._amount - other._amount, self._currency)

    def __mul__(self, factor: Decimal | int) -> Money:
        return Money(self._amount * Decimal(factor), self._currency)

    __rmul__ = __mul__

    def __truediv__(self, divisor: Decimal | int) -> Money:
        return Money(self._amount / Decimal(divisor), self._currency)

    def __neg__(self) -> Money:
        return Money(-self._amount, self._currency)

    def ratio_to(self, other: Money) -> Decimal:
        """This amount as a fraction of ``other``. Returns 0 when other is zero."""
        self._check(other)
        if other._amount == _ZERO:
            return _ZERO
        return self._amount / other._amount

    # ── comparison ──────────────────────────────────────────────────────────
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Money):
            return NotImplemented
        return self._currency is other._currency and self._amount == other._amount

    def __lt__(self, other: Money) -> bool:
        self._check(other)
        return self._amount < other._amount

    def __le__(self, other: Money) -> bool:
        self._check(other)
        return self._amount <= other._amount

    def __gt__(self, other: Money) -> bool:
        self._check(other)
        return self._amount > other._amount

    def __ge__(self, other: Money) -> bool:
        self._check(other)
        return self._amount >= other._amount

    def __hash__(self) -> int:
        return hash((self._amount, self._currency))

    # ── display ─────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Money({self._amount}, {self._currency})"

    def __str__(self) -> str:
        s = self.settled()._amount
        return f"{self._currency.symbol}{s:,.{self._currency.minor_units}f}"

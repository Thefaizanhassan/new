"""Foreign exchange, with the rate recorded.

Running INR and USD positions in one portfolio means every cross-currency
figure depends on a rate.  Phase 0 addendum §4: FX P&L must be separable from
strategy P&L, or a rupee move reads as alpha.  Every conversion therefore names
the rate it used, and rates are point-in-time like any other market data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from trading.core.clock import ensure_utc
from trading.core.types import Currency, Money

__all__ = ["FxConverter", "FxRate", "MissingRateError"]


class MissingRateError(LookupError):
    """No rate available for a required conversion. Fail closed — never guess."""


@dataclass(frozen=True, slots=True)
class FxRate:
    """``rate`` units of ``quote`` per one unit of ``base``."""

    base: Currency
    quote: Currency
    rate: Decimal
    as_of: datetime
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "as_of", ensure_utc(self.as_of))
        if self.rate <= 0:
            raise ValueError(f"FX rate must be positive, got {self.rate}")

    @property
    def inverted(self) -> FxRate:
        return FxRate(
            base=self.quote,
            quote=self.base,
            rate=Decimal(1) / self.rate,
            as_of=self.as_of,
            source=f"{self.source}:inverted",
        )


class FxConverter:
    """Converts Money between currencies using explicitly supplied rates."""

    def __init__(self, rates: list[FxRate] | None = None) -> None:
        self._rates: dict[tuple[Currency, Currency], FxRate] = {}
        for rate in rates or []:
            self.add(rate)

    def add(self, rate: FxRate) -> None:
        self._rates[(rate.base, rate.quote)] = rate
        self._rates[(rate.quote, rate.base)] = rate.inverted

    def rate_for(self, base: Currency, quote: Currency) -> FxRate:
        if base is quote:
            return FxRate(base, quote, Decimal(1), datetime.now().astimezone(), "identity")
        try:
            return self._rates[(base, quote)]
        except KeyError as exc:
            raise MissingRateError(
                f"No {base}->{quote} rate loaded. Refusing to convert without a recorded rate."
            ) from exc

    def convert(self, money: Money, to: Currency) -> tuple[Money, FxRate]:
        """Convert, returning the result *and* the rate used, for the audit trail."""
        rate = self.rate_for(money.currency, to)
        return Money(money.amount * rate.rate, to), rate

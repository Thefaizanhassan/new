"""Position accounting, on FIFO tax lots.

**Why lots rather than an average price.**

Average-cost accounting has to allocate basis proportionally when a position is
partially closed, and that allocation is a division.  Divisions like 5/3 do not
terminate in any finite precision, so the books stop balancing exactly.  Two
property-test failures drove this design:

* storing an average price: buy 1@1, buy 2@2, sell 3@2 realised
  ``0.999999999999999999999999999`` instead of ``1``;
* storing an exact basis total but allocating it by a fraction: a residual of
  ~1e-22 survived a full round trip.

FIFO lots remove the division entirely.  Each lot keeps its own exact quantity
and price, and closing consumes lots by **multiplication only** — so a position
that opens and closes completely returns to exactly zero, at any scale.

This also happens to be the accounting Indian capital-gains treatment expects
for demat holdings, so the lot history the ledger needs for tax reporting later
comes for free rather than being retrofitted.

Realised P&L is recorded **gross**, with costs tracked separately, so the
backtest can plot gross and net equity curves side by side (Phase 0 §11.4).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from trading.core.fill import Fill
from trading.core.instrument import Instrument
from trading.core.types import Money

__all__ = ["Lot", "Position"]

_ZERO = Decimal(0)


@dataclass(frozen=True, slots=True)
class Lot:
    """One acquisition, kept at its exact original price.

    ``quantity`` is signed and every lot in a position shares its sign.
    """

    quantity: Decimal
    price: Decimal


@dataclass(frozen=True, slots=True)
class Position:
    """Holding in one instrument, as an ordered tuple of FIFO lots.

    ``quantity`` and ``cost_basis`` are derived from the lots, never stored
    separately, so they cannot drift out of agreement with them.
    """

    instrument: Instrument
    lots: tuple[Lot, ...] = ()
    realized_pnl: Money | None = None
    total_costs: Money | None = None

    def __post_init__(self) -> None:
        currency = self.instrument.currency
        if self.realized_pnl is None:
            object.__setattr__(self, "realized_pnl", Money.zero(currency))
        if self.total_costs is None:
            object.__setattr__(self, "total_costs", Money.zero(currency))

    # ── state ───────────────────────────────────────────────────────────────
    @property
    def quantity(self) -> Decimal:
        """Signed open quantity — positive long, negative short."""
        return sum((lot.quantity for lot in self.lots), _ZERO)

    @property
    def cost_basis(self) -> Decimal:
        """Signed net cash paid for the open quantity. Exact: no division."""
        return sum(
            (lot.quantity * lot.price * self.instrument.multiplier for lot in self.lots), _ZERO
        )

    @property
    def open_lots(self) -> int:
        return len(self.lots)

    @property
    def is_flat(self) -> bool:
        return self.quantity == _ZERO

    @property
    def is_long(self) -> bool:
        return self.quantity > _ZERO

    @property
    def is_short(self) -> bool:
        return self.quantity < _ZERO

    @property
    def average_price(self) -> Decimal:
        """Derived, for display and reporting. Never used in P&L arithmetic."""
        if self.quantity == _ZERO:
            return _ZERO
        return self.cost_basis / (self.quantity * self.instrument.multiplier)

    @property
    def realized(self) -> Money:
        assert self.realized_pnl is not None
        return self.realized_pnl

    @property
    def costs(self) -> Money:
        assert self.total_costs is not None
        return self.total_costs

    @property
    def net_realized_pnl(self) -> Money:
        """Realised P&L after the costs incurred getting there. The honest number."""
        return self.realized - self.costs

    # ── valuation ───────────────────────────────────────────────────────────
    def market_value(self, price: Decimal) -> Money:
        return Money(self.instrument.contract_value(price, self.quantity), self.instrument.currency)

    def basis_value(self) -> Money:
        return Money(self.cost_basis, self.instrument.currency)

    def unrealized_pnl(self, price: Decimal) -> Money:
        """Mark to market on the open quantity. Correct for longs and shorts alike."""
        return self.market_value(price) - self.basis_value()

    # ── mutation ────────────────────────────────────────────────────────────
    def apply(self, fill: Fill) -> Position:
        """Return the position after applying a fill.

        1. Opening from flat  -> a single new lot
        2. Adding             -> a lot appended; nothing recomputed
        3. Reducing           -> consumes lots oldest-first, realising
                                 ``(fill_price - lot_price) x qty`` per lot,
                                 which is exact because it never divides
        4. **Flipping** (a sell larger than the long, or vice versa)
           -> every lot is consumed, then the remainder opens a fresh lot on
              the other side at the fill price
        """
        if fill.instrument.id != self.instrument.id:
            raise ValueError(
                f"Fill for {fill.instrument.id} cannot be applied to a position "
                f"in {self.instrument.id}"
            )

        multiplier = self.instrument.multiplier
        delta = fill.signed_quantity
        new_costs = self.costs + fill.costs.total

        # 1. opening from flat
        if not self.lots:
            return replace(self, lots=(Lot(delta, fill.price),), total_costs=new_costs)

        same_direction = (self.quantity > _ZERO) == (delta > _ZERO)

        # 2. adding — a new lot, leaving existing lots untouched
        if same_direction:
            return replace(self, lots=(*self.lots, Lot(delta, fill.price)), total_costs=new_costs)

        # 3 & 4. closing against existing lots, oldest first
        direction = Decimal(1) if self.quantity > _ZERO else Decimal(-1)
        to_close = abs(delta)
        realized_total = _ZERO
        remaining_lots: list[Lot] = []

        for lot in self.lots:
            lot_qty = abs(lot.quantity)
            if to_close <= _ZERO:
                remaining_lots.append(lot)
                continue
            consumed = min(to_close, lot_qty)
            # Multiplication only — this is what keeps the books exact.
            realized_total += (fill.price - lot.price) * consumed * direction * multiplier
            to_close -= consumed
            leftover = lot_qty - consumed
            if leftover > _ZERO:
                remaining_lots.append(Lot(leftover * direction, lot.price))

        # 4. flip — the fill outlasted every lot, so open the rest on the other side
        if to_close > _ZERO:
            remaining_lots.append(Lot(to_close * -direction, fill.price))

        return replace(
            self,
            lots=tuple(remaining_lots),
            realized_pnl=self.realized + Money(realized_total, self.instrument.currency),
            total_costs=new_costs,
        )

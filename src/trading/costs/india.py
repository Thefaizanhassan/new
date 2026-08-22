"""Indian equity transaction costs (NSE/BSE).

Rates are **inputs, not constants** — Indian charges change with each Budget
and by broker.  Every rate here is a constructor argument with a documented
default, and the whole object is versioned so a backtest records exactly which
schedule it assumed.

⚠️  Defaults reflect a Zerodha-style delivery account as researched on
2026-08-22 and MUST be re-verified against your broker's live schedule before
any result is trusted.  See the Phase 0 addendum for sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from trading.core.fill import CostBreakdown
from trading.core.types import Currency, Money, Side
from trading.costs.model import CostContext

__all__ = ["IndiaDeliveryEquityCosts", "IndiaIntradayEquityCosts"]

_PCT = Decimal("0.01")


@dataclass(frozen=True)
class IndiaDeliveryEquityCosts:
    """Delivery (CNC) equity costs.

    Charge structure, and why each matters:

    * **Brokerage** — ₹0 for delivery at several discount brokers.
    * **STT** 0.1% on **both** buy and sell. The single largest charge, and the
      reason Indian strategies need roughly 0.35% gross edge per round trip
      just to break even.
    * **Exchange transaction charge** — percentage of turnover, both sides.
    * **SEBI turnover fee** — ₹10 per crore, both sides.
    * **Stamp duty** 0.015% — **buy side only.**
    * **GST** 18% on (brokerage + exchange charge + SEBI fee) — not on STT.
    * **Depository (DP) charge** — flat ₹15.34 incl. GST per scrip **per
      session, sell side only, regardless of quantity.** Flat fees dominate at
      small capital: 0.04% on a ₹40,000 position, **0.3% on a ₹5,000 one.**
    """

    model_id: str = "india_delivery_equity"
    version: str = "2026.08.1"
    brokerage_rate: Decimal = Decimal("0")
    brokerage_cap: Decimal = Decimal("20")
    stt_rate: Decimal = Decimal("0.1") * _PCT
    exchange_txn_rate: Decimal = Decimal("0.00297") * _PCT
    sebi_turnover_rate: Decimal = Decimal("0.0001") * _PCT
    stamp_duty_rate: Decimal = Decimal("0.015") * _PCT
    gst_rate: Decimal = Decimal("18") * _PCT
    depository_fee: Decimal = Decimal("15.34")

    def compute(self, ctx: CostContext) -> CostBreakdown:
        inr = Currency.INR
        turnover = ctx.turnover
        is_buy = ctx.side is Side.BUY

        brokerage = min(turnover * self.brokerage_rate, self.brokerage_cap)
        stt = turnover * self.stt_rate
        exchange = turnover * self.exchange_txn_rate
        sebi = turnover * self.sebi_turnover_rate
        stamp = turnover * self.stamp_duty_rate if is_buy else Decimal(0)
        gst = (brokerage + exchange + sebi) * self.gst_rate

        # The charge a (size, price) signature cannot express: flat, sell-side,
        # once per scrip per session.
        depository = Decimal(0)
        if not is_buy and not ctx.session.already_charged_depository(ctx.instrument.id):
            depository = self.depository_fee
            ctx.session.note_depository_charge(ctx.instrument.id)

        return CostBreakdown(
            currency=inr,
            brokerage=Money(brokerage, inr),
            exchange_fees=Money(exchange, inr),
            transaction_tax=Money(stt, inr),
            stamp_duty=Money(stamp, inr),
            regulatory_fees=Money(sebi, inr),
            depository_fees=Money(depository, inr),
            gst=Money(gst, inr),
        )

    def describe(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "brokerage": f"{self.brokerage_rate:%} of turnover, capped at ₹{self.brokerage_cap}",
            "stt": f"{self.stt_rate:%} on buy and sell",
            "exchange_txn": f"{self.exchange_txn_rate:%} of turnover",
            "sebi_turnover": f"{self.sebi_turnover_rate:%} of turnover",
            "stamp_duty": f"{self.stamp_duty_rate:%} on buy only",
            "gst": f"{self.gst_rate:%} on brokerage + exchange + SEBI fees",
            "depository": f"₹{self.depository_fee} flat, per scrip per session, sell only",
            "caveat": "Re-verify against your broker's live schedule before trusting results.",
        }


@dataclass(frozen=True)
class IndiaIntradayEquityCosts:
    """Intraday (MIS) equity costs.

    Differs from delivery in three ways: brokerage applies (₹20 or 0.03%,
    whichever is lower, per executed order), STT is charged on the **sell side
    only** at a lower rate, and there is no DP charge because nothing is
    delivered.
    """

    model_id: str = "india_intraday_equity"
    version: str = "2026.08.1"
    brokerage_rate: Decimal = Decimal("0.03") * _PCT
    brokerage_cap: Decimal = Decimal("20")
    stt_sell_rate: Decimal = Decimal("0.025") * _PCT
    exchange_txn_rate: Decimal = Decimal("0.00297") * _PCT
    sebi_turnover_rate: Decimal = Decimal("0.0001") * _PCT
    stamp_duty_rate: Decimal = Decimal("0.003") * _PCT
    gst_rate: Decimal = Decimal("18") * _PCT

    def compute(self, ctx: CostContext) -> CostBreakdown:
        inr = Currency.INR
        turnover = ctx.turnover
        is_buy = ctx.side is Side.BUY

        brokerage = min(turnover * self.brokerage_rate, self.brokerage_cap)
        stt = Decimal(0) if is_buy else turnover * self.stt_sell_rate
        exchange = turnover * self.exchange_txn_rate
        sebi = turnover * self.sebi_turnover_rate
        stamp = turnover * self.stamp_duty_rate if is_buy else Decimal(0)
        gst = (brokerage + exchange + sebi) * self.gst_rate
        zero = Money.zero(inr)

        return CostBreakdown(
            currency=inr,
            brokerage=Money(brokerage, inr),
            exchange_fees=Money(exchange, inr),
            transaction_tax=Money(stt, inr),
            stamp_duty=Money(stamp, inr),
            regulatory_fees=Money(sebi, inr),
            depository_fees=zero,
            gst=Money(gst, inr),
        )

    def describe(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "brokerage": f"{self.brokerage_rate:%}, capped at ₹{self.brokerage_cap} per order",
            "stt": f"{self.stt_sell_rate:%} on sell only",
            "depository": "none — nothing is delivered",
            "caveat": "Re-verify against your broker's live schedule before trusting results.",
        }

"""Indian cost model.

The DP-charge tests exist because that charge is the concrete reason the
backtrader `getcommission(size, price)` signature is insufficient — see
docs/technology-evaluation.md §4.1.
"""

from decimal import Decimal

from trading.core.types import Side
from trading.costs.india import IndiaDeliveryEquityCosts, IndiaIntradayEquityCosts
from trading.costs.model import CostContext


def _ctx(reliance, session, now, side, qty="100", price="1400"):
    return CostContext(reliance, side, Decimal(qty), Decimal(price), now, session)


def test_stt_charged_on_both_sides_for_delivery(reliance, session, now):
    m = IndiaDeliveryEquityCosts()
    buy = m.compute(_ctx(reliance, session, now, Side.BUY))
    sell = m.compute(_ctx(reliance, session, now, Side.SELL))
    turnover = Decimal("140000")
    assert buy.transaction_tax.amount == turnover * Decimal("0.001")
    assert sell.transaction_tax.amount == turnover * Decimal("0.001")


def test_stamp_duty_is_buy_side_only(reliance, session, now):
    m = IndiaDeliveryEquityCosts()
    assert m.compute(_ctx(reliance, session, now, Side.BUY)).stamp_duty.amount > 0
    assert m.compute(_ctx(reliance, session, now, Side.SELL)).stamp_duty.is_zero


def test_depository_fee_is_charged_once_per_scrip_per_session(reliance, session, now):
    """A flat, session-scoped, sell-side charge — not derivable from size and price."""
    m = IndiaDeliveryEquityCosts()
    first = m.compute(_ctx(reliance, session, now, Side.SELL, qty="50"))
    second = m.compute(_ctx(reliance, session, now, Side.SELL, qty="50"))
    assert first.depository_fees.amount == m.depository_fee
    assert second.depository_fees.is_zero, "DP fee must not be charged twice in one session"


def test_depository_fee_not_charged_on_buys(reliance, session, now):
    m = IndiaDeliveryEquityCosts()
    assert m.compute(_ctx(reliance, session, now, Side.BUY)).depository_fees.is_zero


def test_flat_fees_dominate_at_small_position_sizes(reliance, session, now):
    """The addendum's claim, asserted rather than asserted-in-prose.

    Round-trip cost as a fraction of turnover must be materially worse for a
    small position than a large one, because the DP fee is flat.
    """
    m = IndiaDeliveryEquityCosts()

    def round_trip_pct(value: str) -> Decimal:
        session.depository_charged.clear()
        qty = Decimal(value) / Decimal("1400")
        buy = m.compute(_ctx(reliance, session, now, Side.BUY, qty=str(qty)))
        sell = m.compute(_ctx(reliance, session, now, Side.SELL, qty=str(qty)))
        return (buy.total + sell.total).amount / Decimal(value)

    small, large = round_trip_pct("5000"), round_trip_pct("200000")
    assert small > large * 2, f"small={small:.4%} large={large:.4%}"
    assert Decimal("0.002") < large < Decimal("0.004")  # ~0.23% as researched


def test_intraday_stt_is_sell_side_only_and_has_no_dp_charge(reliance, session, now):
    m = IndiaIntradayEquityCosts()
    assert m.compute(_ctx(reliance, session, now, Side.BUY)).transaction_tax.is_zero
    assert m.compute(_ctx(reliance, session, now, Side.SELL)).transaction_tax.amount > 0
    assert m.compute(_ctx(reliance, session, now, Side.SELL)).depository_fees.is_zero


def test_brokerage_is_capped(reliance, session, now):
    m = IndiaIntradayEquityCosts()
    huge = m.compute(_ctx(reliance, session, now, Side.BUY, qty="10000"))
    assert huge.brokerage.amount == m.brokerage_cap


def test_model_describes_its_assumptions_for_the_run_manifest():
    d = IndiaDeliveryEquityCosts().describe()
    assert d["version"] and "per scrip per session" in d["depository"]
    assert "Re-verify" in d["caveat"]

"""The core safety property: an Order cannot exist without risk approval.

Phase 0 §4.4(b) and §13.3. If any of these tests fail, the platform's central
guarantee is broken — treat it as a release blocker, not a flaky test.
"""

from decimal import Decimal

import pytest

from trading.core.order import Order, OrderRequest
from trading.core.risk import RiskDecision, RiskRuleResult, UnauthorizedOrderError
from trading.core.types import Side


def _pass(rule_id: str = "max_position") -> RiskRuleResult:
    return RiskRuleResult(rule_id=rule_id, passed=True, observed="0.04", limit="0.10")


def _fail(rule_id: str = "max_exposure") -> RiskRuleResult:
    return RiskRuleResult(rule_id=rule_id, passed=False, observed="0.94", limit="0.80")


@pytest.fixture
def request_(reliance) -> OrderRequest:
    return OrderRequest(instrument=reliance, side=Side.BUY, quantity=Decimal(10))


def test_risk_decision_cannot_be_forged(now):
    with pytest.raises(UnauthorizedOrderError):
        RiskDecision(decision_id="fake", verdict="APPROVED", issued_at=now, issued_by="attacker")


def test_order_cannot_be_constructed_directly(request_, now):
    decision = RiskDecision.approve("risk_engine", now, (_pass(),))
    with pytest.raises(UnauthorizedOrderError):
        Order(client_order_id="x", request=request_, decision=decision, created_at=now)


def test_order_refuses_a_rejected_decision(request_, now):
    rejected = RiskDecision.reject("risk_engine", now, (_pass(), _fail()))
    with pytest.raises(UnauthorizedOrderError, match="rejected risk decision"):
        Order.create(request_, rejected, now)


def test_approved_decision_produces_an_order_with_a_client_order_id(request_, now):
    approved = RiskDecision.approve("risk_engine", now, (_pass(),))
    order = Order.create(request_, approved, now)
    assert order.client_order_id.startswith("coid_")
    assert order.decision.approved


def test_client_order_ids_are_unique_across_orders(request_, now):
    approved = RiskDecision.approve("risk_engine", now, (_pass(),))
    ids = {Order.create(request_, approved, now).client_order_id for _ in range(500)}
    assert len(ids) == 500, "duplicate client_order_id defeats idempotent submission"


def test_cannot_approve_a_decision_containing_a_failed_rule(now):
    with pytest.raises(ValueError, match="failed rules"):
        RiskDecision.approve("risk_engine", now, (_pass(), _fail()))


def test_cannot_reject_a_decision_where_everything_passed(now):
    with pytest.raises(ValueError, match="every rule passed"):
        RiskDecision.reject("risk_engine", now, (_pass(),))


def test_rejection_reason_names_the_rule_and_the_numbers(now):
    rejected = RiskDecision.reject("risk_engine", now, (_fail("max_exposure"),))
    assert "max_exposure" in rejected.rejection_reason
    assert "0.94" in rejected.rejection_reason and "0.80" in rejected.rejection_reason

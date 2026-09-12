"""Lifecycle gates.

These enforce that a strategy earns each status. The most important assertions
are the negative ones: a gate that never blocks is decoration.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from trading.data.provider import DataTier
from trading.strategies.base import LifecycleStatus as L
from trading.strategies.lifecycle import (
    CriterionStatus,
    Evidence,
    evaluate_promotion,
    evidence_from_validation,
)


def strong_evidence(**overrides) -> Evidence:
    """Everything a strategy could possibly have going for it."""
    base = {
        "data_tier": DataTier.PRODUCTION,
        "backtest_completed": True,
        "trades": 350,
        "net_expectancy": Decimal("0.004"),
        "benchmark_compared": True,
        "out_of_sample_passed": True,
        "walk_forward_passed": True,
        "parameter_plateau": True,
        "monte_carlo_drawdown_ok": True,
        "trial_count": 12,
        "deflated_sharpe": 0.97,
        "risk_limits_defined": True,
        "kill_switch_drilled": True,
        "reconciliation_clean": True,
        "paper_trading_days": 90,
        "divergence_within_tolerance": True,
        "human_signoff_at": datetime(2026, 8, 22, tzinfo=UTC),
        "human_signoff_by": "rajeshwar",
    }
    return Evidence(**{**base, **overrides})


# ── the tier cap ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("tier", [DataTier.SYNTHETIC, DataTier.PROTOTYPE])
def test_untrustworthy_data_caps_a_strategy_at_backtesting(tier):
    """However good the numbers, prototype data cannot support a promotion."""
    decision = evaluate_promotion("s", L.BACKTESTING, L.PROMISING, strong_evidence(data_tier=tier))
    assert not decision.approved
    assert any(r.criterion_id == "GATE_000_data_tier" for r in decision.blockers)


@pytest.mark.parametrize("tier", [DataTier.SYNTHETIC, DataTier.PROTOTYPE])
def test_reaching_backtesting_is_still_allowed_on_untrusted_data(tier):
    decision = evaluate_promotion("s", L.RESEARCH, L.BACKTESTING, strong_evidence(data_tier=tier))
    assert decision.approved


def test_production_data_with_full_evidence_is_approved():
    assert evaluate_promotion("s", L.BACKTESTING, L.PROMISING, strong_evidence()).approved
    assert evaluate_promotion("s", L.PROMISING, L.VALIDATED, strong_evidence()).approved
    assert evaluate_promotion("s", L.VALIDATED, L.PAPER, strong_evidence()).approved
    assert evaluate_promotion("s", L.PAPER, L.LIVE_APPROVED, strong_evidence()).approved


# ── unavailable blocks ─────────────────────────────────────────────────────
def test_an_unimplemented_check_blocks_rather_than_passes():
    """A gate that passes because nobody wrote its check manufactures confidence."""
    decision = evaluate_promotion(
        "s", L.PROMISING, L.VALIDATED, strong_evidence(walk_forward_passed=None)
    )
    assert not decision.approved
    blocker = next(r for r in decision.blockers if "walk_forward" in r.criterion_id)
    assert blocker.status is CriterionStatus.UNAVAILABLE


def test_an_unmeasured_expectancy_blocks():
    decision = evaluate_promotion(
        "s", L.BACKTESTING, L.PROMISING, strong_evidence(net_expectancy=None)
    )
    assert not decision.approved


def test_a_negative_net_expectancy_fails_rather_than_being_unavailable():
    decision = evaluate_promotion(
        "s", L.BACKTESTING, L.PROMISING, strong_evidence(net_expectancy=Decimal("-0.001"))
    )
    blocker = next(r for r in decision.blockers if "expectancy" in r.criterion_id)
    assert blocker.status is CriterionStatus.FAIL


# ── specific criteria ──────────────────────────────────────────────────────
def test_too_few_trades_blocks_promotion():
    decision = evaluate_promotion("s", L.BACKTESTING, L.PROMISING, strong_evidence(trades=40))
    assert any("trade_count" in r.criterion_id for r in decision.blockers)


def test_an_undrilled_kill_switch_blocks_paper_trading():
    decision = evaluate_promotion(
        "s", L.VALIDATED, L.PAPER, strong_evidence(kill_switch_drilled=False)
    )
    assert any("kill_switch" in r.criterion_id for r in decision.blockers)


def test_too_few_paper_days_blocks_live_approval():
    decision = evaluate_promotion(
        "s", L.PAPER, L.LIVE_APPROVED, strong_evidence(paper_trading_days=20)
    )
    assert any("paper_days" in r.criterion_id for r in decision.blockers)


def test_live_approval_requires_a_recorded_human_signoff():
    decision = evaluate_promotion(
        "s", L.PAPER, L.LIVE_APPROVED, strong_evidence(human_signoff_at=None)
    )
    assert any("human_signoff" in r.criterion_id for r in decision.blockers)


# ── transitions ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("start", "target"),
    [(L.RESEARCH, L.PAPER), (L.RESEARCH, L.LIVE), (L.BACKTESTING, L.VALIDATED)],
)
def test_skipping_a_stage_is_refused_however_strong_the_evidence(start, target):
    decision = evaluate_promotion("s", start, target, strong_evidence())
    assert not decision.approved
    assert any(r.criterion_id == "GATE_100_transition" for r in decision.blockers)


def test_going_backwards_is_refused():
    assert not evaluate_promotion("s", L.PAPER, L.PROMISING, strong_evidence()).approved


def test_live_is_unreachable_by_design_until_phase_12():
    decision = evaluate_promotion("s", L.LIVE_APPROVED, L.LIVE, strong_evidence())
    assert not decision.approved
    assert decision.blockers[0].status is CriterionStatus.UNAVAILABLE


@pytest.mark.parametrize("target", [L.PAUSED, L.RETIRED])
def test_stopping_a_strategy_never_requires_evidence(target):
    """Halting must never be gated — that is how a bad strategy keeps running."""
    decision = evaluate_promotion(
        "s", L.LIVE_APPROVED, target, Evidence(data_tier=DataTier.SYNTHETIC)
    )
    assert decision.approved


def test_the_decision_names_every_criterion_it_checked():
    decision = evaluate_promotion("s", L.PROMISING, L.VALIDATED, strong_evidence())
    assert len(decision.results) >= 5
    assert "VALIDATED" in decision.summary()
    for result in decision.results:
        assert result.observed and result.required


# ── Phase 6: the validation gates are now answerable ────────────────────────
def test_missing_deflated_sharpe_blocks_validation():
    """A Sharpe with no trial count behind it is not evidence.

    The whole experiment ledger exists to make this number computable. If it is
    absent the gate must block, not wave the strategy through on the raw ratio.
    """
    decision = evaluate_promotion(
        "s", L.PROMISING, L.VALIDATED, strong_evidence(deflated_sharpe=None)
    )
    assert not decision.approved
    gate = next(r for r in decision.results if r.criterion_id == "GATE_015_deflated_sharpe")
    assert gate.status is CriterionStatus.UNAVAILABLE


def test_deflated_sharpe_below_threshold_fails_rather_than_blocks():
    """Measured-and-bad is a FAIL; not-measured is UNAVAILABLE. The distinction matters.

    UNAVAILABLE says "go and run the check". FAIL says "you ran it and the answer
    was no". Collapsing them would hide which of those happened.
    """
    decision = evaluate_promotion(
        "s", L.PROMISING, L.VALIDATED, strong_evidence(deflated_sharpe=0.42)
    )
    gate = next(r for r in decision.results if r.criterion_id == "GATE_015_deflated_sharpe")
    assert gate.status is CriterionStatus.FAIL
    assert "0.420" in gate.observed


def test_evidence_from_validation_cannot_invent_a_pass():
    """A report missing an artefact must yield None, which blocks.

    This is the load-bearing property of the Phase 6 wiring: there is no path
    from "no sweep was run" to "the parameters are on a plateau".
    """

    class BareReport:
        def evidence_fields(self):
            return {
                "data_tier": DataTier.PRODUCTION,
                "out_of_sample_passed": None,
                "walk_forward_passed": False,
                "parameter_plateau": None,
                "monte_carlo_drawdown_ok": None,
                "trial_count": None,
                "deflated_sharpe": None,
            }

    evidence = evidence_from_validation(BareReport())
    decision = evaluate_promotion("s", L.PROMISING, L.VALIDATED, evidence)
    assert not decision.approved
    unavailable = {
        r.criterion_id for r in decision.results if r.status is CriterionStatus.UNAVAILABLE
    }
    assert unavailable == {
        "GATE_010_out_of_sample",
        "GATE_012_parameter_plateau",
        "GATE_013_monte_carlo_drawdown",
        "GATE_014_trial_count_recorded",
        "GATE_015_deflated_sharpe",
    }


def test_evidence_from_validation_preserves_fields_it_cannot_answer():
    """Paper-trading days and a human sign-off have no artefact; a report must not clear them."""

    class Report:
        def evidence_fields(self):
            return {
                "data_tier": DataTier.PRODUCTION,
                "out_of_sample_passed": True,
                "walk_forward_passed": True,
                "parameter_plateau": True,
                "monte_carlo_drawdown_ok": True,
                "trial_count": 40,
                "deflated_sharpe": 0.98,
            }

    merged = evidence_from_validation(Report(), base=strong_evidence())
    assert merged.paper_trading_days == 90
    assert merged.human_signoff_by == "rajeshwar"
    assert merged.trial_count == 40
    assert evaluate_promotion("s", L.PROMISING, L.VALIDATED, merged).approved

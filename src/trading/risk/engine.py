"""The risk engine.

An adversarial component whose job is to assume every other part of the system
is buggy.  It is the only thing that can mint a :class:`RiskDecision`, and
therefore the only route to an :class:`Order` existing at all.

Two rules govern it:

* **Fail closed.** If a rule cannot be evaluated — missing data, an exception, an
  unreachable dependency — the trade is rejected. Never "skip the check and
  proceed" (Phase 0 §13.2).
* **Every evaluation is recorded**, pass or fail, with the rule id, the observed
  value and the limit. That record is what makes the decision chain in Phase 0
  §16 reconstructable, and what tells a well-calibrated limit from one quietly
  strangling a strategy.

Phase 1 implements the pre-trade rules that need no live-market state. The
continuous monitors (daily loss, drawdown, reconciliation drift, broker
connectivity) arrive with the full engine in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from trading.config.compliance import ComplianceProfile
from trading.core.instrument import Instrument
from trading.core.order import OrderRequest
from trading.core.risk import RiskDecision, RiskRuleResult
from trading.core.types import Money, Side, TradingMode

__all__ = ["RiskEngine", "RiskLimits"]


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """Deliberately conservative defaults. Loosening one should be a decision."""

    max_position_weight: Decimal = Decimal("0.25")
    max_gross_exposure: Decimal = Decimal("0.80")
    max_order_value: Money | None = None
    max_open_positions: int = 10
    min_order_value: Decimal = Decimal("1000")
    """Below this, fixed costs dominate — at ₹5,000 the flat DP charge alone is
    0.3% per round trip. Rejecting tiny orders is a cost control, not pedantry."""


@dataclass(frozen=True, slots=True)
class RiskInputs:
    """Everything the engine needs. Assembled by the caller, never fetched here."""

    request: OrderRequest
    reference_price: Decimal
    equity: Money
    cash: Money
    gross_exposure_value: Money
    open_positions: int
    mode: TradingMode
    now: datetime


class RiskEngine:
    """Evaluates every rule, then mints an approval or a rejection."""

    ISSUER = "risk_engine"

    def __init__(
        self,
        limits: RiskLimits,
        compliance: ComplianceProfile,
        *,
        kill_switch_path: Path | None = None,
    ) -> None:
        self.limits = limits
        self.compliance = compliance
        # Layer 3 of the kill switch (Phase 0 §13.4): a filesystem sentinel
        # checked at the lowest level of the order path, so it still works when
        # the strategy layer is looping or wedged.
        self.kill_switch_path = kill_switch_path or Path("KILL")

    def evaluate(self, inputs: RiskInputs) -> RiskDecision:
        try:
            results = tuple(self._run_rules(inputs))
        except Exception as exc:
            failure = RiskRuleResult(
                rule_id="RISK_000_engine_error",
                passed=False,
                observed=f"{type(exc).__name__}: {exc}",
                limit="no exception",
                detail="rules could not be evaluated; failing closed",
            )
            return RiskDecision.reject(self.ISSUER, inputs.now, (failure,))

        if all(r.passed for r in results):
            return RiskDecision.approve(self.ISSUER, inputs.now, results)
        return RiskDecision.reject(self.ISSUER, inputs.now, results)

    # ── rules ───────────────────────────────────────────────────────────────
    def _run_rules(self, i: RiskInputs) -> list[RiskRuleResult]:
        instrument: Instrument = i.request.instrument
        order_value = instrument.contract_value(i.reference_price, i.request.quantity)
        equity = i.equity.amount

        return [
            self._kill_switch(),
            self._mode_allows_trading(i.mode),
            self._max_order_value(order_value),
            self._min_order_value(order_value),
            self._max_position_weight(order_value, equity),
            self._max_gross_exposure(order_value, i.gross_exposure_value.amount, equity),
            self._max_open_positions(i.open_positions),
            self._sufficient_cash(i, order_value),
        ]

    def _kill_switch(self) -> RiskRuleResult:
        engaged = self.kill_switch_path.exists()
        return RiskRuleResult(
            rule_id="RISK_001_kill_switch",
            passed=not engaged,
            observed="ENGAGED" if engaged else "clear",
            limit="clear",
            detail=f"sentinel {self.kill_switch_path}" if engaged else "",
        )

    def _mode_allows_trading(self, mode: TradingMode) -> RiskRuleResult:
        allowed = mode in (TradingMode.BACKTEST, TradingMode.PAPER, TradingMode.LIVE)
        return RiskRuleResult(
            rule_id="RISK_002_trading_mode",
            passed=allowed,
            observed=str(mode),
            limit="BACKTEST | PAPER | LIVE",
            detail="RESEARCH mode does not place orders" if not allowed else "",
        )

    def _max_order_value(self, order_value: Decimal) -> RiskRuleResult:
        cap = self.limits.max_order_value
        passed = cap is None or order_value <= cap.amount
        return RiskRuleResult(
            rule_id="RISK_003_max_order_value",
            passed=passed,
            observed=f"{order_value:.2f}",
            limit="unbounded" if cap is None else f"{cap.amount:.2f}",
        )

    def _min_order_value(self, order_value: Decimal) -> RiskRuleResult:
        passed = order_value >= self.limits.min_order_value
        return RiskRuleResult(
            rule_id="RISK_004_min_order_value",
            passed=passed,
            observed=f"{order_value:.2f}",
            limit=f"{self.limits.min_order_value:.2f}",
            detail="below this, fixed costs dominate the trade" if not passed else "",
        )

    def _max_position_weight(self, order_value: Decimal, equity: Decimal) -> RiskRuleResult:
        weight = order_value / equity if equity else Decimal(1)
        return RiskRuleResult(
            rule_id="RISK_005_max_position_weight",
            passed=weight <= self.limits.max_position_weight,
            observed=f"{weight:.4f}",
            limit=f"{self.limits.max_position_weight}",
        )

    def _max_gross_exposure(
        self, order_value: Decimal, current_gross: Decimal, equity: Decimal
    ) -> RiskRuleResult:
        projected = (current_gross + order_value) / equity if equity else Decimal(1)
        return RiskRuleResult(
            rule_id="RISK_006_max_gross_exposure",
            passed=projected <= self.limits.max_gross_exposure,
            observed=f"{projected:.4f}",
            limit=f"{self.limits.max_gross_exposure}",
        )

    def _max_open_positions(self, open_positions: int) -> RiskRuleResult:
        return RiskRuleResult(
            rule_id="RISK_007_max_open_positions",
            passed=open_positions < self.limits.max_open_positions,
            observed=str(open_positions),
            limit=str(self.limits.max_open_positions),
        )

    def _sufficient_cash(self, i: RiskInputs, order_value: Decimal) -> RiskRuleResult:
        if i.request.side is Side.SELL:
            return RiskRuleResult(
                rule_id="RISK_008_buying_power",
                passed=True,
                observed="n/a",
                limit="n/a",
                detail="sell releases cash",
            )
        return RiskRuleResult(
            rule_id="RISK_008_buying_power",
            passed=order_value <= i.cash.amount,
            observed=f"{order_value:.2f}",
            limit=f"{i.cash.amount:.2f}",
        )

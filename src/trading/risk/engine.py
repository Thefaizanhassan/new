"""The risk engine.

An adversarial component whose job is to assume every other part of the system
is buggy.  It is the only thing that can mint a :class:`RiskDecision`, and
therefore the only route to an :class:`Order` existing at all (Phase 0 §13.3).

Three properties it must have:

**Fail closed.**  A rule that raises is a rejection, not a skip.  If the engine
cannot establish that a trade is safe, the trade does not happen.

**Everything is recorded.**  Every rule's observed value and limit is kept
whether it passed or failed, so a rejection is explainable months later and a
limit that is quietly strangling a strategy is visible rather than inferred.

**A halt outranks every rule.**  Once a monitor has halted trading, no
combination of individually-passing rules can produce an order.
"""

from __future__ import annotations

from pathlib import Path

from trading.config.compliance import ComplianceProfile
from trading.core.risk import RiskDecision, RiskRuleResult
from trading.observability.logging import get_logger
from trading.risk.context import RiskContext
from trading.risk.monitors import (
    HaltLevel,
    HaltState,
    MonitorReading,
    MonitorResult,
    RiskMonitorSet,
)
from trading.risk.profiles import RiskLimits
from trading.risk.rules import RiskRule, compliance_rules, standard_rules

__all__ = ["RiskEngine"]

log = get_logger(__name__)


class RiskEngine:
    ISSUER = "risk_engine"

    def __init__(
        self,
        limits: RiskLimits | None = None,
        compliance: ComplianceProfile | None = None,
        *,
        kill_switch_path: Path | None = None,
        rules: list[RiskRule] | None = None,
        monitors: RiskMonitorSet | None = None,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.compliance = compliance
        self.kill_switch_path = kill_switch_path or Path("KILL")

        if rules is not None:
            self.rules = rules
        else:
            self.rules = standard_rules(self.limits, kill_switch_path=self.kill_switch_path)
            if compliance is not None:
                self.rules += compliance_rules(compliance)

        self.monitors = monitors or RiskMonitorSet(self.limits)

    # ── pre-trade ───────────────────────────────────────────────────────────
    def evaluate(self, ctx: RiskContext) -> RiskDecision:
        """Judge one proposed order against every rule."""
        results: list[RiskRuleResult] = []

        halt = self._halt_result()
        if halt is not None:
            results.append(halt)

        for rule in self.rules:
            try:
                results.append(rule.evaluate(ctx))
            except Exception as exc:  # a rule that cannot answer is never a pass
                results.append(
                    RiskRuleResult(
                        rule_id=getattr(rule, "rule_id", "RISK_000_unknown"),
                        passed=False,
                        observed=f"{type(exc).__name__}: {exc}",
                        limit="no exception",
                        detail="rule failed to evaluate; failing closed",
                    )
                )

        frozen = tuple(results)
        if all(r.passed for r in frozen):
            return RiskDecision.approve(self.ISSUER, ctx.now, frozen)

        decision = RiskDecision.reject(self.ISSUER, ctx.now, frozen)
        log.info(
            "risk_rejected",
            instrument=str(ctx.request.instrument.id),
            strategy=ctx.request.strategy_id,
            reason=decision.rejection_reason[:200],
        )
        return decision

    def _halt_result(self) -> RiskRuleResult | None:
        """A halt is checked first and outranks every other rule."""
        if not self.monitors.halt.engaged:
            return None
        return RiskRuleResult(
            rule_id="RISK_000_halt_engaged",
            passed=False,
            observed=str(self.monitors.halt.level),
            limit="NONE",
            detail=self.monitors.halt.summary(),
        )

    # ── continuous ──────────────────────────────────────────────────────────
    def check_monitors(self, reading: MonitorReading) -> list[MonitorResult]:
        """Evaluate portfolio-level monitors, escalating the halt if any fire."""
        before = self.monitors.halt.level
        results = self.monitors.check(reading)
        if self.monitors.halt.level > before:
            log.warning(
                "trading_halted",
                level=str(self.monitors.halt.level),
                reason=self.monitors.halt.summary(),
            )
        return results

    @property
    def halt(self) -> HaltState:
        return self.monitors.halt

    @property
    def halted(self) -> bool:
        return self.monitors.halt.engaged

    def release_halt(self, *, acknowledged_by: str) -> None:
        """Clear a halt. Deliberately explicit and attributed.

        A halt never clears itself: whatever tripped it needs a human to look,
        and an automatic resume would re-enter the situation that caused it.
        """
        if not self.monitors.halt.engaged:
            return
        log.warning(
            "halt_released",
            previous=str(self.monitors.halt.level),
            reason=self.monitors.halt.summary(),
            acknowledged_by=acknowledged_by,
        )
        self.monitors.halt = self.monitors.halt.release()

    def engage_halt(self, level: HaltLevel, reason: str, now: object) -> None:
        """Manual halt — the operator-facing kill switch (layers 1 and 2)."""
        manual = MonitorResult("MANUAL_halt", level, reason, "NONE", "engaged by operator")
        self.monitors.halt = self.monitors.halt.apply([manual], now)  # type: ignore[arg-type]
        log.warning("trading_halted", level=str(level), reason=reason)

    def describe_rules(self) -> list[dict[str, str]]:
        """Every rule in evaluation order, for the CLI and the run manifest."""
        return [
            {
                "rule_id": getattr(rule, "rule_id", "unknown"),
                "type": type(rule).__name__,
                "doc": (type(rule).__doc__ or "").strip().split("\n")[0],
            }
            for rule in self.rules
        ]

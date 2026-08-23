"""Pre-trade risk rules.

Each rule is a small object with an id, so it can be tested on its own,
switched off deliberately rather than by accident, and named in a rejection
that a human can act on.

Every rule records the **observed value and the limit**, whether it passes or
fails.  That is what makes Phase 0 §16's decision chain readable months later,
and what distinguishes a well-calibrated limit from one quietly strangling a
strategy (§3.10).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Protocol

from trading.config.compliance import ComplianceProfile
from trading.core.risk import RiskRuleResult
from trading.core.types import Side, TradingMode
from trading.risk.context import RiskContext
from trading.risk.correlation import correlation_adjusted_exposure
from trading.risk.profiles import RiskLimits

__all__ = ["RiskRule", "compliance_rules", "standard_rules"]


class RiskRule(Protocol):
    """Implementations are frozen dataclasses, so the id is read-only."""

    @property
    def rule_id(self) -> str: ...

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult: ...


def _result(
    rule_id: str, passed: bool, observed: object, limit: object, detail: str = ""
) -> RiskRuleResult:
    return RiskRuleResult(
        rule_id=rule_id, passed=passed, observed=str(observed), limit=str(limit), detail=detail
    )


# ── safety and mode ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class KillSwitchRule:
    """Layer 3 of the kill switch (Phase 0 §13.4).

    A filesystem sentinel checked at the lowest level of the order path, so it
    still works when the strategy layer is wedged or looping. An in-app flag
    fails exactly when you need it, because the thing that is wrong is the app.
    """

    path: Path
    rule_id: str = "RISK_001_kill_switch"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        engaged = self.path.exists()
        return _result(
            self.rule_id,
            not engaged,
            "ENGAGED" if engaged else "clear",
            "clear",
            f"sentinel present at {self.path}" if engaged else "",
        )


@dataclass(frozen=True)
class TradingModeRule:
    rule_id: str = "RISK_002_trading_mode"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        allowed = ctx.mode in (TradingMode.BACKTEST, TradingMode.PAPER, TradingMode.LIVE)
        return _result(
            self.rule_id,
            allowed,
            ctx.mode,
            "BACKTEST | PAPER | LIVE",
            "RESEARCH mode does not place orders" if not allowed else "",
        )


@dataclass(frozen=True)
class MarketSessionRule:
    """Orders outside a session are rejected or queued by the broker, never filled.

    Only enforced in live-like modes: a backtest replays bars that are, by
    construction, from real sessions.
    """

    rule_id: str = "RISK_003_market_session"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        if not ctx.is_live_like:
            return _result(self.rule_id, True, "n/a", "n/a", "not enforced in backtest")
        if ctx.session_open is None:
            return _result(
                self.rule_id,
                False,
                "unknown",
                "open",
                "the exchange calendar was not consulted; failing closed",
            )
        return _result(self.rule_id, ctx.session_open, ctx.session_open, "open")


@dataclass(frozen=True)
class DataStalenessRule:
    """Trading on a stale quote is trading on a price that no longer exists."""

    limits: RiskLimits
    rule_id: str = "RISK_004_data_staleness"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        if not ctx.is_live_like:
            return _result(self.rule_id, True, "n/a", "n/a", "not enforced in backtest")
        if ctx.data_age is None:
            return _result(
                self.rule_id,
                False,
                "unknown",
                f"{self.limits.max_data_age_seconds}s",
                "data age was not measured; failing closed",
            )
        age = int(ctx.data_age.total_seconds())
        return _result(
            self.rule_id,
            age <= self.limits.max_data_age_seconds,
            f"{age}s",
            f"{self.limits.max_data_age_seconds}s",
        )


# ── order sizing ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MaxOrderValueRule:
    limits: RiskLimits
    rule_id: str = "RISK_005_max_order_value"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        cap = self.limits.max_order_value
        passed = cap is None or ctx.order_value <= cap
        return _result(self.rule_id, passed, f"{ctx.order_value:.2f}", cap or "unbounded")


@dataclass(frozen=True)
class MinOrderValueRule:
    limits: RiskLimits
    rule_id: str = "RISK_006_min_order_value"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        passed = ctx.order_value >= self.limits.min_order_value
        return _result(
            self.rule_id,
            passed,
            f"{ctx.order_value:.2f}",
            self.limits.min_order_value,
            "below this, fixed costs dominate the trade" if not passed else "",
        )


@dataclass(frozen=True)
class MaxPositionWeightRule:
    limits: RiskLimits
    rule_id: str = "RISK_007_max_position_weight"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        projected = abs(ctx.projected_position_value()) / ctx.equity_amount
        return _result(
            self.rule_id,
            projected <= self.limits.max_position_weight,
            f"{projected:.4f}",
            self.limits.max_position_weight,
        )


@dataclass(frozen=True)
class LimitPriceSanityRule:
    """A limit price far from the market is usually a bug, not an intention.

    A fat-fingered decimal point or a stale reference price produces an order
    that either never fills or fills catastrophically.
    """

    limits: RiskLimits
    rule_id: str = "RISK_008_limit_price_sanity"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        limit_price = ctx.request.limit_price
        if limit_price is None:
            return _result(self.rule_id, True, "market order", "n/a")
        if ctx.reference_price <= 0:
            return _result(self.rule_id, False, "no reference price", "a positive reference")
        deviation = abs(limit_price - ctx.reference_price) / ctx.reference_price
        return _result(
            self.rule_id,
            deviation <= self.limits.max_limit_price_deviation,
            f"{deviation:.4f}",
            self.limits.max_limit_price_deviation,
            f"limit {limit_price} vs reference {ctx.reference_price}",
        )


# ── portfolio exposure ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class MaxGrossExposureRule:
    limits: RiskLimits
    rule_id: str = "RISK_010_max_gross_exposure"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        projected = ctx.projected_gross_exposure() / ctx.equity_amount
        return _result(
            self.rule_id,
            projected <= self.limits.max_gross_exposure,
            f"{projected:.4f}",
            self.limits.max_gross_exposure,
        )


@dataclass(frozen=True)
class MaxNetExposureRule:
    limits: RiskLimits
    rule_id: str = "RISK_011_max_net_exposure"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        projected = abs(ctx.projected_net_exposure()) / ctx.equity_amount
        return _result(
            self.rule_id,
            projected <= self.limits.max_net_exposure,
            f"{projected:.4f}",
            self.limits.max_net_exposure,
        )


@dataclass(frozen=True)
class CorrelatedExposureRule:
    """What the book is actually betting, rather than what it nominally holds.

    Without return history this assumes perfect correlation — the conservative
    reading — because assuming unmeasured diversification is how a concentrated
    book passes an exposure check.
    """

    limits: RiskLimits
    rule_id: str = "RISK_012_correlated_exposure"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        weights = ctx.projected_weights()
        breakdown = correlation_adjusted_exposure(weights, ctx.returns)
        detail = f"nominal {breakdown.nominal:.4f}" + (
            " — no return history, assumed fully correlated"
            if breakdown.assumed_correlated
            else f", {breakdown.observations} observations"
        )
        return _result(
            self.rule_id,
            breakdown.adjusted <= self.limits.max_correlated_exposure,
            f"{breakdown.adjusted:.4f}",
            self.limits.max_correlated_exposure,
            detail,
        )


@dataclass(frozen=True)
class MaxLeverageRule:
    limits: RiskLimits
    rule_id: str = "RISK_013_max_leverage"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        projected = ctx.projected_gross_exposure() / ctx.equity_amount
        return _result(
            self.rule_id,
            projected <= self.limits.max_leverage,
            f"{projected:.4f}x",
            f"{self.limits.max_leverage}x",
        )


@dataclass(frozen=True)
class MaxOpenPositionsRule:
    limits: RiskLimits
    rule_id: str = "RISK_014_max_open_positions"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        projected = ctx.projected_open_positions()
        return _result(
            self.rule_id,
            projected <= self.limits.max_open_positions,
            projected,
            self.limits.max_open_positions,
        )


@dataclass(frozen=True)
class StrategyAllocationRule:
    """One strategy must not consume the portfolio."""

    limits: RiskLimits
    rule_id: str = "RISK_015_strategy_allocation"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        strategy_id = ctx.request.strategy_id
        allocated = ctx.portfolio.strategy_allocations.get(strategy_id)
        used = allocated.amount if allocated else Decimal(0)
        # A closing order frees allocation rather than consuming more of it.
        projected = max(used + ctx.signed_order_value, Decimal(0)) / ctx.equity_amount
        return _result(
            self.rule_id,
            projected <= self.limits.max_strategy_allocation,
            f"{projected:.4f}",
            self.limits.max_strategy_allocation,
            f"strategy {strategy_id}",
        )


@dataclass(frozen=True)
class BuyingPowerRule:
    rule_id: str = "RISK_016_buying_power"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        if ctx.request.side is Side.SELL:
            return _result(self.rule_id, True, "n/a", "n/a", "a sell releases cash")
        cash = ctx.portfolio.cash.amount
        return _result(
            self.rule_id, ctx.order_value <= cash, f"{ctx.order_value:.2f}", f"{cash:.2f}"
        )


@dataclass(frozen=True)
class LiquidityRule:
    """Above roughly 1% of average daily volume your own order moves the price.

    This is also what stops a backtest from 'buying' ₹20 lakh of a stock that
    trades ₹5 lakh a day and reporting a spectacular return.
    """

    limits: RiskLimits
    rule_id: str = "RISK_017_liquidity"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        adv = ctx.average_daily_volume
        if adv is None or adv <= 0:
            return _result(
                self.rule_id,
                True,
                "unknown",
                self.limits.max_adv_participation,
                "average daily volume unavailable; not enforced",
            )
        participation = ctx.request.quantity / adv
        return _result(
            self.rule_id,
            participation <= self.limits.max_adv_participation,
            f"{participation:.5f}",
            self.limits.max_adv_participation,
        )


@dataclass(frozen=True)
class SymbolUniverseRule:
    limits: RiskLimits
    rule_id: str = "RISK_018_symbol_universe"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        symbol = str(ctx.request.instrument.id)
        if symbol in self.limits.denied_symbols:
            return _result(self.rule_id, False, symbol, "not on the denylist", "explicitly denied")
        if self.limits.allowed_symbols and symbol not in self.limits.allowed_symbols:
            return _result(
                self.rule_id,
                False,
                symbol,
                f"one of {len(self.limits.allowed_symbols)} allowed",
                "an allowlist is exhaustive",
            )
        return _result(self.rule_id, True, symbol, "permitted")


@dataclass(frozen=True)
class DuplicateOrderRule:
    """Catches the same order being submitted twice in quick succession.

    Not a substitute for client-order-id idempotency at the broker layer — that
    handles the network-timeout case (Phase 0 §3.5). This catches the *other*
    cause: a strategy or a UI firing twice.
    """

    window_seconds: int = 60
    rule_id: str = "RISK_019_duplicate_order"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        fingerprint = (
            str(ctx.request.instrument.id),
            str(ctx.request.side),
            str(ctx.request.quantity),
        )
        duplicates = [
            record
            for record in ctx.recent_orders
            if record.fingerprint() == fingerprint
            and (ctx.now - record.submitted_at).total_seconds() <= self.window_seconds
        ]
        return _result(
            self.rule_id,
            not duplicates,
            len(duplicates),
            0,
            f"identical order within {self.window_seconds}s: {duplicates[0].client_order_id}"
            if duplicates
            else "",
        )


# ── compliance, expressed as risk rules ─────────────────────────────────────
@dataclass(frozen=True)
class OrderRateRule:
    """SEBI's retail algo threshold, enforced rather than documented.

    Staying below it is what keeps a self-developed algo exempt from exchange
    registration. Crossing it is a regulatory event, so it is a hard stop.
    """

    compliance: ComplianceProfile
    rule_id: str = "COMP_001_order_rate"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        cap = self.compliance.max_orders_per_second
        if cap is None:
            return _result(self.rule_id, True, "n/a", "unrestricted")
        in_window = sum(
            1
            for record in ctx.recent_orders
            if (ctx.now - record.submitted_at).total_seconds() < 1.0
        )
        return _result(
            self.rule_id,
            in_window + 1 <= cap,
            in_window + 1,
            cap,
            self.compliance.max_orders_per_second_note[:120],
        )


@dataclass(frozen=True)
class PatternDayTraderRule:
    """The US PDT rule: >3 day trades in 5 business days needs $25,000 equity.

    A capital gate rather than an engineering one — the architecture cannot
    solve it, so the risk engine simply refuses the trade that would breach it.
    """

    compliance: ComplianceProfile
    rule_id: str = "COMP_002_pattern_day_trader"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        if not self.compliance.pattern_day_trader_rule:
            return _result(self.rule_id, True, "n/a", "not applicable in this jurisdiction")
        minimum = self.compliance.pdt_min_equity or Decimal(0)
        maximum = self.compliance.pdt_max_day_trades or 3
        if ctx.portfolio.equity.amount >= minimum:
            return _result(
                self.rule_id,
                True,
                f"{ctx.portfolio.equity.amount:.0f}",
                f">= {minimum}",
                "equity is above the PDT threshold",
            )
        return _result(
            self.rule_id,
            ctx.day_trades_in_window < maximum,
            ctx.day_trades_in_window,
            maximum,
            f"equity {ctx.portfolio.equity.amount:.0f} is below the {minimum} PDT minimum",
        )


@dataclass(frozen=True)
class LiveConnectivityRule:
    """SEBI requires API orders from a whitelisted static IP.

    Live trading is not implemented, so this refuses outright rather than
    pretending the requirement is satisfied.
    """

    compliance: ComplianceProfile
    rule_id: str = "COMP_003_live_connectivity"

    def evaluate(self, ctx: RiskContext) -> RiskRuleResult:
        if ctx.mode is not TradingMode.LIVE:
            return _result(self.rule_id, True, ctx.mode, "n/a outside LIVE")
        if not self.compliance.requires_static_ip:
            return _result(self.rule_id, True, "not required", "n/a")
        return _result(
            self.rule_id,
            False,
            "unverified",
            "a whitelisted static IP",
            "static-IP verification arrives with the broker adapter in Phase 9",
        )


def standard_rules(limits: RiskLimits, *, kill_switch_path: Path) -> list[RiskRule]:
    """The pre-trade rule set, in evaluation order."""
    return [
        KillSwitchRule(kill_switch_path),
        TradingModeRule(),
        MarketSessionRule(),
        DataStalenessRule(limits),
        MaxOrderValueRule(limits),
        MinOrderValueRule(limits),
        MaxPositionWeightRule(limits),
        LimitPriceSanityRule(limits),
        MaxGrossExposureRule(limits),
        MaxNetExposureRule(limits),
        CorrelatedExposureRule(limits),
        MaxLeverageRule(limits),
        MaxOpenPositionsRule(limits),
        StrategyAllocationRule(limits),
        BuyingPowerRule(),
        LiquidityRule(limits),
        SymbolUniverseRule(limits),
        DuplicateOrderRule(),
    ]


def compliance_rules(compliance: ComplianceProfile) -> list[RiskRule]:
    """Jurisdiction rules, enforced by the same engine as everything else."""
    return [
        OrderRateRule(compliance),
        PatternDayTraderRule(compliance),
        LiveConnectivityRule(compliance),
    ]

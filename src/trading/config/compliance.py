"""Compliance profiles.

Regulatory constraints are **risk rules**, not documentation.  They live here as
data so the risk engine can enforce them the same way it enforces a position
limit, and so a run manifest records which regime a backtest assumed.

India: SEBI's retail algo framework has been mandatory for all brokers since
1 April 2026.  US: the Pattern Day Trader rule.  Both are verified facts, with
sources in the Phase 0 addendum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from trading.core.types import Currency, Market

__all__ = ["INDIA_SEBI", "US_RETAIL", "ComplianceProfile", "profile_for"]


@dataclass(frozen=True, slots=True)
class ComplianceProfile:
    """Jurisdiction-specific constraints the risk engine must enforce."""

    profile_id: str
    market: Market
    regulator: str
    currency: Currency
    version: str

    # ── order-rate limits ───────────────────────────────────────────────────
    max_orders_per_second: int | None = None
    max_orders_per_second_note: str = ""

    # ── day-trading limits ──────────────────────────────────────────────────
    pattern_day_trader_rule: bool = False
    pdt_min_equity: Decimal | None = None
    pdt_max_day_trades: int | None = None
    pdt_window_days: int | None = None

    # ── connectivity and session obligations ────────────────────────────────
    requires_static_ip: bool = False
    requires_daily_session_logout: bool = False
    requires_algo_id_tagging: bool = False
    requires_domestic_hosting: bool = False

    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict[str, str]:
        return {
            "profile_id": self.profile_id,
            "regulator": self.regulator,
            "version": self.version,
            "max_orders_per_second": str(self.max_orders_per_second or "unrestricted"),
            "pattern_day_trader_rule": str(self.pattern_day_trader_rule),
            "requires_static_ip": str(self.requires_static_ip),
            "requires_domestic_hosting": str(self.requires_domestic_hosting),
        }


INDIA_SEBI = ComplianceProfile(
    profile_id="india_sebi_retail_algo",
    market=Market.INDIA,
    regulator="SEBI",
    currency=Currency.INR,
    version="2026.04.01",
    max_orders_per_second=10,
    max_orders_per_second_note=(
        "Self-developed algos trading your own account need no exchange registration "
        "below this threshold, measured per exchange within any calendar second. "
        "Above it, registration through your broker and an exchange-assigned Algo-ID "
        "are required."
    ),
    requires_static_ip=True,
    requires_daily_session_logout=True,
    requires_algo_id_tagging=True,
    requires_domestic_hosting=True,
    notes=(
        "Mandatory for all brokers since 2026-04-01.",
        "API orders must originate from an IP whitelisted with your broker.",
        "API sessions must log out daily before each trading day.",
        "Retail algos are expected to be hosted on Indian servers — a laptop on "
        "home broadband is not a compliant live-trading host.",
    ),
)

US_RETAIL = ComplianceProfile(
    profile_id="us_retail",
    market=Market.US,
    regulator="SEC/FINRA",
    currency=Currency.USD,
    version="2026.01.01",
    pattern_day_trader_rule=True,
    pdt_min_equity=Decimal("25000"),
    pdt_max_day_trades=3,
    pdt_window_days=5,
    notes=(
        "More than 3 day trades in 5 business days in a margin account requires "
        "$25,000 minimum equity.",
        "A cash account is PDT-exempt but settles T+1, which caps you near one "
        "round trip per dollar per day.",
        "Wash sale rule: losses disallowed if repurchased within 30 days.",
    ),
)

_PROFILES = {Market.INDIA: INDIA_SEBI, Market.US: US_RETAIL}


def profile_for(market: Market) -> ComplianceProfile:
    return _PROFILES[market]

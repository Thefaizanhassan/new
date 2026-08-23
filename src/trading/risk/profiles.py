"""Risk limits as configuration.

Limits are data, not constants in the engine, so they can be versioned per
strategy and recorded in a run manifest.  A backtest that does not record the
limits it ran under is not reproducible: the same strategy under a 10% position
cap and a 25% cap is, in every way that matters, two different strategies.

Defaults are deliberately tight.  Loosening one should be a decision someone
made, not something that happened by inheriting a permissive default.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["RiskLimits", "load_risk_profile"]


class RiskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = "default"
    version: str = "1.0.0"

    # ── position and exposure ───────────────────────────────────────────────
    max_position_weight: Decimal = Field(default=Decimal("0.25"), gt=0, le=1)
    max_gross_exposure: Decimal = Field(default=Decimal("0.80"), gt=0)
    max_net_exposure: Decimal = Field(default=Decimal("0.80"), gt=0)
    max_correlated_exposure: Decimal = Field(default=Decimal("0.60"), gt=0)
    """Cap on `sqrt(wᵀCw)` — what the book is actually betting, not what it holds."""
    max_leverage: Decimal = Field(default=Decimal("1.0"), ge=1)
    max_open_positions: int = Field(default=10, ge=1)
    max_strategy_allocation: Decimal = Field(default=Decimal("0.50"), gt=0, le=1)

    # ── order sizing ────────────────────────────────────────────────────────
    max_order_value: Decimal | None = None
    min_order_value: Decimal = Field(default=Decimal("1000"), ge=0)
    """Below this, fixed costs dominate: a flat ₹15 depository fee on a ₹5,000
    position is 0.3% per round trip. Rejecting tiny orders is cost control."""
    max_limit_price_deviation: Decimal = Field(default=Decimal("0.05"), gt=0)
    max_adv_participation: Decimal = Field(default=Decimal("0.01"), gt=0, le=1)
    """Order size as a fraction of average daily volume. Above ~1% your own
    order starts moving the price against you."""

    # ── continuous limits ───────────────────────────────────────────────────
    max_daily_loss: Decimal = Field(default=Decimal("0.02"), gt=0)
    """Mark-to-market, including unrealised. A realised-only limit is a hole
    you can drive a portfolio through."""
    max_drawdown: Decimal = Field(default=Decimal("0.15"), gt=0)
    max_consecutive_losses: int = Field(default=8, ge=1)
    max_error_rate: int = Field(default=5, ge=1)
    max_data_age_seconds: int = Field(default=300, ge=1)

    # ── universe ────────────────────────────────────────────────────────────
    allowed_symbols: tuple[str, ...] = ()
    """Empty means no allowlist. A non-empty list is exhaustive."""
    denied_symbols: tuple[str, ...] = ()

    def manifest_entry(self) -> dict[str, str]:
        return {
            "risk_profile_id": self.profile_id,
            "risk_profile_version": self.version,
            **{
                f"risk_{k}": str(v)
                for k, v in self.model_dump().items()
                if k not in ("profile_id", "version")
            },
        }


def load_risk_profile(path: Path | str) -> RiskLimits:
    """Load limits from YAML. Unknown keys are errors, not silent no-ops."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return RiskLimits(**raw)

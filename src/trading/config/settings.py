"""Application configuration.

Environment-driven and validated at startup, so a misconfiguration fails
immediately and loudly rather than at the moment an order is about to be sent.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from trading.config.compliance import ComplianceProfile, profile_for
from trading.core.types import Currency, Market, TradingMode

__all__ = ["Settings", "load_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRADING_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    mode: TradingMode = TradingMode.RESEARCH
    market: Market = Market.INDIA
    base_currency: Currency = Currency.INR
    starting_capital: Decimal = Decimal("400000")

    database_url: str | None = None
    data_dir: Path = Path("./data")

    log_level: str = "INFO"
    log_format: str = Field(default="console", pattern="^(console|json)$")

    @field_validator("starting_capital")
    @classmethod
    def _positive_capital(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("starting_capital must be positive")
        return v

    @model_validator(mode="after")
    def _guard_live_mode(self) -> Settings:
        """LIVE is not implemented and must not be reachable by config alone.

        Phase 0 §15.2: reaching LIVE requires an explicit config value, a
        separate credential set, a per-strategy lifecycle status, a capital cap,
        a clean startup reconciliation and an interactive confirmation. None of
        that exists yet, so the mode is refused outright rather than partially
        honoured.
        """
        if self.mode is TradingMode.LIVE:
            raise ValueError(
                "TRADING_MODE=LIVE is refused: live trading is not implemented "
                "(roadmap Phase 12) and its safety gates do not exist yet."
            )
        return self

    @model_validator(mode="after")
    def _currency_matches_market(self) -> Settings:
        expected = self.market.default_currency
        if self.base_currency is not expected:
            raise ValueError(
                f"base_currency {self.base_currency} does not match market "
                f"{self.market} (expected {expected}). Multi-currency reporting "
                f"arrives with the second market adapter in Phase 9."
            )
        return self

    @property
    def compliance(self) -> ComplianceProfile:
        return profile_for(self.market)

    @property
    def uses_postgres(self) -> bool:
        return bool(self.database_url)

    def describe(self) -> dict[str, str]:
        return {
            "mode": self.mode,
            "market": self.market,
            "base_currency": self.base_currency,
            "starting_capital": str(self.starting_capital),
            "compliance_profile": self.compliance.profile_id,
            "event_store": "postgres" if self.uses_postgres else "jsonl",
        }


def load_settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]

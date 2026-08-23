"""Strategies defined in YAML rather than Python.

Phase 0 §10.7 route ②.  A config strategy is a real :class:`Strategy` — it goes
through the same risk engine, the same lifecycle gates and the same run
manifest.  What it avoids is a code deploy for a parameter change, and it is the
**only** form an AI-drafted strategy may take, because a validated config is
auditable in a way that generated Python is not.

Rules are written in the restricted expression language
(:mod:`trading.strategies.expressions`), so a config file cannot execute
arbitrary code.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Flat, Intent, TargetWeight
from trading.core.types import Timeframe
from trading.features.engine import FeatureSpec
from trading.strategies.base import LifecycleStatus, StrategyContext, StrategySpec
from trading.strategies.expressions import ExpressionError, compile_expression

__all__ = ["ConfigStrategy", "StrategyConfig", "load_strategy"]


class RuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    when: str = Field(min_length=1)

    @field_validator("when")
    @classmethod
    def _must_compile(cls, value: str) -> str:
        """Compile at load time so a bad rule fails now, not mid-backtest."""
        compile_expression(value)
        return value


class EntryConfig(RuleConfig):
    target_weight: Decimal = Field(gt=0, le=1)


class StrategyConfig(BaseModel):
    """The YAML schema. ``extra="forbid"`` so a typo is an error, not a silent no-op."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    name: str
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: str = ""
    author: str = "unknown"
    universe: list[str] = Field(min_length=1)
    timeframe: Timeframe = Timeframe.DAY_1
    warmup_bars: int = Field(default=200, ge=1)
    entry: EntryConfig
    exit: RuleConfig
    lifecycle_status: LifecycleStatus = LifecycleStatus.RESEARCH

    @field_validator("universe")
    @classmethod
    def _parse_instruments(cls, value: list[str]) -> list[str]:
        for symbol in value:
            InstrumentId.parse(symbol)
        return value

    @property
    def instruments(self) -> tuple[InstrumentId, ...]:
        return tuple(InstrumentId.parse(s) for s in self.universe)


class ConfigStrategy:
    """A :class:`Strategy` built from a :class:`StrategyConfig`."""

    ENTRY_FEATURE = "_entry"
    EXIT_FEATURE = "_exit"

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self._entry = compile_expression(config.entry.when)
        self._exit = compile_expression(config.exit.when)
        self.spec = StrategySpec(
            id=config.id,
            name=config.name,
            version=config.version,
            description=config.description,
            universe=config.instruments,
            params={
                "entry": config.entry.when,
                "exit": config.exit.when,
                "target_weight": config.entry.target_weight,
                "warmup_bars": config.warmup_bars,
            },
            lifecycle_status=config.lifecycle_status,
            author=config.author,
        )

    def warmup_bars(self) -> int:
        return self.config.warmup_bars

    def feature_specs(self) -> list[FeatureSpec]:
        """Declared so the engine can precompute them once instead of per bar."""
        return [
            FeatureSpec(self.ENTRY_FEATURE, self.config.entry.when, as_condition=True),
            FeatureSpec(self.EXIT_FEATURE, self.config.exit.when, as_condition=True),
        ]

    def _evaluate(self, ctx: StrategyContext, instrument: Instrument, name: str) -> bool:
        """Read a precomputed feature, falling back to computing it here.

        Precomputed is the fast path; the fallback keeps the strategy correct
        when it is used outside the engine, such as in a unit test.
        """
        features = ctx.features(instrument.id)
        if not features.empty and name in features.columns:
            return bool(features[name].iloc[-1])

        history = ctx.history(instrument.id)
        compiled = self._entry if name == self.ENTRY_FEATURE else self._exit
        return bool(compiled(history).iloc[-1])

    def on_bar(self, ctx: StrategyContext, instrument: Instrument) -> list[Intent]:
        history = ctx.history(instrument.id)
        if len(history) < self.config.warmup_bars:
            return []

        holding = ctx.has_position(instrument.id)
        try:
            entry = self._evaluate(ctx, instrument, self.ENTRY_FEATURE)
            exit_ = self._evaluate(ctx, instrument, self.EXIT_FEATURE)
        except ExpressionError:
            # Fail closed: a rule that cannot be evaluated produces no intent
            # rather than a default action.
            return []

        evidence: dict[str, Any] = {
            "entry_rule": self.config.entry.when,
            "exit_rule": self.config.exit.when,
            "entry_true": entry,
            "exit_true": exit_,
        }

        if entry and not holding:
            return [
                Intent(
                    instrument_id=instrument.id,
                    target=TargetWeight(self.config.entry.target_weight),
                    strategy_id=self.spec.id,
                    reason=f"entry rule matched: {self.config.entry.when}",
                    evidence=evidence,
                )
            ]
        if exit_ and holding:
            return [
                Intent(
                    instrument_id=instrument.id,
                    target=Flat(),
                    strategy_id=self.spec.id,
                    reason=f"exit rule matched: {self.config.exit.when}",
                    evidence=evidence,
                )
            ]
        return []

    def explain(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Where each rule fired over a history. For inspecting a rule before trusting it."""
        return pd.DataFrame(
            {"entry": self._entry(bars), "exit": self._exit(bars)}, index=bars.index
        )


def load_strategy(path: Path | str) -> ConfigStrategy:
    """Load and validate a YAML strategy. Raises before anything can trade."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return ConfigStrategy(StrategyConfig(**raw))

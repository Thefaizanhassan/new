"""Strategies defined in YAML rather than Python.

Phase 0 §10.7 route ②.  A config strategy is a real :class:`Strategy` — it goes
through the same risk engine, the same lifecycle gates and the same run
manifest.  What it avoids is a code deploy for a parameter change, and it is the
**only** form an AI-drafted strategy may take, because a validated config is
auditable in a way that generated Python is not.

Rules are written in the restricted expression language
(:mod:`trading.strategies.expressions`), so a config file cannot execute
arbitrary code.

**Parameters and why substitution is safe here.**  A ``parameters:`` block
declares named numbers that ``{placeholders}`` in the rule text resolve against,
so a sweep can vary ``{fast}`` from 10 to 100 without twenty near-identical YAML
files.  Substituting text into a rule is exactly the shape of an injection bug,
so two things constrain it: parameter **values must be numeric**, which means a
substituted value cannot contain an operator, a call or a parenthesis; and the
rendered rule still goes through the expression allowlist, which is the same gate
a hand-written rule passes.  Either alone would be enough; both is deliberate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Flat, Intent, TargetWeight
from trading.core.types import Timeframe
from trading.features.engine import FeatureSpec
from trading.strategies.base import LifecycleStatus, StrategyContext, StrategySpec
from trading.strategies.expressions import ExpressionError, compile_expression

__all__ = [
    "ConfigStrategy",
    "ParameterError",
    "StrategyConfig",
    "load_strategy",
    "render_rule",
]


_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_PARAM_NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

ParameterValue = int | float | Decimal
"""Numeric only. See the module docstring: this is what makes substitution safe."""


class ParameterError(ValueError):
    """A parameter block or override that cannot be applied."""


def _render_number(value: ParameterValue) -> str:
    """Render a parameter for substitution into a rule.

    Integral values render without a decimal point so ``sma(close, {fast})``
    becomes ``sma(close, 50)`` rather than ``sma(close, 50.0)`` — TA-Lib periods
    are integers, and the expression layer would reject a float there.
    """
    as_decimal = Decimal(str(value))
    if as_decimal == as_decimal.to_integral_value():
        return str(int(as_decimal))
    return format(as_decimal.normalize(), "f")


def render_rule(template: str, parameters: Mapping[str, ParameterValue]) -> str:
    """Substitute ``{name}`` placeholders in a rule template.

    Every placeholder must have a value and every value must be used: an
    unresolved placeholder would reach the expression compiler as a syntax error,
    and an unused parameter is almost always a typo in one of the two names — a
    sweep over ``{fast}`` that silently varies nothing is worse than a crash,
    because it produces a flat surface that reads as a perfect plateau.
    """
    referenced = set(_PLACEHOLDER.findall(template))
    missing = referenced - set(parameters)
    if missing:
        raise ParameterError(
            f"rule {template!r} references undeclared parameter(s) "
            f"{sorted(missing)}; declared: {sorted(parameters) or 'none'}"
        )
    rendered = template
    for name in referenced:
        rendered = rendered.replace("{" + name + "}", _render_number(parameters[name]))
    return rendered


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
    parameters: dict[str, Decimal] = Field(default_factory=dict)
    """Named numbers that ``{placeholders}`` in the rules resolve against."""
    entry: EntryConfig
    exit: RuleConfig
    lifecycle_status: LifecycleStatus = LifecycleStatus.RESEARCH

    templates: dict[str, str] = Field(default_factory=dict)
    """The pre-substitution rule text, kept for the manifest. Populated by the
    renderer, not written by hand — a config that sets it is rejected."""

    @model_validator(mode="before")
    @classmethod
    def _render_parameters(cls, data: Any) -> Any:
        """Substitute parameters into the rules *before* anything else validates.

        Order matters: ``RuleConfig`` compiles its rule on assignment, and a rule
        still containing ``{fast}`` is a syntax error. Rendering here means the
        compiled rule, the feature spec, the manifest and the content hash all
        refer to the concrete expression that ran — there is no second, templated
        form floating around that somebody could mistake for what was tested.
        """
        if not isinstance(data, dict):
            return data
        if data.get("templates"):
            raise ParameterError(
                "'templates' is populated by the loader from the rule text; "
                "remove it from the config"
            )
        raw = data.get("parameters") or {}
        if not isinstance(raw, dict):
            raise ParameterError(f"'parameters' must be a mapping, got {type(raw).__name__}")

        parameters: dict[str, Decimal] = {}
        for name, value in raw.items():
            if not _PARAM_NAME.match(str(name)):
                raise ParameterError(
                    f"parameter name {name!r} must be lowercase letters, digits and "
                    "underscores, starting with a letter or underscore"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
                raise ParameterError(
                    f"parameter {name!r} must be a number, got {type(value).__name__} — "
                    "non-numeric values would let a parameter rewrite the rule's shape"
                )
            try:
                parameters[str(name)] = Decimal(str(value))
            except ArithmeticError as exc:
                raise ParameterError(f"parameter {name!r} is not a number: {value!r}") from exc

        templates: dict[str, str] = {}
        for section, key in (("entry", "when"), ("exit", "when")):
            block = data.get(section)
            if not (isinstance(block, dict) and isinstance(block.get(key), str)):
                continue
            template = block[key]
            rendered = render_rule(template, parameters)
            if rendered != template:
                # Only a rule that actually held a placeholder gets a template
                # recorded. Storing one for every rule would put a duplicate of
                # each rule in the manifest and make ``templates`` mean nothing.
                templates[section] = template
                block[key] = rendered

        unused = set(parameters) - {n for t in templates.values() for n in _PLACEHOLDER.findall(t)}
        if unused:
            raise ParameterError(
                f"parameter(s) {sorted(unused)} are declared but never referenced by "
                "a rule; a sweep over an unused parameter varies nothing and its flat "
                "surface reads as a perfect plateau"
            )

        data["parameters"] = parameters
        data["templates"] = templates
        return data

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
                # Both forms travel with the spec: the templates say what was
                # swept, the values say which point of the sweep actually ran.
                **{f"param_{k}": v for k, v in sorted(config.parameters.items())},
                **{f"template_{k}": v for k, v in sorted(config.templates.items())},
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


_OVERRIDABLE_FIELDS = frozenset({"warmup_bars", "target_weight"})
"""Scalars a sweep may set directly, distinct from ``parameters``.

``warmup_bars`` is here for a reason worth knowing: sweeping ``slow`` up to 300
while ``warmup_bars`` stays at 201 leaves the strategy evaluating an average that
is still NaN, which produces no trades and an apparently flat, apparently robust
surface. A sweep over indicator periods must move the warmup with them.
"""


def load_strategy(
    path: Path | str,
    overrides: Mapping[str, ParameterValue] | None = None,
) -> ConfigStrategy:
    """Load and validate a YAML strategy. Raises before anything can trade.

    ``overrides`` sets declared ``parameters`` and the scalars in
    ``_OVERRIDABLE_FIELDS``; anything else is rejected rather than ignored, so a
    sweep over a misspelled name fails instead of silently running the defaults
    twenty times and reporting a perfect plateau.
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")

    if overrides:
        declared = set(raw.get("parameters") or {})
        unknown = set(overrides) - declared - _OVERRIDABLE_FIELDS
        if unknown:
            raise ParameterError(
                f"{path}: cannot override {sorted(unknown)} — declared parameters are "
                f"{sorted(declared) or 'none'} and settable fields are "
                f"{sorted(_OVERRIDABLE_FIELDS)}"
            )
        for name, value in overrides.items():
            if name == "warmup_bars":
                raw["warmup_bars"] = int(value)
            elif name == "target_weight":
                raw.setdefault("entry", {})["target_weight"] = Decimal(str(value))
            else:
                raw.setdefault("parameters", {})[name] = value

    return ConfigStrategy(StrategyConfig(**raw))

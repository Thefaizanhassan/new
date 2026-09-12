"""YAML-defined strategies, the feature engine, and the registry."""

from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from trading.core.fill import CostBreakdown, Fill
from trading.core.instrument import Instrument, InstrumentId
from trading.core.intent import Flat, TargetWeight
from trading.core.position import Position
from trading.core.types import Currency, InstrumentClass, Money, Side
from trading.features.engine import FeatureEngine, FeatureSpec, verify_causality
from trading.strategies.base import LifecycleStatus, StrategyContext
from trading.strategies.config_strategy import (
    ConfigStrategy,
    ParameterError,
    StrategyConfig,
    load_strategy,
)
from trading.strategies.registry import (
    DuplicateStrategyError,
    StrategyNotFoundError,
    StrategyRegistry,
    default_registry,
)

RELIANCE = InstrumentId("NSE", "RELIANCE")

VALID = {
    "id": "test_strategy",
    "name": "Test",
    "version": "1.0.0",
    "universe": ["NSE:RELIANCE"],
    "warmup_bars": 50,
    "entry": {"when": "close > sma(close, 20)", "target_weight": "0.2"},
    "exit": {"when": "close < sma(close, 20)"},
}


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 300
    close = np.linspace(100, 200, n) + np.sin(np.arange(n) / 8) * 15
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=pd.date_range("2024-01-01", periods=n, tz="UTC"),
    )


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=RELIANCE,
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


def write(tmp_path, config: dict) -> str:
    path = tmp_path / "strategy.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


# ── config validation ──────────────────────────────────────────────────────
def test_a_valid_config_loads(tmp_path):
    strategy = load_strategy(write(tmp_path, VALID))
    assert strategy.spec.id == "test_strategy"
    assert strategy.spec.lifecycle_status is LifecycleStatus.RESEARCH


def test_a_malformed_rule_fails_at_load_not_at_trade_time(tmp_path):
    broken = {**VALID, "entry": {"when": "__import__('os')", "target_weight": "0.2"}}
    with pytest.raises(ValueError, match=r"Unknown function"):
        load_strategy(write(tmp_path, broken))


def test_an_attribute_access_in_a_rule_is_rejected_at_load(tmp_path):
    broken = {**VALID, "entry": {"when": "close.values > 0", "target_weight": "0.2"}}
    with pytest.raises(ValueError, match=r"not allowed"):
        load_strategy(write(tmp_path, broken))


def test_an_unknown_field_is_an_error_rather_than_a_silent_no_op(tmp_path):
    typo = {**VALID, "warmup_barz": 50}
    with pytest.raises(ValueError, match=r"warmup_barz|Extra inputs"):
        load_strategy(write(tmp_path, typo))


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", "1.0"), ("id", "Test-Strategy"), ("universe", []), ("warmup_bars", 0)],
)
def test_invalid_metadata_is_rejected(tmp_path, field, value):
    with pytest.raises(ValueError):
        load_strategy(write(tmp_path, {**VALID, field: value}))


def test_a_target_weight_above_full_equity_is_rejected(tmp_path):
    bad = {**VALID, "entry": {"when": "close > 0", "target_weight": "1.5"}}
    with pytest.raises(ValueError):
        load_strategy(write(tmp_path, bad))


def test_an_unqualified_symbol_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"EXCHANGE:SYMBOL"):
        load_strategy(write(tmp_path, {**VALID, "universe": ["RELIANCE"]}))


def test_non_yaml_content_is_reported_clearly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("just a string")
    with pytest.raises(ValueError, match=r"must contain a YAML mapping"):
        load_strategy(path)


# ── behaviour ──────────────────────────────────────────────────────────────
def _context(bars, instrument, *, holding: bool, features=None) -> StrategyContext:
    position = Position(instrument=instrument)
    if holding:
        position = position.apply(
            Fill.create(
                client_order_id="c",
                instrument=instrument,
                side=Side.BUY,
                quantity=Decimal(10),
                price=Decimal(100),
                costs=CostBreakdown.zero(Currency.INR),
                timestamp=bars.index[-1].to_pydatetime(),
            )
        )
    return StrategyContext(
        now=bars.index[-1].to_pydatetime(),
        history={instrument.id: bars},
        positions={instrument.id: position},
        equity=Money.inr("400000"),
        features={instrument.id: features} if features is not None else None,
    )


def test_entry_produces_a_target_weight_when_flat(bars, instrument):
    strategy = ConfigStrategy(StrategyConfig(**VALID))
    rising = bars.iloc[:200]  # trending up: entry rule true
    intents = strategy.on_bar(_context(rising, instrument, holding=False), instrument)
    assert len(intents) == 1
    assert isinstance(intents[0].target, TargetWeight)
    assert intents[0].evidence["entry_rule"] == VALID["entry"]["when"]


def test_no_intent_is_produced_before_warmup(bars, instrument):
    strategy = ConfigStrategy(StrategyConfig(**VALID))
    assert strategy.on_bar(_context(bars.iloc[:10], instrument, holding=False), instrument) == []


def test_exit_flattens_when_holding(bars, instrument):
    config = {
        **VALID,
        "entry": {"when": "close > 1e9", "target_weight": "0.2"},
        "exit": {"when": "close > 0"},
    }
    strategy = ConfigStrategy(StrategyConfig(**config))
    intents = strategy.on_bar(_context(bars, instrument, holding=True), instrument)
    assert len(intents) == 1
    assert isinstance(intents[0].target, Flat)


def test_precomputed_features_and_recomputation_agree(bars, instrument):
    """The optimisation must not change behaviour."""
    strategy = ConfigStrategy(StrategyConfig(**VALID))
    features = FeatureEngine(strategy.feature_specs()).compute(bars)

    without = strategy.on_bar(_context(bars, instrument, holding=False), instrument)
    with_ = strategy.on_bar(
        _context(bars, instrument, holding=False, features=features), instrument
    )
    assert [str(i.target) for i in without] == [str(i.target) for i in with_]


def test_explain_shows_where_each_rule_fired(bars):
    fired = ConfigStrategy(StrategyConfig(**VALID)).explain(bars)
    assert set(fired.columns) == {"entry", "exit"}
    assert fired["entry"].sum() > 0


# ── feature engine ─────────────────────────────────────────────────────────
def test_feature_engine_computes_named_columns(bars):
    engine = FeatureEngine(
        [
            FeatureSpec("fast", "sma(close, 10)"),
            FeatureSpec("above", "close > sma(close, 50)", as_condition=True),
        ]
    )
    frame = engine.compute(bars)
    assert list(frame.columns) == ["fast", "above"]
    assert len(frame) == len(bars)


def test_a_bad_expression_fails_when_the_engine_is_built(bars):
    with pytest.raises(ValueError):
        FeatureEngine([FeatureSpec("bad", "close.__class__")])


@pytest.mark.parametrize(
    "expression",
    [
        "sma(close, 20)",
        "rsi(close, 14)",
        "atr(close, 14)",
        "zscore(close, 20)",
        "prior_max(high, 20)",
        "crossed_above(sma(close, 5), sma(close, 20))",
    ],
)
def test_every_feature_in_the_language_is_causal(bars, expression):
    """Precomputing over full history is only safe if features cannot see ahead."""
    assert verify_causality(FeatureSpec("f", expression), bars)


def test_the_causality_check_actually_catches_a_leak(bars):
    """A guard that never fails is not a guard. `prev(x, -1)` reads one bar ahead."""
    assert not verify_causality(FeatureSpec("leak", "prev(close, -1)"), bars)


def test_causality_needs_enough_bars(bars):
    with pytest.raises(ValueError, match=r"at least 3 bars"):
        verify_causality(FeatureSpec("f", "close"), bars.iloc[:2])


# ── registry ───────────────────────────────────────────────────────────────
def test_registering_the_same_id_and_version_twice_is_an_error(tmp_path):
    registry = StrategyRegistry()

    def factory() -> ConfigStrategy:
        return ConfigStrategy(StrategyConfig(**VALID))

    registry.register(factory)
    with pytest.raises(DuplicateStrategyError, match=r"already registered"):
        registry.register(factory)


def test_get_returns_the_highest_version_by_default():
    registry = StrategyRegistry()
    for version in ("1.0.0", "1.10.0", "1.2.0"):
        registry.register(
            lambda v=version: ConfigStrategy(StrategyConfig(**{**VALID, "version": v}))
        )
    assert registry.get("test_strategy").spec.version == "1.10.0"  # not "1.2.0"


def test_an_unknown_strategy_lists_what_is_registered():
    registry = StrategyRegistry()
    registry.register(lambda: ConfigStrategy(StrategyConfig(**VALID)))
    with pytest.raises(StrategyNotFoundError, match=r"test_strategy"):
        registry.get("nonexistent")


def test_an_unknown_version_lists_the_available_ones():
    registry = StrategyRegistry()
    registry.register(lambda: ConfigStrategy(StrategyConfig(**VALID)))
    with pytest.raises(StrategyNotFoundError, match=r"1.0.0"):
        registry.get("test_strategy", "9.9.9")


def test_loading_a_directory_skips_bad_configs_without_hiding_good_ones(tmp_path):
    (tmp_path / "good.yaml").write_text(yaml.safe_dump(VALID))
    (tmp_path / "bad.yaml").write_text("id: [not, a, string")
    (tmp_path / "worse.yaml").write_text(
        yaml.safe_dump(
            {**VALID, "id": "other", "entry": {"when": "os.system('x')", "target_weight": "0.2"}}
        )
    )
    registry = StrategyRegistry()
    assert registry.load_directory(tmp_path) == 1
    assert registry.get("test_strategy")


def test_the_default_registry_finds_both_python_and_yaml_strategies():
    registry = default_registry()
    ids = {entry.spec.id for entry in registry}
    assert {"buy_and_hold", "sma_cross"} <= ids  # Python
    assert any(entry.origin.endswith(".yaml") for entry in registry)  # YAML


def test_registry_entries_expose_where_they_came_from():
    for row in default_registry().summary():
        assert row["origin"]
        assert row["content_hash"]


# ── parameters (Phase 6) ────────────────────────────────────────────────────
PARAM_YAML = """
id: swept
name: Swept
version: 1.0.0
universe: [NSE:RELIANCE]
warmup_bars: 201
parameters:
  fast: 50
  slow: 200
entry:
  when: crossed_above(sma(close, {fast}), sma(close, {slow}))
  target_weight: 0.20
exit:
  when: crossed_below(sma(close, {fast}), sma(close, {slow}))
"""


def write_param_strategy(tmp_path, **replacements):
    text = PARAM_YAML
    for old, new in replacements.items():
        text = text.replace(old, new)
    path = tmp_path / "swept.yaml"
    path.write_text(text)
    return path


def test_placeholders_are_rendered_before_the_rule_is_compiled(tmp_path):
    strategy = load_strategy(write_param_strategy(tmp_path))
    assert strategy.config.entry.when == "crossed_above(sma(close, 50), sma(close, 200))"
    assert strategy.config.templates["entry"] == (
        "crossed_above(sma(close, {fast}), sma(close, {slow}))"
    )


def test_overrides_change_the_rendered_rule_and_the_content_hash(tmp_path):
    """Two points of a sweep must not share an identity, or their results are
    indistinguishable in the ledger and the manifest."""
    path = write_param_strategy(tmp_path)
    default = load_strategy(path)
    swept = load_strategy(path, overrides={"fast": 20, "slow": 100, "warmup_bars": 101})
    assert swept.config.entry.when == "crossed_above(sma(close, 20), sma(close, 100))"
    assert swept.warmup_bars() == 101
    assert swept.spec.content_hash != default.spec.content_hash


def test_integral_parameters_render_without_a_decimal_point(tmp_path):
    """TA-Lib periods are integers; ``sma(close, 50.0)`` would be rejected."""
    strategy = load_strategy(write_param_strategy(tmp_path), overrides={"fast": 20.0})
    assert "sma(close, 20)" in strategy.config.entry.when
    assert "20.0" not in strategy.config.entry.when


def test_fractional_parameters_survive_rendering(tmp_path):
    path = write_param_strategy(
        tmp_path,
        **{
            "  fast: 50": "  threshold: -2.0",
            "  slow: 200": "  trend: 200",
            "crossed_above(sma(close, {fast}), sma(close, {slow}))": (
                "zscore(close, 20) < {threshold} and close > sma(close, {trend})"
            ),
            "crossed_below(sma(close, {fast}), sma(close, {slow}))": "zscore(close, 20) > 0.0",
        },
    )
    strategy = load_strategy(path, overrides={"threshold": Decimal("-1.75")})
    assert "zscore(close, 20) < -1.75" in strategy.config.entry.when


def test_a_non_numeric_parameter_is_rejected(tmp_path):
    """The injection guard. A numeric value cannot contain an operator or a call."""
    path = write_param_strategy(tmp_path, **{"  fast: 50": "  fast: 50) or (close > 0"})
    with pytest.raises(ValidationError, match="not a number"):
        load_strategy(path)


def test_a_boolean_parameter_is_rejected(tmp_path):
    path = write_param_strategy(tmp_path, **{"  fast: 50": "  fast: true"})
    with pytest.raises(ValidationError, match="must be a number"):
        load_strategy(path)


def test_an_undeclared_placeholder_is_rejected(tmp_path):
    path = write_param_strategy(tmp_path, **{"{slow}": "{missing}"})
    with pytest.raises(ValidationError, match="undeclared parameter"):
        load_strategy(path)


def test_a_declared_but_unreferenced_parameter_is_rejected(tmp_path):
    """A sweep over a parameter no rule reads varies nothing, and its flat surface
    reads as a perfect plateau — worse than a crash."""
    path = write_param_strategy(tmp_path, **{"  slow: 200": "  slow: 200\n  unused: 7"})
    with pytest.raises(ValidationError, match="never referenced"):
        load_strategy(path)


def test_overriding_an_undeclared_name_is_rejected(tmp_path):
    path = write_param_strategy(tmp_path)
    with pytest.raises(ParameterError, match="cannot override"):
        load_strategy(path, overrides={"fst": 20})


def test_target_weight_is_overridable(tmp_path):
    strategy = load_strategy(write_param_strategy(tmp_path), overrides={"target_weight": 0.05})
    assert strategy.config.entry.target_weight == Decimal("0.05")


def test_templates_cannot_be_supplied_by_hand(tmp_path):
    path = write_param_strategy(tmp_path, **{"parameters:": "templates:\n  entry: x\nparameters:"})
    with pytest.raises(ValidationError, match="populated by the loader"):
        load_strategy(path)


def test_a_strategy_with_no_parameters_block_still_loads(tmp_path):
    """Parameters are additive; every strategy written before them must keep working."""
    text = PARAM_YAML.replace("parameters:\n  fast: 50\n  slow: 200\n", "")
    text = text.replace("{fast}", "50").replace("{slow}", "200")
    path = tmp_path / "plain.yaml"
    path.write_text(text)
    strategy = load_strategy(path)
    assert strategy.config.parameters == {}
    assert strategy.config.templates == {}


def test_the_shipped_parameterised_configs_load_and_sweep():
    for path in (
        "configs/strategies/sma_cross_param.yaml",
        "configs/strategies/mean_reversion_param.yaml",
    ):
        base = load_strategy(path)
        assert base.config.parameters, f"{path} declares no parameters"
        assert "{" not in base.config.entry.when
        assert "{" not in base.config.exit.when

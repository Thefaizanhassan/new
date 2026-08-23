"""The restricted expression language.

The rejection tests are the important ones: this evaluator is what stands
between a YAML file — or an LLM-drafted one — and arbitrary code execution.
"""

import numpy as np
import pandas as pd
import pytest

from trading.features import indicators as ind
from trading.strategies.expressions import (
    ExpressionError,
    compile_expression,
    describe_language,
)


@pytest.fixture
def bars() -> pd.DataFrame:
    n = 300
    close = np.linspace(100, 200, n) + np.sin(np.arange(n) / 8) * 10
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, tz="UTC"),
    )


# ── rejection: this is the security boundary ───────────────────────────────
@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').system('rm -rf /')",
        "().__class__.__bases__[0].__subclasses__()",
        "open('/etc/passwd').read()",
        "close.__class__",
        "exec('x=1')",
        "eval('1+1')",
        "globals()",
        "[c for c in ().__class__.__base__.__subclasses__()]",
        "lambda: __import__('os')",
        "f'{close.__class__}'",
        "(close := 5)",
    ],
)
def test_code_execution_attempts_are_rejected_at_compile_time(attack):
    with pytest.raises(ExpressionError):
        compile_expression(attack)


@pytest.mark.parametrize(
    ("expression", "fragment"),
    [
        ("close.values", "Attribute"),
        ("close[0]", "Subscript"),
        ("lambda: 1", "Lambda"),
        ("[x for x in range(3)]", "ListComp"),
        ('f"{close}"', "JoinedStr"),
        ("sma(close, 50) if True else 0", "IfExp"),
        ("unknown_fn(close)", "Unknown function"),
        ("mystery > 5", "Unknown name"),
        ("close > 'text'", "numeric and boolean literals"),
        ("1 < close < 2", "Chained comparisons"),
        ("", "empty"),
        ("close >", "Cannot parse"),
    ],
)
def test_rejections_explain_themselves(expression, fragment):
    """An error a user cannot act on is only marginally better than a crash."""
    with pytest.raises(ExpressionError, match=fragment):
        compile_expression(expression)


def test_a_rejected_expression_never_reaches_evaluation():
    """Validation happens at compile time, not when a strategy is about to trade."""
    with pytest.raises(ExpressionError):
        compile_expression("__import__('os')")


# ── correctness ────────────────────────────────────────────────────────────
def test_comparison_produces_a_boolean_series_aligned_to_the_bars(bars):
    result = compile_expression("close > 150")(bars)
    assert isinstance(result, pd.Series)
    assert result.dtype == bool
    assert len(result) == len(bars)


def test_and_or_not_work_despite_pandas_refusing_bool_on_a_series(bars):
    """Python's and/or call __bool__, which a Series rejects. The AST is rewritten."""
    both = compile_expression("close > 120 and close < 180")(bars)
    either = compile_expression("close < 120 or close > 180")(bars)
    negated = compile_expression("not (close > 150)")(bars)

    assert both.equals((bars["close"] > 120) & (bars["close"] < 180))
    assert either.equals((bars["close"] < 120) | (bars["close"] > 180))
    assert negated.equals(~(bars["close"] > 150))


def test_operator_precedence_survives_the_rewrite(bars):
    assert compile_expression("close > 100 or close > 150 and close < 120")(bars).equals(
        (bars["close"] > 100) | ((bars["close"] > 150) & (bars["close"] < 120))
    )


def test_indicators_match_a_direct_computation(bars):
    result = compile_expression("sma(close, 20)", as_condition=False)(bars)
    assert result.equals(ind.compute("SMA", bars, timeperiod=20)["value"])


def test_an_indicator_can_be_applied_to_a_named_series(bars):
    on_close = compile_expression("sma(close, 20)", as_condition=False)(bars)
    on_high = compile_expression("sma(high, 20)", as_condition=False)(bars)
    assert not on_close.equals(on_high)
    assert on_high.iloc[-1] > on_close.iloc[-1]


def test_multi_input_indicators_read_the_columns_they_declared(bars):
    """ATR needs high, low and close — it cannot work from one Series."""
    atr = compile_expression("atr(close, 14)", as_condition=False)(bars)
    assert atr.notna().sum() > 0
    assert (atr.dropna() > 0).all()


def test_warmup_nans_are_false_not_true(bars):
    """A strategy must not trade on an unconverged indicator value."""
    result = compile_expression("sma(close, 50) > 0")(bars)
    assert not result.iloc[:49].any()
    assert result.iloc[100]


def test_crossed_above_fires_only_on_the_crossing_bar(bars):
    fires = compile_expression("crossed_above(sma(close, 5), sma(close, 20))")(bars)
    above = compile_expression("sma(close, 5) > sma(close, 20)")(bars)
    assert fires.sum() < above.sum()
    assert (fires & ~above).sum() == 0


def test_rolling_max_includes_the_current_bar_and_prior_max_does_not(bars):
    """The trap: `close > rolling_max(high, N)` can never be true."""
    assert compile_expression("close > rolling_max(high, 20)")(bars).sum() == 0
    assert compile_expression("close > prior_max(high, 20)")(bars).sum() > 0


def test_zscore_is_centred_and_scaled(bars):
    z = compile_expression("zscore(close, 50)", as_condition=False)(bars).dropna()
    assert abs(z.mean()) < 2
    assert z.abs().max() < 10


def test_missing_columns_are_reported(bars):
    with pytest.raises(ExpressionError, match="missing column"):
        compile_expression("close > 100")(bars.drop(columns=["volume"]))


def test_the_language_documents_what_it_refuses_and_why():
    doc = describe_language()
    assert "Deliberately NOT supported" in doc
    assert "Attribute" in doc and "sandbox escapes" in doc
    assert "rolling_max" in doc and "prior_max" in doc  # the trap is documented

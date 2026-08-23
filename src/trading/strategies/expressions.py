"""A restricted expression language for config-defined strategies.

Config strategies let you write a rule in YAML instead of Python::

    entry: "sma(close, 50) > sma(close, 200) and rsi(close, 14) < 70"

Evaluating that with :func:`eval` would be a code-injection hole: a YAML file —
or worse, an LLM-generated one (Phase 0 §10.7) — could import ``os`` and run
anything.  So expressions are parsed to an AST, **every node is checked against
an allowlist**, and evaluation happens with no builtins and a fixed namespace.

The allowlist is deliberately narrow.  Attribute access, subscripting, lambdas,
comprehensions, f-strings, imports and assignment are all rejected — not because
each is individually dangerous, but because ``().__class__.__bases__`` is how
every sandbox escape starts.

Expressions evaluate over pandas Series, so a rule is naturally vectorised and
its truth value at the current bar is the last element.  Because every function
here is **causal** — a value at time *t* depends only on data at or before *t* —
an expression can be evaluated over a whole history and sliced safely.

One accommodation for pandas: Python's ``and``/``or``/``not`` short-circuit by
calling ``__bool__``, which a Series refuses to answer.  Writing ``&`` and ``|``
instead works but reads badly in a config file and has surprising precedence, so
the validated AST is **rewritten** — ``and`` becomes ``&``, ``or`` becomes ``|``,
``not`` becomes ``~``.  Precedence is already fixed by the tree shape, so the
rewrite cannot change what an expression means.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any, Final

import numpy as np
import pandas as pd

from trading.features import indicators as ind

__all__ = [
    "FUNCTIONS",
    "SERIES_NAMES",
    "ExpressionError",
    "compile_expression",
    "describe_language",
]


class ExpressionError(ValueError):
    """An expression is malformed, or uses something outside the allowlist."""


# ── the allowlist ───────────────────────────────────────────────────────────
_ALLOWED_NODES: Final[tuple[type[ast.AST], ...]] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.keyword,
    # Accepted so an author may write & | ~ directly if they prefer, and
    # produced by the boolean rewrite below.
    ast.BitAnd,
    ast.BitOr,
    ast.Invert,
)

#: Everything below is rejected explicitly so the reason can be explained.
_FORBIDDEN_REASONS: Final[dict[type[ast.AST], str]] = {
    ast.Attribute: "attribute access is how sandbox escapes begin",
    ast.Subscript: "indexing is not needed and widens the surface",
    ast.Lambda: "anonymous functions are not permitted",
    ast.ListComp: "comprehensions are not permitted",
    ast.DictComp: "comprehensions are not permitted",
    ast.SetComp: "comprehensions are not permitted",
    ast.GeneratorExp: "comprehensions are not permitted",
    ast.JoinedStr: "f-strings can evaluate arbitrary expressions",
    ast.Starred: "argument unpacking is not permitted",
    ast.NamedExpr: "assignment inside an expression is not permitted",
    ast.Await: "async is not permitted",
    ast.IfExp: "conditional expressions are not permitted; use and/or",
}

#: Price and volume series a rule may refer to.
SERIES_NAMES: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


# ── functions available to expressions ──────────────────────────────────────
def _bind_indicator(name: str, output: str, bars: pd.DataFrame) -> Callable[..., pd.Series]:
    """Bind an indicator to the frame being evaluated.

    Series names in an expression resolve to real Series, so ``sma(close, 50)``
    passes one. But multi-input indicators such as ATR need high, low *and*
    close. Binding the frame lets the indicator read whatever columns it
    declared, while the passed Series overrides its primary input — which is
    what makes ``sma(high, 50)`` mean what it looks like it means.
    """

    def call(series: Any = None, timeperiod: int | None = None, **kwargs: Any) -> pd.Series:
        spec = ind.INDICATORS[name]
        frame = bars
        if series is not None:
            if not isinstance(series, pd.Series):
                raise ExpressionError(
                    f"{name.lower()}() expects a price series such as "
                    f"{', '.join(SERIES_NAMES)}, got {type(series).__name__}"
                )
            frame = bars.copy()
            frame[spec.inputs[0]] = series
        params = dict(kwargs)
        if timeperiod is not None:
            params["timeperiod"] = int(timeperiod)
        return ind.compute(name, frame, **params)[output]

    return call


def _require_series(value: Any, function: str) -> pd.Series:
    if not isinstance(value, pd.Series):
        raise ExpressionError(
            f"{function}() expects a price series such as {', '.join(SERIES_NAMES)}, "
            f"got {type(value).__name__}"
        )
    return value


def _crossed_above(fast: Any, slow: Any) -> pd.Series:
    """True on the bar where ``fast`` moves from at-or-below to above ``slow``."""
    fast = _require_series(fast, "crossed_above")
    slow = _require_series(slow, "crossed_above")
    return (fast > slow) & (fast.shift(1) <= slow.shift(1))


def _crossed_below(fast: Any, slow: Any) -> pd.Series:
    fast = _require_series(fast, "crossed_below")
    slow = _require_series(slow, "crossed_below")
    return (fast < slow) & (fast.shift(1) >= slow.shift(1))


def _zscore(series: Any, timeperiod: int = 20) -> pd.Series:
    """How many standard deviations the current value sits from its own mean.

    The workhorse of mean-reversion rules. Uses a rolling window, so it is
    causal — it never sees a value later than the bar it is computed for.
    """
    series = _require_series(series, "zscore")
    window = int(timeperiod)
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def _pct_change(series: Any, timeperiod: int = 1) -> pd.Series:
    return _require_series(series, "pct_change").pct_change(int(timeperiod))


def _rolling_max(series: Any, timeperiod: int = 20) -> pd.Series:
    return _require_series(series, "rolling_max").rolling(int(timeperiod)).max()


def _rolling_min(series: Any, timeperiod: int = 20) -> pd.Series:
    return _require_series(series, "rolling_min").rolling(int(timeperiod)).min()


def _prior_max(series: Any, timeperiod: int = 20) -> pd.Series:
    """Highest value over the previous N bars, **excluding the current one**.

    This is what a breakout rule means. ``close > rolling_max(high, 20)`` is
    never true, because that window includes today and a close cannot exceed
    its own bar's high — a trap worth naming rather than leaving to be
    discovered by a strategy that silently never trades.
    """
    return _require_series(series, "prior_max").rolling(int(timeperiod)).max().shift(1)


def _prior_min(series: Any, timeperiod: int = 20) -> pd.Series:
    """Lowest value over the previous N bars, excluding the current one."""
    return _require_series(series, "prior_min").rolling(int(timeperiod)).min().shift(1)


def _prev(series: Any, timeperiod: int = 1) -> pd.Series:
    """The value N bars ago."""
    return _require_series(series, "prev").shift(int(timeperiod))


#: Indicator functions, bound to the frame at evaluation time.
_INDICATOR_FUNCTIONS: Final[dict[str, tuple[str, str]]] = {
    "sma": ("SMA", "value"),
    "ema": ("EMA", "value"),
    "rsi": ("RSI", "value"),
    "atr": ("ATR", "value"),
    "adx": ("ADX", "value"),
    "obv": ("OBV", "value"),
    "macd": ("MACD", "macd"),
    "macd_signal": ("MACD", "signal"),
    "bb_upper": ("BBANDS", "upper"),
    "bb_middle": ("BBANDS", "middle"),
    "bb_lower": ("BBANDS", "lower"),
}

#: Frame-independent functions, usable as-is.
_PLAIN_FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    "crossed_above": _crossed_above,
    "crossed_below": _crossed_below,
    "zscore": _zscore,
    "pct_change": _pct_change,
    "rolling_max": _rolling_max,
    "rolling_min": _rolling_min,
    "prior_max": _prior_max,
    "prior_min": _prior_min,
    "prev": _prev,
    "abs": abs,
}

#: Every callable name an expression may use.
FUNCTIONS: Final[dict[str, Callable[..., Any]]] = {
    **_PLAIN_FUNCTIONS,
    **dict.fromkeys(_INDICATOR_FUNCTIONS, _require_series),
}


# ── validation and compilation ──────────────────────────────────────────────
class _VectorizeBooleans(ast.NodeTransformer):
    """Rewrite and/or/not into their element-wise equivalents."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        self.generic_visit(node)
        operator: ast.operator = ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr()
        combined = node.values[0]
        for right in node.values[1:]:
            combined = ast.BinOp(left=combined, op=operator, right=right)
        return combined

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            return ast.UnaryOp(op=ast.Invert(), operand=node.operand)
        return node


def _validate(tree: ast.AST, expression: str) -> None:
    for node in ast.walk(tree):
        reason = _FORBIDDEN_REASONS.get(type(node))
        if reason:
            raise ExpressionError(
                f"{type(node).__name__} is not allowed in {expression!r}: {reason}"
            )
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"{type(node).__name__} is not allowed in {expression!r}. "
                f"The expression language is intentionally small — see "
                f"`trading strategy-language` for what it supports."
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError(f"Only direct function calls are allowed in {expression!r}")
            if node.func.id not in FUNCTIONS:
                raise ExpressionError(
                    f"Unknown function {node.func.id!r} in {expression!r}. "
                    f"Available: {', '.join(sorted(FUNCTIONS))}"
                )
        if isinstance(node, ast.Name) and node.id not in FUNCTIONS and node.id not in SERIES_NAMES:
            raise ExpressionError(
                f"Unknown name {node.id!r} in {expression!r}. "
                f"Available series: {', '.join(SERIES_NAMES)}"
            )
        if isinstance(node, ast.Compare) and len(node.ops) > 1:
            raise ExpressionError(
                f"Chained comparisons are not supported in {expression!r}. "
                f"Write `a > b and b > c` rather than `a > b > c`."
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ExpressionError(
                f"Only numeric and boolean literals are allowed, got "
                f"{type(node.value).__name__} in {expression!r}"
            )


def compile_expression(
    expression: str, *, as_condition: bool = True
) -> Callable[[pd.DataFrame], pd.Series]:
    """Validate an expression and return a callable that evaluates it over bars.

    With ``as_condition`` the result is coerced to booleans, which is what an
    entry or exit rule needs. Without it the raw numeric Series is returned,
    which is what a named feature needs.

    Raises :class:`ExpressionError` at **compile** time — never at the moment a
    strategy is about to trade.
    """
    if not expression or not expression.strip():
        raise ExpressionError("Expression is empty")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Cannot parse {expression!r}: {exc.msg}") from exc

    _validate(tree, expression)
    tree = ast.fix_missing_locations(_VectorizeBooleans().visit(tree))
    code = compile(tree, filename="<strategy-expression>", mode="eval")

    def evaluate(bars: pd.DataFrame) -> pd.Series:
        missing = [c for c in SERIES_NAMES if c not in bars.columns]
        if missing:
            raise ExpressionError(f"Bars are missing column(s) {missing}")

        namespace: dict[str, Any] = {
            **_PLAIN_FUNCTIONS,
            # Indicators are bound per evaluation so they can read whatever
            # columns they declared from this frame.
            **{
                name: _bind_indicator(indicator, output, bars)
                for name, (indicator, output) in _INDICATOR_FUNCTIONS.items()
            },
            **{name: bars[name] for name in SERIES_NAMES},
            "__builtins__": {},
        }
        try:
            result = eval(code, namespace)
        except ExpressionError:
            raise
        except Exception as exc:
            raise ExpressionError(f"Evaluating {expression!r} failed: {exc}") from exc

        if isinstance(result, pd.DataFrame):
            raise ExpressionError(f"{expression!r} produced a frame, expected a series")
        if isinstance(result, pd.Series):
            if not as_condition:
                return result
            # NaN through an indicator's warm-up means "not yet true", never
            # "true" — a strategy must not trade on an unconverged value.
            return result.astype("boolean").fillna(value=False).astype(bool)
        if as_condition:
            return pd.Series(bool(result), index=bars.index)
        return pd.Series(result, index=bars.index)

    evaluate.__doc__ = f"Compiled strategy expression: {expression}"
    return evaluate


def describe_language() -> str:
    """Documentation for the expression language, for the CLI and error messages."""
    lines = [
        "Series available:  " + ", ".join(SERIES_NAMES),
        "",
        "Functions:",
        *(f"  {name}(...)" for name in sorted(FUNCTIONS)),
        "",
        "Operators:  and  or  not  +  -  *  /  %  **  <  <=  >  >=  ==  !=",
        "",
        "Deliberately NOT supported, and why:",
        *(
            f"  {node.__name__:<14} {reason}"
            for node, reason in sorted(_FORBIDDEN_REASONS.items(), key=lambda kv: kv[0].__name__)
        ),
        "",
        "Examples:",
        "  crossed_above(sma(close, 50), sma(close, 200))",
        "  zscore(close, 20) < -2.0 and rsi(close, 14) < 30",
        "  close > prior_max(high, 20) and adx(close, 14) > 25",
        "",
        "A trap worth knowing:",
        "  rolling_max(high, 20) includes the current bar, so `close > rolling_max(high, 20)`",
        "  is never true — a close cannot exceed its own bar's high. Breakout rules want",
        "  prior_max(), which excludes the current bar.",
    ]
    return "\n".join(lines)

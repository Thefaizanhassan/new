"""The feature engine.

Two jobs, and the second one is the interesting one.

**Speed.** A strategy that recomputes a 200-bar moving average from scratch on
every bar is O(n²): the Phase 1 skeleton did exactly that, and it does not scale
past a few thousand bars. Features are computed **once over the whole history**
and sliced per bar instead.

**Proving that is safe.** Precomputing over the full series sounds like exactly
the mistake this platform exists to prevent — but it is safe *if and only if*
every feature is **causal**: its value at time *t* depends only on data at or
before *t*. Moving averages, RSI, ATR and z-scores all are. A centred moving
average or a negative shift is not.

So causality is not assumed, it is **checked**: :func:`verify_causality`
recomputes a feature on truncated history and asserts it matches the
precomputed value. Any feature that fails is one that can see the future.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from trading.strategies.expressions import compile_expression

__all__ = ["FeatureEngine", "FeatureSpec", "verify_causality"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """A named column derived from bars by an expression."""

    name: str
    expression: str
    as_condition: bool = False

    def compiled(self) -> Callable[[pd.DataFrame], pd.Series]:
        return compile_expression(self.expression, as_condition=self.as_condition)


class FeatureEngine:
    """Computes named features over a bar history."""

    def __init__(self, specs: list[FeatureSpec]) -> None:
        self.specs = specs
        # Compiling up front means a malformed expression fails at construction
        # rather than mid-backtest.
        self._compiled = {spec.name: spec.compiled() for spec in specs}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._compiled)

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """All features, aligned to ``bars.index``. Warm-up rows are NaN/False."""
        if bars.empty:
            return pd.DataFrame(index=bars.index, columns=list(self._compiled))
        return pd.DataFrame(
            {name: fn(bars) for name, fn in self._compiled.items()}, index=bars.index
        )


def verify_causality(
    spec: FeatureSpec, bars: pd.DataFrame, *, at: int | None = None, tolerance: float = 1e-9
) -> bool:
    """Check that a feature at bar ``at`` is unchanged by hiding later bars.

    A feature that fails this is reading the future, and any strategy using it
    is producing results that cannot be achieved in live trading.
    """
    if len(bars) < 3:
        raise ValueError("Need at least 3 bars to check causality")
    index = at if at is not None else int(len(bars) * 0.75)

    compiled = spec.compiled()
    full = compiled(bars)
    truncated = compiled(bars.iloc[: index + 1])

    a, b = full.iloc[index], truncated.iloc[index]
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return abs(float(a) - float(b)) <= tolerance

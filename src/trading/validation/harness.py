"""What validation needs from a backtest, and how it asks for it.

Validation must not depend on the engine.  Walk-forward analysis, parameter
sweeps and Monte Carlo resampling are all just "run this thing over that window
and hand me the numbers", and pinning them to :class:`WalkingSkeletonRunner`
would make every one of them untestable without a data provider, a risk engine
and a cost model.

So the contract is a callable: give validation a :class:`SegmentEvaluator` and it
will call it.  :mod:`trading.validation.engine_adapter` builds one from the real
engine; a test builds one from three lines of arithmetic.  That is the whole
reason this module exists.

**The objective is deliberately awkward to game.**  A parameter sweep maximising
raw return picks whichever setting happened to hold the one stock that tripled.
A sweep maximising Sharpe over a window with three trades picks noise.  So
:class:`Objective` refuses to score a segment that did not trade enough to mean
anything: below ``min_trades`` the score is negative infinity, and the optimiser
cannot select it however good the ratio looks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol

import pandas as pd

from trading.validation.splits import Window

__all__ = [
    "Objective",
    "ParameterSet",
    "ParameterValue",
    "SegmentEvaluator",
    "SegmentOutcome",
]

ParameterValue = int | float | Decimal
ParameterSet = Mapping[str, ParameterValue]
"""One candidate configuration.

**Numeric only, and the type says so deliberately.**  A parameter reaches a YAML
strategy by being substituted into a rule, and a non-numeric value could contain
an operator or a call and rewrite the rule's shape.
:mod:`trading.strategies.config_strategy` rejects that at runtime; narrowing the
type here means the mistake cannot be written in the first place, rather than
being caught only when a sweep runs.

The cost is that a categorical parameter — sweeping which indicator to use, say —
is not expressible.  That is the right trade: such a sweep is a search over
strategies rather than over parameters, and it belongs in the registry with its
own lifecycle entry, not hidden inside one config's grid."""


@dataclass(frozen=True, slots=True)
class SegmentOutcome:
    """The result of running one parameter set over one window.

    ``returns`` is the per-bar return series *of the measured window only* — the
    warmup prefix is already dropped.  Everything downstream (concatenated
    out-of-sample curves, bootstrap resampling, deflated Sharpe) is computed from
    this series, so if it silently included warmup bars every out-of-sample
    number in the system would be diluted by a period the strategy was not even
    trading in.
    """

    params: ParameterSet
    window: Window
    returns: pd.Series
    trades: int
    trade_pnl: tuple[float, ...]
    metrics: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def net_return(self) -> float:
        """Compounded return over the measured window."""
        if self.returns.empty:
            return 0.0
        return float((1.0 + self.returns.astype(float)).prod() - 1.0)

    @property
    def bars(self) -> int:
        return len(self.returns)

    def metric(self, name: str) -> float | None:
        value = self.metrics.get(name)
        if value is None:
            return None
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None


class SegmentEvaluator(Protocol):
    """Run one parameter set over one window.

    Implementations must measure only ``window``, warming up from whatever
    earlier bars they need.  ``Split.run`` carries the extended window for
    exactly this purpose.
    """

    def __call__(
        self, params: ParameterSet, window: Window, measure_from: Window
    ) -> SegmentOutcome:
        """``window`` is what to feed the engine; ``measure_from`` is what counts."""
        ...


@dataclass(frozen=True, slots=True)
class Objective:
    """What "better" means during a sweep, stated explicitly rather than assumed.

    The default is Sharpe, because it is the ratio most readers can interpret,
    with a floor on trade count so a two-trade fluke cannot win.  ``calmar``
    is the better choice when survivable drawdown matters more than smoothness,
    and ``net_return`` is available but should be treated as a diagnostic: a
    sweep that maximises return alone reliably selects the most leveraged noise
    on the grid.
    """

    metric: str = "sharpe"
    minimise: bool = False
    min_trades: int = 5

    REJECTED: float = field(default=-math.inf, init=False, repr=False)

    def score(self, outcome: SegmentOutcome) -> float:
        """Score a segment, or ``-inf`` if it is not scoreable.

        Returning ``-inf`` rather than ``0.0`` matters: zero is a *neutral*
        result that could still beat a losing alternative, so a segment with two
        trades would win a grid where everything else lost money.
        """
        if outcome.trades < self.min_trades:
            return self.REJECTED
        value = outcome.metric(self.metric)
        if value is None:
            return self.REJECTED
        return -value if self.minimise else value

    def best(self, outcomes: Sequence[SegmentOutcome]) -> SegmentOutcome | None:
        """The best scoreable outcome, or ``None`` if none of them qualified.

        ``None`` is a real answer and must not be papered over with a fallback to
        the first candidate: it means no setting on the grid traded enough to be
        judged, and a fold that reports a winner anyway is fabricating one.
        """
        scored = [(self.score(o), o) for o in outcomes]
        qualified = [(s, o) for s, o in scored if s > self.REJECTED]
        if not qualified:
            return None
        return max(qualified, key=lambda pair: pair[0])[1]

    def describe(self) -> str:
        direction = "minimise" if self.minimise else "maximise"
        return f"{direction} {self.metric} (segments with < {self.min_trades} trades rejected)"

"""Parameter sensitivity surfaces: the plateau-versus-spike test.

Phase 0 §11.5.  If you only ever run one overfitting check, run this one.  It is
the cheapest, the hardest to argue with, and the easiest to explain to somebody
who has never seen a backtest.

Sweep a parameter across its plausible range and plot the result.  Two shapes
come back, and they mean opposite things.

**A plateau.**  SMA(45), SMA(50) and SMA(55) all earn roughly the same.  The
strategy is exploiting something that does not care about the exact number, which
is what a real market effect looks like.  Pick the middle of the plateau and it
will still be roughly right next year.

**A spike.**  SMA(50) earns 40%, SMA(45) loses 5%, SMA(55) loses 8%.  Nothing
about the market changes between a 49-day and a 50-day average, so a result that
collapses between them is not measuring the market — it is measuring which
setting happened to line up with the noise in this particular sample.  Deploying
it is betting that next year's noise lines up the same way.

This is why the single most useful output here is **not** the best point.  It is
:attr:`SensitivitySurface.robust_best`: the parameter set with the best
*neighbourhood*.  Deploying the peak of a spike is the mistake the whole
technique exists to prevent, and reporting the peak while quietly deploying it is
how a validated strategy becomes a live loss.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from trading.validation.harness import (
    Objective,
    ParameterSet,
    ParameterValue,
    SegmentEvaluator,
    SegmentOutcome,
)
from trading.validation.splits import Window

__all__ = [
    "GridPoint",
    "ParameterGrid",
    "SensitivitySurface",
    "run_sensitivity",
]

ParamKey = tuple[tuple[str, ParameterValue], ...]


def _key(params: ParameterSet) -> ParamKey:
    return tuple(sorted(params.items()))


@dataclass(frozen=True, slots=True)
class ParameterGrid:
    """A named sweep. ``axes`` maps a parameter to the values to try.

    Order within each axis matters: neighbourhood is defined as adjacency in the
    listed order, so values must be listed in their natural progression.  A grid
    listing ``[10, 200, 50]`` would call 10 and 200 neighbours and make every
    plateau test meaningless, so the constructor rejects unordered numeric axes
    rather than trusting the caller to have been careful.
    """

    axes: Mapping[str, Sequence[ParameterValue]]

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("a grid needs at least one axis")
        for name, values in self.axes.items():
            if len(values) < 1:
                raise ValueError(f"axis {name!r} has no values")
            if len(set(values)) != len(values):
                raise ValueError(f"axis {name!r} has duplicate values")
            ordered = sorted(values, key=Decimal)
            if [Decimal(v) for v in values] != [Decimal(v) for v in ordered]:
                raise ValueError(
                    f"axis {name!r} is not in ascending order; neighbourhood is "
                    "adjacency in the listed order, so unordered values would "
                    "make the plateau test meaningless"
                )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.axes)

    @property
    def size(self) -> int:
        return math.prod(len(v) for v in self.axes.values())

    def points(self) -> tuple[ParameterSet, ...]:
        names = self.names
        return tuple(
            dict(zip(names, combo, strict=True))
            for combo in itertools.product(*(self.axes[n] for n in names))
        )

    def coordinates(self, params: ParameterSet) -> tuple[int, ...]:
        """Position of a point on each axis, for neighbourhood arithmetic."""
        return tuple(list(self.axes[n]).index(params[n]) for n in self.names)

    def at(self, coords: Sequence[int]) -> ParameterSet | None:
        """The point at these coordinates, or ``None`` if off the grid."""
        names = self.names
        if len(coords) != len(names):
            return None
        out: dict[str, ParameterValue] = {}
        for name, position in zip(names, coords, strict=True):
            values = list(self.axes[name])
            if not 0 <= position < len(values):
                return None
            out[name] = values[position]
        return out

    def neighbours(self, params: ParameterSet) -> tuple[ParameterSet, ...]:
        """Points one step away along a single axis.

        Deliberately not diagonal: a diagonal neighbour differs in two
        parameters, so including them would blur "this parameter is insensitive"
        with "this combination happens to also work".
        """
        base = list(self.coordinates(params))
        found: list[ParameterSet] = []
        for axis in range(len(base)):
            for step in (-1, 1):
                shifted = list(base)
                shifted[axis] += step
                point = self.at(shifted)
                if point is not None:
                    found.append(point)
        return tuple(found)

    def describe(self) -> str:
        return ", ".join(f"{n}={list(v)}" for n, v in self.axes.items())


@dataclass(frozen=True, slots=True)
class GridPoint:
    params: ParameterSet
    score: float
    outcome: SegmentOutcome

    @property
    def scoreable(self) -> bool:
        return math.isfinite(self.score)


@dataclass(frozen=True, slots=True)
class SensitivitySurface:
    """Scores across the whole grid, plus the shape tests that read them."""

    grid: ParameterGrid
    points: tuple[GridPoint, ...]
    objective: Objective
    plateau_ratio_threshold: float = 0.5
    """How much of the peak's score the neighbourhood must retain to count as a
    plateau. 0.5 is lenient — it allows the neighbours to be half as good — and
    a strategy that cannot clear even that is fitted to one exact setting."""
    min_positive_fraction: float = 0.5
    """Fraction of the whole grid that must be profitable. A strategy profitable
    at one setting out of twenty found a coincidence, not an effect."""

    _by_key: dict[ParamKey, GridPoint] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._by_key.update({_key(p.params): p for p in self.points})

    @property
    def scoreable(self) -> tuple[GridPoint, ...]:
        return tuple(p for p in self.points if p.scoreable)

    def point_for(self, params: ParameterSet) -> GridPoint | None:
        return self._by_key.get(_key(params))

    @property
    def best(self) -> GridPoint | None:
        """The peak. Interesting, and the wrong thing to deploy on its own."""
        scoreable = self.scoreable
        return max(scoreable, key=lambda p: p.score) if scoreable else None

    @property
    def median_score(self) -> float:
        scores = [p.score for p in self.scoreable]
        return float(np.median(scores)) if scores else 0.0

    @property
    def positive_fraction(self) -> float:
        """Fraction of the grid that made money at all.

        Unscoreable points — settings that never traded enough to judge — count
        against this. They are not neutral: a grid where most settings never
        trade has not been swept, it has been mostly skipped.
        """
        if not self.points:
            return 0.0
        return sum(1 for p in self.points if p.outcome.net_return > 0) / len(self.points)

    def neighbourhood_score(self, params: ParameterSet) -> float:
        """Mean score of the immediate neighbours, excluding the point itself.

        Excluding the point is what makes this a test rather than a smoothing:
        a spike's own score cannot prop up its neighbourhood.
        """
        neighbours = [self.point_for(n) for n in self.grid.neighbours(params)]
        scores = [n.score for n in neighbours if n is not None and n.scoreable]
        return float(np.mean(scores)) if scores else -math.inf

    @property
    def plateau_ratio(self) -> float:
        """Neighbourhood score of the peak, over the peak's own score.

        1.0 means the neighbours are as good as the peak — a flat plateau.  Near
        zero means the peak stands alone.  Negative means its neighbours lose
        money, which is the clearest possible statement that the peak is noise.
        """
        best = self.best
        if best is None or best.score <= 0:
            return 0.0
        neighbourhood = self.neighbourhood_score(best.params)
        if not math.isfinite(neighbourhood):
            return 0.0
        return neighbourhood / best.score

    @property
    def robust_best(self) -> GridPoint | None:
        """The point whose *neighbourhood* is strongest — the one to deploy.

        Scored as the mean of the point and its neighbours, so a setting is only
        chosen if the settings either side of it also work.  On a genuine plateau
        this lands near the middle; on a spiky surface it deliberately refuses
        the spike.
        """
        candidates = [
            (
                (p.score + self.neighbourhood_score(p.params) * len(self.grid.neighbours(p.params)))
                / (1 + len(self.grid.neighbours(p.params))),
                p,
            )
            for p in self.scoreable
            if math.isfinite(self.neighbourhood_score(p.params))
        ]
        if not candidates:
            return self.best
        return max(candidates, key=lambda pair: pair[0])[1]

    @property
    def is_plateau(self) -> bool:
        """The peak sits on a shelf, and the grid is broadly profitable."""
        return (
            self.best is not None
            and self.plateau_ratio >= self.plateau_ratio_threshold
            and self.positive_fraction >= self.min_positive_fraction
        )

    @property
    def is_spike(self) -> bool:
        return self.best is not None and not self.is_plateau

    @property
    def failures(self) -> tuple[str, ...]:
        if self.best is None:
            return ("no grid point traded enough to be scored",)
        checks = (
            (
                self.plateau_ratio >= self.plateau_ratio_threshold,
                f"the best setting is a spike, not a plateau: its neighbours score "
                f"{self.plateau_ratio:.0%} of its own score "
                f"(need >= {self.plateau_ratio_threshold:.0%})",
            ),
            (
                self.positive_fraction >= self.min_positive_fraction,
                f"only {self.positive_fraction:.0%} of the grid was profitable "
                f"(need >= {self.min_positive_fraction:.0%})",
            ),
        )
        return tuple(message for ok, message in checks if not ok)

    def axis_profile(self, name: str) -> list[tuple[ParameterValue, float]]:
        """Mean score along one axis, marginalising the others.

        This is the one-dimensional view worth reading first: it shows whether
        the strategy cares about this parameter at all.
        """
        profile: list[tuple[ParameterValue, float]] = []
        for value in self.grid.axes[name]:
            scores = [p.score for p in self.scoreable if p.params[name] == value]
            profile.append((value, float(np.mean(scores)) if scores else -math.inf))
        return profile

    def summary_lines(self) -> list[tuple[str, str]]:
        best, robust = self.best, self.robust_best
        return [
            ("grid size", f"{len(self.points)} points ({self.grid.describe()})"),
            ("scoreable points", f"{len(self.scoreable)} of {len(self.points)}"),
            (
                "best point",
                f"{dict(best.params)} score {best.score:.3f} ({best.outcome.net_return:+.2%})"
                if best
                else "none",
            ),
            (
                # Naming a setting to deploy when the surface is a spike would be
                # recommending the least-bad point on a grid that failed.
                "robust best" if self.is_plateau else "robust best (do NOT deploy)",
                f"{dict(robust.params)} score {robust.score:.3f} ({robust.outcome.net_return:+.2%})"
                if robust
                else "none",
            ),
            ("median score", f"{self.median_score:.3f}"),
            ("plateau ratio", f"{self.plateau_ratio:.0%}"),
            ("profitable fraction", f"{self.positive_fraction:.0%}"),
            ("shape", "PLATEAU" if self.is_plateau else "SPIKE"),
        ]

    def verdict(self) -> str:
        if self.is_plateau:
            robust = self.robust_best
            return (
                f"plateau: neighbours retain {self.plateau_ratio:.0%} of the peak, "
                f"{self.positive_fraction:.0%} of the grid profitable — deploy "
                f"{dict(robust.params) if robust else '{}'}, not the peak"
            )
        return "spike: " + "; ".join(self.failures)


def run_sensitivity(
    evaluate: SegmentEvaluator,
    grid: ParameterGrid,
    window: Window,
    *,
    measure_from: Window | None = None,
    objective: Objective | None = None,
) -> SensitivitySurface:
    """Evaluate every point on the grid over the same window.

    One window for every point, deliberately: the surface is about the shape of
    the parameter response, so varying the window as well would confound the two.
    Run this on the *training* portion of the data.  Sweeping the out-of-sample
    period and then reporting the best point is not validation — it is tuning on
    the test set with extra steps.
    """
    objective = objective or Objective()
    measured = measure_from or window
    points: list[GridPoint] = []
    for params in grid.points():
        outcome = evaluate(params, window, measured)
        points.append(GridPoint(params=params, score=objective.score(outcome), outcome=outcome))
    return SensitivitySurface(grid=grid, points=tuple(points), objective=objective)

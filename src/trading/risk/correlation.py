"""Correlation-adjusted exposure.

Phase 0 §15.1: three strategies each taking a "reasonable" 5% position in AAPL,
MSFT and NVDA are not diversified.  Those names move together, so the portfolio
is really one 15% bet on large-cap tech, and a gross-exposure limit that sums
weights will happily wave it through.

The adjustment is the quadratic form ``sqrt(wᵀ C w)`` — the exposure of a single
hypothetical asset carrying the same risk as the book:

* all correlations 1 → equals the plain sum of weights (nothing is diversified)
* all correlations 0 → equals the root-sum-square (fully diversified)
* correlations −1 → approaches zero (the positions offset)

So one number tells the risk engine what the portfolio is *actually* betting,
rather than what it nominally holds.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd

from trading.core.instrument import InstrumentId

__all__ = ["ExposureBreakdown", "correlation_adjusted_exposure"]

_MIN_OBSERVATIONS = 30


class ExposureBreakdown:
    """Nominal and correlation-adjusted exposure, plus why they differ."""

    __slots__ = ("adjusted", "assumed_correlated", "nominal", "observations")

    def __init__(
        self,
        nominal: Decimal,
        adjusted: Decimal,
        observations: int,
        assumed_correlated: bool,
    ) -> None:
        self.nominal = nominal
        self.adjusted = adjusted
        self.observations = observations
        self.assumed_correlated = assumed_correlated

    @property
    def diversification_benefit(self) -> Decimal:
        """How much correlation reduces the effective bet. 0 means none."""
        return self.nominal - self.adjusted

    def __str__(self) -> str:
        note = " (assumed fully correlated)" if self.assumed_correlated else ""
        return f"nominal {self.nominal:.4f} → adjusted {self.adjusted:.4f}{note}"


def correlation_adjusted_exposure(
    weights: dict[InstrumentId, Decimal],
    returns: pd.DataFrame | None = None,
    *,
    min_observations: int = _MIN_OBSERVATIONS,
) -> ExposureBreakdown:
    """Effective gross exposure given how the held instruments actually co-move.

    **Fails closed.** Without enough return history to estimate correlation, the
    positions are assumed *perfectly correlated* — the most conservative
    assumption — rather than treated as independent. Assuming diversification
    you have not measured is how a concentrated book passes an exposure check.
    """
    nominal = sum((abs(w) for w in weights.values()), Decimal(0))
    if not weights:
        return ExposureBreakdown(Decimal(0), Decimal(0), 0, assumed_correlated=False)

    symbols = [str(i) for i in weights]
    usable = (
        returns is not None
        and not returns.empty
        and all(s in returns.columns for s in symbols)
        and len(returns.dropna()) >= min_observations
    )
    if not usable:
        return ExposureBreakdown(nominal, nominal, 0, assumed_correlated=True)

    assert returns is not None
    aligned = returns[symbols].dropna()

    # A series with no variance has no correlation to anything — computing it
    # anyway divides by zero and yields a silent NaN. Treat those instruments as
    # fully correlated, which is the same conservative reading as no history.
    if bool((aligned.std(ddof=0) == 0).any()):
        return ExposureBreakdown(nominal, nominal, len(aligned), assumed_correlated=True)
    matrix = aligned.corr().to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=1.0)  # an unestimable pair is treated as correlated

    vector = np.array([float(weights[i]) for i in weights], dtype=float)
    variance = float(vector @ matrix @ vector)
    adjusted = Decimal(str(np.sqrt(max(variance, 0.0))))

    return ExposureBreakdown(
        nominal=nominal,
        adjusted=adjusted,
        observations=len(aligned),
        assumed_correlated=False,
    )

"""The leakage canary.

Phase 0 §11.3.  A cheap, automated check for a whole class of bugs that code
review does not catch:

    Run the strategy on data whose temporal structure has been destroyed.
    If it still makes money, something is reading the future.

The synthetic series is built by **bootstrapping the real returns** — sampling
them with replacement — so it keeps the instrument's actual volatility, drift
and fat tails, but destroys every pattern a strategy could legitimately exploit.
A trend-following rule has nothing to follow in that series; a rule that peeks
at tomorrow's bar does not care.

**What it catches, and what it does not.**  Measured against three deliberate
look-ahead strategies on the same data:

===========================================  ============  ================
Leak                                          Real (gross)  Shuffled (gross)
===========================================  ============  ================
Peeks 2 bars ahead                                  +863%            +795%
Peeks at next bar's open-to-open move              +3333%          +28976%
Peeks 1 bar ahead, trades the following open       +2280%              −6%
===========================================  ============  ================

The first two are caught emphatically: the leak still pays on data with no
structure. The third **evades the test** — its peeked value only becomes
exploitable through real autocorrelation, which the bootstrap destroys, so the
cheat earns nothing on shuffled data despite being a blatant look-ahead.

So this proves a *class* of failure is absent, not that a strategy is correct.
A canary whose limitation is documented is worth much more than one assumed to
be complete.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CanaryResult", "run_leakage_canary", "shuffle_bars"]


def shuffle_bars(bars: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Bootstrap a synthetic series with the same return distribution, no structure.

    The OHLC relationships within each bar are rebuilt from the original bar's
    proportions, so the result still passes the validation gate — a canary that
    produced invalid bars would be testing the validator, not the strategy.
    """
    if len(bars) < 3:
        raise ValueError("Need at least 3 bars to bootstrap")

    rng = np.random.default_rng(seed)
    close = bars["close"].astype(float)
    returns = close.pct_change().dropna().to_numpy()
    sampled = rng.choice(returns, size=len(bars) - 1, replace=True)

    prices = np.empty(len(bars))
    prices[0] = float(close.iloc[0])
    prices[1:] = prices[0] * np.cumprod(1 + sampled)

    # Reuse each original bar's intra-bar geometry so highs stay above closes.
    original = bars.astype(float)
    high_ratio = (original["high"] / original["close"]).to_numpy()
    low_ratio = (original["low"] / original["close"]).to_numpy()
    open_ratio = (original["open"] / original["close"]).to_numpy()

    synthetic = pd.DataFrame(
        {
            "open": prices * open_ratio,
            "high": prices * np.maximum(high_ratio, np.maximum(open_ratio, 1.0)),
            "low": prices * np.minimum(low_ratio, np.minimum(open_ratio, 1.0)),
            "close": prices,
            "volume": original["volume"].to_numpy(),
        },
        index=bars.index,
    )
    return synthetic


@dataclass(frozen=True, slots=True)
class CanaryResult:
    real_return: float
    shuffled_returns: tuple[float, ...]
    trials: int

    @property
    def median_shuffled(self) -> float:
        return float(np.median(self.shuffled_returns)) if self.shuffled_returns else 0.0

    @property
    def profitable_shuffles(self) -> int:
        return sum(1 for r in self.shuffled_returns if r > 0)

    @property
    def percentile_of_real(self) -> float:
        """Where the real result sits in the shuffled distribution.

        A genuine edge should sit high. A result buried inside the noise
        distribution is not evidence of anything.
        """
        if not self.shuffled_returns:
            return 0.0
        below = sum(1 for r in self.shuffled_returns if r < self.real_return)
        return below / len(self.shuffled_returns)

    @property
    def leaking(self) -> bool:
        """True when the strategy profits on structureless data.

        Profiting on most of the shuffles means the returns come from something
        other than market structure — and on data that has none, that means the
        strategy is seeing what it should not.

        A ``False`` here is not a clean bill of health; see the module docstring
        for the leak signature this cannot detect.
        """
        if not self.shuffled_returns:
            return False
        return self.profitable_shuffles > len(self.shuffled_returns) * 0.5

    @property
    def implausible_magnitude(self) -> bool:
        """Real result far outside the noise distribution.

        Not a leak test — a genuinely good strategy should also sit high. It is
        a prompt to look, and it is the signal that catches the leak the
        profitability test misses.
        """
        if not self.shuffled_returns:
            return False
        # Scale by the spread of the noise distribution, falling back to the
        # magnitude of the noise itself when every shuffle landed identically —
        # otherwise a degenerate distribution silently disables the check.
        scale = float(np.std(self.shuffled_returns))
        if scale <= 0:
            scale = max(abs(self.median_shuffled), 0.01)
        return (self.real_return - self.median_shuffled) > scale * 5

    def summary(self) -> str:
        if self.leaking:
            verdict = "LEAK SUSPECTED — profits on structureless data"
        elif self.implausible_magnitude:
            verdict = "WORTH A LOOK — result far outside the noise distribution"
        else:
            verdict = "no leak detected"
        return (
            f"{verdict}: real {self.real_return:+.2%} vs median shuffled "
            f"{self.median_shuffled:+.2%} "
            f"({self.profitable_shuffles}/{self.trials} shuffles profitable, "
            f"real at the {self.percentile_of_real:.0%} percentile)"
        )


def run_leakage_canary(
    run: Callable[[pd.DataFrame], float],
    bars: pd.DataFrame,
    *,
    trials: int = 20,
    seed: int = 0,
) -> CanaryResult:
    """Compare a strategy's real result against its result on structureless data.

    ``run`` takes a bar frame and returns the net return it produced.
    """
    real = run(bars)
    shuffled = tuple(run(shuffle_bars(bars, seed=seed + i)) for i in range(trials))
    return CanaryResult(real_return=real, shuffled_returns=shuffled, trials=trials)

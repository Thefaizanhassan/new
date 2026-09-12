"""Monte Carlo resampling: the drawdown you did not happen to see.

Phase 0 §11.5 and §12.  A backtest produces **one** path through history.  The
maximum drawdown on that path is not "the strategy's maximum drawdown" — it is
the worst run of bad luck that happened to occur in the order it occurred in.
Deal the same trades in a different order and the drawdown changes, often by a
lot.  Sizing a position or setting a halt limit from the single observed figure
is therefore setting it from a sample of one.

Two resamplings live here because they answer two different questions, and
conflating them produces a number that means neither.

**Trade resampling** (:func:`resample_trades`) draws the observed trades with
replacement into new sequences.  It asks: *given this distribution of trade
outcomes, how bad could the ordering have been?*  It holds position sizing fixed
and accumulates P&L additively, so it describes the drawdown distribution of the
trade sizes that were actually taken — not of a compounding account that would
have sized up after wins.  That is the conservative reading and it is stated
plainly rather than dressed up as a forecast.

**Block bootstrap of returns** (:func:`block_bootstrap_returns`) draws
*contiguous blocks* of daily returns and compounds them.  Blocks, not individual
days, because volatility clusters: shuffling day by day destroys the tendency of
bad days to arrive together, which is precisely the thing that creates drawdowns.
A day-by-day bootstrap systematically understates drawdown, and would hand back a
comfortable number that is an artefact of the method.

**What none of this does.**  Every path is drawn from the returns that actually
occurred, so the worst case here is bounded by the worst regime in the sample.
If the history contains no 2008 and no March 2020, no amount of resampling will
invent one.  A p99 drawdown from this module is "bad luck within the regimes I
have seen", never "the worst that can happen".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MonteCarloResult",
    "block_bootstrap_returns",
    "resample_trades",
]


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """The distribution of outcomes across resampled paths."""

    method: str
    paths: int
    observed_max_drawdown: float
    observed_net_return: float
    drawdowns: tuple[float, ...]
    net_returns: tuple[float, ...]
    ruin_threshold: float
    seed: int
    note: str = ""

    def drawdown_percentile(self, pct: float) -> float:
        """Drawdown at a percentile of the resampled distribution.

        ``95`` gives the level only 5% of paths exceeded — a defensible place to
        set a halt, unlike the single observed path.
        """
        if not self.drawdowns:
            return 0.0
        return float(np.percentile(self.drawdowns, pct))

    @property
    def median_drawdown(self) -> float:
        return self.drawdown_percentile(50)

    @property
    def p95_drawdown(self) -> float:
        return self.drawdown_percentile(95)

    @property
    def p99_drawdown(self) -> float:
        return self.drawdown_percentile(99)

    @property
    def worst_drawdown(self) -> float:
        return max(self.drawdowns) if self.drawdowns else 0.0

    @property
    def observed_percentile(self) -> float:
        """Where the one real path sits in the resampled distribution.

        A real drawdown near the median means the backtest saw typical luck.  Near
        the 5th percentile means the historical path was unusually *kind*, and the
        observed drawdown badly understates what to prepare for — which is the
        single most useful thing this module can tell you.
        """
        if not self.drawdowns:
            return 0.0
        below = sum(1 for d in self.drawdowns if d < self.observed_max_drawdown)
        return below / len(self.drawdowns)

    @property
    def probability_of_loss(self) -> float:
        if not self.net_returns:
            return 0.0
        return sum(1 for r in self.net_returns if r <= 0) / len(self.net_returns)

    @property
    def probability_of_ruin(self) -> float:
        """Fraction of paths whose drawdown breached ``ruin_threshold``.

        "Ruin" is not zero equity — it is the drawdown at which a real operator
        stops, which is much shallower and is the number that actually ends a
        strategy's life.
        """
        if not self.drawdowns:
            return 0.0
        return sum(1 for d in self.drawdowns if d >= self.ruin_threshold) / len(self.drawdowns)

    def recommended_drawdown_limit(self, headroom: float = 1.2) -> float:
        """A halt level set from the distribution rather than the observed path.

        The p95 drawdown with headroom: tight enough to stop a genuine failure,
        loose enough that ordinary bad luck does not trip it.  A limit set at the
        observed drawdown will be breached roughly half the time by chance alone,
        and a halt that fires on normal variation trains its operator to ignore it.
        """
        return self.p95_drawdown * headroom

    def summary_lines(self) -> list[tuple[str, str]]:
        return [
            ("method", self.method),
            ("paths", str(self.paths)),
            ("observed max drawdown", f"{self.observed_max_drawdown:.2%}"),
            ("median resampled drawdown", f"{self.median_drawdown:.2%}"),
            ("p95 drawdown", f"{self.p95_drawdown:.2%}"),
            ("p99 drawdown", f"{self.p99_drawdown:.2%}"),
            ("worst resampled drawdown", f"{self.worst_drawdown:.2%}"),
            ("observed path percentile", f"{self.observed_percentile:.0%}"),
            ("probability of a losing path", f"{self.probability_of_loss:.0%}"),
            (
                f"probability of >= {self.ruin_threshold:.0%} drawdown",
                f"{self.probability_of_ruin:.0%}",
            ),
            ("recommended halt level", f"{self.recommended_drawdown_limit():.2%}"),
        ]

    def verdict(self, limit: float) -> str:
        """Read against a risk limit: would that limit have survived the distribution?"""
        if not self.drawdowns:
            return "no paths generated — nothing to conclude"
        if self.p95_drawdown <= limit:
            return (
                f"p95 drawdown {self.p95_drawdown:.2%} is inside the {limit:.2%} limit "
                f"(p99 {self.p99_drawdown:.2%}, worst {self.worst_drawdown:.2%})"
            )
        return (
            f"p95 drawdown {self.p95_drawdown:.2%} EXCEEDS the {limit:.2%} limit — "
            f"one in twenty orderings of these same trades breaches it"
        )


def _max_drawdown(equity: np.ndarray) -> float:
    """Peak-to-trough decline as a fraction of the running peak."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    safe = np.where(peak > 0, peak, np.nan)
    drawdown = (peak - equity) / safe
    finite = drawdown[np.isfinite(drawdown)]
    return float(np.max(finite)) if finite.size else 0.0


def resample_trades(
    trade_pnl: Sequence[float],
    *,
    starting_capital: float,
    paths: int = 2_000,
    seed: int = 0,
    ruin_threshold: float = 0.25,
    trades_per_path: int | None = None,
) -> MonteCarloResult:
    """Re-deal the observed trades into new orderings and measure each path.

    Sampling is *with replacement*, which means a path can contain the worst
    trade twice.  That is the point: the historical sequence contained it once
    because of how the sample fell, not because the strategy is incapable of
    hitting it twice in a row.
    """
    pnl = np.asarray([float(p) for p in trade_pnl], dtype=float)
    if pnl.size < 2:
        raise ValueError(
            f"need at least 2 trades to resample, got {pnl.size} — "
            "a drawdown distribution from one trade is not a distribution"
        )
    if starting_capital <= 0:
        raise ValueError("starting_capital must be positive")
    if paths < 1:
        raise ValueError("paths must be at least 1")

    count = trades_per_path or pnl.size
    rng = np.random.default_rng(seed)
    draws = rng.choice(pnl, size=(paths, count), replace=True)
    curves = starting_capital + np.cumsum(draws, axis=1)

    drawdowns = tuple(_max_drawdown(curve) for curve in curves)
    net_returns = tuple(float(curve[-1] / starting_capital - 1.0) for curve in curves)

    observed = starting_capital + np.cumsum(pnl)
    return MonteCarloResult(
        method="trade resampling (with replacement, fixed position sizing)",
        paths=paths,
        observed_max_drawdown=_max_drawdown(observed),
        observed_net_return=float(observed[-1] / starting_capital - 1.0),
        drawdowns=drawdowns,
        net_returns=net_returns,
        ruin_threshold=ruin_threshold,
        seed=seed,
        note=(
            f"{pnl.size} observed trades, {count} per path; P&L accumulated "
            "additively, so this describes the observed trade sizes rather than a "
            "compounding account"
        ),
    )


def block_bootstrap_returns(
    returns: pd.Series,
    *,
    block_bars: int = 20,
    paths: int = 2_000,
    seed: int = 0,
    ruin_threshold: float = 0.25,
) -> MonteCarloResult:
    """Resample contiguous blocks of returns and compound them.

    ``block_bars`` should be long enough to contain a typical stressed stretch —
    roughly a month of trading days is a reasonable default for daily bars.  Too
    short and volatility clustering is destroyed, understating drawdown; too long
    and there are too few distinct blocks to resample, so every path looks like
    the original.
    """
    clean = returns.dropna().astype(float)
    if len(clean) < block_bars * 2:
        raise ValueError(
            f"need at least {block_bars * 2} returns for {block_bars}-bar blocks, got {len(clean)}"
        )
    if paths < 1:
        raise ValueError("paths must be at least 1")

    values = clean.to_numpy()
    n = values.size
    blocks_per_path = int(np.ceil(n / block_bars))
    max_start = n - block_bars
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start + 1, size=(paths, blocks_per_path))

    offsets = np.arange(block_bars)
    # (paths, blocks, block_bars) -> flatten to a return path, then trim to n.
    sampled = values[starts[:, :, None] + offsets[None, None, :]]
    drawn = sampled.reshape(paths, -1)[:, :n]
    curves = np.cumprod(1.0 + drawn, axis=1)

    observed = np.cumprod(1.0 + values)
    return MonteCarloResult(
        method=f"block bootstrap ({block_bars}-bar blocks, compounded)",
        paths=paths,
        observed_max_drawdown=_max_drawdown(observed),
        observed_net_return=float(observed[-1] - 1.0),
        drawdowns=tuple(_max_drawdown(curve) for curve in curves),
        net_returns=tuple(float(curve[-1] - 1.0) for curve in curves),
        ruin_threshold=ruin_threshold,
        seed=seed,
        note=(
            f"{n} observed bars, {blocks_per_path} blocks per path; blocks preserve "
            "volatility clustering, which a day-by-day shuffle destroys"
        ),
    )

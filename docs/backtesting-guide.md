# Backtesting

> A backtest is a **hypothesis**, not evidence. It is the weakest form of evidence in this
> system: one draw from one historical path, on data you chose, with parameters you tuned.

Everything here exists to stop a backtest flattering you.

---

## Where backtests lie: the fill model

Everything else can be correct and the result still be fiction, because the fill model decides
what price you got and whether you got it at all. Four things it refuses to pretend:

**You cannot trade at a price the bar never printed.** A bar is a lossy summary — within it the
price may have visited the high and the low in either order. Assuming favourable intra-bar
ordering is the single most common way a backtest manufactures alpha. Fills are clamped to the
bar's range no matter what the slippage model says.

**You pay the spread.** Every round trip loses roughly the full spread before anything else.
The default `SpreadSlippage` charges half on each side.

**Your own order moves the price.** `SquareRootImpact` grows impact with the square root of
participation — the standard academic model, where doubling an order costs about 1.4× rather
than 2×.

| Order size | Slippage | In bps |
|---:|---:|---:|
| 100 | 0.42 | 3.0 |
| 1,000 | 0.72 | 5.2 |
| 10,000 | 1.68 | 12.0 |
| 100,000 | 4.71 | 33.6 |

**You cannot buy more than traded.** `max_participation` caps a fill at a share of the bar's
volume; the rest becomes a partial fill. This is what stops a backtest "buying" ₹20 lakh of a
stock that trades ₹5 lakh a day and reporting a spectacular return.

```bash
uv run trading backtest --slippage none    # fiction — for isolating other effects
uv run trading backtest --slippage spread  # default
uv run trading backtest --slippage impact  # spread + square-root impact
```

### Limit orders

Touching a price is **not** being filled at it — someone was ahead of you in the queue. Limit
fills take a configurable haircut, without which limit-order backtests are systematically
optimistic. A limit the bar never touched simply does not fill, and a bar with no volume fills
nothing at all. Both are real outcomes, not edge cases.

---

## Read this metric first

```
Deflated Sharpe   0.0013  (50 trial(s) claimed)
```

Test 500 strategy variants and report the best, and its Sharpe estimates **the maximum of 500
draws**, not that strategy's edge. The Deflated Sharpe Ratio corrects for exactly that, plus
the non-normality of returns — a strategy with fat left tails needs a much higher raw Sharpe to
be believable.

The same 1.21 Sharpe, honestly discounted:

| Trials claimed | DSR | Noise alone could reach | Read as |
|---:|---:|---:|---|
| 1 | 0.992 | — | Credible |
| 10 | 0.798 | 0.79 | Not distinguishable from luck |
| 100 | 0.453 | 1.27 | Not distinguishable from luck |
| 1,000 | 0.200 | 1.63 | Not distinguishable from luck |

**You must supply the trial count.** `--trials N` is a claim about your own process, and the
system cannot verify it. Under-reporting it is lying to yourself, which is the only person a
backtest can deceive.

---

## The full report

```bash
uv run trading backtest --strategy sma_cross --full-report --trials 50
```

Returns, risk, risk-adjusted ratios, trade statistics, portfolio exposure and a benchmark
comparison — plus a monthly returns grid, and automated flags:

```
Worth looking at before believing this
  • Only 10 trades — the error bars swamp the estimate
  • Deflated Sharpe 0.001 — not distinguishable from the best of 50 trials
  • Underperforms buy-and-hold of the same instrument, net of costs
```

Flags are prompts to look harder, not conclusions. Every one has an innocent explanation;
several together rarely do.

### Gross and net are always both reported

The gap between them is the cost drag, and for most strategies it is the whole story. A report
showing only one of them is hiding the interesting half.

### The benchmark is buy-and-hold of the same instrument

That is the bar every strategy has to clear **after costs**, and most do not. On the synthetic
fixture, the SMA crossover loses 9.5% while buy-and-hold gains 44.7% over the same period.
That is the intended lesson, not a claim about any market.

---

## The leakage canary

```bash
uv run trading canary --strategy sma_cross --trials 10
```

Run the strategy on data whose temporal structure has been destroyed by bootstrapping its own
returns. The synthetic series keeps the instrument's real volatility, drift and fat tails but
has no pattern a strategy could legitimately exploit. **If it still makes money, something is
reading the future.**

### What it catches, and what it does not

Measured against three deliberate look-ahead strategies on the same data:

| Leak | Real (gross) | Shuffled (gross) | Caught? |
|---|---:|---:|:--:|
| Peeks 2 bars ahead | +863% | **+795%** | ✅ |
| Peeks at next bar's open-to-open move | +3,333% | **+28,976%** | ✅ |
| Peeks 1 bar ahead, trades the following open | +2,280% | −6% | ❌ |

The third **evades the profitability test.** Its peeked value only becomes exploitable through
real autocorrelation, which the bootstrap destroys — so the cheat earns nothing on shuffled
data despite being a blatant look-ahead. A second signal catches it: the real result sits far
outside the noise distribution, reported separately as *worth a look* rather than as a leak,
because a genuinely good strategy would also sit high.

> This proves a **class** of failure is absent, not that a strategy is correct. A canary whose
> limitation is documented is worth much more than one assumed to be complete.

---

## The backtrader cross-check

Phase 0 kept backtrader out of production — no Indian broker, and a cost model that cannot
express a per-scrip-per-session fee — but gave it the job it is genuinely good at:
**an independent implementation to check ours against.**

An independent engine disagreeing is the single most effective check on a backtester. Both run
the same fixed-quantity SMA cross, with no slippage and a flat 0.1% commission, over the same
800 bars:

```
ours         1,146,433.18   8 fills
backtrader   1,146,891.98   8 fills
difference         458.80   = 0.046% of starting capital
```

```bash
uv run pytest tests/crosscheck -q
```

backtrader is a **dev dependency only** — it never enters the production dependency tree. If
this test starts failing, one of the two engines has a bug, and it is worth finding out which
before trusting either with a promotion decision.

---

## Conventions, fixed once

| Convention | Value | Why |
|---|---|---|
| Signal timing | Bar N's close | It is the last thing knowable |
| Fill timing | Bar N+1's **open** | At bar N's close the market is shut. Same-bar fills are the #1 source of fake alpha |
| Unfilled DAY orders | Expire | The market moved on; carrying them forward invents liquidity |
| Warm-up bars | Skipped | A strategy must not trade on an unconverged indicator |

All of it is recorded in the run manifest, alongside the data version, strategy content hash,
cost model version and risk profile — so a changed number is always attributable to exactly one
input.

---

## What is still missing

- **Walk-forward and out-of-sample validation** — Phase 6. Until then no strategy can pass the
  `VALIDATED` lifecycle gate, which is the correct answer rather than a limitation.
- **Parameter sensitivity surfaces** — the plateau-versus-spike test, the single best
  overfitting detector.
- **Monte Carlo trade resampling**, for a distribution of drawdowns rather than the one path
  history happened to take.
- **Bid/ask data.** Spread is currently an assumption in basis points, not a measurement.

# Validation

*Phase 6. Read `docs/backtesting-guide.md` first — this document assumes you know
what a backtest produces.*

## What this layer is for

A backtest produces a number. Validation asks whether that number is **evidence**
or an **artefact of how hard you looked**.

That distinction is the whole subject. It is not a formality, and it is not
pessimism for its own sake. It is the difference between the two most common
outcomes in retail algorithmic trading:

- You test 200 variations, report the best one, and go live on a result that was
  the maximum of 200 draws from noise. It stops working immediately, and you
  conclude the market changed.
- You test 200 variations, measure how much of the best result survives on data
  it never saw, find that almost none of it does, and save your capital.

Everything in `trading/validation/` exists to make the second outcome the default.
**Nothing in this layer makes a strategy better. Everything in it makes a claim
about a strategy harder to make.**

---

## The five checks, and the question each one answers

| Check | Question | Module |
|---|---|---|
| Walk-forward | Would I actually have made money doing this? | `walkforward.py` |
| Purged CV | Do these parameters work across regimes, or just one? | `purged_cv.py` |
| Sensitivity | Does the result survive changing the parameters slightly? | `sensitivity.py` |
| Monte Carlo | How bad could the drawdown have been? | `montecarlo.py` |
| Trial count | How many things did I try before finding this? | `experiments.py` |

Run them all with one command:

```bash
trading validate --symbol RELIANCE --source fixture \
  --axis "fast=20,30,40,50,60" --axis "slow=100,150,200"
```

Expect it to fail. It is built to.

---

## 1. Walk-forward analysis — the only deployment-realistic check

### The problem it solves

A single backtest tunes parameters on the whole history and reports the result on
the same history. That number is not a prediction. It is a description of how well
the tuning worked, which is a different thing and is always flattering.

### How it works

Walk-forward repeats the honest procedure:

```
choose parameters using bars    1..756     trade bars  757..1008 with them
choose parameters using bars  253..1008    trade bars 1009..1260 with them
choose parameters using bars  505..1260    trade bars 1261..1512 with them
...
```

Each test segment is traded with parameters chosen **before it began**. Stitch
those segments together and you have the equity curve a disciplined operator
re-tuning on that schedule would have had.

> **Only the stitched out-of-sample curve is quotable.** The in-sample column in
> the report exists to be compared against. Quoting it is the error this entire
> layer exists to prevent.

```bash
trading walk-forward --axis "fast=20,50" --axis "slow=100,200" \
  --train-bars 756 --test-bars 252
```

### What the numbers mean

**Walk-forward efficiency** — out-of-sample performance over in-sample
performance. Sharpe 2.0 in-sample and 0.2 out-of-sample gives 0.1: the tuning
found noise. Above about 0.5 the edge survived the transition — and even that is
a weak claim. It says the strategy is not *purely* fitted, not that it is
profitable.

A **negative** efficiency is left as it is, and means something specific: the
training windows found a positive objective and the test windows delivered the
opposite sign. That is a stronger failure than zero.

**Consistency** — the fraction of test segments that made money. A strategy whose
entire return comes from one fold is one regime's luck, however good the total.

**Parameter stability** — how often the folds agreed on the best setting. This is
the quieter and often more damning signal. If every fold picks a wildly different
optimum, there is no stable optimum to find, and whichever value a full-history
fit lands on is arbitrary. A strategy can post acceptable efficiency and still
fail this.

### Two traps this implementation handles explicitly

**Warmup.** A strategy needing a 200-bar average does nothing for the first 200
bars it sees. Hand it a 252-bar test window and most of the window is dead. So
every test window is *run* with a warmup prefix reaching back into earlier bars,
and only the bars from the window's own start are *measured*. Re-reading earlier
bars to warm an average is not look-ahead — it is what a live system does every
morning.

**Carried positions.** Because of that prefix, a position can be carried into the
test window from the tuning period. Its mark-to-market movement is real, and a
live operator switching parameters does hold whatever the previous run left. But a
fold with **no completed round trip** inside its test window is measuring that
carried position rather than any out-of-sample decision, so such a fold is
**skipped, not scored**.

This matters more than it sounds. On the reference strategy, counting those folds
turned a **−1.29%** out-of-sample result into **+0.84%** — the entire apparent
profit came from positions entered while the strategy was being fitted.

A consequence worth knowing rather than discovering: a strategy holding for six
months reports zero complete trades on a three-month test window, and every fold
is skipped. That is the correct answer — the window is shorter than the holding
period and cannot evaluate the strategy — and the skip reason says so.

---

## 2. Purged, embargoed cross-validation — a second opinion

### Why ordinary k-fold is wrong here

Standard k-fold shuffles rows and trains on a random 80%. On a price series that
means training on 2025 to predict 2023. The model is allowed to know the future,
it will score beautifully, and it will tell you nothing.

### What purged CV adds

Walk-forward can never test its own first training window: the earliest years are
only ever used for tuning. If those years hold the one regime the strategy cannot
survive, walk-forward will not find out. Purged k-fold uses **every** fold as a
test fold in turn, so every regime in the sample gets tested.

### What it is not

For any fold but the first, the training data includes bars that come **after** the
test fold. That is intrinsic to the method. It means a number from here is **not**
evidence the strategy could have been traded. Only walk-forward is that.

The verdict string says so, and a test asserts the wording, so a future edit
cannot quietly upgrade the claim.

### Purging and embargo

Two adjacent bars are not independent observations:

- A trade opened a week before a test window **closes inside it**, so the two
  windows share an outcome. *Purging* drops the bars immediately before each test
  fold from training. Set `purge_bars` to at least the longest holding period the
  strategy permits.
- Volatility clusters, so the days following a stressed window still carry its
  regime. The *embargo* drops the bars immediately after.

Without them, "out of sample" overlaps in-sample by the length of a trade.

### The honest limitation

Purged k-fold was designed for a feature/label matrix, where dropping rows is
exact. A backtest is a **path**: positions carry between bars. Training on a
non-contiguous sample therefore means running the backtest separately over each
contiguous chunk and combining the objectives, which **restarts position state at
each chunk boundary**. Every result records how many restarts that involved
(`mid-sample restarts` in the report) so it cannot be forgotten.

---

## 3. Parameter sensitivity — the plateau-versus-spike test

**If you only ever run one overfitting check, run this one.** It is the cheapest,
the hardest to argue with, and the easiest to explain to someone who has never
seen a backtest.

Sweep a parameter across its plausible range and look at the shape.

```
A PLATEAU — a real effect                A SPIKE — fitted to noise

sharpe                                   sharpe
1.2 │      ╭────────╮                    2.0 │          ╷
1.0 │   ╭──╯        ╰──╮                  1.0 │          │
0.8 │ ╭─╯                ╰─╮              0.0 │──────────┴──────────
    └─┴──┴──┴──┴──┴──┴──┴──┴─           -0.5 │
     20 30 40 50 60 70 80                     └─┴──┴──┴──┴──┴──┴──┴─
                                                20 30 40 50 60 70 80
```

**A plateau.** SMA(45), SMA(50) and SMA(55) all earn roughly the same. The
strategy is exploiting something that does not care about the exact number, which
is what a real market effect looks like.

**A spike.** SMA(50) earns 40%; SMA(45) loses 5%; SMA(55) loses 8%. Nothing about
the market changes between a 49-day and a 50-day average. A result that collapses
between them is not measuring the market — it is measuring which setting happened
to line up with the noise in this sample. Deploying it is betting that next
year's noise lines up the same way.

```bash
trading sensitivity --axis "fast=10:60:10" --axis "slow=100,150,200"
```

Two numbers decide the verdict:

- **plateau ratio** — the peak's *neighbourhood* score divided by the peak's own
  score. The neighbourhood deliberately excludes the point itself, so a spike
  cannot prop up its own neighbourhood. 1.0 is a flat shelf; near zero means the
  peak stands alone; **negative** means its neighbours lose money, which is the
  clearest possible statement that the peak is noise.
- **profitable fraction** — how much of the grid made money at all. A strategy
  profitable at one setting in twenty found a coincidence.

### The most useful output is not the best point

It is **`robust best`**: the setting whose *neighbourhood* is strongest. On a
genuine plateau this lands near the middle. On a spiky surface it deliberately
refuses the spike.

The report labels it **`robust best (do NOT deploy)`** when the surface is a
spike, because naming a setting to deploy on a grid that failed would be
recommending the least-bad point of a bad grid.

### Two things the sweep gets right that are easy to get wrong

- **Neighbours are axis-aligned, never diagonal.** A diagonal neighbour differs in
  two parameters, which blurs "this parameter is insensitive" with "this
  combination also happens to work".
- **Warmup moves with the parameters.** Sweep `slow` up to 300 with warmup left at
  201 and the strategy evaluates an average that is still NaN, trades nothing, and
  produces a flat surface that reads as a *perfect plateau*. The CLI derives
  warmup from the largest period it is sweeping.

### Run it on the training period

Sweeping the out-of-sample window and reporting the best point is tuning on the
test set with extra steps. `trading sensitivity` defaults its end date well before
the backtest default for this reason, and `trading validate` confines the sweep to
the first training block.

---

## 4. Monte Carlo — the drawdown you did not happen to see

A backtest produces **one** path through history. Its maximum drawdown is not "the
strategy's maximum drawdown" — it is the worst run of bad luck that happened to
occur, in the order it occurred. Deal the same trades in a different order and the
drawdown changes, often by a lot.

Sizing a position or setting a halt from that single figure is setting it from a
**sample of one**.

Two resamplings, answering two different questions:

### Trade resampling

Draws the observed trades with replacement into new sequences. *Given this
distribution of trade outcomes, how bad could the ordering have been?*

Sampling with replacement means a path can contain the worst trade twice. That is
the point: history contained it once because of how the sample fell, not because
the strategy cannot hit it twice in a row.

It holds position sizing fixed and accumulates P&L additively, so it describes the
drawdown distribution of the **trade sizes actually taken** — not of a compounding
account that would have sized up after wins. That is the conservative reading, and
it is stated rather than dressed up as a forecast.

### Block bootstrap of returns

Draws **contiguous blocks** of daily returns and compounds them.

Blocks, not individual days, because volatility clusters: bad days arrive
together, and that clustering is what creates drawdowns. A day-by-day bootstrap
destroys it and **systematically understates drawdown** — it would hand back a
comfortable number that is an artefact of the method. A test in
`tests/unit/test_montecarlo.py` builds returns with deliberate clustering and
asserts the 40-bar estimate comes out deeper than the 1-bar one.

*(On genuinely i.i.d. returns the two agree, which is the correct behaviour — there
is no clustering to preserve.)*

### What to read

| Number | What it tells you |
|---|---|
| `observed path percentile` | Where the one real path sits in the distribution. Near the 5th percentile means history was unusually **kind**, and the observed drawdown badly understates what to prepare for. |
| `p95 drawdown` | The level only 5% of orderings exceeded. **Set risk limits from this, not from the observed path.** |
| `probability of >= N% drawdown` | "Ruin" is not zero equity — it is the drawdown at which a real operator stops, which is much shallower and is the number that actually ends a strategy's life. |
| `recommended halt level` | p95 with headroom. A halt set at the observed drawdown is breached about half the time by chance alone, and a halt that fires on normal variation trains its operator to ignore it. |

### What none of this does

Every path is drawn from returns that actually occurred, so the worst case is
bounded by the worst regime in the sample. **If your history contains no 2008 and
no March 2020, no amount of resampling will invent one.** A p99 drawdown here means
"bad luck within the regimes I have seen", never "the worst that can happen".

---

## 5. The experiment ledger — an honest denominator

The **Deflated Sharpe Ratio** corrects a reported Sharpe for how many things were
tried before it was found. Test 500 variants and report the best, and its Sharpe
estimates the *maximum of 500 draws*, not that strategy's edge.

That correction is only as good as the trial count — and a trial count typed in by
the person reporting the result is worth nothing. It will be 1.

So trials are counted by the machine that runs them. Every evaluation goes into an
append-only SQLite table keyed by a hash of the parameters:

```bash
trading experiments --strategy sma_cross_param
```

```
  strategy          instrument     trials   evaluations   best    last seen
  sma_cross_param   NSE:RELIANCE        8           199   0.958   2026-09-12T19…
```

Eight distinct parameter sets across 199 evaluations. The **8** is what deflates
the Sharpe. Note that it accumulates **across sessions**: a sweep you ran last
week still counts against a result you find today, which is exactly right and is
the part a human ledger always gets wrong.

Three design decisions:

- **Append-only.** No update, no delete. A failed experiment is evidence, and the
  natural thing to delete is precisely the run that makes the winner look lucky. A
  test asserts the methods do not exist.
- **Parameters are canonicalised.** `10`, `10.0` and `Decimal("10")` are the same
  parameter to a strategy and hash to one trial. Counting them separately sounds
  conservative but makes the uniqueness claim false and inflates one strategy's
  count relative to another's.
- **SQLite, not Parquet.** The bar store is columnar because it answers range scans
  over millions of rows. The ledger answers "how many distinct parameter sets has
  this strategy seen?", wants a uniqueness constraint, and is written one row at a
  time. That is a transactional workload, SQLite is in the standard library, and
  the file opens in any sqlite client.

### Its honest limitation

The ledger counts what was recorded **through it**. Trials run in a notebook, by
hand, or in a previous project are invisible, so the count is a **floor** rather
than the truth. A floor is still far better than a self-report.

---

## Parameterised strategies

Sweeping needs something to sweep. A YAML strategy can declare parameters and
reference them from its rules:

```yaml
parameters:
  fast: 50
  slow: 200
entry:
  when: crossed_above(sma(close, {fast}), sma(close, {slow}))
  target_weight: 0.20
exit:
  when: crossed_below(sma(close, {fast}), sma(close, {slow}))
```

### Why substituting text into a rule is safe here

Text substitution into an expression is exactly the shape of an injection bug. Two
things constrain it, and either alone would be enough:

1. **Parameter values must be numeric.** A number cannot contain an operator, a
   call, or a parenthesis. `fast: "50) or (close > 0"` is rejected at load time.
2. **The rendered rule still goes through the expression allowlist** — the same
   gate a hand-written rule passes.

Both are in place deliberately. `tests/unit/test_config_strategy.py` includes the
injection attempt as a test.

### Three further guards

- An **undeclared placeholder** is an error, not a syntax failure later.
- A **declared but unreferenced** parameter is an error. A sweep over a parameter
  no rule reads varies nothing, and its flat surface reads as a perfect plateau —
  worse than a crash.
- **Overriding a misspelled name** is an error. Otherwise a sweep runs the defaults
  twenty times and reports a perfect plateau.

The rendered rule — not the template — is what gets compiled, hashed and recorded
in the manifest, so there is no second templated form that could be mistaken for
what was tested. Both forms travel with the spec: `template_entry` says what was
swept, `param_fast` says which point ran.

---

## How this answers the lifecycle gate

Before Phase 6, promotion to `VALIDATED` refused everything: the checks did not
exist, and a gate that passes for want of a check is worse than no gate.

Now each criterion is derived from an artefact that ran:

| Gate | Answered by |
|---|---|
| `GATE_010_out_of_sample` | pooled walk-forward test segments: profitable **and** Sharpe above a floor |
| `GATE_011_walk_forward` | efficiency, consistency, stability, pooled trade count |
| `GATE_012_parameter_plateau` | the sensitivity surface's shape |
| `GATE_013_monte_carlo_drawdown` | resampled p95 drawdown against the risk profile's limit |
| `GATE_014_trial_count_recorded` | the experiment ledger |
| `GATE_015_deflated_sharpe` | pooled Sharpe, deflated by that count |

A missing artefact yields `None`, which the gate reads as `UNAVAILABLE` and treats
as blocking. **There is deliberately no way to hand the gate a bare `True`** — every
field is computed from something that ran, and a test asserts that a report
missing a sweep cannot produce "the parameters are on a plateau".

`GATE_015` distinguishes two failures that are easy to collapse:

- `UNAVAILABLE` — "go and run the check."
- `FAIL` — "you ran it and the answer was no."

### The data tier still blocks, and that is correct

`GATE_000_data_tier` refuses promotion past `BACKTESTING` on `SYNTHETIC` or
`PROTOTYPE` data, so a strategy validated on the fixture provider or on yfinance
**cannot reach `VALIDATED` however good its numbers are.** That is the intended
behaviour, not a bug to work around. Passing every Phase 6 check on
prototype-tier data means the *pipeline* works; it does not mean the *strategy*
does, because the data carries no point-in-time guarantee.

---

## Worked example: the reference strategy fails, and why

```bash
trading validate --source fixture --axis "fast=20,40,60" --axis "slow=100,200"
```

```
sma_cross_param is NOT validated:
  - walk-forward FAILED: out-of-sample return -6.88% (need > +0.00%);
    efficiency -2.05 (need >= 0.50); consistency 14% (need >= 50%);
    parameter stability 43% (need >= 50%); only 9 out-of-sample trades (need >= 30)
  - pooled out-of-sample record fails: -6.88% return, Sharpe -0.29
  - spike: its neighbours score 0% of its own score; 0% of the grid was profitable
  - resampled p95 drawdown 22.22% exceeds the 15.00% risk limit
  - deflated Sharpe 0.020 below 0.95 — not distinguishable from the best of 6 trials
  - data tier is SYNTHETIC; a validated strategy needs PRODUCTION data
```

Read that as six independent reasons, not one. Every check failed, for a different
reason, and the strategy would have been rejected by any one of them alone.

This is the expected result. SMA crossover on a synthetic random walk has no edge
to find, and `tests/e2e/test_validation_pipeline.py` asserts the failure — a
pipeline that passed it would be broken.

---

## Choosing the knobs

None of these have a correct answer, which is why none of them is buried in the
code. Whatever you choose lands in the report.

| Knob | Trade-off |
|---|---|
| `--train-bars` | Long windows give a stabler parameter estimate but assume the market has not changed. Short ones adapt faster but tune on less evidence. |
| `--test-bars` | Few long windows give fewer, more reliable folds. Many short ones give a longer out-of-sample record made of noisier pieces — and short windows may be shorter than the holding period, which skips folds. |
| `--anchored` | Anchored grows the training window from a fixed origin (older data still applies). Rolling discards it (the regime has changed). |
| `--objective` | `sharpe` is interpretable. `calmar` is better when survivable drawdown matters more than smoothness. `net_return` is a diagnostic only — maximising return alone reliably selects the most leveraged noise on the grid. |
| `--min-trades` | The floor below which a segment is not scored at all. Too low and a two-trade fluke wins a fold; too high and a slow strategy has no scoreable folds. |
| `purge_bars` | At least the longest holding period, or a trade links the training and test samples. |
| `block_bars` | Long enough to contain a typical stressed stretch (~a month of daily bars). Too short destroys clustering; too long and every path resembles the original. |

---

## What is still missing

Stated plainly, because a validation layer that hides its own gaps is the thing it
exists to prevent:

- **Point-in-time index membership.** Survivorship bias is not yet defensible for
  universe strategies. Single-instrument validation is unaffected.
- **Combinatorially purged CV** (multiple test folds per split, giving a
  distribution of paths rather than one) is not implemented. Plain purged k-fold is.
- **Multiple-testing correction across strategies.** The deflated Sharpe corrects
  for trials within one strategy. Comparing twenty strategies and picking the best
  is a second selection this does not yet correct for.
- **Regime labelling.** Purged CV shows *that* a fold failed, not *what regime* it
  failed in. Reading that still requires looking at the data.
- **Bid/ask spread is still an assumption in basis points, not a measurement**, so
  every cost figure downstream of it inherits that assumption.

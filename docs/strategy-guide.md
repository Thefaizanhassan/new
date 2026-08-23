# Writing Strategies

Two ways to define a strategy, and they run through **exactly the same pipeline** — the same
risk engine, the same cost model, the same lifecycle gates, the same run manifest. There is no
fast path that skips a check.

---

## 1. The quick way: YAML

Drop a file in `configs/strategies/`:

```yaml
id: mean_reversion_reliance
name: Z-Score Mean Reversion — Reliance
version: 1.0.0
author: rajeshwar
universe: [NSE:RELIANCE]
warmup_bars: 201
entry:
  when: zscore(close, 20) < -2.0 and close > sma(close, 200)
  target_weight: 0.20
exit:
  when: zscore(close, 20) > 0.0
```

```bash
uv run trading validate-strategy configs/strategies/mean_reversion_reliance.yaml
uv run trading backtest --strategy mean_reversion_reliance --source store
```

Changing a number in that file gives you a **different strategy** with a different content hash
and therefore a different backtest identity. That is deliberate: two strategies quietly sharing
an identity make every result ambiguous.

### About that trend filter

`close > sma(close, 200)` is not decoration. Buying every dip is how mean-reversion strategies
die — in a downtrend the price is cheap *for a reason*, and "oversold" stays oversold for weeks.
The filter restricts the trade to instruments still in an uptrend, which is the difference
between mean reversion and catching a falling knife.

---

## 2. The expression language

Rules are **not** Python. They are a deliberately small language, parsed to an AST and checked
node by node against an allowlist before anything runs.

```bash
uv run trading strategy-language
```

**Series:** `open` `high` `low` `close` `volume`

**Indicators:** `sma` `ema` `rsi` `atr` `adx` `obv` `macd` `macd_signal` `bb_upper` `bb_middle`
`bb_lower`

**Transforms:** `crossed_above` `crossed_below` `zscore` `pct_change` `rolling_max` `rolling_min`
`prior_max` `prior_min` `prev` `abs`

**Operators:** `and` `or` `not` `+ - * / % **` `< <= > >= == !=`

### Why it is this small

`eval()` on a config file is a code-injection hole. It matters more than usual here because
Phase 0 §10.7 makes YAML the **only** form an AI-drafted strategy may take — a validated config
is auditable in a way that generated Python is not.

So the following are rejected, each with a reason:

| Rejected | Why |
|---|---|
| `close.values` | Attribute access is how every sandbox escape begins — `().__class__.__bases__` |
| `close[0]` | Indexing isn't needed and widens the surface |
| `lambda`, comprehensions | Not needed; both can smuggle in execution |
| f-strings | They evaluate arbitrary expressions |
| `a if b else c` | Use `and`/`or` |
| `1 < close < 2` | Chained comparisons don't vectorise; write `a > b and b > c` |
| Any name not in the lists above | Including every builtin |

```
>>> compile_expression("__import__('os').system('rm -rf /')")
ExpressionError: Only direct function calls are allowed
```

### A trap worth knowing

```yaml
when: close > rolling_max(high, 20)     # never true
when: close > prior_max(high, 20)       # what you meant
```

`rolling_max` includes the current bar, and a close cannot exceed its own bar's high. The rule
compiles, validates, and silently never fires. `validate-strategy` warns when an entry rule
never fires over the sample, which is how you catch this in seconds rather than after a
confusing backtest.

### One accommodation for pandas

`and`/`or`/`not` short-circuit by calling `__bool__`, which a Series refuses to answer. Writing
`&`/`|` works but reads badly and has surprising precedence — so the validated AST is
**rewritten**: `and` → `&`, `or` → `|`, `not` → `~`. Precedence is already fixed by the tree
shape, so the rewrite cannot change meaning. Both forms are accepted.

---

## 3. The Python way

For anything the expression language can't say — multi-instrument logic, custom state, ML
inference. Implement two methods:

```python
class MyStrategy:
    spec: StrategySpec

    def warmup_bars(self) -> int: ...
    def on_bar(self, ctx: StrategyContext, instrument: Instrument) -> list[Intent]: ...
```

Two rules that are not negotiable:

**Emit `Intent`, never orders.** A strategy declares a *target* ("be 4% long"), not an action
("buy 21 shares"). Targets are idempotent, they compose across strategies, and they keep sizing
in one place so two strategies cannot unknowingly stack exposure.

**Read only from `ctx`.** The context is built per bar and **physically does not contain future
data**. That is what makes look-ahead bias structurally impossible rather than a discipline you
have to maintain. Reaching around it to a dataframe you closed over defeats the entire design.

---

## 4. Features and why precomputation is safe

A strategy recomputing a 200-bar average on every bar is O(n²). Declaring features lets the
engine compute them **once** over the full history and slice per bar — measured at **21× faster
over 2,868 bars, with byte-identical results.**

```python
def feature_specs(self) -> list[FeatureSpec]:
    return [FeatureSpec("trend", "sma(close, 50) > sma(close, 200)", as_condition=True)]
```

Computing over the full series then slicing sounds exactly like the mistake this platform exists
to prevent. It is safe **if and only if** every feature is *causal* — its value at time *t*
depends only on data at or before *t*. Moving averages, RSI, ATR and z-scores are. A centred
average or a negative shift is not.

So causality is checked rather than assumed:

```bash
uv run trading validate-strategy configs/strategies/my_strategy.yaml
```

```
Causality (does a rule read the future?)
  _entry   PASS
  _exit    PASS
```

`verify_causality` recomputes a feature on truncated history and asserts it matches the
precomputed value. A test proves the guard has teeth by feeding it `prev(close, -1)` — a
one-bar look-ahead — and asserting it fails.

---

## 5. Lifecycle: statuses you have to earn

```
RESEARCH → BACKTESTING → PROMISING → VALIDATED → PAPER → LIVE_APPROVED → LIVE
```

```bash
uv run trading lifecycle --strategy-id sma_cross --from-status BACKTESTING --to-status PROMISING
```

Three rules govern promotion:

**One step at a time.** Skipping a stage is refused however strong the evidence — it is how a
strategy reaches paper trading without ever having been validated.

**Data trust caps ambition.** A strategy backtested on `SYNTHETIC` or `PROTOTYPE` data cannot be
promoted past `BACKTESTING`, whatever the numbers say. Phase 2 made the tier travel with the
bars precisely so this gate could read it. **This currently blocks everything**, because
yfinance is prototype-tier — which is the correct answer, not a bug.

**An unimplemented check blocks rather than passes.**

```
[UNAVAILABLE] GATE_011_walk_forward: not assessed (required: True)
              — the check that would answer this is not implemented yet
```

Walk-forward validation arrives in Phase 6. Until then the criterion reports `UNAVAILABLE` and
refuses promotion. A gate that silently passes because nobody wrote its check doesn't just fail
to help — it manufactures confidence.

Pausing and retiring are never gated. Stopping a strategy must not require evidence; that is how
a bad strategy keeps running.

---

## 6. The registry

```bash
uv run trading strategies
```

Everything registered, from Python and YAML alike, with its content hash and origin. Registering
the same `(id, version)` twice is an error rather than an overwrite. `get()` without a version
returns the highest, compared numerically — so `1.10.0` beats `1.2.0`.

A malformed config in the directory is reported and skipped rather than aborting the load, so one
bad file cannot hide every good one.

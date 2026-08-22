# Technology Evaluation

**Required by:** your Technology Standards §6 and §9 — evaluate before the first production
implementation, and verify current maintenance status rather than assuming.
**Date:** 2026-08-22 · **Verified against:** live PyPI metadata and actual installs in a
Python 3.12 environment, not from memory.

**Evidence legend:** `[FACT]` verified in this session · `[ASSUMPTION]` unverified ·
`[HYPOTHESIS]` to be tested

---

## 0. Verdict summary

| Your default | Verdict | Note |
|---|---|---|
| **Pandas** | ✅ **Accepted** | Already the design's choice. No argument. |
| **TA-Lib** | ✅ **Accepted** | My earlier reservation was based on stale information. Corrected below. |
| **Backtrader** | ⚠️ **Contested** — approval requested | Runs fine; but cannot be the *production* engine for an India-first system. Full case in §4. |

One correction I owe you up front: in the Phase 0 document I dismissed TA-Lib's install
friction and steered toward hand-written indicators. **I tested it, and I was wrong** — TA-Lib
now ships binary wheels and installs in under a second with no C toolchain. Your default is the
right call and I've adopted it.

---

## 1. The evaluation table you asked for

| Area | Your default | Evaluated against | Decision |
|---|---|---|---|
| Data analysis | Pandas | Polars, DuckDB | **Pandas** as the analytical interface; **DuckDB + Parquet** as the storage layer beneath it |
| Numerical | NumPy | — | **NumPy** (transitively via pandas and TA-Lib) |
| Technical indicators | TA-Lib | pandas-ta, custom | **TA-Lib** ✅ |
| Backtesting | Backtrader | vectorbt, NautilusTrader, custom | **Contested — see §4** |
| Storage | PostgreSQL / DuckDB / Parquet | SQLite, Supabase, TimescaleDB | **Postgres** (ledger) + **DuckDB/Parquet** (market data) |
| API | FastAPI | — | **FastAPI**, from Phase 6 |
| ML | scikit-learn / PyTorch | — | **scikit-learn** from Phase 9. **PyTorch deferred** — see §6 |
| UI | Streamlit | NiceGUI, React | **Streamlit** |

---

## 2. Pandas — accepted

**1. What it does.** Provides the DataFrame: a labelled, indexed 2-D table with vectorised
operations, time-series-aware indexing, resampling, joins and grouping.

**2. Why we need it.** Every step of your §1 list — parsing timestamps, normalising schemas,
detecting missing candles and duplicate timestamps, aligning series, resampling, joining,
preparing features, analysing trades — is a DataFrame operation. Writing that by hand would be
slower to build and far more likely to be wrong.

**3. What uses it.** The data layer (loading, validation, normalisation), the feature layer, and
all post-run analysis and reporting. **It does not appear in the domain core** — see the
boundary note below.

**4. Alternatives.** *Polars* is genuinely faster with a stricter API and better memory
behaviour, but a much smaller finance ecosystem. *DuckDB* is superb for analytical SQL over
large files but is not an in-memory manipulation library.

**5. Why selected.** Ecosystem gravity. Every finance example, every tutorial and every library
you'll encounter assumes pandas, which matters a great deal while you're learning. It is also
what TA-Lib, scikit-learn and Streamlit all expect.

**6. Limitations.** Memory-hungry (roughly 5–10× the on-disk size); single-threaded for most
operations; silent dtype coercion can turn a Decimal column into float without warning; and the
index/copy semantics are a genuine source of subtle bugs.

**7. Suitability.** Research ✅ · Backtesting ✅ (analysis and reporting) · Paper ⚠️ (fine for
periodic work, not per-tick) · Live ❌ **not in the order path.**

> ### The boundary I'm enforcing, and why
>
> Your §1 warns against reading "pandas handles millions of rows" as permission to put unlimited
> data in DataFrames. I want to go one step further, because it matters for correctness rather
> than performance:
>
> **Pandas columns are float64. The ledger is `Decimal`.** Money, quantities and prices in
> `trading.core` never pass through a DataFrame. Pandas is the *analytical* layer — it loads,
> validates, resamples and reports. The moment a value becomes a position or a cash balance, it
> converts to `Decimal` and stays there. This is the single most important boundary in the data
> design, and it directly serves your §7 hierarchy: correctness before convenience.

**Workload classification** (your §1 requirement):

| Workload | Rows (rough) | Tool |
|---|---|---|
| Daily bars, 50 instruments, 20 yrs | ~250k | **Pandas** — trivial |
| Minute bars, 50 instruments, 5 yrs | ~23M | **DuckDB/Parquet** to filter → pandas to analyse |
| Tick data | billions | **Not in scope.** Would need DuckDB/Arrow throughout |
| Real-time streaming | — | **Never pandas.** Plain Python objects in the event loop |

Phase 1–8 sit entirely in row one. `[ASSUMPTION]` You will not hit a pandas performance wall
before Phase 13 (intraday).

---

## 3. TA-Lib — accepted, and I was wrong to doubt it

**1. What it does.** A C library (with a Python wrapper) implementing ~150 standard technical
indicators and candlestick patterns, operating on NumPy arrays.

**2. Why we need it.** Your §4 is right: don't reinvent well-tested indicator maths. Indicator
bugs are insidious — a subtly wrong RSI doesn't crash, it just quietly produces a strategy that
backtests differently from every reference implementation, and you'd never know.

**3. What uses it.** The feature/indicator layer only. Never the domain core, never the risk
engine.

**4. Alternatives evaluated `[FACT]`:**

| Option | Status verified 2026-08-22 | Verdict |
|---|---|---|
| **TA-Lib** | **v0.7.1, released 2026-07-16.** 54 wheels including macOS-arm64 for cp310–cp314 | ✅ **Selected** |
| pandas-ta | v0.4.71**b0** — a beta, last upload 2025-09-14. Maintainer has publicly flagged funding sustainability risk | ❌ Beta + funding risk is the wrong foundation |
| pandas-ta-classic | Community fork, ~253 indicators, no C dependency | 🔶 Credible fallback if TA-Lib ever breaks |
| Custom implementations | Full control, no dependency | ❌ Reinventing tested maths, exactly what your §4 forbids |

**5. Why selected.** Actively maintained (a release five weeks ago), the de-facto reference
implementation that other libraries validate *against*, C-speed, and — the point I got wrong —
**installation is now trivial.**

> `[FACT]` **Correcting my Phase 0 position.** TA-Lib historically required
> `brew install ta-lib` before `pip install TA-Lib` would compile, which is where its
> difficult-install reputation comes from. Since v0.6.5 it ships prebuilt binary wheels. I
> tested it in this session on Python 3.12:
>
> ```
> $ uv pip install TA-Lib
> Installed 5 packages in 32ms
>  + ta-lib==0.7.1
> ```
>
> No Homebrew step, no compiler, no C headers. **The objection no longer applies.**

**6. Limitations.** Operates on NumPy float64 arrays, so it inherits float semantics — fine,
because indicators are *signals*, not money. It silently returns `NaN` for the warm-up period,
which must be handled explicitly rather than dropped (dropping creates gaps, and gaps create
look-ahead). Some functions have undocumented lookback requirements. And it will happily compute
an indicator over data containing a future value — **TA-Lib does not protect you from
look-ahead bias; the `StrategyContext` does.**

**7. Suitability.** Research ✅ · Backtesting ✅ · Paper ✅ · Live ✅ (deterministic and fast —
safe in the hot path, unlike anything statistical or AI-driven).

**Indicator set for the MVP** — driven by strategy requirements per your §4, not by availability.
Each will carry the documentation block you specified (name, purpose, input, parameters,
timeframe, interpretation, limitations):

`SMA` · `EMA` · `RSI` · `ATR` · `MACD` · `BBANDS` · `ADX` · `OBV` — eight, not a hundred.

---

## 4. Backtrader — contested, approval requested

Following your §2 protocol exactly: why, what the replacement solves, comparison, migration
implications, then your decision.

### 4.0 First, what I verified — including in backtrader's favour

I installed and ran it rather than repeating its reputation. `[FACT]`

| Test | Result |
|---|---|
| Installs on Python 3.12 | ✅ Clean, pure-Python wheel |
| Runs an SMA-cross strategy on Python 3.12 + pandas 3.0.5 | ✅ **Ran correctly** |
| Deprecation/Future warnings raised | ✅ **Zero** |
| Last PyPI release | ⚠️ **1.9.78.123, 19 April 2023** — 3 years 4 months ago |
| `requires_python` metadata | ⚠️ Absent |
| Live brokers shipped | ⚠️ **IB, Oanda, VisualChart only** |
| Broker money representation | ⚠️ `float` |
| Community fork `backtrader2` | ⚠️ No PyPI release in over 12 months |

**Its reputation for Python 3.12 breakage is overstated** — it ran clean for me. I'm not going
to argue from a problem that didn't reproduce.

### 4.1 Why I recommend against it *as the production engine*

Three findings, in descending order of how much they actually matter.

**① It has no Indian broker, and you chose India-first. `[FACT]`**

```
backtrader stores : ibstore, oandastore, vchartfile, vcstore
backtrader brokers: bbroker, ibbroker, oandabroker, vcbroker
```

Interactive Brokers, Oanda and VisualChart. **No Zerodha Kite, no Dhan, no Upstox, no Angel
One.** Nothing that trades NSE.

This is the decisive one. If strategies are written as `bt.Strategy` subclasses, then live
trading through Kite Connect requires **a second implementation of every strategy** — which is
precisely the backtest/live divergence that Phase 0 §4.2 exists to prevent, and the failure mode
where "the strategy you validated is not the strategy you are running."

**② Its cost model cannot express Indian charges. `[FACT]`**

```
backtrader.CommInfoBase.getcommission(self, size, price)
```

Two arguments. No date, no instrument identity, no session context.

India's delivery-equity DP charge is **a flat ~₹13–16 per scrip per day, charged once on the
sell side regardless of quantity.** Expressing "have I already been charged for this scrip
today?" requires state that this signature cannot reach — you'd have to smuggle in a mutable
object and hope the call ordering cooperates.

This is not a corner case. Per the Phase 0 addendum, at ₹40,000 per position that flat fee is
0.04%; at ₹5,000 it's **0.3%** — and with your starting capital, flat fees are the charge that
actually decides whether a strategy is viable. **A backtest that cannot model it is a backtest
that will lie to you in the specific way that matters most at your capital scale.**

**③ Unmaintained, under a ledger. `[FACT]`** No release since April 2023; the community fork is
also dormant. That is acceptable for a research toy. For the component that decides what a
strategy is worth before you commit real money to it, an unmaintained ~50k-line dependency you
cannot fix is a maintainability risk — your §7 ranks maintainability above performance and
convenience.

**A note on float money.** backtrader's broker uses `float`. I measured the actual drift:
10,000 additions of 0.1 lands 1.6e-10 off. **That is negligible and I won't pretend otherwise.**
The real issue isn't drift, it's that `Decimal` in the ledger and `float` in the engine means a
conversion boundary on every fill — and reconciliation against a broker that reports exact paise
gets fuzzy in a way that's tedious to debug. It's a supporting argument, not a decisive one.

### 4.2 What the replacement solves

A custom event-driven core (~1,200–1,800 lines) delivers three things backtrader structurally
cannot:

1. **One strategy implementation across backtest, paper and live** — including Kite Connect,
   because the broker is an adapter behind a protocol rather than a framework built-in.
2. **A cost model that matches Indian reality** — stateful, per-instrument, per-session, so STT,
   stamp duty, GST and the per-scrip-per-day DP charge are all expressible and *versioned*.
3. **Reproducible run manifests** — every backtest keyed by strategy hash, dataset version, cost
   model version and git SHA, so a changed result is explainable. backtrader has no concept of
   this.

### 4.3 Honest comparison

| Criterion (your §7 hierarchy) | Backtrader | Custom event-driven | Winner |
|---|---|---|---|
| **1. Correctness** | float money; cannot model Indian DP charge | Decimal throughout; exact cost model | **Custom** |
| **2. Reproducibility** | no run manifests or versioning | manifest on every run | **Custom** |
| **3. Testability** | strategies need a `Cerebro` to run; awkward to unit test | strategies are pure functions of a context | **Custom** |
| **4. Maintainability** | 50k LOC, unmaintained since 2023, no NSE broker | ~1.5k LOC you own and understand | **Custom** |
| **5. Performance** | known-slow event loop | comparable; vectorbt for sweeps | Tie |
| **6. Convenience** | **batteries included** — indicators, sizers, analyzers, plotting, huge tutorial base | must be built | **Backtrader** |

By your own decision hierarchy, backtrader wins the criterion you ranked *last* and loses the
four you ranked first. I'd have made the same call before you wrote the hierarchy down, but I
want to be clear that I'm applying your rule, not overriding it.

### 4.4 Migration and maintenance implications

**If you approve the custom engine:**
- Cost: ~2–3 weeks in Phase 5 (already budgeted in the roadmap — no schedule change).
- We own the fill model. That's the risk *and* the point: it's where every subtle bias hides,
  and writing it is how you learn where they hide.
- Zero migration cost later — the adapter design means swapping to NautilusTrader at Phase 13,
  if intraday fill realism demands it, is contained rather than a rewrite.

**If you prefer backtrader as primary, here's what it actually costs** — I'll build it if you
say so, and this is the honest bill:
- A second strategy implementation for live trading via Kite (**ongoing divergence risk**, and
  it compounds with every strategy you add).
- An approximated Indian cost model that under-reports charges at small position sizes.
- A dependency that has been unmaintained for three years sitting under your money.
- ~1 week *saved* in Phase 5. This is a real saving and I'm not dismissing it.

### 4.5 What I propose instead — you keep backtrader's real value

Not "reject backtrader." **Two legitimate roles for it, neither in the production path:**

**① A learning and prototyping sandbox.** When you want to sketch an idea in twenty minutes,
backtrader is genuinely the fastest tool for it, and its tutorial ecosystem is excellent while
you're learning. Nothing about the architecture stops you, and I'd encourage it.

**② A cross-validation oracle in Phase 5** — and this one I think is genuinely valuable.

> Implement the same reference strategy **twice**: once in our engine, once in backtrader.
> Feed both identical bars and identical simple costs. **Assert the equity curves agree within
> tolerance, as an automated test.**
>
> This is how you find out our engine has a bug. An independent implementation disagreeing is
> the single most effective check on a backtester, and it turns backtrader from a dependency
> into a *test asset* — used, respected, and not load-bearing.

**Requested decision.** Approve the custom event-driven engine as production, with backtrader in
roles ① and ②? Or overrule me and make backtrader primary — in which case I'll build it and
document the two consequences above rather than relitigate.

---

## 5. Storage — Postgres + DuckDB/Parquet

Covered fully in [Phase 0 §6](phase-0-discovery.md#6-database-and-storage-architecture).
Summary: **Postgres** for the transactional ledger, event log and registries (ACID, real
constraints, JSONB, concurrent readers); **DuckDB + Parquet** for market data and research
(columnar, 5–10× smaller than CSV, queries files directly, zero ops). SQLite rejected as the
ledger foundation because Phase 7 has three concurrent processes; TimescaleDB rejected as
unnecessary ops complexity; Supabase rejected for the hot path because order state behind a
network hop means a fill you cannot record during an outage.

---

## 6. Deferred with reasons

**FastAPI** — accepted, but **not until Phase 6**. Nothing needs an HTTP surface before there's
something to control. Adding it now would be a dependency without a purpose, which your §6
forbids.

**scikit-learn** — accepted, **Phase 9**. Needed for the meta-labelling and regime work in
Phase 0 §14.

**PyTorch — deferred indefinitely, and I'd push back if asked to add it.** `[FACT]` You have
roughly 5,000 daily bars per instrument against a signal-to-noise ratio near zero. A neural
network on that data memorises rather than generalises. Per Phase 0 §14.3, the rule is to beat
the simple baseline first and report the delta. If gradient boosting ever beats a linear model
by a margin that survives walk-forward validation, we'll revisit — but adding PyTorch before
that point is exactly the "popular library for its own sake" your standards warn against.

**vectorbt** — Phase 6, research sidecar only for parameter sweeps. Findings re-verified
event-driven before any strategy is promoted.

---

## 7. Dependencies actually installed in Phase 1

Every one, with its justification. Your §6: no dependency without a clear purpose.

| Package | Purpose | Layer |
|---|---|---|
| `pandas` | Market-data loading, validation, resampling, analysis | Data, features, reporting |
| `numpy` | Numerical arrays (transitive, and TA-Lib's interface) | Features |
| `TA-Lib` | Standard technical indicators | Features |
| `pydantic` + `pydantic-settings` | Typed validated config and boundary contracts | Config, adapters |
| `structlog` | Structured, queryable JSON logs | Observability |
| `sqlalchemy` + `alembic` | Ledger schema and migrations | Storage |
| `psycopg` | Postgres driver | Storage |
| `exchange-calendars` | NSE/BSE/NYSE sessions, holidays, half-days | Data |
| `typer` + `rich` | CLI and readable terminal output | Interface |
| `pytest`, `hypothesis`, `ruff`, `mypy` | Tests, property tests, lint, types | Dev only |

**Not installed:** backtrader (pending your decision), vectorbt, FastAPI, scikit-learn, PyTorch,
Streamlit, Redis, Celery — each has a phase, and none is here yet.

---

## 8. How this maps to your layer separation (§5)

Your four-layer separation is already the project structure, with one addition — a **domain
core** beneath all four, holding the money, instrument and position types that every layer
shares but none owns.

| Your layer | Package | Pandas? | Decimal? |
|---|---|---|---|
| **Data** — download → validate → clean → normalise → store → retrieve | `trading/data/` | ✅ yes | on exit |
| **Indicator / Feature** — clean data → indicators → features | `trading/features/` | ✅ yes | no (float is correct here) |
| **Strategy** — features → rules → signal | `trading/strategies/` | ❌ no | ✅ yes |
| **Backtesting** — signal → orders → execution → portfolio → performance | `trading/engine/`, `trading/backtest/` | reporting only | ✅ yes |
| *(added)* **Domain core** — Money, Instrument, Position, Order, RiskDecision | `trading/core/` | ❌ never | ✅ always |

---

## Sources

- [TA-Lib on PyPI](https://pypi.org/project/TA-Lib/) · [ta-lib-python releases](https://github.com/ta-lib/ta-lib-python/releases) · [install docs](https://ta-lib.github.io/ta-lib-python/install.html) — v0.7.1, wheel availability
- [backtrader on GitHub](https://github.com/mementum/backtrader) · [Is Backtrader dead?](https://community.backtrader.com/topic/3702/is-backtrader-dead) · [backtrader2 health](https://snyk.io/advisor/python/backtrader2) — maintenance status
- [pandas-ta on PyPI](https://pypi.org/project/pandas-ta/) · [pandas-ta-classic](https://pypi.org/project/pandas-ta-classic/) — fork status
- [Python backtesting landscape 2026](https://bullalert.ai/blog/best-python-backtest-engines-2026/) · [framework comparison](https://quanttradingtools.com/python-backtesting-frameworks/)

Version numbers, release dates and wheel tags were read from live PyPI metadata and from
installs performed in this session, not from these articles.

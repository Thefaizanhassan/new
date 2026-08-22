# Phase 0 — Discovery & Architecture Proposal

**Status:** Proposal. Decisions §25 resolved — see the [Addendum](phase-0-addendum-decisions.md). No implementation has begun.
**Date:** 2026-08-22

**Evidence legend:** `[FACT]` verifiable · `[ASSUMPTION]` unverified · `[HYPOTHESIS]` untested
· `[BACKTEST]` simulation only · `[PAPER]` simulated live · `[LIVE]` real money

---

## 1. My understanding of what you are building

You want a **personal quantitative research and trading platform** — a system you own, run
locally, and understand — that takes an idea from "I wonder if this works" to "this is
running with real money" through a pipeline that refuses to skip steps.

Three things I want to name explicitly, because they shape every decision below.

**This is not a trading bot.** A bot is the last 5% of this project. The other 95% is the
apparatus that tells you whether the bot deserves to exist. Writing code that buys when a
50-day moving average crosses a 200-day moving average takes an afternoon. Knowing whether
that rule has a real edge — or whether you fitted it to noise, or forgot that your broker
charges you, or accidentally used tomorrow's price to make today's decision — is the actual
engineering problem. I am going to keep steering effort toward that.

**You are three different users of this system, and they want opposite things.**

| Role | Wants | Optimises for |
|---|---|---|
| **Researcher** (you, exploring) | Fast iteration, loose constraints, many experiments | Speed, flexibility |
| **Operator** (you, trading) | Nothing surprising ever happens | Safety, predictability |
| **Auditor** (you, at 11pm asking "why did it do that?") | Complete history, reconstructable reasoning | Traceability |

Most retail trading projects serve only the Researcher, then get promoted to live trading
and fail as the Operator. The architecture below deliberately separates these concerns —
research code is allowed to be fast and loose, production code is not, and there is a hard
boundary between them.

**The honest base rate.** `[FACT]` Most retail algorithmic strategies do not beat a simple
buy-and-hold benchmark after transaction costs and taxes. This is not pessimism, it is the
starting prior, and the system should be built to detect that outcome quickly and cheaply
rather than to hide it. A platform that spends six months telling you "no durable edge here"
while risking nothing has done its job. I will not, at any point, tell you a strategy works
because a backtest looked good.

### What "AI-powered" should and should not mean here

You asked for an AI platform, and I want to set the expectation precisely, because this is
where most of the money in this space gets wasted.

`[FACT]` Large language models cannot forecast price direction from price history. They have
no access to order flow, their outputs are non-deterministic, and — critically — for any
historical date you ask about, the answer is contaminated: the model was trained on text
describing what actually happened next. A backtest of an LLM "predicting" 2023 is not a
backtest, it is the model remembering.

That does not make AI useless here. It relocates it. AI in this system earns its place in
**research, explanation, and classification** — not prediction of returns, and never in the
path between a signal and an order. Section 13 lays out exactly which jobs AI gets, ranked by
how much value each actually delivers.

---

## 2. Terminology you need — explained properly

You said not to assume you know these. I won't. This section covers the terms that
**drive architectural decisions**; the [full glossary](glossary.md) covers the rest.

### 2.1 The market plumbing

**Bar / OHLCV.** Market data is usually delivered as *bars*: a summary of all trading during
a fixed window. Each bar has an **O**pen (first trade price), **H**igh, **L**ow, **C**lose
(last trade price), and **V**olume (shares traded). A "daily bar" summarises one trading day;
a "1-minute bar" summarises one minute.

*Why it matters architecturally:* a bar is a **lossy summary**. Within a single daily bar the
price may have visited the high and the low in either order. A backtest that assumes it could
have bought at the low and sold at the high of the same bar is fiction. Our fill model must
never assume favourable intra-bar ordering.

**Bid, ask, and spread.** At any moment there is a highest price someone will *buy* at (the
**bid**) and a lowest price someone will *sell* at (the **ask**). The gap is the **spread**.
If AAPL is bid $189.50 / ask $189.52, you buy at $189.52 and sell at $189.50 — you lose 2
cents instantly on a round trip.

*Why it matters:* the "price" in your OHLCV data is usually the last *traded* price, which is
somewhere between bid and ask. Every backtest that fills at the closing price is
systematically optimistic by roughly half the spread per side. For liquid stocks that's
negligible. For a small-cap with a 1% spread and a strategy trading 200 times a year, that
alone is −200% of gross return. **The spread is not a rounding error; it is often the whole
edge.**

**Slippage.** The difference between the price you expected and the price you got. Sources:
the spread, price movement between decision and execution, and **market impact** — your own
order pushing the price against you when it is large relative to available liquidity.

**Liquidity.** How much you can trade without moving the price. Usually proxied by **ADV**
(average daily volume). Rule of thumb `[ASSUMPTION]`: staying under ~1% of ADV keeps impact
modest for retail size. Our backtester will cap fills at a configurable share of bar volume
so that "buy $2M of a stock that trades $500k/day" cannot silently succeed in simulation.

### 2.2 The four biases that make backtests lie

These are the reason this project has an architecture rather than a script. Each one makes a
worthless strategy look excellent, and each is prevented **structurally** in the design below.

**Look-ahead bias** — using information that was not available at decision time. The classic
version: you compute a signal from a daily bar's *close*, then execute at that same bar's
close. In reality, at the moment the close is known, the market is shut. Subtler versions:
using a value that was later revised, or a corporate action applied retroactively.

> *Structural prevention:* strategies never receive a dataframe. They receive a
> `StrategyContext` that can only expose data with a timestamp `<= ctx.now`. Asking for
> future data is not "discouraged" — it is impossible, because the object does not hold it.

**Survivorship bias** — testing on today's list of companies. If you backtest "buy the S&P
500" using today's members, you are buying companies *selected for having survived*. Enron,
Lehman and Wirecard are not in your universe, so your simulated portfolio never went to zero.
`[FACT]` This inflates historical equity returns by roughly 1–4% annually depending on
universe and period.

> *Structural prevention:* the universe is resolved from **point-in-time index membership**
> (what was in the index on that date), including delisted symbols. If we cannot source that,
> we restrict to a fixed hand-picked universe and document the limitation, rather than
> pretending.

**Data leakage** — training information crossing into your test set. In machine learning on
time series this is endemic: normalising features using statistics from the whole dataset,
or using overlapping labels, or standard k-fold cross-validation (which trains on the future
to predict the past).

**Overfitting** — finding a pattern in noise. Given enough parameter combinations, *any*
random series yields a beautiful equity curve. `[FACT]` If you test 500 strategy variants at
a 5% significance level, ~25 will look "significant" purely by chance. This is the single
biggest reason retail algo trading fails, and it is not solved by more data or better models
— it is solved by counting your attempts and discounting accordingly (Section 12).

### 2.3 How performance is judged

**Return, CAGR, and why "total return" is a bad headline.** *Total return* is (end − start)/
start. **CAGR** (Compound Annual Growth Rate) converts that to a per-year rate, which is the
only way to compare strategies tested over different periods. Conceptually:
`CAGR = (end/start)^(1/years) − 1`.

**Drawdown and Maximum Drawdown (MDD).** A *drawdown* is how far you are below your previous
peak equity. **MDD** is the worst such fall in the period.

*Why it matters more than return:* MDD is the number that determines whether you can actually
run the strategy. A strategy with 25% CAGR and 60% MDD is, for almost everyone, unrunnable —
you will abandon it at the bottom, converting a paper drawdown into a realised loss. MDD is
also asymmetric: a 50% drawdown requires a 100% gain to recover. **Design your risk limits
around drawdown, not return.**

**Volatility.** The standard deviation of returns, annualised. It measures dispersion, not
danger — it treats a 5% gain and a 5% loss identically. That limitation is why the ratios
below exist.

**Sharpe ratio.**
- *What it means:* excess return per unit of volatility. Conceptually,
  `(strategy return − risk-free rate) / volatility of returns`, annualised.
- *Why traders use it:* it makes strategies with different risk levels comparable. Doubling
  your leverage doubles return and doubles volatility, leaving Sharpe unchanged — so Sharpe
  measures the quality of the edge rather than how hard you pressed on it.
- *How to read it `[ASSUMPTION]` (rough retail conventions):* <0.5 weak · 0.5–1.0 plausible ·
  1.0–2.0 good · >2.0 on a retail backtest — **suspect a bug or overfitting before believing
  it.**
- *Limitations, and they are serious:* it penalises upside volatility as much as downside; it
  assumes roughly normally-distributed returns, which market returns are not (crashes are far
  more common than a bell curve predicts); it is trivially inflated by strategies that make
  small consistent gains and rare catastrophic losses (selling options, martingale sizing);
  and a Sharpe computed on 60 trades has enormous error bars. **Never judge on Sharpe alone.**

**Sortino ratio.** Sharpe, but dividing only by *downside* deviation — volatility computed
from losing periods only. Fixes Sharpe's "punishes big wins" flaw. Higher is better; read it
alongside Sharpe, and be suspicious when they diverge wildly.

**Calmar ratio.** `CAGR / Maximum Drawdown`. Directly answers "how much return do I get per
unit of the pain that would make me quit?" `[ASSUMPTION]` Above 0.5 is respectable; above 1.0
is strong. Best of the three for judging *runnability*.

**Win rate, profit factor, expectancy — and why win rate is the least useful.**
- *Win rate* — % of trades profitable. **Actively misleading in isolation.** A strategy
  winning 95% of the time that loses 30× on the 5% is a catastrophe; trend-following
  typically wins 35–40% of trades and is highly profitable.
- *Profit factor* — gross profits / gross losses. Above 1.0 is profitable; `[ASSUMPTION]`
  1.5+ is decent, >3 on retail data invites suspicion.
- *Expectancy* — average profit per trade: `(win rate × avg win) − (loss rate × avg loss)`.
  **The number that actually matters**, because it tells you what one more trade is worth.
  Multiply by trade count for the whole picture. Expectancy must be computed **net of all
  costs** — a strategy with positive gross expectancy and negative net expectancy is the most
  common failure mode in retail algo trading.

### 2.4 Concepts that create architecture

**Corporate actions, splits, dividends, and "which price series?"** When a company does a 4:1
split, the share price quarters overnight. A raw price series shows a −75% crash; a strategy
would fire a signal on a non-event. The fix is *adjusted* prices, which restate history as if
the split had always applied. Dividends are similar: the price drops by the dividend on the
ex-date, and *total-return* adjusted series add it back.

*Why it matters:* you need **both** series, for different purposes. Signals and returns must
use adjusted prices (so history is continuous); order placement and position accounting must
use raw prices (because you buy real shares at real prices). Storing only one is a bug you
find months later. `[ASSUMPTION]` — and this is important — adjusted history *changes* every
time a new corporate action occurs, which means the "close of 2019-06-03" is a different
number today than it was last year. Backtests must therefore record which data version they
used, or they are not reproducible.

**Position sizing and exposure.** *Position sizing* is how much to buy — the decision that
dominates outcomes far more than entry timing. *Exposure* is how much of your capital is at
risk: 60% exposure means 60% of equity is in positions. *Gross* exposure sums absolute
positions (longs + shorts); *net* subtracts (longs − shorts). A market-neutral book can be
150% gross and 0% net.

*Architecturally:* sizing must **not** live inside strategies. If three strategies each
independently size positions, they can unknowingly stack to 300% exposure in correlated
names. Strategies express *what they want*; a portfolio layer decides *how much*.

**Reconciliation.** Periodically comparing your system's belief about positions and cash
against the broker's record — the broker being the source of truth, always. Divergence means
a bug, a missed fill, or a manual trade you forgot. `[FACT]` Unreconciled state is how
automated systems quietly double a position.

**Paper trading.** Running the full system against live market data with simulated money.
Its purpose is *not* to prove profitability — the sample is far too short. Its purpose is to
find the ~40 integration bugs, timezone errors, and API surprises that no backtest can
surface, and to measure **backtest-vs-live divergence**, which is the real signal.

---

## 3. Missing requirements I identified

You asked me to find what you hadn't mentioned. These are ordered by how much damage they do
if discovered late. The first two are the ones I'd genuinely lose sleep over.

### 3.1 🔴 Jurisdiction and regulation — an unanswered question that changes the architecture

You never said which country you trade in or which market. This is the largest open variable
in the entire design, because the two most likely answers produce **materially different
systems**.

`[FACT]` **If you trade Indian markets:** SEBI's retail algorithmic trading framework became
fully mandatory for all brokers on **1 April 2026** — it is live now, not upcoming. The
relevant provisions for someone building their own system:

- Self-developed algos trading **your own account** are permitted **without exchange
  registration**, provided you stay under the order-rate threshold — set at **10 orders per
  second** per exchange within any calendar second. Above that, the algo must be registered
  through your broker and receives an exchange-assigned **Algo-ID**.
- Every API-originated order must come from a **static IP whitelisted with your broker**.
  A home broadband connection with a dynamic IP will not work; a static IP service costs
  roughly ₹1,500–6,000/year.
- Orders placed by algos must carry the exchange-assigned identifier for traceability.
- API sessions must **log out daily** before each trading day — persistent sessions are out.
- Retail algos are expected to be **hosted on Indian servers**, and 2FA plus audit-trail
  requirements apply.

The practical consequence: **"runs on my Mac" is viable for research and paper trading, but
likely not for live API trading in India.** Live execution would need a small always-on
Indian host (a Mumbai-region VM) with a static IP. That is not a problem — it's a €5–15/month
line item — but it must be in the design from the start, because it changes deployment,
secrets handling, and the data-locality story.

`[FACT]` **If you trade US markets:** personal algorithmic trading is essentially
unregulated, but two rules bite:
- **Pattern Day Trader (PDT):** more than 3 day trades in 5 business days in a margin account
  requires a **$25,000 minimum equity**. Below that, your account gets restricted. This
  effectively rules out intraday strategies for smaller accounts — a hard constraint on
  strategy selection, not a technicality.
- **Wash sale rule:** losses are disallowed if you repurchase a "substantially identical"
  security within 30 days. High-turnover strategies can generate large *taxable* gains
  alongside disallowed losses — a real cash-flow risk.

**My recommendation:** the core platform is designed jurisdiction-agnostic — all
market-specific behaviour lives behind the broker adapter, the calendar, and a
`ComplianceProfile` config object. But I need your answer before Phase 9 (broker integration),
and preferably before Phase 2 (market data), because data providers differ sharply by market.

### 3.2 🔴 Your Mac is not a server

This is the operational gap that most personal trading projects discover the expensive way.
A MacBook sleeps when you close it, drops Wi-Fi when you walk between rooms, reboots for OS
updates without asking, and throttles background processes on battery.

The failure that matters isn't "the strategy stopped" — it's **"the strategy stopped holding
a half-filled position with a working stop order the system no longer knows about."**

Required in the design, from Phase 1:
- **Crash-safe state.** Trading state is persisted before it is acted on, never held only in
  memory. Restart must be able to answer "what was I doing?"
- **Startup reconciliation.** On every start, before doing anything else: query the broker,
  compare to local state, and if they disagree, **halt and alert** rather than proceed.
- **An `UNKNOWN` order state.** "I sent it and the connection died" is a real state that must
  block further orders for that symbol until resolved by query, never by assumption.
- **A defined deployment target for live.** `[ASSUMPTION]` A small always-on Linux VM in the
  relevant region, with the Mac as the research and control station. Not needed for Phases
  1–8; needs to be true before Phase 11.

### 3.3 🟠 Point-in-time data ("what did I know, and when did I know it?")

Market data is not immutable. Prices are revised, fundamentals are restated (often months
later), index membership changes, and adjusted prices are rewritten by every new corporate
action. A backtest run against today's snapshot silently uses knowledge from the future.

The fix is a **bitemporal** store: every record carries both the time it *refers to*
(`event_time`) and the time we *learned it* (`ingested_at`). A backtest declares an `as_of`
date and sees only what was knowable then. This is unglamorous and it is the difference
between a reproducible research platform and a very expensive random number generator.

### 3.4 🟠 Taxes, and why they change strategy selection

You listed taxes as "where relevant." They are more than relevant — they are frequently the
deciding factor between two strategies with identical gross returns.

`[FACT]` Tax treatment is a function of **holding period and turnover**, which means it
attacks high-frequency strategies hardest — exactly the strategies that look best in a
naive backtest. In India, intraday equity trading is typically treated as *speculative
business income* taxed at slab rates, plus STT, stamp duty and exchange charges on every
trade; delivery-based gains attract STCG/LTCG at different rates by holding period. In the
US, short-term capital gains are taxed as ordinary income while long-term (>1 year) gains get
a preferential rate.

The consequence for design: **the backtester reports gross *and* net-of-cost equity curves,
and the cost model includes a configurable, jurisdiction-specific tax and levy component.**
Not a tax-filing engine — that's out of scope — but enough that a strategy is never selected
on pre-tax numbers. `[ASSUMPTION]` This will kill more strategies than any other single check.

### 3.5 🟠 Idempotency and exactly-once order submission

You listed "duplicate orders" as a risk. I want to name the precise mechanism, because it is
the most common way automated systems lose real money:

> You send an order. The network times out. You do not know whether the broker received it.
> You retry. The broker had received it. You now hold twice the intended position.

The only robust defence is **client-generated order IDs**. Every order carries a
`client_order_id` you create before sending. On any uncertainty, you **query by that ID**
rather than resending. A retry is therefore *never* a blind resend — it is always
query-then-decide. `[FACT]` Every serious broker API supports this; most tutorials ignore it.

### 3.6 🟠 Time, clocks, and bar-labelling conventions

Ambiguous timestamps cause silent, hard-to-find errors. Decisions to fix once, in Phase 1,
and never revisit:
- All timestamps stored in **UTC**, with the exchange and its timezone stored alongside.
- Bars labelled by their **close** time (a bar labelled 09:31 covers 09:30:00–09:30:59.999).
  The alternative convention is equally valid; mixing them is not.
- **Market calendars** from the `exchange_calendars` library — half-days, holidays, and
  session boundaries are data, not `if weekday < 5`.
- A single injected **`Clock`** abstraction. Nothing in the codebase calls `datetime.now()`
  directly. This is what makes the backtest and live paths identical.

### 3.7 🟠 Multiple testing — counting your attempts

If you test 500 strategy variants and report the best one, its backtest Sharpe is an estimate
of *the maximum of 500 random draws*, not of that strategy's edge. `[FACT]` The correction —
the **Deflated Sharpe Ratio** — requires knowing the number of trials, which means the system
must **count and persist every experiment you run**, including the failures you'd rather
forget. If experiment tracking is optional, this correction becomes impossible. It is
therefore mandatory infrastructure, not a nice-to-have.

### 3.8 🟡 A kill switch that works when the application is broken

An in-app kill switch fails exactly when you need it, because the thing that's wrong is the
app. Layered design:
1. **In-app** — a flag in the risk engine (fastest, least reliable).
2. **Filesystem** — a `KILL` sentinel file checked at the lowest level of the order path,
   before any strategy logic. Works even if the strategy layer is looping.
3. **Broker-side** — API key revocation and, where supported, broker-level trading disable.
   The only one that works if the process is unresponsive.
4. **A documented manual runbook** — the phone number, the web login, the steps. Written
   before you need it, at 3am, in a panic.

### 3.9 🟡 Strategy capacity and the tail of small numbers

*Capacity* is how much money a strategy can absorb before its own trading destroys the edge.
`[ASSUMPTION]` At retail size this is usually only binding for illiquid instruments — but
it is a genuine trap in one specific case: backtests on small-caps or low-volume names show
spectacular returns *because* nobody could actually trade them at those prices. The
volume-participation cap in the fill model is the defence.

### 3.10 🟡 Observing what did *not* happen

Rejected signals are as informative as executed trades. If the risk engine blocked 40 trades
last month, you need to know which rule fired, how often, and what the counterfactual P&L
would have been — otherwise you cannot tell a well-calibrated risk limit from one that is
quietly strangling the strategy. **Every rejection is a logged, queryable, first-class event.**

### 3.11 🟡 Strategy decay monitoring

A validated strategy is not permanently valid. Markets change regime; edges get arbitraged
away. Once running, the system must continuously compare **live/paper performance against the
backtest distribution** and auto-pause on statistically significant divergence — rather than
waiting for you to notice a bad quarter.

### 3.12 🟡 Backup, restore, and secrets escrow

An untested backup is not a backup. `[ASSUMPTION]` Required: nightly encrypted backup of the
operational database off-machine, a **tested** restore procedure, and a documented answer to
"my Mac was stolen" — including how you regain broker access if your Keychain is gone. The
recovery path for credentials matters as much as their protection.

### 3.13 🟡 Data quality gates

Bad ticks, zero-volume bars, stale quotes, missing sessions and unapplied splits all generate
phantom signals. Data enters through a **validation gate** that checks OHLC internal
consistency (`high >= max(open, close)`, `low <= min(open, close)`, prices > 0), session
alignment against the calendar, gap detection, and duplicate timestamps. Failures are
**quarantined and alerted**, never silently dropped — silent dropping creates gaps, and gaps
create look-ahead when a strategy skips a bar it should have traded.

### 3.14 🟡 Legal and licensing boundaries

`[FACT]` Market data licences generally forbid redistribution — fine for personal use, but it
constrains any future sharing or cloud-hosting of raw data. Scraping TradingView violates
their Terms of Service and is excluded from this design (Section 8). If you ever advise others
using this system, registration requirements may apply (SEBI RA registration in India, RIA in
the US) — out of scope, but worth knowing the line exists.

### 3.15 🟡 The psychological risk

Underrated and genuinely dangerous: **manual intervention destroys the statistics.** If you
override the system during a drawdown, you no longer have the strategy you validated — you
have an untested hybrid, and your backtest no longer describes anything. The system should
make intervention *possible* (it is your money) but **logged, deliberate, and visible in
performance attribution**, so you can see what your overrides actually cost you.

---

## 4. Recommended architecture

### 4.1 The shape: a modular monolith, not microservices

**Recommendation: a single Python application with strictly enforced internal module
boundaries, deployed as one or two processes.**

| Option | Verdict | Reasoning |
|---|---|---|
| Single script / notebook | ❌ | Cannot be tested, versioned, or trusted with money |
| **Modular monolith** | ✅ **Recommended** | One transaction boundary, trivial debugging, no network partitions between your own components, deployable to one small VM. Boundaries enforced by import rules so components *could* be split later. |
| Microservices | ❌ (for now) | You are one person. Distributed transactions, service discovery, and inter-service failure modes would add substantial operational risk in exchange for scaling you do not need. Premature. |
| Event-streaming (Kafka etc.) | ❌ (for now) | Solves a throughput problem you will not have. Postgres `LISTEN/NOTIFY` covers your messaging needs at zero operational cost. |

Internally the design follows **ports and adapters** (hexagonal): a pure domain core with no
I/O, surrounded by adapters that talk to the outside world. This is what makes the same
strategy code run against a historical file, a paper broker, and a live broker.

### 4.2 The single most important decision: one core, three drivers

Everything else follows from this. The difference between backtesting, paper trading and live
trading is **only which clock and which broker are plugged in.** Nothing else changes — not
the strategy, not the risk engine, not the portfolio accounting.

```
                     ┌───────────────────────────────────┐
                     │   Strategy + Risk + Portfolio     │   written once,
                     │        (pure domain core)         │   never forked
                     └─────────────────┬─────────────────┘
                                       │  identical interface
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
   ┌─────┴──────┐                ┌─────┴──────┐                ┌─────┴──────┐
   │  BACKTEST  │                │   PAPER    │                │    LIVE    │
   ├────────────┤                ├────────────┤                ├────────────┤
   │ SimClock   │                │ RealClock  │                │ RealClock  │
   │ FileFeed   │                │ LiveFeed   │                │ LiveFeed   │
   │ SimBroker  │                │PaperBroker │                │ LiveBroker │
   │ (fill model)│               │(broker sim)│                │  (real $)  │
   └────────────┘                └────────────┘                └────────────┘
     minutes                       real time                     real time
     no money                      no money                      REAL MONEY
```

`[FACT]` The overwhelming majority of retail trading projects maintain separate backtest and
live code paths, and they diverge. When they diverge, the strategy you validated is not the
strategy you are running, and you find out with money on the table. This design makes that
divergence structurally impossible for the core logic — the only untested surface is the
adapter layer, which is exactly where paper trading is aimed.

### 4.3 The decision pipeline

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ① MARKET DATA                                                           │
 │     Provider Adapter → Normaliser → Validation Gate → Bar Store          │
 │     (Tiingo/Polygon/broker)  (UTC,      (OHLC sanity,  (Parquet +        │
 │                               close-      calendar,      DuckDB,          │
 │                               labelled)   gaps)          bitemporal)      │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ② FEATURE ENGINE      point-in-time safe · cached · versioned            │
 │     SMA, RSI, ATR, realised vol, z-scores, regime labels …                │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ③ STRATEGY          on_bar(ctx) → list[Intent]                           │
 │     ctx exposes ONLY data with timestamp <= ctx.now  ← look-ahead killed  │
 │     ┌────────────────────────────────────────────────────────────┐        │
 │     │ optional: AI/ML advisory input (regime label, P(win))       │        │
 │     │ — an input to the strategy, never a route to the broker     │        │
 │     └────────────────────────────────────────────────────────────┘        │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
                    ╔═════════════════════════════════════╗
                    ║   INTENT  "I want to be long AAPL   ║
                    ║            at 4% of equity"         ║
                    ║   — a wish, not an order            ║
                    ╚══════════════════┬══════════════════╝
                                       ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ④ PORTFOLIO SIZER    combines intents from ALL strategies                │
 │     nets opposing positions · applies capital allocation · → target book  │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ╔══════════════════════════════════════════════════════════════════════════╗
 ║  ⑤ RISK ENGINE        deterministic · synchronous · FAILS CLOSED         ║
 ║     position cap · exposure cap · leverage · daily loss · drawdown ·      ║
 ║     open-position count · correlation · liquidity · stale data ·         ║
 ║     duplicate check · session check · kill switch                        ║
 ║                                                                          ║
 ║        ┌──────────────────┐          ┌───────────────────────────┐       ║
 ║        │ RiskDecision.OK  │          │ RiskDecision.REJECTED     │       ║
 ║        │  (signed token)  │          │  rule_id · inputs ·       │       ║
 ║        └────────┬─────────┘          │  threshold · reason       │       ║
 ║                 │                    └───────────┬───────────────┘       ║
 ╚═════════════════│════════════════════════════════│═══════════════════════╝
                   ▼                                ▼
 ┌───────────────────────────────┐      ┌──────────────────────────────────┐
 │  ⑥ ORDER FACTORY              │      │  logged as a first-class event   │
 │  CANNOT construct an Order    │      │  — visible in the dashboard,     │
 │  without a valid RiskDecision │      │    counted, and reviewable       │
 └───────────────┬───────────────┘      └──────────────────────────────────┘
                 ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⑦ ORDER ROUTER    client_order_id (idempotency) · dedupe · state machine │
 │     PENDING_NEW → NEW → PARTIALLY_FILLED → FILLED│CANCELED│REJECTED       │
 │                        └→ UNKNOWN  (blocks symbol until resolved)         │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⑧ BROKER ADAPTER          SimBroker │ PaperBroker │ LiveBroker           │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⑨ FILL EVENTS → POSITION LEDGER → PORTFOLIO STATE                       │
 │     immutable append-only event log; state is DERIVED, never mutated      │
 └────────────────────────────────────┬─────────────────────────────────────┘
                                      ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  ⑩ RECONCILER   broker is the source of truth · divergence → HALT + ALERT │
 └──────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Three design decisions worth defending

**(a) `Intent` is separate from `Order`.** Your draft had `generate_signal()` returning a
signal. I want strategies to emit a *target exposure* ("I want to be 4% long AAPL"), not an
action ("buy 21 shares"). Four consequences, all good:

- **Idempotent.** Re-running the same bar produces the same target, so a missed message or a
  restart self-heals. Delta-based systems ("buy 21 shares") drift permanently after one
  missed message. This is the same reason Kubernetes is declarative.
- **Composable.** Two strategies wanting +4% and −3% of the same symbol net to +1% at the
  portfolio layer, with one order and one commission — instead of two orders that pay the
  spread twice and briefly hold contradictory positions.
- **Sizing lives in one place.** Strategies cannot individually mis-size and stack exposure.
- **Restart-safe.** On startup you compare current book to target book and converge. There is
  no "what orders was I in the middle of sending?" question.

**(b) The risk engine cannot be bypassed — enforced by types.** A convention that says "always
call the risk engine" gets forgotten at 1am during a hotfix. Instead, `Order.__init__` is
private; orders can only be created via a factory that requires a `RiskDecision` object,
which only the risk engine can mint. Bypassing risk requires *deliberately editing the domain
core*, which shows up in a diff and in a code review. **The safety property is a compile-time
property, not a runtime hope.**

**(c) Event-sourced state.** Portfolio state is *derived* from an immutable, append-only
event log rather than stored and mutated. This buys three things at once, which is why it's
worth the modest extra complexity:
- The "why did it trade?" audit chain is free — it *is* the log.
- Crash recovery is free — replay the log.
- Reproducibility is free — the log is the record of what happened, and it cannot be
  retroactively edited without breaking a hash chain.

---

## 5. Recommended technology stack

Chosen, not defaulted. Where I rejected something you named, I say why.

### 5.1 Language and tooling

| Choice | Recommendation | Reasoning |
|---|---|---|
| **Python** | **3.12** | `[ASSUMPTION]` Best balance of modern typing (`type` statement, better generics) and universal numeric-stack support. 3.13 is fine if you prefer; avoid the bleeding edge — a scientific-stack wheel that isn't built yet costs you a day. |
| **Package/venv** | **`uv`** | Order-of-magnitude faster than pip/poetry, proper lockfile, single tool for venv + deps + Python versions. Replaces poetry, pip-tools, pyenv. |
| **Validation/config** | **Pydantic v2** + `pydantic-settings` | Runtime validation at every boundary (config, API payloads, broker responses, LLM output). Strategy parameters become typed, validated, serialisable objects — which is what makes strategies hashable and versionable. |
| **Lint/format** | **Ruff** | Replaces flake8 + isort + black + a dozen plugins, ~100× faster. One tool, one config. |
| **Types** | **mypy --strict on `core/`**, relaxed elsewhere | Strict typing where money is calculated; pragmatic in research code. |
| **Testing** | **pytest** + **Hypothesis** | Hypothesis (property-based testing) is unusually well-suited here: "for *any* sequence of fills, cash + position value must equal equity" catches accounting bugs that example-based tests miss entirely. |
| **Logging** | **structlog** → JSON | Queryable logs. `logger.info("order_rejected", rule="max_exposure", value=0.72, limit=0.60)` is greppable and analysable; a formatted string is not. |
| **Money** | **`decimal.Decimal`** for cash/prices/quantities | `[FACT]` Floats cannot represent 0.1 exactly. Float accumulation over thousands of fills produces position drift and reconciliation failures. `float` for statistics, `Decimal` for anything that touches the ledger. |

### 5.2 Numerics and data

**pandas + numpy** as the primary research stack — ecosystem gravity wins; every finance
example, every library, every Stack Overflow answer assumes it. **DuckDB** for analytical
queries over Parquet (it queries files directly, no import step). **Polars** deferred — 
genuinely faster, but the ecosystem tax isn't worth it until pandas is measurably the
bottleneck. **scipy** and **statsmodels** as needed for statistics. **scikit-learn** from
Phase 7. **PyTorch: not recommended** — see Section 13; with ~5,000 daily bars per symbol,
deep learning will memorise, not learn.

### 5.3 Backtesting engine — the significant decision

| Option | Assessment | Verdict |
|---|---|---|
| **backtrader** | `[FACT]` In long-term maintenance mode since 2023; the author has stated no major features are coming. | ❌ Not for a new 2026 project |
| **vectorbt** | Extremely fast vectorised sweeps — thousands of parameter combinations in seconds. But realistic order simulation is hard, and it cannot drive a live broker. | ✅ **As a research sidecar only** |
| **NautilusTrader** | `[FACT]` Actively developed, Rust core, genuinely excellent execution realism, backtest/live parity by design — the professional choice. But a steep learning curve and a large framework commitment. | ⏳ **Revisit at Phase 9+** |
| **Custom event-driven core** | ~1,200–1,800 lines. You understand every line. Shares code with live by construction. | ✅ **Recommended for MVP** |

**Recommendation: build a custom event-driven core, use vectorbt for parameter sweeps.**

My reasoning, stated honestly because this is the choice I'd most expect you to challenge:
the shared-code-path property in §4.2 is non-negotiable, and only two options deliver it —
building it, or adopting NautilusTrader wholesale. Given that you are simultaneously learning
the domain, I think building it is the better trade. The backtester is where every subtle
bias hides; writing it yourself is how you learn where they hide, and 1,500 lines of code you
fully understand is safer with your money than 50,000 lines you don't. Nautilus's abstractions
are excellent, but learning them is learning *Nautilus*, not learning *trading*.

**The explicit trigger to switch:** if you move to intraday strategies where fill modelling
against order-book data becomes the accuracy bottleneck, migrate to NautilusTrader. The
adapter architecture means that's a contained change, not a rewrite.

### 5.4 Application framework

**FastAPI** from Phase 6 — needed for webhooks, the control API, and health endpoints; async,
typed, Pydantic-native. **SQLAlchemy 2.0** (typed ORM) + **Alembic** (migrations — from day
one, because "I'll add migrations later" means hand-editing production schema at some point).
**APScheduler** for scheduling, with the firm rule that **every scheduled job must be
idempotent**, so a double-fire is harmless. Celery/Temporal rejected as premature.

**asyncio** for the live trading loop (I/O-bound: waiting on websockets and HTTP), plain
synchronous code for backtests (CPU-bound), with the domain core written **sync and pure** so
it works under both. `[ASSUMPTION]` This keeps async complexity confined to the adapter layer,
where it belongs.

### 5.5 User interface

| Option | Assessment | Verdict |
|---|---|---|
| **Streamlit** | Pure Python, fastest path to a useful research dashboard, excellent charting. Weak for real-time control surfaces; re-runs the whole script on interaction. | ✅ **Phases 4–8** |
| **NiceGUI** | Pure Python with real reactivity and websockets. Smaller ecosystem. | ⏳ Good middle option if Streamlit chafes |
| **React + FastAPI** | Full control, real-time, production-grade. Substantially more work; a second language and toolchain. | ⏳ **Only if you outgrow the above** |

**Recommendation: Streamlit for the MVP**, with one architectural rule that matters more than
the framework choice:

> ⚠️ **The UI must never be the process that trades.** The trading engine is a separate
> long-running process. The dashboard is a *read-only view* over the database plus a
> *command sender* over the API. If Streamlit crashes, or you close the tab, or it re-runs
> the script on a widget click, **nothing about your positions changes.**

`[FACT]` Coupling the UI to the trading loop is a common and expensive beginner mistake —
a browser refresh should never be able to duplicate an order.

---

## 6. Database and storage architecture

### 6.1 Four workloads with genuinely different needs

| Workload | Pattern | Needs |
|---|---|---|
| Market data | Append-heavy, wide analytical scans, immutable | Columnar, compressed, cheap |
| Trading state | Small, transactional, correctness-critical | ACID, constraints, concurrent access |
| Event/audit log | Append-only, high write, occasional replay | Durable, ordered, tamper-evident |
| Research artifacts | Large blobs, write-once | Filesystem |

No single store is best at all four. Forcing them into one is the most common storage mistake.

### 6.2 Recommendation: a two-store split

**① DuckDB + Parquet — market data and research.**

`[FACT]` Parquet is columnar and compressed (typically 5–10× smaller than CSV); DuckDB
queries Parquet files directly with no import step, and is fast enough that "scan 15 years of
daily bars for 500 symbols" is sub-second on a laptop. Zero operational overhead — it is a
library, not a server. Files are trivially backed up and version-controlled by content hash.

Layout: `data/bars/{provider}/{asset_class}/{symbol}/{year}.parquet`, with a metadata
manifest in Postgres recording what exists, from when, from which provider, and at which
dataset version.

**② PostgreSQL — operational state, event log, and registries.**

| Alternative | Why not |
|---|---|
| **SQLite** | Genuinely tempting for zero-ops simplicity, and fine for Phases 1–4. But by Phase 6 you have three concurrent processes (engine, API, dashboard); SQLite's single-writer model, weak type affinity, and limited constraint support make it the wrong foundation for the *ledger*. Migrating later is real work. |
| **TimescaleDB** | Adds operational complexity for continuous aggregates and compression you don't need — DuckDB+Parquet beats it for our read pattern. Revisit only if you store tick data at scale. |
| **Supabase / cloud PG** | ❌ for the hot path. Your order state behind a network hop means a broker fill you cannot record during an outage. Excellent as an *encrypted backup target*. |
| **Redis** | Not needed. Postgres `LISTEN/NOTIFY` handles inter-process messaging at zero extra ops cost. Add Redis only if you measure a genuine hot-cache need. |

**Recommendation: Postgres from Phase 1, via Docker Compose.** The cost is one
`docker compose up`; the benefit is JSONB event payloads, real `timestamptz`, proper foreign
keys and check constraints (your accounting invariants become *database-enforced*, not
hoped-for), and painless concurrency later. SQLite's simplicity is a false economy here.

### 6.3 Data classification

| Data | Where | Retention | Backup |
|---|---|---|---|
| Historical bars | Local Parquet | Permanent | Weekly → encrypted cloud |
| Intraday/tick (later) | Local Parquet, partitioned | Rolling window, configurable | Selective |
| Orders, fills, positions | Postgres | **Permanent — never deleted** | Nightly, encrypted, off-machine |
| Event / audit log | Postgres, hash-chained | **Permanent — append-only** | Nightly |
| Backtest runs + manifests | Postgres (metadata) + Parquet (curves) | Permanent | Weekly |
| Feature cache | Local Parquet | Regenerable — safe to delete | None |
| AI/LLM call records | Postgres | Permanent | Nightly |
| Secrets | **macOS Keychain** — never the DB, never `.env` | — | Documented escrow |
| Logs | Local files, rotated | 90 days hot | Optional |

---

## 7. Market data architecture

### 7.1 Two interfaces, not one

Historical and real-time data have different semantics and must not share an interface —
conflating them is how look-ahead bias sneaks in through the back door.

```python
class HistoricalDataProvider(Protocol):
    def get_bars(symbols, start, end, timeframe, *, as_of: datetime) -> BarSet: ...
    def get_corporate_actions(symbol, start, end) -> list[CorporateAction]: ...

class RealtimeDataProvider(Protocol):
    async def stream_bars(symbols, timeframe) -> AsyncIterator[Bar]: ...
    def last_update_time() -> datetime:  # staleness detection — a risk input
```

Note `as_of` on the historical interface: **it is not optional.** A backtest declares the date
it is pretending to be, and the store returns only what was known by then.

### 7.2 The ingestion pipeline

```
Provider API ──► Normaliser ──► Validation Gate ──► Bitemporal Store ──► Feature Cache
                     │                │                    │
              UTC, close-labelled,   quarantine     event_time +
              symbol canonicalised,  + alert on     ingested_at
              Decimal prices         failure        (never overwrite)
```

**Validation gate rules** (fail → quarantine + alert, never silent drop):
`high >= max(open, close)` · `low <= min(open, close)` · all prices > 0 · volume >= 0 ·
no duplicate timestamps · session exists in the exchange calendar · gap detection against
expected sessions · price-change sanity (>50% move without a corporate action is flagged).

**Corporate actions.** Store **raw prices immutably**, plus a separate adjustment-factor
table. Adjusted series are *computed on read*, never written back. This means old backtests
stay reproducible even after a new split rewrites the adjusted history — and it gives you both
series (adjusted for signals, raw for order pricing) from one source of truth.

### 7.3 Provider recommendations

Deferred pending your market (§3.1), but the shortlist `[FACT]`, prices as of research date:

**US equities:** *Tiingo* (~$10/mo, 30+ years EOD + fundamentals — **best starting point for
daily strategies**) · *EODHD* (~€20/mo, 150k+ tickers globally, strong bulk download) ·
*Polygon* (~$199/mo Advanced, for intraday/tick) · *Databento* (institutional, metered — model
the cost before committing) · *Alpaca* (data bundled with the brokerage account).

**Indian equities:** the practical route is your broker's historical API (Kite Connect, Dhan,
Upstox), supplemented by a paid vendor. `[ASSUMPTION]` Long, clean, survivorship-free Indian
history is harder and costlier to obtain than US equivalents — budget for this being the
single most annoying part of an India-focused build.

**On `yfinance`:** fine for a Phase-2 prototype, **unacceptable for anything you will trust.**
Silent adjustment changes, no SLA, no guarantee of point-in-time correctness, and ToS
ambiguity around programmatic use. Use it to get the pipeline running, then replace it before
any strategy is validated.

**Recommendation: start with one EOD provider and daily bars.** Do not buy intraday data until
a daily strategy has survived validation. `[ASSUMPTION]` Intraday data is ~20× more expensive,
~400× larger, and dramatically harder to backtest honestly — and if you cannot find an edge in
daily data, minute data will mostly give you a faster way to lose to transaction costs.

---

## 8. TradingView integration

You asked me to investigate this properly rather than assume. Here is what I found and what
I'd recommend.

### 8.1 The four possible roles, assessed

**① Charting and visual research only — ✅ recommended now.**
You use TradingView as a human, to look at charts and form hypotheses. Zero integration, zero
risk, zero cost beyond your existing subscription. This is genuinely where most of its value
is for you.

**② Alerts → webhook → your system, as a *signal source* — ⏳ possible later.**
`[FACT]` TradingView webhook alerts require a **paid plan** (webhooks are not available on the
free tier, which is limited to a handful of price alerts and no technical alerts), with 2FA
enabled. Active-alert counts are capped by tier. Alert messages should be valid JSON —
TradingView then sends `application/json`; otherwise it sends plain text, which most endpoints
reject.

**③ Pine Script as your strategy language — ❌ rejected.**
Pine is proprietary and sandboxed: you cannot backtest it against *your* cost model, cannot
unit test it, cannot version it alongside your Python, and cannot reuse its logic in your risk
engine. TradingView's own strategy tester has well-known fill-model optimism. Adopting Pine
means owning two divergent implementations of every strategy — precisely the failure mode
§4.2 exists to prevent.

**④ Scraping TradingView — ❌ rejected.**
ToS violation, brittle against UI changes, and building your capital allocation on an
adversarial data source. Excluded from this design.

### 8.2 The structural problem with webhooks, and its fix

`[FACT]` TradingView alerts are **fire-and-forget HTTP POSTs**. There is no delivery
guarantee, no ordering guarantee, no replay, and no acknowledgement. If your endpoint is down,
your Wi-Fi drops, or the request times out, **that signal is simply gone forever.**

This makes one design mistake catastrophic and one design choice safe:

> ❌ **Never** let a webhook carry a *delta*: `{"action": "buy", "qty": 100}`. Miss one
> message and your position is permanently wrong, silently, with no error anywhere.
>
> ✅ **Always** carry *target state*: `{"symbol": "AAPL", "target_weight": 0.04}`. A missed
> message self-heals on the next one, because the system converges toward a declared target
> rather than accumulating instructions.

This is the same declarative/idempotent principle as §4.4(a), and it is why the `Intent`
abstraction pays for itself twice.

### 8.3 If you do adopt webhooks (Phase 9+), the required controls

The endpoint is **hostile input** — the URL will be scanned, and anyone who learns it can post
to it. Non-negotiable:

- **HMAC signature** over the payload with a shared secret; constant-time comparison.
- **IP allowlist** of TradingView's published egress ranges (defence in depth, not primary —
  IPs change).
- **Replay protection** — nonce plus a tight timestamp window; reject anything stale.
- **Strict schema validation** (Pydantic); reject-and-log on any deviation, never coerce.
- **Rate limiting** per source.
- **The signal enters as an `Intent` and goes through the full risk pipeline.** A webhook is
  a *signal source*, exactly equivalent to a Python strategy. It gets no privileged path to
  the broker, ever.
- **Public reachability**: your Mac is not addressable from the internet. You need a Cloudflare
  Tunnel or a small VPS — a real operational cost to weigh against the benefit.

**Recommendation:** keep TradingView at role ① through Phase 8. Design the `SignalSource`
interface now so that role ② is a ~150-line adapter later if you want it, but **do not put
TradingView in the critical path of a system that manages your money.**

---

## 9. Broker integration architecture

### 9.1 The interface

```python
class BrokerAdapter(Protocol):
    capabilities: BrokerCapabilities        # what this broker can actually do

    # --- read ---
    async def get_account(self) -> Account            # cash, buying power, equity
    async def get_positions(self) -> list[Position]
    async def get_orders(self, *, open_only: bool) -> list[BrokerOrder]
    async def get_order(self, client_order_id: str) -> BrokerOrder | None   # ← idempotency
    async def get_fills(self, since: datetime) -> list[Fill]

    # --- write ---
    async def place_order(self, req: OrderRequest) -> BrokerOrder
    async def cancel_order(self, client_order_id: str) -> None
    async def cancel_all(self) -> None                 # kill-switch path

    # --- stream ---
    async def stream_updates(self) -> AsyncIterator[BrokerEvent]

    # --- safety ---
    async def health(self) -> BrokerHealth
```

**`BrokerCapabilities` is the piece most designs omit.** Brokers differ in supported order
types, fractional shares, shorting, extended hours, and rate limits. Declaring capabilities
explicitly lets the execution layer *degrade gracefully at plan time* — "this broker has no
native trailing stop, so synthesise one" — instead of discovering it via a rejected order
during live trading.

### 9.2 Idempotency and the order state machine

```
                    ┌──────────────┐
                    │ PENDING_NEW  │  we are about to send
                    └──────┬───────┘
             ack ──────────┼────────── timeout / network error
                    ┌──────▼──────┐         ┌──────────────────────────┐
                    │     NEW     │         │        UNKNOWN           │
                    └──────┬──────┘         │  ⚠ BLOCKS this symbol    │
                           │                │  resolve by QUERY only,  │
       ┌───────────────────┼──────────┐     │  never by assumption or  │
       ▼                   ▼          ▼     │  by blind resend         │
┌─────────────┐    ┌────────────┐  ┌────────┴─────┐                    │
│PARTIALLY_   │───►│   FILLED   │  │  REJECTED    │◄───────────────────┘
│  FILLED     │    └────────────┘  └──────────────┘
└──────┬──────┘    ┌────────────┐  ┌──────────────┐
       └──────────►│  CANCELED  │  │   EXPIRED    │
                   └────────────┘  └──────────────┘
```

Three rules that prevent the classic duplicate-order loss (§3.5):

1. **Every order carries a `client_order_id`** generated *before* sending and persisted
   *before* sending. If the process dies mid-send, restart knows the ID to ask about.
2. **A retry is never a resend.** On any uncertainty: `get_order(client_order_id)` first,
   then decide. This is the single most important safety mechanism in the entire system.
3. **`UNKNOWN` is a real state that blocks.** No new orders for that symbol until resolved.
   Fail closed.

### 9.3 Two kinds of paper trading — you need both

This distinction is usually missed, and each catches bugs the other cannot:

| | **Internal simulator** (`PaperBroker`) | **Broker's paper account** |
|---|---|---|
| What it is | Your fill model against live data | Real broker API, fake money |
| Tests | Fill assumptions, slippage model, strategy behaviour | Auth, rate limits, order lifecycle, payload quirks, reconnection |
| Misses | Every real-world API surprise | Whether your fill model is honest |

**Recommendation: run both, sequentially.** Internal simulator from Phase 6, broker paper
account from Phase 9. Comparing the two is itself informative — where they disagree is where
your fill model is wrong.

### 9.4 Candidate brokers `[FACT]`

**US:** *Alpaca* — lowest friction by a wide margin; REST + WebSocket, free unlimited paper
trading with no gateway process and no minimum balance, commission-free stocks/ETFs. Covers
equities and ETFs only. `[ASSUMPTION]` The best *learning* broker even if you later move.
*Interactive Brokers* — far broader (options, futures, forex, global), but requires running
the TWS/IB Gateway desktop process with a daily restart, which is a real operational burden
on a Mac. *Tradier* — options-friendly middle ground.

**India:** *Zerodha Kite Connect* (mature docs, largest community), *Dhan*, *Upstox*, *Fyers*,
*Angel One SmartAPI*, *Finvasia Shoonya* (free API tier). `[FACT]` All now require static-IP
whitelisting, daily session logout, and exchange algo-identifier tagging under the SEBI
framework live since 1 April 2026 (§3.1).

**Recommendation: defer the choice, but exercise a real broker's paper API early.** The point
of Phase 9 is not to pick your permanent broker — it is to hit a real order lifecycle before
your assumptions calcify.

---

## 10. Strategy framework

### 10.1 Why your proposed interface needs to change

You suggested:

```python
class Strategy:
    def generate_signal(self, market_data): ...
```

You told me not to use it blindly, so: this interface has four problems, and the fourth is
the one that would cost you money.

1. **No clock.** The strategy cannot know what "now" is, so it cannot reason about sessions,
   staleness, or time-based exits.
2. **No position awareness.** It cannot express "I'm already long, hold" — and cannot make
   exit decisions, which is where most of a strategy's P&L actually lives.
3. **No identity or metadata.** Nothing to version, register, or attribute results to.
4. **`market_data` is unbounded.** If it's a DataFrame, it almost certainly contains future
   rows, and **look-ahead bias becomes one careless index away.** This is the big one.

### 10.2 The proposed interface

```python
class Strategy(Protocol):
    spec: StrategySpec                       # versioned, content-hashed identity

    def warmup_bars(self) -> int: ...        # bars needed before first valid signal
    def on_bar(self, ctx: StrategyContext) -> list[Intent]: ...
    def on_fill(self, ctx: StrategyContext, fill: Fill) -> list[Intent]: ...
```

The safety property lives in `StrategyContext`:

```python
class StrategyContext:
    now: datetime                            # the ONLY source of time
    def history(self, symbol, n: int, field: str) -> Series:
        """Last n values with timestamp <= self.now. Future data is not held
           by this object, so it cannot be returned."""
    def feature(self, name: str, symbol: str) -> Decimal: ...
    def position(self, symbol: str) -> Position | None: ...
    def portfolio(self) -> PortfolioSnapshot: ...
    def regime(self) -> RegimeLabel | None:  # optional AI/ML advisory input
```

> **Look-ahead bias is eliminated by construction.** The context object is built per-bar and
> physically does not contain future data. There is no discipline to forget and no code review
> to miss it — the future is simply not reachable from inside a strategy.

### 10.3 `Intent`: target exposure, not an action

```python
@dataclass(frozen=True)
class Intent:
    symbol: str
    target: TargetWeight | TargetQty | Flat  # DECLARATIVE — a destination, not a step
    confidence: Decimal | None               # 0–1, optional; used for sizing, never for bypass
    urgency: Urgency                         # PATIENT | NORMAL | IMMEDIATE → order type
    reason: str                              # human-readable, shown in the dashboard
    evidence: dict[str, Any]                 # the feature values that drove it — the audit trail
```

`evidence` is what makes "why did it trade?" answerable months later, without re-running
anything.

### 10.4 Identity, versioning, and reproducibility

```python
@dataclass(frozen=True)
class StrategySpec:
    id: str; name: str; version: str                # semver
    params: BaseModel                               # typed Pydantic model
    universe: UniverseSpec                          # point-in-time resolvable
    timeframe: Timeframe
    data_requirements: list[DataRequirement]
    cost_model_id: str                              # assumptions are part of identity
    lifecycle_status: LifecycleStatus
    author: str; created_at: datetime; description: str

    @property
    def content_hash(self) -> str:
        """sha256 over params + code AST + universe + timeframe."""
```

**Versioning rule: any change to code *or* parameters produces a new `content_hash`, and
therefore a new strategy version.** A backtest result is keyed by:

```
(strategy_content_hash, dataset_version, cost_model_version, code_git_sha, config_hash, seed)
```

If you rerun and get a different number, exactly one of those six changed, and the system can
tell you which. That is what §20's reproducibility requirement actually means in practice.

### 10.5 Lifecycle — gates, not labels

Your proposed statuses, made **enforceable**: the system refuses a transition whose criteria
are unmet.

```
RESEARCH ──► BACKTESTING ──► PROMISING ──► VALIDATED ──► PAPER ──► LIVE_APPROVED ──► LIVE
                                 │              │           │             │
                                 └──────────────┴───────────┴─────────────┴──► PAUSED ──► RETIRED
```

| Transition | Enforced gate |
|---|---|
| → PROMISING | Full backtest complete; positive net-of-cost expectancy; benchmark comparison recorded |
| → VALIDATED | Out-of-sample + walk-forward passed; parameter *plateau* (not spike); Monte Carlo drawdown within limits; trial count recorded and deflated Sharpe applied |
| → PAPER | Risk limits defined; kill-switch drill performed; reconciliation clean |
| → LIVE_APPROVED | ≥60 trading days paper; live-vs-backtest divergence within tolerance; **explicit human sign-off recorded with a timestamp and a reason** |
| → LIVE | Separate global mode flag + capital cap + a second confirmation |
| → PAUSED | Automatic on drawdown breach, divergence, or repeated errors |

### 10.6 Supported strategy families

The framework must accommodate all of these without special-casing: **technical** (MA cross,
breakout, momentum) · **mean reversion** (z-score, Bollinger, pairs) · **trend following** ·
**statistical arbitrage** (cointegration-based pairs) · **factor** (cross-sectional ranking) ·
**ML-based** (Phase 7) · **portfolio-level** (rebalancing, risk parity) · **external signal
sources** (TradingView webhooks as a `SignalSource` implementing the same interface).

`[ASSUMPTION]` Cross-sectional strategies (rank 500 stocks, buy the top 20) need the context
to expose the *whole universe* at `now`, not one symbol — the interface above supports this
via `ctx.portfolio()` and universe-scoped feature access.

### 10.7 Strategy authoring — three routes

**① Code strategies** — a Python class. Full power. Requires you to write Python.
**② Config strategies** — YAML declaring a composition of primitives:

```yaml
strategy: {id: sma_cross_spy, version: 1.0.0}
universe: {symbols: [SPY]}
timeframe: 1d
entry:  {when: "sma(close,50) > sma(close,200)", target_weight: 0.95}
exit:   {when: "sma(close,50) < sma(close,200)", target: flat}
risk:   {stop_loss_atr: 3.0, max_position_weight: 0.95}
```
Parsed into a real `Strategy` object via a **restricted expression evaluator** — a
whitelisted AST, not `eval()`. This is what makes iteration fast without a code deploy.

**③ AI-assisted** — you describe a strategy in English; the LLM emits a **YAML spec**
(route ②), not arbitrary Python. This is a deliberate constraint: generating a validated
config is auditable and safe, while generating executable code is a code-injection surface.
Where Python truly is required, it runs through: static analysis (no imports, no I/O, no
`eval`/`exec`, no network) → sandboxed subprocess → mandatory backtest → the same validation
gates as everything else. **No AI-authored strategy ever skips a gate, and none is
auto-promoted.**

---

## 11. Backtesting architecture

### 11.1 Event-driven, because vectorised cannot share a code path

A *vectorised* backtest computes signals over an entire price array at once — extremely fast,
and the natural home of look-ahead bias, since every row can see every other row. An
*event-driven* backtest walks bar by bar, feeding one bar at a time, exactly as live trading
does.

**Recommendation: event-driven as the source of truth; vectorbt for exploration only, with
findings re-verified event-driven before any promotion.** Sweep 10,000 parameter combinations
vectorised in seconds; validate the shortlist honestly.

### 11.2 The fill model — where backtests lie

Default conventions, all configurable and all recorded in the run manifest:

| Concern | Default | Why |
|---|---|---|
| **Signal→execution delay** | Signal on bar N close → fill at bar N+1 open | At bar N's close the market is shut. Same-bar fills are the #1 source of fake alpha. |
| **Market order fill price** | Next open ± slippage | Honest |
| **Limit order fill** | Only if bar range touched the price, **and** a configurable queue-position haircut applies | Touching a price ≠ being filled at it — someone was ahead of you |
| **Slippage** | Configurable: fixed bps · spread-proportional · **square-root volume participation** | Square-root impact is the standard academic model `[ASSUMPTION]` |
| **Volume cap** | Max % of the bar's volume per fill → forces partial fills | Prevents "buy $2M of a $500k/day stock" |
| **Gaps / halts / limit up-down** | Explicit no-fill handling | Real markets stop trading |
| **Shorting** | Borrow availability flag + borrow cost | Shorts are not symmetric to longs |
| **Costs** | Commission + exchange/regulatory fees + spread + borrow + financing + jurisdiction levies (STT/stamp duty/SEC fees) | §3.4 |

> ⚠️ **The backtest runs through the same risk engine as live.** This is an unusual choice and
> I want to flag it explicitly: most backtesters ignore risk limits, which means the strategy
> you validate is *not* the strategy you deploy — the live one gets throttled by rules the
> backtest never saw. Running risk inline means backtested performance already reflects the
> constraints you will actually trade under.

### 11.3 Anti-bias mechanisms as code

- Look-ahead — prevented by `StrategyContext` (§10.2).
- Survivorship — universes resolved from point-in-time membership including delisted symbols;
  where unavailable, a fixed universe **with the limitation documented in the run manifest.**
- Data leakage — the feature engine rejects any transform with a negative time shift.
- **Leakage canary** — run every strategy against shuffled/randomised returns. If it still
  looks profitable, there is a bug. Cheap, automated, and it catches a whole class of errors
  that code review does not.
- Determinism — every run records its seed; identical inputs must produce byte-identical
  outputs, verified in CI.

### 11.4 Metrics and outputs

**Returns:** total, CAGR, annualised, monthly/annual tables · **Risk:** volatility, downside
deviation, max drawdown, average drawdown, drawdown duration, recovery period, VaR/CVaR ·
**Risk-adjusted:** Sharpe, Sortino, Calmar, **Deflated Sharpe Ratio** (§12) · **Trades:**
count, win/loss rate, profit factor, expectancy, average/largest win and loss, holding period
· **Portfolio:** exposure over time, turnover, **cost drag**, concentration · **Comparison:**
benchmark (buy & hold + equal-weight universe), alpha, beta, correlation, tracking error.

**Charts:** equity curve **(gross and net on the same axes — the most educational chart you
can produce)** · underwater/drawdown curve · monthly heatmap · trade P&L distribution ·
rolling Sharpe · exposure timeline · cost breakdown.

**Every run emits a manifest** — the six-part key from §10.4 plus data provenance, universe
resolution, all cost assumptions, wall-clock time and library versions. Without it, a backtest
is an anecdote.

---

## 12. Strategy validation

### 12.1 The premise

> A backtest is a **hypothesis**, not evidence. It is the *weakest* form of evidence in this
> system — one draw from one historical path, on data you chose, with parameters you tuned.

Validation exists to answer one question: **is this an edge, or did I fit noise?**

### 12.2 The techniques, explained

**Out-of-sample holdout.** Reserve the most recent ~20–30% of history and *never look at it*
until final validation. **Enforced by the data layer**: out-of-sample data is unavailable to
research runs until a run is registered as final. Discipline fails; a flag in the data
provider does not.

**Walk-forward analysis.** Optimise on a window, test on the next, roll forward, repeat:

```
│──── train ────│─test─│
        │──── train ────│─test─│
                │──── train ────│─test─│
                        │──── train ────│─test─│
   time ───────────────────────────────────────────►
```
This simulates what you would actually have done — re-tuning periodically with only past data.
**The concatenated out-of-sample segments are the only backtest result worth quoting.**

**Purging and embargo.** When labels span multiple bars (e.g. "return over the next 10 days"),
a training sample near the test boundary overlaps the test period and leaks. *Purging* removes
overlapping samples; an *embargo* drops a small gap after each test window. Without these,
cross-validated ML results in finance are simply wrong.

**Parameter sensitivity — the best overfitting detector, and the easiest to read.** Plot
performance across the parameter grid. A real edge shows a broad **plateau** — neighbouring
parameters also work. Overfitting shows a **spike** — one magic combination surrounded by
mediocrity, because you found the one setting that fit past noise.

```
   Sharpe                              Sharpe
     │      ▁▃▅▆▆▆▅▃▁                    │            ▉
     │   ▁▃▅███████▅▃▁                   │  ▁▂▁ ▁▂▁ ▁▉▁ ▁▂▁
     └────────────────── param           └────────────────── param
     ✅ PLATEAU — robust                  ❌ SPIKE — fitted to noise
```

**Monte Carlo.** Resample the trade sequence thousands of times to get a *distribution* of
outcomes rather than a single path. Answers "how bad could the drawdown plausibly have been?"
— and **your risk limits should be set from that distribution, not from the single historical
path you happened to observe.**

**Regime slicing.** Report performance separately across bull/bear, high/low volatility, and
rising/falling-rate periods. A strategy that only works in one regime is a bet on that regime,
which is fine — as long as you *know* that is the bet you are making.

**Deflated Sharpe Ratio.** Corrects the observed Sharpe for the number of trials, the length
of the track record, and the non-normality of returns. Requires the trial count — which is why
experiment tracking is mandatory infrastructure (§3.7), including the failures.

### 12.3 Overfitting red flags — automated checks

The report flags: Sharpe > 2.0 on retail daily data · a parameter spike rather than a plateau
· out-of-sample degradation > 50% vs in-sample · fewer than ~100 trades (error bars swamp the
estimate) · performance concentrated in one period or a handful of trades · >5 tuned parameters
· results collapsing under small cost-assumption changes · the leakage canary showing profit
on randomised data.

---

## 13. Risk management architecture

### 13.1 An independent subsystem with veto power

The risk engine is not a strategy helper. It is an **adversarial component** whose job is to
assume every other part of the system is buggy.

**Two layers:**

**Pre-trade (synchronous, blocking, deterministic):** kill-switch check · trading-mode check ·
market-session check · **data staleness** · max order value · max position size (absolute and
% of equity) · max gross/net exposure · max leverage · max open positions · max per-strategy
capital · symbol allowlist/denylist · **duplicate-order detection** · price sanity (limit
within X% of last) · liquidity check (order vs ADV) · correlation-adjusted exposure ·
buying-power check.

**Continuous (monitors that can halt):** daily loss limit **(mark-to-market, including
unrealised — a realised-only limit is a hole you can drive a portfolio through)** · max
drawdown · consecutive losses · error rate · data-feed staleness · broker connectivity ·
**reconciliation drift** · unexpected position detected · strategy divergence from backtest.

### 13.2 Fail closed

> If a risk rule cannot be evaluated — missing data, an exception, an unreachable dependency —
> **the trade is rejected.** Never "skip the check and proceed."

Every evaluation, pass or fail, is logged with rule ID, inputs, threshold and verdict. That
log is what makes §19's "why did it trade / why didn't it?" answerable.

### 13.3 Structural enforcement

```python
class Order:
    def __init__(self, *args, _decision: RiskDecision, **kw):
        if not isinstance(_decision, RiskDecision) or not _decision.approved:
            raise UnauthorizedOrderError            # unreachable via the factory
```

`RiskDecision` can only be minted by the risk engine. Constructing an order without one is
not a lint warning — it is a runtime failure, and the *only* way around it is to deliberately
edit the domain core, which appears in a diff.

### 13.4 Kill switch — four independent layers

| Layer | Mechanism | Works when |
|---|---|---|
| 1 · Soft | In-app flag: stop new entries, manage existing | App healthy |
| 2 · Hard | Cancel all open orders, halt everything, **do not auto-liquidate** | App healthy |
| 3 · File | `KILL` sentinel checked at the lowest level of the order path | Strategy layer broken |
| 4 · Broker | API key revocation / broker-side trading disable | **Process unresponsive** |

Plus a **panic flatten** (close everything) that requires explicit manual confirmation.

`[ASSUMPTION]` **Auto-liquidation on a kill is deliberately excluded.** Force-closing every
position during the chaos that triggered the kill is frequently worse than holding — you sell
into the worst liquidity of the day. Halt automatically; liquidate by human decision.

---

## 14. AI/ML architecture

### 14.1 The dividing line

```
┌────────────────────────────────┬───────────────────────────────────────┐
│  DETERMINISTIC — AI may never  │  AI/ML — advisory only                │
│  influence                     │                                        │
├────────────────────────────────┼───────────────────────────────────────┤
│  Risk rules & thresholds       │  Market regime classification          │
│  Position limits               │  Signal quality estimation             │
│  Order validation              │  Feature discovery                     │
│  Portfolio constraints         │  Anomaly detection                      │
│  Kill switch                   │  Research assistance                    │
│  Reconciliation                │  Trade review & explanation             │
│  Accounting & P&L              │  Strategy drafting (gated)              │
└────────────────────────────────┴───────────────────────────────────────┘
        AI output is an INPUT to a deterministic process — never a command.
```

### 14.2 Where AI actually earns its place, ranked

**Tier A — high value, low risk. Build first (Phase 7).**

| Application | Why it works |
|---|---|
| **Decision narration** | An LLM reads the event chain for a trade and writes a plain-English explanation. Read-only, post-hoc, zero risk, and it directly serves your goal of understanding the system. |
| **Post-trade review** | "What did these 12 losing trades have in common?" — pattern-finding over your own history, where the LLM is analysing data rather than predicting. |
| **Research assistant** | Drafting strategy specs, explaining indicators, reviewing code, generating test cases. Value here is on *your* productivity, not on market prediction. |
| **Log & anomaly triage** | Summarising errors and surfacing unusual system behaviour. |

**Tier B — genuine, measurable value. Build with rigour (Phase 7+).**

| Application | Right tool |
|---|---|
| **Market regime classification** | Hidden Markov Model or simple volatility/trend clustering. `[ASSUMPTION]` A 3-state HMM usually matches or beats a neural network here, and you can actually interpret its states. |
| **Meta-labelling** | ⭐ **The highest-value ML application in retail algo trading.** Don't predict direction — let your rule-based strategy generate signals, then train a classifier to predict *whether each signal will be profitable*. Improves precision (fewer bad trades) without touching the primary logic, and gives you a probability to size with. Far easier than direction prediction, because the primary model has already done the hard part. |
| **Probability-scaled sizing** | Use the meta-label probability to scale position size within risk limits. |
| **Execution anomaly detection** | Flag fills that deviate from the expected slippage distribution. |

**Tier C — low value, high risk. Do not build.**

| Anti-pattern | Why not |
|---|---|
| **LLMs predicting price direction** | `[FACT]` No access to order flow; non-deterministic; and for any historical date the model was trained on text describing what happened next — so any "backtest" is contaminated. This is the most expensive mistake in the space. |
| **Deep learning on raw OHLCV** | ~5,000 daily bars per symbol against a signal-to-noise ratio near zero. A transformer will memorise, not generalise. |
| **News sentiment → trades (naively)** | Severe look-ahead risk: news archives are timestamped by publication, not by when the information was actionable, and are frequently revised. Needs genuine point-in-time news data to be honest. |
| **Auto-optimising parameters with AI** | An automated overfitting machine. Nothing in the process stops it from finding the best fit to noise. |

### 14.3 The model ladder

> **Always beat the dumb baseline first, and report the delta — never the absolute number.**

`constant/always-flat → simple rule → linear/logistic regression → gradient boosting →
anything else`

`[ASSUMPTION]` Most complexity in retail quant work adds nothing over a well-specified linear
model on well-chosen features. A model that cannot beat "buy and hold" or "always predict the
majority class" is not a model, regardless of its architecture.

### 14.4 Financial ML specifics

- **Never standard k-fold CV** — it trains on the future. Use purged, embargoed CV (§12.2).
- **Triple-barrier labelling** — instead of "return over 10 days", label by which of three
  barriers is hit first: take-profit, stop-loss, or time limit. This matches how you actually
  trade, and produces far better-behaved labels than fixed-horizon returns.
- **Sample-uniqueness weighting** — overlapping labels are not independent observations;
  weight them down or you will overstate your sample size, and therefore your confidence.
- **Stationary features only** — returns, ratios, z-scores. Never raw price levels: a model
  trained on AAPL at $50 has learned nothing applicable at $250.
- **Distrust feature importance under collinearity** — technical indicators are heavily
  correlated, so importance gets split arbitrarily among near-duplicates.

### 14.5 LLM governance

- **Structured output only** (JSON Schema + Pydantic validation). A parse failure is a hard
  failure — never "retry until it parses", which selects for hallucinations that happen to
  be well-formed.
- **Every call recorded**: versioned prompt, full input/output, model ID, temperature, token
  cost, latency, content hash. LLM calls are experiments and get the same provenance as
  backtests.
- **Hard daily cost cap**, enforced in code.
- **Never in the hot path.** LLM latency (~1–10s) and non-determinism are incompatible with
  order decisions. LLM work is asynchronous, cached, and post-hoc.
- **AI-generated code is untrusted input**: static analysis → sandboxed subprocess → mandatory
  backtest → full validation gates → human sign-off. No exceptions, no auto-promotion.

---

## 15. Portfolio, paper trading, and live-trading safety

### 15.1 Portfolio engine

State tracked: positions (qty, average cost, market value, unrealised P&L) · cash and buying
power · realised P&L (by trade, strategy, symbol, period) · gross/net exposure · allocation by
strategy, symbol, sector and asset class · portfolio volatility · rolling correlation matrix ·
concentration · current drawdown.

**The problem it exists to solve:** three strategies each independently taking a "reasonable"
5% position in AAPL, MSFT and NVDA are not diversified — those names are highly correlated,
and the portfolio is really one 15% bet on large-cap tech. The portfolio engine computes
**correlation-adjusted exposure** so the risk engine can enforce limits on the *real* bet
rather than the nominal one.

**Per-strategy capital allocation** is a hard constraint: each strategy gets a capital budget,
and its intents are scaled to fit. One strategy cannot consume the portfolio.

### 15.2 Trading modes

```python
class TradingMode(StrEnum):
    RESEARCH = "research"    # no orders, no broker connection at all
    BACKTEST = "backtest"    # simulated clock, simulated broker
    PAPER    = "paper"       # real clock, real data, simulated money
    LIVE     = "live"        # REAL MONEY
```

**Default is `RESEARCH`.** `LIVE` requires *all* of: an explicit config value, an environment
variable, a separate credential set, a per-strategy `LIVE` lifecycle status, a capital cap, a
clean startup reconciliation, and an interactive confirmation. `[ASSUMPTION]` Any one of these
being forgettable would be a design flaw — the point is that reaching `LIVE` by accident is
not possible.

### 15.3 Live-trading safety checklist (Phase 11 gate)

Kill switch, all four layers, **drilled at least once** · maximum daily loss (mark-to-market)
· maximum position size and order value · maximum daily trade count · duplicate-order
protection · stale-data detection with automatic halt · broker connectivity monitoring ·
startup and periodic reconciliation · automatic pause on repeated failures · automatic pause
on backtest divergence · complete audit logging · a written manual-intervention runbook ·
a tested database restore.

---

## 16. Observability and reproducibility

### 16.1 Answering "why did the system make this trade?"

Every trade reconstructs to a complete chain, from the event log alone — no re-running:

```
14:31:02  BAR         AAPL 1d close=189.42 vol=52.1M  [provider=tiingo, dsv=2026-08-21]
14:31:02  FEATURES    sma50=185.30 sma200=181.11 atr14=3.82 rvol20=0.19 regime=TRENDING
14:31:02  SIGNAL      strategy=sma_cross@1.2.0 → Intent(AAPL, target_weight=0.04)
                      reason="sma50 crossed above sma200"
                      evidence={sma50_prev: 180.9, sma200_prev: 181.2, bars_since_cross: 1}
14:31:02  AI          regime_classifier@0.3.1 → TRENDING p=0.81   [advisory]
14:31:03  SIZING      target 0.04 × equity 103,420 = $4,136 → 21 shares @ ~189.42
14:31:03  RISK        ✅ max_position_pct   0.040 ≤ 0.10
                      ✅ max_gross_exposure 0.61  ≤ 0.80
                      ✅ daily_loss_limit   -0.4% ≤ 2.0%
                      ✅ liquidity          0.00004% of ADV ≤ 1%
                      ✅ data_staleness     1.2s ≤ 60s
                      → RiskDecision(APPROVED, id=rd_01J...)
14:31:03  ORDER       BUY 21 AAPL MKT  client_order_id=coid_01J...
14:31:04  FILL        21 @ 189.44  commission=$0.00  slippage=+$0.42 (2.2 bps)
14:31:04  POSITION    AAPL 0 → 21 @ 189.44 | cash 45,102 → 41,124 | exposure 0.61 → 0.65
```

And equally, for a rejection:

```
14:28:11  SIGNAL      strategy=mean_rev@2.0.1 → Intent(SPY, target_weight=-0.15)
14:28:11  RISK        ❌ max_gross_exposure 0.94 > 0.80  [rule=RISK_007]
                      → RiskDecision(REJECTED)
14:28:11  NO ORDER    counterfactual tracked for review
```

`[FACT]` This chain is not generated for the log — it *is* the event log. The dashboard renders
it, and the LLM narrator (§14.2) reads it. Nothing is reconstructed after the fact, which means
nothing can drift out of sync with what actually happened.

### 16.2 Health, metrics, alerts

**Health checks:** data-feed freshness · broker connectivity · database · scheduler liveness ·
disk space · last successful reconciliation. **Metrics:** signals generated/rejected, orders by
state, fill latency, slippage vs model, P&L, exposure, error rates, LLM cost. **Alerts** with
routing by severity — a duplicate-order detection and a failed nightly backup are not the same
urgency. `[ASSUMPTION]` Start with local desktop notifications plus email; add push/Telegram
when live.

### 16.3 Reproducibility

Every backtest and experiment records: dataset version + provider + `as_of` · strategy
`content_hash` + semver · **git SHA of the code** · full resolved config · all parameters ·
model version · cost model version + all assumptions · execution assumptions · universe
resolution · random seeds · library versions · wall-clock time.

**Verified in CI:** the same run twice must be byte-identical. If a rerun differs, the system
diffs the manifests and tells you which of those inputs changed.

---

## 17. Security architecture

| Concern | Approach |
|---|---|
| **Secret storage** | **macOS Keychain** via `keyring` for local dev. Never `.env` with real values, never the database, never git. `.env.example` with placeholders is committed; `.env` is git-ignored. For a server: SOPS + age, or the host's secret manager. |
| **Credential separation** | Four distinct key sets: read-only market data · paper trading · **live trading** · admin. Never reuse across modes. The dashboard process gets read-only credentials only. |
| **Least privilege** | ⭐ **Disable withdrawal permission on every trading API key.** `[FACT]` The majority of API-key incidents drain funds via withdrawal, not via bad trades. Verify this setting explicitly at broker setup — it is the single highest-leverage security control in the entire system. |
| **Key rotation** | Quarterly, plus immediately on any suspicion. A documented procedure, not a memory. |
| **Audit log integrity** | Append-only, **hash-chained**: each entry includes a hash of the previous, so silent tampering or deletion is detectable. |
| **Webhook security** | HMAC + IP allowlist + replay protection + strict schema + rate limiting (§8.3). |
| **Network exposure** | The API binds to `127.0.0.1` **only**. Remote access via an authenticated tunnel, never an open port. |
| **Code execution** | AI-generated and config-defined strategies run through a restricted AST evaluator or a sandboxed subprocess with no network or filesystem access. |
| **Dependencies** | Locked (`uv.lock`), `pip-audit` in CI, pinned versions, no `curl \| bash` installs. |
| **Disk** | FileVault as a baseline assumption; encrypted backups off-machine. |
| **Compromise runbook** | Written before it is needed: revoke keys → kill switch → reconcile positions → audit the log → rotate everything. |

---

## 18. Performance and latency

**Determine requirements before optimising anything.**

| Strategy class | Latency budget | Implication |
|---|---|---|
| Position / swing (days–weeks) | Seconds to minutes are **irrelevant** | Python is entirely adequate; EOD data; scheduled runs |
| Intraday (minutes–hours) | Seconds matter mildly | Python fine; needs reliable always-on hosting |
| Minute-level / scalping | 100s of ms matter | Python borderline; needs low-latency hosting; costs rise sharply |
| HFT (microseconds) | ❌ **Explicitly out of scope** | Not achievable with this stack, and not achievable for retail at any price |

`[ASSUMPTION]` **You should start in the top row.** Daily strategies remove the entire latency
problem, which lets you spend your effort on the parts that determine whether this works at all
— validation honesty and execution safety.

**Where latency genuinely matters for you** — and it is not where people expect:
1. **Reaction to fills** — a stop that fires late costs real money.
2. **The kill-switch path** — must be fast and must not depend on anything that might be broken.
3. **Backtest throughput** — this will be your *first* actual performance problem (parameter
   sweeps), long before live latency ever matters. Solved by vectorbt and multiprocessing.

Indicator math will not be your bottleneck. Measure before optimising.

---

## 19. Development roadmap

> ⚠️ **Superseded by the [Phase 0 Addendum](phase-0-addendum-decisions.md).** Market, timeframe and capital have since been decided (India + US · daily, intraday and options as separate tracks · under $5k initially), which revises this section.


Adjusted from your proposal. **Three changes I'd argue for:**

1. **A walking skeleton in Phase 1.** Build the thinnest possible end-to-end slice — one
   symbol, buy-and-hold, one data source, one simulated broker, one log line — before building
   any component properly. `[ASSUMPTION]` Integration pain discovered in week 1 is cheap;
   discovered in month 4, it forces rework of everything built on the wrong assumption.
2. **The risk engine moves before the backtester is finished (Phase 4, not 5).** The backtest
   must run *through* the risk engine (§11.2), so risk has to exist first — otherwise you
   validate a strategy that behaves differently the moment it is deployed.
3. **A dedicated data-correctness gate (Phase 2.5).** Point-in-time correctness, corporate
   actions and validation deserve their own acceptance criteria. Everything downstream inherits
   these errors, and they are nearly invisible once buried.

| Phase | Goal | Key acceptance criteria | Est. `[ASSUMPTION]` |
|---|---|---|---|
| **0** | Discovery & architecture | This document approved; open decisions resolved | ✅ now |
| **1** | Foundation + **walking skeleton** | `uv` project, config, structlog, Postgres + Alembic, CI, pytest. **End-to-end: load 1 symbol → buy-and-hold → simulated fill → equity curve → logged.** | 1 wk |
| **2** | Market data | Provider adapter, normaliser, Parquet/DuckDB store, calendars, incremental backfill | 1–2 wk |
| **2.5** | **Data correctness gate** | Bitemporal `as_of` queries, corporate actions (raw + factors), validation gate, quarantine + alert | 1 wk |
| **3** | Strategy framework | `Strategy`/`StrategyContext`/`Intent`, registry, versioning, feature engine, ~8 tested indicators, 2 reference strategies | 1–2 wk |
| **4** | **Risk engine** *(moved earlier)* | Pre-trade rules, continuous monitors, `RiskDecision` token, kill switch, fail-closed proven by tests | 1 wk |
| **5** | Backtesting | Event-driven engine **running through the risk engine**, fill models, cost models, full metrics, reports, run manifests | 2–3 wk |
| **6** | Validation | Walk-forward, purged CV, sensitivity surfaces, Monte Carlo, deflated Sharpe, leakage canary, experiment tracking | 1–2 wk |
| **7** | Paper trading | Live data + `PaperBroker`, order state machine, reconciliation, crash recovery, **same strategy code as Phase 5** | 2 wk |
| **8** | Dashboard | Streamlit: portfolio, equity, decision log, risk events, backtest comparison, strategy control | 1–2 wk |
| **9** | AI research layer | Decision narration, post-trade review, regime classification, meta-labelling, LLM governance | 2–3 wk |
| **10** | Broker integration | Real broker **paper account**, capability discovery, idempotency proven under induced failures | 1–2 wk |
| **11** | Production hardening | Always-on deployment, monitoring, alerting, backup + **tested restore**, security review, kill-switch drill | 1–2 wk |
| **12** | **Controlled live** | Minimum capital, one validated strategy, tight limits, daily reconciliation, staged scale-up | gated |

> **Phase 12 is gated on evidence, not on time:** ≥60 trading days of paper results, live-vs-
> backtest divergence within tolerance, a clean reconciliation record, a completed kill-switch
> drill, a verified restore, and an explicit written sign-off from you.

---

## 20. Risks and the genuinely hard parts

### 20.1 The three that actually matter

**① Overfitting — the dominant risk.** Everything in §12 exists for this. `[FACT]` It is not
solved by more data or better models; it is solved by counting your attempts, testing
out-of-sample, and preferring plateaus to peaks. Expect most strategies to fail here. That is
the system working.

**② Transaction costs consuming the edge.** `[ASSUMPTION]` A strategy with 0.3% gross edge per
trade and 0.35% round-trip cost is a *guaranteed* loss that looks brilliant in a naive
backtest. Mitigated by making costs first-class, configurable, versioned, and displayed as
gross-vs-net on every equity curve.

**③ Operational failure with money at risk.** Duplicate orders, a laptop sleeping mid-position,
an unreconciled fill. `[FACT]` These lose real money in ways no backtest predicts, and they are
the specific reason for idempotency, the `UNKNOWN` state, reconciliation and the kill switch.

### 20.2 The rest

| Risk | Severity | Mitigation |
|---|---|---|
| Regime change kills a validated strategy | High | Regime slicing in validation; live divergence monitoring; auto-pause |
| Data errors → phantom signals | High | Validation gate; quarantine; multi-provider cross-check later |
| Tax drag underestimated | Medium-High | Jurisdiction-specific cost model; net-of-tax reporting |
| Regulatory non-compliance (SEBI) | **Blocking if India** | §3.1; resolve jurisdiction before Phase 2 |
| **Never actually trading** — building forever | Medium-High | Walking skeleton; time-boxed phases; paper trade early |
| Manual intervention destroying the statistics | Medium-High | Overrides logged and shown in attribution |
| Security / key compromise | High impact | Least privilege; **withdrawals disabled**; rotation; runbook |
| Over-trusting AI-generated code | Medium | Sandbox, static analysis, mandatory gates, no auto-promotion |
| Capacity limits at small size | Low-Medium | Volume-participation cap in the fill model |

### 20.3 The hardest parts, honestly

1. **An honest fill model.** Easy to write, very hard to make accurate. It is the difference
   between a backtest that means something and one that doesn't.
2. **Point-in-time data.** Unglamorous, tedious, and the foundation everything else rests on.
3. **Resisting the urge to skip validation** when a backtest looks fantastic. This is a
   discipline problem more than an engineering one, which is why the gates are enforced in code.
4. **Accepting a negative result.** `[ASSUMPTION]` The most likely outcome of the first year is
   "no durable edge found." A system that establishes that cheaply, without losing money, has
   succeeded — and that framing is worth internalising now rather than at month nine.

---

## 21. Recommended MVP

> ⚠️ **Superseded by the [Phase 0 Addendum](phase-0-addendum-decisions.md).** Market, timeframe and capital have since been decided (India + US · daily, intraday and options as separate tracks · under $5k initially), which revises this section.


> **One strategy family, one small universe, daily bars, backtested honestly, paper traded for
> a quarter, with a dashboard that explains every decision.**

**In scope (Phases 1–8):**

- Project foundation: `uv`, config, structured logging, Postgres + Alembic, CI, tests
- One EOD data provider + Parquet/DuckDB store + validation + calendars + corporate actions
- Feature engine with ~8 indicators (SMA, EMA, RSI, ATR, realised vol, returns, z-score,
  Donchian), each unit-tested against known reference values
- Strategy framework + registry + versioning
- **Three reference strategies**: buy-and-hold benchmark, SMA-cross trend-following, z-score
  mean-reversion — deliberately simple, well-understood, and useful as regression tests
- Event-driven backtester with realistic costs and fills, running through the risk engine
- Full metrics + reports + reproducible run manifests
- Validation framework: out-of-sample, walk-forward, sensitivity, Monte Carlo, leakage canary
- Risk engine: ~10 pre-trade rules, ~5 continuous monitors, all four kill-switch layers
- `SimBroker` + `PaperBroker`, order state machine, reconciliation, crash recovery
- Streamlit dashboard: portfolio, equity curve, decision log, risk events, backtest comparison
- Documentation as it is built

**Definition of done:** you can define a strategy, backtest it honestly, see exactly why every
trade happened or didn't, validate it against overfitting, run it in paper trading for weeks
across restarts, and explain any decision it made — **without having risked a rupee or a
dollar.**

---

## 22. Explicitly NOT in the MVP

Cut deliberately. Each is a real feature; each would slow you down before it helps.

| Excluded | Reason |
|---|---|
| **Live trading with real money** | Requires everything above to be proven first |
| **Any LLM in the decision path** | §14.1. Never, at any phase |
| **ML/AI strategies** | Phase 9. Learn to detect overfitting on simple strategies first, where it is visible |
| **Options, futures, forex, crypto** | Each brings its own accounting (Greeks, roll, margin, 24/7). One asset class first |
| **Intraday / minute / tick data** | ~20× cost, ~400× volume, much harder to backtest honestly. Not until a daily strategy survives validation |
| **Multiple brokers** | The abstraction exists from day one; the second implementation waits |
| **Multi-user, auth, cloud hosting** | Single-user local; a VM only when live requires it |
| **React UI** | Streamlit until it demonstrably chafes |
| **TradingView webhooks** | §8. Charting-only for now |
| **News / sentiment** | Point-in-time news data is expensive and leakage-prone |
| **Portfolio optimisation** (mean-variance, Black-Litterman) | `[FACT]` Notoriously unstable — tiny input changes swing allocations wildly. Fixed weights first |
| **Order book / microstructure** | Only relevant at latencies you are not trading at |
| **Automated hyperparameter search** | ⚠️ An overfitting machine. Deliberately excluded until §12 is fully built and you can see it happening |
| **Tax-lot accounting engine** | Record trades now; compute taxes offline |
| **Microservices / Kafka / distributed anything** | §4.1 |

---

## 23. Proposed project structure

Changed from your draft in four ways: a `src/` layout (prevents import shadowing and
accidental reliance on cwd); a **hard separation of `core/` (pure domain, no I/O) from
`adapters/` (all I/O)**, which is what makes the ports-and-adapters design real rather than
aspirational; `research/` explicitly quarantined from production imports; and `runbooks/` for
operational procedures, because a runbook written during an incident is worthless.

```text
ai-trading-platform/
├── src/trading/
│   ├── core/                    # ⚠ PURE DOMAIN — no I/O, no network, no DB, no clock
│   │   ├── types.py             #   Money, Symbol, Quantity, Timeframe (Decimal-based)
│   │   ├── events.py            #   immutable event definitions
│   │   ├── intent.py            #   Intent, TargetWeight, TargetQty, Flat
│   │   ├── order.py             #   Order (private ctor), OrderRequest, state machine
│   │   ├── position.py          #   Position, Fill, position accounting
│   │   ├── portfolio.py         #   PortfolioState — derived from events
│   │   └── clock.py             #   Clock protocol — nothing calls datetime.now()
│   │
│   ├── adapters/                # ALL I/O lives here
│   │   ├── data/                #   tiingo/, polygon/, broker_feed/, yfinance/
│   │   ├── brokers/             #   base.py, sim.py, paper.py, alpaca.py, kite.py …
│   │   ├── storage/             #   parquet_store.py, duckdb_query.py, postgres/
│   │   └── notify/              #   desktop.py, email.py
│   │
│   ├── engine/                  # the shared execution core (§4.2)
│   │   ├── runner.py            #   the single loop — backtest, paper and live
│   │   ├── fills/               #   fill models + slippage models
│   │   ├── router.py            #   idempotency, dedupe, order state machine
│   │   └── reconciler.py
│   │
│   ├── features/                # indicators + feature engine (point-in-time safe)
│   ├── strategies/              # base.py, registry.py, builtin/, config_dsl/
│   ├── risk/                    # rules/, engine.py, decision.py, killswitch.py, monitors/
│   ├── portfolio/               # sizing.py, allocation.py, correlation.py
│   ├── backtest/                # engine.py, costs.py, metrics/, reports/
│   ├── validation/              # walkforward.py, purged_cv.py, montecarlo.py,
│   │                            # sensitivity.py, deflated_sharpe.py, canary.py
│   ├── ai/                      # llm/ (governance), regime/, metalabel/, narrator.py
│   ├── api/                     # FastAPI: control, health, webhooks/
│   ├── ui/                      # Streamlit — READ-ONLY + command sender (§5.5)
│   ├── observability/           # logging.py, metrics.py, audit.py (hash-chained), health.py
│   └── config/                  # settings.py (Pydantic), profiles/, compliance.py
│
├── tests/
│   ├── unit/                    # mirrors src/ structure
│   ├── integration/             # DB, adapters, engine wiring
│   ├── e2e/                     # full backtest → paper flows
│   ├── failure/                 # ⭐ broker outage, dup orders, stale data, crash recovery
│   └── property/                # Hypothesis: accounting invariants
│
├── research/                    # ⚠ notebooks + throwaway. NEVER imported by src/
├── configs/                     # strategies/*.yaml, cost_models/, risk_profiles/, universes/
├── data/                        # git-ignored: bars/, features/, cache/
├── migrations/                  # Alembic
├── scripts/                     # backfill, run_backtest, run_paper, reconcile, killswitch
├── runbooks/                    # incident.md, compromise.md, restore.md, live_checklist.md
├── docs/                        # architecture, setup, strategy-guide, glossary, ADRs …
├── docker-compose.yml           # Postgres
├── .env.example                 # placeholders only — never real values
├── pyproject.toml
└── README.md
```

**Enforced import rules** (checked in CI, because a boundary that isn't enforced isn't a
boundary): `core/` imports nothing from the project · `strategies/` and `risk/` may import
`core/` and `features/` only · `research/` is never imported by `src/` · adapters are
imported only through protocols defined in `core/`.

---

## 24. Phase 0 acceptance criteria

Phase 0 is complete when **all** of the following are true:

- [ ] You have read this document and can restate, in your own words: look-ahead bias,
      survivorship bias, overfitting, slippage, drawdown, and expectancy. *(If any is still
      fuzzy, that's a gap in my explanation — tell me and I'll rewrite that part.)*
- [ ] **Jurisdiction and market are decided** (§3.1) — this unblocks data and broker choices
- [ ] **Asset class and timeframe are decided** — this determines the entire latency and data
      architecture
- [ ] Approximate capital scale is stated (it determines viable strategies and PDT/sizing)
- [ ] The one-core/three-drivers architecture (§4.2) is understood and agreed
- [ ] The risk-engine-cannot-be-bypassed principle (§4.4b, §13.3) is agreed
- [ ] The technology stack (§5) is agreed, or specific items are challenged
- [ ] The two-store data architecture (§6.2) is agreed
- [ ] The MVP scope (§21) and exclusions (§22) are agreed
- [ ] The roadmap and its three adjustments (§19) are agreed
- [ ] You accept the base rate: `[FACT]` most strategies will fail validation, and that is the
      system working correctly
- [ ] You agree that live trading is gated on evidence, not on a date

**Deliverables produced by Phase 0:** this document · the [glossary](glossary.md) · the
repository initialised with README, `.gitignore` and docs · the open decision list below. *(No application code — Phase 1 builds the project skeleton.)*

---

## 25. Open decisions — I need your input

> ⚠️ **Superseded by the [Phase 0 Addendum](phase-0-addendum-decisions.md).** Market, timeframe and capital have since been decided (India + US · daily, intraday and options as separate tracks · under $5k initially), which revises this section.


Ordered by how much they block. The first two change the architecture; the rest change
defaults I can pick for you if you'd rather not decide yet.

| # | Decision | Why it blocks | My recommendation |
|---|---|---|---|
| **1** | **Which market?** India (NSE/BSE) · US · both · crypto | Determines regulatory constraints (§3.1), data providers, brokers, calendars, cost models, and whether live trading can run on your Mac at all | Pick **one** to start. `[ASSUMPTION]` US equities has the cheapest, cleanest historical data and the lowest-friction paper trading — best for *learning*. India is right if that's where your capital actually is; budget for SEBI compliance and a Mumbai VM. |
| **2** | **Asset class + timeframe?** | Determines data cost, latency architecture, deployment, and whether PDT applies | **Equities/ETFs on daily bars.** Removes the entire latency problem, cheapest data, and honest backtesting is actually achievable |
| **3** | Approximate capital scale? | Drives position sizing, minimum viable strategies, PDT applicability, and whether per-trade costs are material | Needed before Phase 5's cost model |
| 4 | Long-only, or shorting too? | Shorting adds borrow costs, availability, margin and asymmetric risk | **Long-only for the MVP** |
| 5 | Do you want the TradingView webhook path eventually? | Determines whether Phase 1 designs the `SignalSource` seam now | Design the seam, build it later |
| 6 | Postgres via Docker acceptable? | If you'd rather avoid Docker, we start on SQLite and migrate at Phase 7 | **Postgres via Docker** — one command, avoids a migration |
| 7 | Comfortable writing Python strategies, or config/YAML only? | Determines whether the config DSL is MVP or Phase 9 | Both, but **code strategies first** |

---

## 26. Coverage map

Mapping your 20 required Phase 0 items to this document:

| Required | Section | | Required | Section |
|---|---|---|---|---|
| 1 · Understanding | §1 | | 11 · Backtesting architecture | §11 |
| 2 · Terminology | §2 + [glossary](glossary.md) | | 12 · Risk-management architecture | §13 |
| 3 · Missing requirements | §3 | | 13 · AI/ML architecture | §14 |
| 4 · Architecture | §4 | | 14 · Security architecture | §17 |
| 5 · Technology stack | §5 | | 15 · Development roadmap | §19 |
| 6 · Database/storage | §6 | | 16 · Risks and hard parts | §20 |
| 7 · Market data architecture | §7 | | 17 · Recommended MVP | §21 |
| 8 · TradingView integration | §8 | | 18 · NOT in MVP | §22 |
| 9 · Broker integration | §9 | | 19 · Project structure | §23 |
| 10 · Strategy framework | §10 | | 20 · Phase 0 acceptance criteria | §24 |

Additional sections beyond your list: §12 strategy validation · §15 portfolio, paper trading
and live safety · §16 observability and reproducibility · §18 performance and latency ·
§25 open decisions.

---

## Sources

Facts marked `[FACT]` about current platform, regulatory and library status were verified
against these sources on 2026-08-22:

- SEBI retail algo framework, thresholds and timeline — [Upstox](https://upstox.com/news/market-news/financial-regulations/sebi-extends-timeline-for-retail-algo-trading-framework-sets-glide-path-for-brokers/article-182285/), [QuantInsti](https://www.quantinsti.com/articles/algorithmic-trading-india/), [Tradejini](https://www.tradejini.com/blogs/what-sebis-new-algo-trading-rules-mean-for-you), [5paisa](https://www.5paisa.com/news/sebi-tightens-algo-trading-rules-with-mandatory-2fa-and-audit-trails-from-april-1)
- Static IP / broker API compliance — [Zerodha (In The Money)](https://inthemoneybyzerodha.substack.com/p/sebi-algo-trading-changes-april-2026), [Kite Connect forum](https://kite.trade/forum/discussion/15912/preparing-to-comply-with-sebis-retail-algo-rules-static-ip-ratelimits-order-types)
- TradingView webhook plan requirements and limits — [TV-Hub](https://www.tv-hub.org/guide/tradingview-alerts-setup), [TradeFunded](https://tradefunded.org/automate-tradingview-alerts)
- Backtesting library status — [BullAlert](https://bullalert.ai/blog/best-python-backtest-engines-2026/), [QuantTradingTools](https://quanttradingtools.com/python-backtesting-frameworks/)
- Broker API comparison — [BrokerChooser](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading), [TradeAlgo](https://www.tradealgo.com/trading-guides/tools/best-broker-apis-for-algorithmic-trading-in-2026)
- Market data provider pricing — [NB Data](https://www.nb-data.com/p/best-financial-data-apis-in-2026), [AI Fin Hub](https://aifinhub.io/articles/market-data-apis-compared-2026/)

---

**End of Phase 0 proposal. Awaiting your approval before Phase 1.**

# Phase 0 Addendum — Decisions Resolved

**Supersedes:** [phase-0-discovery.md](phase-0-discovery.md) §19 (roadmap), §21 (MVP), §25 (open decisions)
**Date:** 2026-08-22 · **Status:** awaiting Phase 1 approval

---

## 1. What you decided

| # | Decision | Your answer |
|---|---|---|
| 1 | Market | **Both** — India (NSE/BSE) and US |
| 2 | Asset class + timeframe | **Equities/ETFs daily · Equities/ETFs intraday · Options** — "all separately" |
| 3 | Capital | **Under $5,000 / ₹4L initially**, scaling as things work |

You picked the maximally broad scope. I'm going to build for all of it — but I owe you one
number first, because it changes the *sequencing*, not the destination.

---

## 2. The number you need before anything else

You are choosing two markets and three instrument tracks on a capital base under $5,000.
Here is what that costs annually **before a single trade is placed.**

### Fixed infrastructure, live trading `[FACT]` — verified 2026-08-22

| Item | India | US |
|---|---|---|
| Market data | Kite Connect API **₹500/mo = ₹6,000/yr** (historical data now included in the base subscription — it used to be a ₹2,000/mo add-on) | Tiingo **~$10/mo = $120/yr** |
| Broker API | included above | Alpaca **$0** |
| Static IP | **₹1,500–6,000/yr** — mandatory for API orders under SEBI | not required |
| Always-on host | **₹5,000–15,000/yr** — SEBI expects Indian-server hosting | $0–180/yr (optional for daily bars) |
| Demat AMC | **₹354/yr** | $0 |
| **Total** | **₹12,850 – ₹27,350/yr** | **$120 – $300/yr** |

**Both markets, live: roughly $270 – $620/yr (₹23,500 – ₹54,000/yr).**

### What that means against your capital

| Capital | Annual infra drag | Read |
|---|---|---|
| **$5,000 / ₹4L** | **5.4% – 12.4%** | 🔴 A strong retail year is 10–15% gross. Infrastructure alone eats between a third and nearly all of it |
| $25,000 / ₹20L | 1.1% – 2.5% | 🟠 Tolerable |
| $100,000 / ₹85L | 0.3% – 0.6% | 🟢 Negligible |

<br>

> ### The recommendation this produces
>
> **Stay in RESEARCH and PAPER mode until either (a) a strategy passes validation, or
> (b) capital crosses roughly $25,000 / ₹20L.** Paper trading the US pipeline costs
> **~$120/yr or less** — you get the entire research loop, the full decision chain, and
> honest backtesting for the price of a coffee a month. Going live in India costs
> ₹13,000–27,000/yr *whether or not the strategy works.*
>
> This is not me talking you out of live trading. It is the same discipline as the rest of
> the design: **don't pay a fixed cost until the variable return is evidenced.**

### Per-trade costs also differ sharply

`[FACT]` **India, delivery equity (Zerodha as reference):** ₹0 brokerage, but **STT is 0.1% on
buy and 0.1% on sell** — a 0.2% round trip before anything else. Add stamp duty (0.015% buy),
exchange transaction charges, SEBI fees, 18% GST on those, and a **flat DP charge of ~₹13–16
per scrip on every sell**. Round trip lands around **0.25–0.35%**.

`[FACT]` **US equity (Alpaca as reference):** $0 commission, negligible SEC/TAF fees. Your cost
is **spread and slippage only.**

<br>

> **Design consequence:** an Indian equity strategy needs roughly **0.35% gross edge per round
> trip just to break even**; a US one needs a fraction of that. These are not the same strategy
> universe. The cost model is therefore per-market and versioned, and the same strategy will
> correctly show a different net equity curve in each. `[ASSUMPTION]` Expect strategies that
> pass validation in the US to fail it in India on turnover alone.

The flat ₹13–16 DP charge matters more than it looks at your capital: on a ₹40,000 position
it's 0.04%; on a ₹5,000 position it's **0.3%**. **At small capital, flat fees dominate** — which
argues for fewer, larger positions rather than a wide diversified book.

---

## 3. Two hard constraints on the intraday and options tracks

### 🔴 US intraday equities are effectively blocked below $25,000

`[FACT]` The Pattern Day Trader rule: more than 3 day trades in 5 business days in a **margin
account** requires $25,000 minimum equity. Below that the account gets restricted.

The workaround is a **cash account**, which is exempt from PDT — but US equities settle T+1, so
you can only trade with *settled* cash. In practice that caps you at roughly one round trip per
dollar per day, which defeats most intraday strategies.

**This is not something the architecture can solve.** US intraday goes on the roadmap, but it
is gated on capital, not on engineering.

### 🟠 Options roughly double the domain model

Everything built so far assumes an instrument is a share: a quantity, a price, a position. An
option is not that. It has a strike, an expiry, a multiplier (one contract controls 100 shares
or one lot), an underlying, exercise style, and a risk profile that is **non-linear** — measured
in Greeks, not notional value.

Concretely, options require:

| Layer | What has to change |
|---|---|
| Data | Option chains, implied volatility surfaces, open interest — an entirely separate feed, and far larger |
| Position accounting | Multiplier, expiry, assignment, early exercise, pin risk, exercise/expiry cash flows |
| Risk engine | Delta/gamma/vega/theta exposure limits. Notional caps are actively misleading for options |
| Backtesting | Chain reconstruction at every bar, plus a bid/ask model — option spreads are *wide*, often 2–10% |
| Cost model | `[FACT]` STT on options rose to **0.15% on sell premium** from 1 April 2026 |

`[ASSUMPTION]` At ₹4L / $5k, options also have a granularity problem: one NIFTY lot or one SPY
contract is a large fraction of your capital, so position sizing becomes coarse and risk limits
become hard to respect.

<br>

> **What I'm doing about it now, at zero cost:** the `Position`, `Fill` and `Instrument` domain
> types will carry `multiplier`, `expiry`, `underlying` and `instrument_class` **from Phase 1**,
> even though only equities are implemented. Retrofitting options into a position model that
> assumes shares is genuinely painful; designing the seam now is nearly free.

---

## 4. What "both markets" changes in the architecture

Your answer adds real scope. Here is exactly what, so it is not a surprise later.

### New in the MVP because of this decision

| Addition | Why it's now unavoidable | Cost |
|---|---|---|
| **Multi-currency portfolio accounting** | Positions in USD and INR, one portfolio. Needs a base currency, an FX rate source (itself point-in-time), and **FX P&L separated from strategy P&L** — otherwise a rupee move looks like alpha | ~1 wk |
| **`ComplianceProfile` as a first-class object** | SEBI's 10 orders/sec cap, static-IP requirement, daily session logout and Algo-ID tagging are *risk rules*, enforced by the risk engine. The US profile has PDT day-trade counting instead | ~2 days |
| **Canonical instrument identity** | `AAPL` and `RELIANCE` need namespacing, and the same conceptual instrument may exist on NSE and BSE. A symbol string is not an identity | ~2 days |
| **Two market calendars** | NSE 09:15–15:30 IST · NYSE 09:30–16:00 ET. Different holidays, different half-days | included in Phase 2 |
| **Two cost models** | Per §2 above — versioned, per-market, and applied per trade | included in Phase 5 |
| **Two deployment targets** | India live needs a Mumbai host with a static IP; US daily can run from your Mac | Phase 11 |

`[FACT]` One piece of good news: **the sessions do not overlap.** NSE closes at 10:00 UTC; US
opens at 13:30 UTC. A single-threaded scheduler can serve both markets without contention, and
you never have two markets live at once. That removes a whole class of concurrency problems.

### Not changed

The core is untouched. Strategies, the risk engine, the backtester, the event log and the
portfolio engine are all market-agnostic already — that is what the ports-and-adapters design
in §4 bought us. **Adding India after the US is an adapter exercise, not a rewrite.**

---

## 5. Revised sequencing

You said "all separately." That instinct is right — they *are* separate tracks sharing one core.
But they should be built **sequentially**, because each one adds a distinct layer of domain
complexity, and building three at once means debugging three unfamiliar domains simultaneously.

```
                    ┌──────────────────────────────────────┐
                    │  ONE CORE  — built once, Phases 1–8  │
                    │  strategies · risk · backtest ·      │
                    │  portfolio · event log · dashboard   │
                    └───────────────┬──────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  ┌───────────┐              ┌────────────┐             ┌────────────┐
  │ TRACK A   │              │  TRACK B   │             │  TRACK C   │
  │ Equities  │              │  Equities  │             │  Options   │
  │  DAILY    │─────────────▶│  INTRADAY  │────────────▶│            │
  └───────────┘              └────────────┘             └────────────┘
   Phases 1–8                  Phase 13                   Phase 14
   cheapest data               +always-on host            +chains, IV,
   no latency problem          +minute data                Greeks, expiry
   ✅ start here               ⚠ US blocked <$25k         ⚠ 2× domain model
                               ⚠ India: static IP,        ⚠ STT 0.15% on
                                 10 OPS cap                 sell premium

  Markets:  US implemented first (≈$120/yr, zero compliance overhead)
            India added at Phase 9 as an adapter — architecture carries both from day 1
```

### Why US first, when your capital is likely in INR

Not because the US is the better market for you — because it is the **cheaper place to learn.**
Alpaca paper trading is free, needs no static IP, no Indian VM, no ₹6,000/yr API subscription,
and no SEBI compliance surface. You can exercise the entire pipeline — real API, real order
lifecycle, real rate limits, real reconnection handling — for essentially nothing.

India then arrives at Phase 9 as a broker adapter plus a compliance profile, against a core that
is already proven. **You pay the Indian infrastructure cost once you have something worth
running, not while you are still learning what a fill model is.**

If you'd rather go India-first, say so — it's a Phase-2 provider swap, not an architectural
change. It just costs ~₹13,000/yr more to reach the same learning milestone.

### Revised roadmap

| Phase | Goal | Change from original | Est. |
|---|---|---|---|
| 1 | Foundation + walking skeleton | **+ multi-currency types, instrument taxonomy, ComplianceProfile skeleton** | 1.5 wk |
| 2 | Market data (US, daily) | + canonical symbols, two calendars | 1–2 wk |
| 2.5 | Data correctness gate | unchanged | 1 wk |
| 3 | Strategy framework | unchanged | 1–2 wk |
| 4 | Risk engine | **+ compliance rules as risk rules** | 1–1.5 wk |
| 5 | Backtesting | **+ per-market cost models, FX handling** | 2–3 wk |
| 6 | Validation | unchanged | 1–2 wk |
| 7 | Paper trading | unchanged | 2 wk |
| 8 | Dashboard | **+ multi-currency portfolio view** | 1–2 wk |
| 9 | **India market adapter** | **new** — Kite Connect, NSE calendar, Indian cost model, SEBI compliance profile | 1–2 wk |
| 10 | AI research layer | was Phase 9 | 2–3 wk |
| 11 | Broker paper accounts (both) | was Phase 10 | 1–2 wk |
| 12 | Production hardening | + Mumbai host for India | 1–2 wk |
| 13 | **Track B — intraday** | new, gated on capital (US) / compliance (India) | 2–3 wk |
| 14 | **Track C — options** | new, largest single addition | 3–4 wk |
| — | Controlled live | gated on evidence **and** on capital crossing the infra-drag threshold | — |

**Net effect of your three answers:** roughly **+3 weeks** to the MVP (multi-currency,
compliance profiles, instrument taxonomy), and **+5–7 weeks** for the intraday and options
tracks after it. The MVP itself still ships as one market, one track — which is the only way
any of this gets validated rather than merely built.

---

## 6. Revised MVP scope

**Unchanged in spirit, extended in scaffolding.** The MVP is still: *one strategy family, one
small universe, daily bars, backtested honestly, paper traded for a quarter, with a dashboard
that explains every decision* — now with the multi-market and multi-instrument seams built in
but only one of each implemented.

**Added to MVP scope:** multi-currency portfolio accounting · `ComplianceProfile` enforced by the
risk engine · canonical instrument identity with `instrument_class`, `multiplier`, `expiry`,
`underlying` · two market calendars · per-market versioned cost models.

**Still explicitly out of MVP:** live trading · intraday · options · India implementation (Phase
9) · any LLM in the decision path · ML strategies.

---

## 7. What I need from you to start Phase 1

One question, then I build:

**Do you want US-first (cheapest learning loop, ~$120/yr) or India-first (~₹13,000/yr more, but
it's where your capital is)?** Either works; the architecture is identical.

If you have no strong preference, I will proceed **US-first** and add India at Phase 9.

---

## Sources

`[FACT]` claims in this addendum were verified 2026-08-22 against:

- [Zerodha brokerage &amp; charges breakdown](https://stockcalc.in/blog/zerodha-brokerage-charges-2026-full-breakdown) · [Chittorgarh](https://www.chittorgarh.com/brokerage_charges/zerodha/18/) — STT, DP charges, delivery brokerage, AMC
- [Kite Connect pricing](https://support.zerodha.com/category/trading-and-markets/general-kite/kite-api/articles/historical-data-and-live-market-data-payment-plan) · [Kite Connect forum](https://kite.trade/forum/discussion/14806/historical-data-is-now-free-with-base-kite-connect-subscription) — ₹500/mo, historical data now included
- [Zerodha on SEBI static IP](https://inthemoneybyzerodha.substack.com/p/sebi-algo-trading-changes-april-2026) · [QuantInsti](https://www.quantinsti.com/articles/algorithmic-trading-india/) — static IP, 10 OPS, Algo-ID
- [BrokerChooser](https://brokerchooser.com/best-brokers/best-brokers-for-algo-trading) — Alpaca commission structure and paper trading

> ⚠️ **Fee schedules change.** These figures are inputs to a *versioned* cost model, not
> constants in code. Phase 5 re-verifies every number against the live broker schedule and
> records which version each backtest used.

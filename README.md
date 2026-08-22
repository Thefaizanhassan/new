# AI Algorithmic Trading Platform

A modular, locally-run platform for researching, validating, and (eventually) executing
algorithmic trading strategies — built so that **AI never directly controls money**.

## Status

| | |
|---|---|
| **Current phase** | Phase 2 — Market data layer *(complete)* |
| **Market** | India-first (NSE/BSE), US adapter at Phase 9 |
| **Tests** | 172 passing · ruff, ruff-format and mypy clean |
| **Trading mode** | `RESEARCH` — `LIVE` is refused at startup and stays refused until Phase 12 |

## Start here

1. **[Phase 0 — Discovery & Architecture Proposal](docs/phase-0-discovery.md)** — the full design.
2. **[Phase 0 Addendum — Decisions Resolved](docs/phase-0-addendum-decisions.md)** — market, timeframe and
   capital decided; the cost math those produce; revised roadmap. **Supersedes §19, §21 and §25 of the
   main document.**
3. **[Technology Evaluation](docs/technology-evaluation.md)** — pandas, TA-Lib and Backtrader assessed
   against verified current status. **Contains one decision awaiting your approval.**
4. **[The Data Layer](docs/data-layer.md)** — bitemporal storage, corporate actions, and the two
   different questions "as of" can mean.
5. **[Local setup on macOS](docs/setup-local-mac.md)** — from nothing to a green test run in ~5 minutes.
6. **[Glossary](docs/glossary.md)** — every trading term used in this project, in plain language.

## Quick start

```bash
brew install uv
uv sync --all-groups
uv run pytest -q          # 75 tests
uv run ruff check . && uv run mypy
```

Full walkthrough, including what to do when something breaks:
[docs/setup-local-mac.md](docs/setup-local-mac.md).

## What exists today

**Built and tested:** the domain core — exact `Decimal` money with currency safety, FIFO-lot
position accounting, and the risk-approval token that makes an `Order` impossible to construct
without a passing risk check. Plus the pandas data pipeline with a full validation gate, Indian
delivery and intraday cost models, NSE/BSE calendars, SEBI and US compliance profiles, and
environment-driven config that refuses `LIVE` mode outright.

Plus the walking skeleton: eight documented TA-Lib indicators, two reference strategies, a
pre-trade risk engine with a filesystem kill switch, and an end-to-end runner producing an
equity curve, a full decision chain and a reproducible run manifest.

And the Phase 2 data layer: a bitemporal Parquet store queried through DuckDB, corporate
actions applied on read while raw prices stay immutable, resumable chunked backfill with
quarantine, and a data-trust tier that travels with the bars and cannot be laundered by
storage. See [docs/data-layer.md](docs/data-layer.md).

**Not built yet:** the real backtesting engine (slippage, spread, partial fills, volume caps),
validation and walk-forward testing, paper trading, broker connections, and the dashboard.
**You cannot place a trade with this, by design.**

```bash
uv run trading status                            # how this instance is configured
uv run trading ingest  --symbol RELIANCE         # backfill into the local store (resumable)
uv run trading catalog                           # what data you actually hold
uv run trading actions --symbol RELIANCE         # record and list corporate actions
uv run trading check-data --symbol RELIANCE      # run the validation gate
uv run trading backtest --source store           # end to end, split-adjusted, from disk
uv run trading indicators --name RSI             # what it means and where it misleads
```

## The one-paragraph version

This is not a trading bot. It is a **research and control system** whose primary job is to
make it hard to fool yourself. Anyone can write code that buys when a moving average crosses.
The difficulty is knowing whether that rule has a real edge or whether you accidentally
fitted it to noise, and then executing it without a bug or an outage costing you more than
the edge was worth. Almost all of the engineering in this project exists to answer those two
questions honestly.

## Non-negotiable design principles

1. **One code path.** The same strategy code runs in backtest, paper, and live. Only the
   clock and the broker are swapped.
2. **The risk engine cannot be bypassed.** An `Order` is structurally impossible to construct
   without a passing `RiskDecision`. This is enforced by types, not by convention.
3. **Fail closed.** If a safety check cannot be evaluated, the trade is rejected.
4. **The data layer cannot serve the future.** Look-ahead bias is prevented structurally,
   not by discipline.
5. **Every decision is explainable.** Market data → features → signal → risk verdict → order
   → fill is reconstructable for any trade, from an immutable event log.
6. **AI is advisory, never authoritative.** Model output is an input to a deterministic
   process, never a command.

## Evidence labelling

Throughout this project's docs, claims are tagged so that speculation is never mistaken
for results:

`[FACT]` verifiable now · `[ASSUMPTION]` taken as true, unverified · `[HYPOTHESIS]` to be tested
`[BACKTEST]` historical simulation only · `[PAPER]` simulated live · `[LIVE]` real money

A `[BACKTEST]` result is **never** described as profitable. It is described as
*not yet falsified*.

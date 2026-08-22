# AI Algorithmic Trading Platform

A modular, locally-run platform for researching, validating, and (eventually) executing
algorithmic trading strategies — built so that **AI never directly controls money**.

## Status

| | |
|---|---|
| **Current phase** | Phase 0 — Discovery & Architecture *(awaiting approval)* |
| **Code written** | None yet, by design |
| **Trading mode** | `RESEARCH` — live trading is not implemented and is disabled by default |

## Start here

1. **[Phase 0 — Discovery & Architecture Proposal](docs/phase-0-discovery.md)** — the full design.
2. **[Glossary](docs/glossary.md)** — every trading term used in this project, in plain language.

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

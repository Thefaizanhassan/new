# Glossary

Every trading term used in this project, in plain language. Terms that drive architectural
decisions are explained in more depth in [Phase 0](phase-0-discovery.md) §2 — this file is the
complete reference.

**How to read this:** if a term appears in a design document and you don't recognise it, it
should be here. If it isn't, that's a bug in the docs — tell me and I'll add it.

---

## Market structure

**Bar** — A summary of trading over a fixed window (1 minute, 1 day). Contains OHLCV.

**OHLCV** — Open, High, Low, Close, Volume. The five numbers that summarise a bar. Note that a
bar is *lossy*: it doesn't tell you whether the high came before the low.

**Tick** — A single trade or quote update. The rawest form of market data. Enormous in volume:
one liquid US stock can produce millions of ticks per day.

**Order book / Level 2** — The list of all resting buy and sell orders at each price. Shows
depth of liquidity. Relevant only at latencies we are not trading at.

**Bid** — The highest price a buyer is currently willing to pay. **You sell at the bid.**

**Ask (or Offer)** — The lowest price a seller is currently willing to accept. **You buy at
the ask.**

**Spread** — Ask minus bid. An immediate, unavoidable cost on every round trip. Wide spreads
can exceed a strategy's entire edge.

**Liquidity** — How much you can trade without moving the price. Usually proxied by ADV.

**ADV (Average Daily Volume)** — Average shares/contracts traded per day. Used to size orders
so you don't move the market against yourself.

**Market impact** — The price movement your own order causes. Grows roughly with the square
root of order size relative to volume.

**Slippage** — The gap between the price you expected and the price you got. Combines spread,
market movement and impact.

**Trading session** — The hours an exchange accepts orders. Includes pre-market and after-hours
in some markets. Not simply "9:30 to 4:00 on weekdays" — see *market calendar*.

**Market calendar** — The authoritative record of which days an exchange is open, including
holidays and half-days. Handled by the `exchange_calendars` library, never by weekday arithmetic.

**Halt** — Trading in an instrument is suspended (news pending, extreme move). Orders can't fill.

**Limit up / limit down** — Regulatory price bands beyond which trading pauses.

---

## Instruments and positions

**Instrument / Symbol / Ticker** — A tradable thing (AAPL, SPY, RELIANCE).

**Asset class** — A category of instrument: equities, ETFs, options, futures, forex, crypto.
Each has different accounting, and mixing them adds real complexity.

**ETF (Exchange Traded Fund)** — A fund that trades like a stock. SPY tracks the S&P 500.
Convenient for index strategies.

**Long** — You own it; you profit if the price rises.

**Short** — You borrowed and sold it; you profit if the price falls. Asymmetric risk: losses are
theoretically unlimited, and you pay a *borrow cost*.

**Position** — Your current holding in an instrument: quantity, average cost, market value.

**Flat** — No position.

**Exposure** — How much capital is at risk. *Gross* = longs + |shorts|. *Net* = longs − |shorts|.

**Leverage** — Trading with borrowed money. 2× leverage doubles gains *and* losses.

**Margin** — Collateral your broker requires for leveraged or short positions.

**Margin call** — Your broker demanding more collateral, or force-closing positions. Something
the risk engine exists to prevent.

**Buying power** — How much you can actually deploy right now, given cash and margin rules.

---

## Orders and execution

**Market order** — Execute immediately at whatever price is available. Guarantees execution,
not price.

**Limit order** — Execute only at a specified price or better. Guarantees price, not execution.

**Stop order (stop-loss)** — Becomes a market order once price crosses a trigger. Used to cap
losses. Note it does *not* guarantee your exit price — in a gap it can fill far worse.

**Stop-limit order** — Becomes a limit order at the trigger. Guards price but may not fill at all.

**Trailing stop** — A stop that follows the price up (for longs), locking in gains.

**Take-profit** — An order that closes a position at a target profit.

**Time in force** — How long an order stays live: DAY, GTC (good till cancelled), IOC
(immediate or cancel), FOK (fill or kill).

**Fill** — An execution. One order may produce several.

**Partial fill** — Only part of your order executed. Common in illiquid names, and a source of
accounting bugs if unhandled.

**Rejected order** — The broker refused it: insufficient funds, invalid price, market closed,
restricted symbol.

**Order lifecycle** — The states an order moves through. See [Phase 0 §9.2](phase-0-discovery.md#92-idempotency-and-the-order-state-machine).

**Client order ID** — An identifier *you* generate before sending, so you can ask "did my order
arrive?" without risking a duplicate. The single most important safety mechanism in execution.

**Idempotency** — The property that doing something twice has the same effect as doing it once.
Critical for orders: a retry must never create a second position.

**Reconciliation** — Comparing your system's beliefs about positions and cash against the
broker's records. The broker is always the source of truth.

---

## Corporate actions

**Corporate action** — An event changing a security's shares or price: split, dividend, merger,
spin-off, delisting.

**Stock split** — Shares multiply and price divides (4:1 split: 100 shares at $200 → 400 at $50).
Economically neutral; catastrophic to a naive price series.

**Dividend** — A cash payment to shareholders. The price drops by roughly the dividend on the
ex-date.

**Ex-dividend date** — The first day a buyer does *not* receive the upcoming dividend.

**Adjusted price** — Historical prices restated so splits and dividends don't appear as jumps.
Required for computing returns and signals.

**Raw / unadjusted price** — What actually traded. Required for order placement and position
accounting. **You need both series.**

**Total return** — Return including dividends, not just price change.

**Delisting** — A security stops trading. If your backtest universe excludes delisted companies,
you have survivorship bias.

---

## Performance metrics

**P&L** — Profit and loss. **Realised** = from closed positions. **Unrealised** = mark-to-market
on open positions. Risk limits must use both.

**Return** — Percentage change in value.

**CAGR** — Compound Annual Growth Rate. Return expressed as a per-year rate, so strategies over
different periods can be compared.

**Volatility** — Standard deviation of returns, annualised. Measures dispersion, not danger —
it treats gains and losses identically.

**Drawdown** — How far below your previous equity peak you currently are.

**Maximum drawdown (MDD)** — The worst peak-to-trough fall in the period. **Arguably the most
important number**, because it determines whether you can psychologically stay in the strategy.
Asymmetric: a 50% drawdown needs a 100% gain to recover.

**Underwater curve** — A chart of drawdown over time. Shows both depth and duration of pain.

**Recovery period** — How long it took to reach a new high after a drawdown.

**Sharpe ratio** — Excess return per unit of volatility. See [§2.3](phase-0-discovery.md#23-how-performance-is-judged)
for full treatment including its serious limitations.

**Sortino ratio** — Like Sharpe, but only penalises *downside* volatility.

**Calmar ratio** — CAGR ÷ maximum drawdown. Return per unit of maximum pain. Best single number
for judging whether a strategy is *runnable*.

**Deflated Sharpe Ratio (DSR)** — Sharpe corrected for how many strategies you tested. Essential
because testing many variants guarantees some look good by luck.

**Win rate** — Percentage of profitable trades. **Misleading alone** — many excellent strategies
win under 40% of the time.

**Profit factor** — Gross profit ÷ gross loss. Above 1.0 is profitable.

**Expectancy** — Average profit per trade: `(win rate × avg win) − (loss rate × avg loss)`. **The
number that actually matters.** Must be computed net of all costs.

**Exposure (time)** — What fraction of the time you held positions. Low exposure with good
returns implies capital efficiency.

**Turnover** — How much you trade relative to portfolio size. High turnover multiplies costs and
worsens tax treatment.

**Benchmark** — What you compare against, usually buy-and-hold of an index. If you can't beat it
after costs and taxes, the strategy isn't worth running.

**Alpha** — Return beyond what the benchmark explains. **Beta** — Sensitivity to the benchmark.

**VaR / CVaR** — Value at Risk: the loss threshold exceeded only X% of the time. CVaR: the
average loss *when* that threshold is breached.

---

## Research and validation

**Backtest** — Simulating a strategy on historical data. A **hypothesis**, never evidence.

**Paper trading** — Live data, live timing, simulated money. Finds integration bugs, not edges.

**Forward testing** — Testing on data after the strategy was designed. The only honest test.

**In-sample** — Data used to develop and tune. **Out-of-sample** — Data held back, untouched,
for final validation.

**Walk-forward analysis** — Repeatedly optimise on a window and test on the next, rolling
forward. Simulates what you'd actually have done.

**Cross-validation** — Splitting data to test generalisation. Standard k-fold is **wrong** for
time series — it trains on the future.

**Purging** — Removing training samples that overlap the test period. **Embargo** — Additionally
dropping a gap after each test window. Both prevent leakage.

**Overfitting** — Finding patterns in noise. The dominant failure mode in algorithmic trading.

**Look-ahead bias** — Using information not available at decision time.

**Survivorship bias** — Testing only on entities that survived to today.

**Data leakage** — Test-set information contaminating training.

**Monte Carlo simulation** — Resampling to get a *distribution* of outcomes rather than one path.

**Parameter sensitivity** — How performance varies across parameter values. A **plateau**
suggests a real edge; a **spike** suggests overfitting.

**Regime** — A persistent market state: trending, ranging, high-volatility, crisis.

**Regime change** — When market behaviour shifts and a previously working strategy stops working.

**Model drift / strategy degradation** — Performance decaying as conditions change or an edge
gets arbitraged away.

**Capacity** — How much money a strategy can absorb before its own trading destroys the edge.

**Multiple testing problem** — Testing many hypotheses guarantees false positives. Fixed by
counting attempts and applying corrections like DSR.

**Walk-forward efficiency** — Out-of-sample performance divided by in-sample performance. 0.1
means the tuning found noise; around 0.5 or better means some of the edge survived the
transition — which is a weaker claim than it sounds.

**Parameter stability** — How often the folds of a walk-forward agree on the best setting. Low
stability means there is no stable optimum, so whichever value a full-history fit lands on is
arbitrary. A strategy can post acceptable efficiency and still fail this.

**Plateau ratio** — A peak's neighbourhood score divided by its own score, with the peak itself
excluded from its neighbourhood. Near 1.0 is a plateau; near 0 is a spike; negative means the
peak's neighbours lose money.

**Robust best** — The parameter setting with the strongest *neighbourhood*, rather than the
highest single score. The one to deploy: on a genuine plateau it lands near the middle, and on a
spiky surface it refuses the spike.

**Block bootstrap** — Resampling *contiguous* stretches of returns rather than individual days,
so volatility clustering survives. A day-by-day shuffle breaks up the runs of bad days that
create drawdowns and therefore understates drawdown.

**Trial count** — How many distinct parameter sets were evaluated before a result was reported.
The denominator of the Deflated Sharpe Ratio. Only meaningful if counted by the machine — a
self-reported trial count is always 1.

**Ruin** — Not zero equity, but the drawdown at which a real operator stops. Much shallower, and
the number that actually ends a strategy's life.

---

## Strategy types

**Trend following** — Buy what's rising. Typically low win rate, large average wins.

**Mean reversion** — Buy what's fallen, expecting a bounce. High win rate, occasional large losses.

**Momentum** — Buy recent relative outperformers. Related to trend following but usually
cross-sectional (ranking many instruments).

**Breakout** — Enter when price exits a range.

**Pairs trading / statistical arbitrage** — Trade the spread between two historically related
instruments. Relies on **cointegration** — a statistical relationship that can and does break.

**Factor investing** — Selecting on characteristics (value, momentum, quality, size, low volatility).

**Market neutral** — Balanced long and short exposure, aiming to profit from relative moves.

**Rebalancing** — Periodically restoring target weights.

---

## Costs

**Commission** — Broker's per-trade or per-share fee.

**Exchange / regulatory fees** — Small per-trade charges. In India: STT, stamp duty, exchange
transaction charges, GST. In the US: SEC and TAF fees.

**Borrow cost** — What you pay to short a security. High for hard-to-borrow names.

**Financing cost** — Interest on margin.

**Cost drag** — Total costs as a percentage of returns. The gap between gross and net equity
curves. Often the entire difference between a "profitable" backtest and a losing strategy.

---

## Risk management

**Position sizing** — Deciding how much to buy. Matters more to outcomes than entry timing.

**Kelly criterion** — A mathematically optimal sizing formula. ⚠️ Extremely aggressive in
practice, highly sensitive to input errors, and can produce catastrophic drawdowns. Fractional
Kelly (¼ or ½) is what practitioners actually use, if they use it at all.

**Risk/reward ratio** — Expected gain vs expected loss on a trade.

**Stop-loss / take-profit / trailing stop** — See *Orders*.

**Maximum daily loss** — A hard limit that halts trading for the day. Must include unrealised P&L.

**Correlation** — How much two instruments move together (−1 to +1). Two 5% positions in highly
correlated names is really one 10% bet.

**Diversification** — Spreading risk across uncorrelated exposures. Note that correlations tend
to converge toward 1 during crises — exactly when diversification is most needed.

**Concentration risk** — Too much in one position, sector or factor.

**Kill switch** — An emergency stop. See [§13.4](phase-0-discovery.md#134-kill-switch--four-independent-layers).

**Fail closed** — When uncertain, refuse the action. The opposite (fail open) is how automated
systems cause damage.

---

## Regulatory

**PDT (Pattern Day Trader)** — US rule: >3 day trades in 5 business days in a margin account
requires $25,000 minimum equity. A hard constraint on intraday strategies for smaller accounts.

**Wash sale** — US rule disallowing a loss if you repurchase a substantially identical security
within 30 days.

**SEBI** — India's securities regulator. Its retail algo trading framework became mandatory for
all brokers on 1 April 2026.

**Algo-ID** — An exchange-assigned identifier that registered algos must attach to orders in
India, for traceability.

**Static IP whitelisting** — SEBI requirement that API orders originate from an IP registered
with your broker.

**STCG / LTCG** — Short-term / long-term capital gains, taxed at different rates depending on
holding period.

**STT (Securities Transaction Tax)** — An Indian per-transaction tax.

---

## Systems and engineering

**Point-in-time data** — Data as it was known at a past moment, not as revised since. Required
for honest backtesting.

**Bitemporal** — Storing both when something *happened* and when you *learned it*.

**Event sourcing** — Storing an immutable log of events and deriving state from it, rather than
mutating state in place. Gives audit trail, crash recovery and reproducibility.

**Audit log** — An append-only record of everything the system did. Hash-chained here, so
tampering is detectable.

**Structured logging** — Logging key-value data rather than formatted strings, so logs are
queryable.

**Idempotent** — Safe to repeat. See *Idempotency* under Orders.

**Fail closed** — See *Risk management*.

**Reconciliation** — See *Orders*.

**Walking skeleton** — A minimal end-to-end implementation built first, to surface integration
problems while they're still cheap to fix.

**Ports and adapters (hexagonal architecture)** — A pure domain core surrounded by swappable
I/O adapters. What makes one strategy run against a file, a simulator and a live broker.

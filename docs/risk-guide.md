# Risk Management

The risk engine is not a helper for strategies. It is an **adversarial component** whose job is
to assume every other part of the system is buggy — including the strategy, the data layer, and
the code that called it.

It is also the only thing that can mint a `RiskDecision`, and therefore the only route to an
`Order` existing at all.

---

## Three properties

**Fail closed.** A rule that raises is a *rejection*, not a skip. If the engine cannot establish
that a trade is safe, the trade does not happen. This is tested by feeding it a rule that throws
and asserting the order is refused.

**Everything is recorded.** Every rule's observed value and limit is kept whether it passed or
failed:

```
[PASS] RISK_007_max_position_weight: 0.1750 vs limit 0.25
[FAIL] RISK_010_max_gross_exposure: 0.9400 vs limit 0.80
```

That is what makes a rejection explainable months later, and what distinguishes a
well-calibrated limit from one quietly strangling a strategy.

**A halt outranks every rule.** Once a monitor has stopped trading, no combination of
individually-passing rules can produce an order.

---

## The bug this section exists to prevent

> A risk rule that stops you **closing** a position is more dangerous than no rule at all,
> because it traps you in the trade.

The first version of the exposure rules added an order's value to the existing position
regardless of direction. A sell that closed a long therefore read as *doubling* it:

```
position:  100 shares, 37.5% of equity
order:     SELL 100  (closes it entirely)
projected: 0.3796 vs limit 0.25   →  REJECTED
```

Every exit was rejected. The strategy opened one position and could never close it.

Rules now judge the book **after** the order, with direction:

```python
def projected_position_value(self) -> Decimal:
    return self.portfolio.market_value(instrument) + self.signed_order_value
```

A regression test asserts that closing an oversized position passes while *adding* to it still
fails.

---

## Pre-trade rules

Twenty-one, evaluated in order on every proposed trade. `trading risk-rules` prints them.

| Group | Rules |
|---|---|
| **Safety** | kill switch (filesystem sentinel), trading mode, market session, data staleness |
| **Sizing** | max/min order value, max position weight, limit-price sanity |
| **Exposure** | gross, net, **correlation-adjusted**, leverage, open positions, per-strategy allocation |
| **Execution** | buying power, liquidity vs ADV, symbol allow/denylist, duplicate order |
| **Compliance** | SEBI order rate, US pattern day trader, live connectivity |

A few worth explaining:

**Kill switch (`RISK_001`)** — a `KILL` file checked at the lowest level of the order path. An
in-app flag fails exactly when you need it, because the thing that is wrong *is* the app.

**Minimum order value (`RISK_006`)** — cost control, not pedantry. A flat ₹15 depository fee on
a ₹5,000 position is 0.3% per round trip. The conservative profile sets this to ₹10,000.

**Liquidity (`RISK_017`)** — order size against average daily volume. Above roughly 1% your own
order moves the price against you. It is also what stops a backtest "buying" ₹20 lakh of a stock
that trades ₹5 lakh a day and reporting a spectacular return.

**Limit-price sanity (`RISK_008`)** — a limit far from the market is usually a fat-fingered
decimal point or a stale reference price, producing an order that either never fills or fills
catastrophically.

---

## Correlation-adjusted exposure

Three strategies each taking a "reasonable" 20% position in correlated names are not
diversified — that is one 60% bet, and a limit that sums weights waves it through.

The adjustment is `sqrt(wᵀCw)`: the exposure of a single hypothetical asset carrying the same
risk as the whole book.

| Correlation | Two 10% positions | Reads as |
|---|---|---|
| +1.0 | 0.2000 | Nothing is diversified — one 20% bet |
| ~0.9 | 0.1973 | Barely diversified |
| 0.0 | 0.1438 | Genuinely diversified |
| −1.0 | 0.0000 | The positions offset |

**Without enough return history it assumes perfect correlation.** Assuming diversification you
have not measured is how a concentrated book passes an exposure check.

---

## Continuous monitors

Pre-trade rules judge one order. Monitors watch the **portfolio** and can stop trading entirely
— a different job, because what ruins an account is rarely one bad order. It is fifteen
reasonable-looking ones during a day that was going wrong.

| Monitor | Halts at | Level |
|---|---|---|
| `MON_001_daily_loss` | Day's loss, **mark-to-market including unrealised** | HARD |
| `MON_002_drawdown` | Distance below equity peak | HARD |
| `MON_003_consecutive_losses` | A run of losers | SOFT |
| `MON_004_error_rate` | Repeated failures | HARD |
| `MON_005_data_feed` | Stale or unmeasured feed age | HARD / SOFT |

The daily loss limit deliberately includes **unrealised** P&L. A realised-only limit is a hole
you can drive a portfolio through: hold every loser open and it never fires while the account
bleeds.

---

## Halt levels, and one deliberate omission

```
SOFT   stop opening; keep managing what is already held
HARD   cancel resting orders, stop everything, DO NOT liquidate
PANIC  flatten everything — manual confirmation only
```

**Auto-liquidation on a hard halt is excluded on purpose.** Force-closing every position during
the chaos that triggered the halt means selling into the worst liquidity of the day, and is
frequently worse than holding. Halt automatically; liquidate by human decision.

**A halt never clears itself.** Whatever tripped it needs a human to look, and an automatic
resume would just re-enter the situation that caused it. Releasing one is explicit and
attributed:

```python
engine.release_halt(acknowledged_by="rajeshwar")
```

### Drill it

```bash
uv run trading halt-drill
```

An undrilled kill switch is an assumption, not a control — which is why
`GATE_021_kill_switch_drilled` blocks promotion to paper trading until you have run this.

---

## Compliance is enforced, not documented

Regulatory constraints are risk rules, evaluated by the same engine as everything else.

**`COMP_001_order_rate`** — SEBI's 10 orders/second retail threshold. Staying below it is what
keeps a self-developed algo exempt from exchange registration, so crossing it is a hard stop
rather than a warning.

**`COMP_002_pattern_day_trader`** — >3 day trades in 5 business days needs $25,000. A capital
gate rather than an engineering one; the engine simply refuses the trade that would breach it.
Not applicable in India.

**`COMP_003_live_connectivity`** — SEBI requires API orders from a whitelisted static IP. Live
trading is not implemented, so this **refuses outright** rather than pretending the requirement
is satisfied.

---

## Profiles

Limits are versioned configuration, recorded in every run manifest.

```bash
uv run trading risk-profile configs/risk/conservative.yaml
uv run trading backtest --risk-profile-path configs/risk/conservative.yaml
```

| | `default` | `conservative` |
|---|---|---|
| Max position | 25% | 10% |
| Max gross exposure | 80% | 50% |
| Max correlated exposure | 60% | 35% |
| Max drawdown | 15% | 8% |
| Min order value | ₹1,000 | ₹10,000 |
| Max open positions | 10 | 5 |

> The same strategy under a 10% cap and a 25% cap is, in every way that matters, **two different
> strategies**. A run that does not record its limits is not reproducible — so the manifest
> carries the profile id, version and rule count.

---

## What is still missing

- **Broker-side kill switch (layer 4)** — API key revocation, the only layer that works when
  the process is unresponsive. Needs a broker: Phase 9.
- **Reconciliation drift and unexpected-position monitors** — need a real broker to reconcile
  against: Phase 7.
- **Strategy divergence from backtest** — needs the validation distribution: Phase 6.
- **Day-trade counting** is plumbed but not yet populated by the execution layer, so the PDT
  rule currently only enforces the equity threshold.

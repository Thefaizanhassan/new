# The Data Layer

> *"We do not build a strategy first and then look for data that makes it work. We understand,
> validate, and structure the data first."* — your Technology Standards §10

This is Phase 2. It is the least glamorous part of the platform and the one everything else
rests on: every bias, every phantom signal and every irreproducible result originates here.

---

## The shape of it

```
provider ──► normalise ──► VALIDATE ──► bitemporal store ──► adjust on read ──► strategy
(yfinance,    UTC, bar-      quarantine    raw prices,          splits and
 fixture,     close          on failure    immutable,           dividends,
 Kite later)  labelled,                    append-only          computed never
              canonical                                         written back
              columns
```

Four rules hold the whole thing together.

### 1. Raw prices are immutable; adjusted prices are computed on read

When a company splits 4:1, the raw price quarters overnight. A raw series shows a −75% crash
and a strategy fires on a non-event. Adjusted prices restate history as if the split had
always applied.

You need **both**, for different jobs — signals need adjusted prices so history is continuous,
order placement needs raw prices because you buy real shares at real prices. And critically:

> Adjusted history is **rewritten by every new corporate action.** If it were the stored truth,
> every past backtest would silently change. Keeping raw immutable and deriving adjusted on
> read means an old run stays reproducible.

```bash
trading actions --symbol RELIANCE --add-split 2024-10-28:4
trading actions --symbol RELIANCE          # nothing on disk was rewritten
```

### 2. Every bar carries when we learned it

Market data is not immutable. Prices get revised; adjusted history gets restated. So every row
stores both the time it *refers to* and the time we *ingested* it, and a read can ask for
either the latest view or an earlier one.

```python
store.read(RELIANCE)                                   # latest known
store.read(RELIANCE, as_of=datetime(2024, 7, 15, ...)) # what we knew that day
```

### 3. "As of" is two questions, not one

This distinction cost a debugging session and is worth stating plainly.

| Parameter | Question | Always available? |
|---|---|---|
| `as_of` | **The simulation clock.** At simulated time T, which corporate actions had happened? Depends on the action's ex-date. | ✅ Yes |
| `revision_as_of` | **The knowledge cutoff.** Which *version* of a bar did I hold? Depends on `ingested_at`. | ⚠️ Only once you have ingestion history |

If you backfill twenty years this morning, every bar carries today's `ingested_at` — so asking
what you knew in 2024 correctly returns **nothing**. Conflating the two silently empties every
backtest, which looks like a bug in the strategy. They are separate parameters for that reason.

### 4. Bad data is quarantined, never dropped

A silent drop creates a gap, and a gap makes a strategy skip a session it should have traded.
Failed chunks go to `data/quarantine/` with the reason attached, and stay inspectable.

---

## Trust travels with the data

Each provider declares a tier, the tier is **stored alongside the bars**, and it cannot be
laundered by passing through the store.

| Tier | Meaning | Can support a validated strategy? |
|---|---|---|
| `SYNTHETIC` | Generated. Deterministic and free; says nothing about any market. | ❌ |
| `PROTOTYPE` | Real data, no guarantees. No SLA, silent revisions, no point-in-time correctness. **yfinance is here.** | ❌ |
| `PRODUCTION` | A paid, contracted source with defined revision behaviour. | ✅ |

Mixing sources resolves to the **least** trustworthy tier present. `DataTier.
may_support_a_validated_strategy` returns `False` for the first two, so the limitation is
enforced in code rather than remembered.

---

## Storage

```
data/
├── bars/{provider}/{exchange}/{symbol}/{year}/{ingest_date}.parquet
├── corporate_actions/{exchange}/{symbol}.parquet
└── quarantine/{provider}/{exchange}/{symbol}_{timestamp}.parquet
```

**Parquet** because it is columnar and typically 5–10× smaller than CSV. **DuckDB** because it
queries those files in place — no server, no import step, and a bitemporal read is one window
function.

Files from earlier ingestion dates are never touched. Within a single date the file
accumulates, because a chunked backfill makes several writes for the same year on the same day.

> **A bug worth recording.** The first version named files `{year}/{ingest_date}.parquet` and
> *replaced* on write. A chunked backfill therefore silently discarded every chunk but the
> last — 1,640 rows became 130 with no error anywhere. Writes now merge, and a regression test
> asserts that two same-day writes for one year keep both.

---

## Backfilling

```bash
trading ingest --symbol RELIANCE --source yfinance --start 2015-01-01
trading catalog
```

Resumable, idempotent, and chunked. Progress is derived from **what is actually in the store**
rather than a separate checkpoint file, so there is no state to get out of sync — re-running is
always safe, and an interrupted backfill picks up where it stopped.

| Flag | Use |
|---|---|
| `--chunk-days N` | Smaller chunks for rate-limited providers |
| `--pause N` | Seconds between requests. Free endpoints rate-limit aggressively |
| `--force` | Re-fetch data you already hold, to pick up provider revisions. The old view survives |

One bad chunk is quarantined and the rest continue. One provider exception is recorded and the
rest continue. Neither loses the whole run.

---

## What still isn't here

- **Point-in-time index membership**, so survivorship bias is not yet defensible for
  universe-based strategies. Fixed universes only until then.
- **Automatic corporate action ingestion** — actions are recorded by hand via `trading actions`.
  A provider feed arrives with Kite Connect in Phase 9.
- **Intraday bars.** The store holds daily only, and says so rather than silently mishandling
  them.
- **Bid/ask and spread data**, which the backtest's fill model will want in Phase 5.

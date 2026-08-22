# Running this on your Mac

Everything below has been tested on the code in this repository. Commands you type are shown
after a `$`.

**Time to first green test run: about 5 minutes.** Nothing here costs money, needs a broker
account, or touches the internet beyond downloading packages.

---

## 1. What you need first

### Homebrew

Open **Terminal** (⌘-Space, type "Terminal") and check:

```
$ brew --version
```

If that errors, install it:

```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the "Next steps" it prints at the end — on Apple Silicon it asks you to add Homebrew to
your `PATH`, and skipping that is the most common reason the next command fails.

### uv — the Python manager

```
$ brew install uv
$ uv --version
```

**What uv is and why we use it.** `uv` installs Python itself, creates the isolated environment,
and installs packages — replacing `pyenv`, `venv`, `pip` and `poetry` with one tool. It is
roughly ten times faster than pip and writes a lockfile, so the exact versions that pass tests
here are the exact versions you get. You do not need Python installed already; uv fetches it.

> **You do not need Docker yet.** Phase 1 runs entirely on local files. Docker only becomes
> relevant when the Postgres ledger arrives — §5 covers it when you get there.

---

## 2. Get the code

```
$ git clone https://github.com/Thefaizanhassan/new.git ai-trading-platform
$ cd ai-trading-platform
$ git checkout claude/ai-algorithmic-trading-platform-a2fsmg
```

---

## 3. Install

```
$ uv sync --all-groups
```

This creates `.venv/` in the project folder, installs Python 3.12 if needed, and installs every
dependency including the dev tools. First run takes a minute or two; later runs are seconds.

**You never need to "activate" anything.** Prefix commands with `uv run` and it uses the right
environment automatically:

```
$ uv run python --version
Python 3.12.11
```

---

## 4. Verify it works

Run the full check suite — the same four commands CI runs:

```
$ uv run pytest -q                 # tests
$ uv run ruff check .              # lint
$ uv run ruff format --check .     # formatting
$ uv run mypy                      # types
```

Expected on a clean checkout:

```
...........................................................................  [100%]
All checks passed!
42 files already formatted
Success: no issues found in 30 source files
```

**If all four pass, your machine is set up correctly.** That is the whole of Phase 1
verification.

### Seeing the interesting parts

Watch the accounting stay exact where floats would not:

```
$ uv run pytest tests/property -v
```

Those are property tests — they generate hundreds of random fill sequences and assert the books
balance for every one. They are the tests that caught two real bugs in the position accounting
while it was being written.

Watch the safety gate refuse to be bypassed:

```
$ uv run pytest tests/unit/test_risk_gate.py -v
```

Every test there asserts that an order cannot come into existence without risk approval. If one
of those ever fails, the platform's central guarantee is broken.

Watch the data validator find planted faults:

```
$ uv run pytest tests/unit/test_validation.py -v
```

---

## 5. Postgres — not yet, but here's how when you need it

Phase 1 writes nothing to a database. From **Phase 7 (paper trading)** you'll want the Postgres
ledger, because by then three processes run at once.

Install **OrbStack** (lighter and faster than Docker Desktop on a Mac, and free for personal
use):

```
$ brew install --cask orbstack
```

Then, from the project folder:

```
$ docker compose up -d
$ docker compose ps          # should show trading-postgres as healthy
```

This runs Postgres on **port 5433**, not the default 5432, so it cannot collide with anything
else you have. To point the app at it, copy the example config and uncomment one line:

```
$ cp .env.example .env
```

then in `.env`:

```
TRADING_DATABASE_URL=postgresql+psycopg://trading:trading@localhost:5433/trading
```

To stop it: `docker compose down`. To delete its data entirely: `docker compose down -v`.

---

## 6. Secrets — read this before you ever add a broker key

**Never put a real API key in `.env`, in the database, or in any file in this repository.**
`.env` is git-ignored, which protects you from an accidental commit but not from a backup, a
screen share, or a stolen laptop.

Broker credentials go in the **macOS Keychain**, which is encrypted at rest and unlocked by your
login. From Phase 9:

```
$ uv run python -c "import keyring; keyring.set_password('trading', 'kite_api_key', 'PASTE_HERE')"
```

Three rules that matter more than the mechanism:

1. **Turn on FileVault** if it isn't already — System Settings → Privacy & Security → FileVault.
   Keychain protection assumes the disk is encrypted.
2. **Keep separate credentials for paper and live.** Never one key for both.
3. **Disable withdrawal permission on every trading API key.** Most API-key incidents drain funds
   through withdrawal rather than through bad trades. This is the highest-value security setting
   you will ever change, and it takes thirty seconds in your broker's dashboard.

---

## 7. India-specific setup — Phase 9 onward, not now

Recorded here so it isn't a surprise later. **None of it is needed for research or backtesting.**

| What | When | Cost |
|---|---|---|
| **Kite Connect subscription** — historical data + order API | Phase 9 | ₹500/month. Historical data is now included; it used to be a ₹2,000/month add-on |
| **Static IP** — SEBI requires API orders to originate from an IP whitelisted with your broker | Phase 12 (live only) | ₹1,500–6,000/year |
| **Indian-hosted server** — SEBI expects retail algos on Indian servers, and a laptop is not a compliant live host anyway | Phase 12 (live only) | ₹5,000–15,000/year |

> Your Mac is the **research and control station** — writing strategies, running backtests,
> reading dashboards. That is all it needs to be, possibly forever. The Mumbai host only enters
> the picture if and when a strategy earns the right to trade real money.

---

## 8. Everyday commands

| Command | Does |
|---|---|
| `uv run pytest -q` | Run all tests |
| `uv run pytest tests/unit -v` | Run one folder, verbosely |
| `uv run pytest -k position` | Run tests matching a name |
| `uv run pytest --cov=trading` | Tests with a coverage report |
| `uv run ruff check . --fix` | Lint and auto-fix |
| `uv run ruff format .` | Reformat everything |
| `uv run mypy` | Type-check |
| `uv sync --all-groups` | Reinstall after dependencies change |
| `uv run python` | A Python shell with everything importable |

Poking at it directly:

```
$ uv run python
>>> from trading.costs.india import IndiaDeliveryEquityCosts
>>> IndiaDeliveryEquityCosts().describe()
```

---

## 9. When something goes wrong

**`command not found: brew`** — Homebrew isn't on your `PATH`. On Apple Silicon:
`echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile` then open a new Terminal.

**`command not found: uv`** — same cause. Open a new Terminal window after installing.

**`uv sync` fails downloading packages** — usually a VPN or corporate proxy intercepting TLS.
Try it off the VPN first.

**Tests fail right after cloning** — run `uv sync --all-groups` again; you may have an
out-of-date environment. If they still fail, that's a genuine bug and worth reporting rather
than working around.

**`ModuleNotFoundError: No module named 'trading'`** — you ran `python` instead of
`uv run python`, so you got the system Python without the project installed.

**Anything involving `ta-lib` failing to build** — shouldn't happen. TA-Lib ships prebuilt
wheels for macOS arm64 on Python 3.12, verified 2026-08-22, so no Homebrew C library is needed.
If you somehow hit a source build, `brew install ta-lib` first, then `uv sync` again.

---

## 10. What actually exists right now

Phase 1 is the foundation, not a usable trading system. Honestly:

**Works today:** the domain core (exact money, FIFO-lot position accounting, the risk-approval
gate that makes orders unforgeable), the pandas data pipeline with the full validation gate,
Indian cost models for delivery and intraday, NSE/BSE market calendars, SEBI and US compliance
profiles, config with a hard refusal of `LIVE` mode, structured logging, and 75 tests.

**Does not exist yet:** loading real market data, indicators, strategies, the backtesting
engine, the risk engine itself (only its token type exists), paper trading, any broker
connection, and the dashboard.

**You cannot place a trade with this, on purpose.** `TRADING_MODE=LIVE` is refused at startup
with an explanatory error, and will stay refused until Phase 12.

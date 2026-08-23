"""Command line interface."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from trading.backtest.canary import run_leakage_canary
from trading.backtest.fills import (
    FixedBpsSlippage,
    NoSlippage,
    RealisticFillModel,
    SpreadSlippage,
    SquareRootImpact,
)
from trading.backtest.reports import build_report
from trading.calendars.base import calendar_for
from trading.config.compliance import profile_for
from trading.config.settings import Settings
from trading.core.instrument import EXCHANGES, Instrument, InstrumentId
from trading.core.types import InstrumentClass, Money, TradingMode
from trading.costs.india import IndiaDeliveryEquityCosts
from trading.data.backfill import BackfillService
from trading.data.corporate_actions import (
    ActionType,
    AdjustmentMode,
    CorporateAction,
    CorporateActionStore,
)
from trading.data.fixture import FixtureProvider
from trading.data.provider import DataTier, HistoricalDataProvider
from trading.data.store import ParquetBarStore
from trading.data.store_provider import StoreBackedProvider
from trading.data.validation import Severity, validate_ohlcv
from trading.engine.runner import RunResult, WalkingSkeletonRunner
from trading.features import indicators as ind
from trading.features.engine import verify_causality
from trading.observability.logging import configure_logging
from trading.risk.engine import RiskEngine
from trading.risk.monitors import HaltLevel
from trading.risk.profiles import RiskLimits, load_risk_profile
from trading.strategies import expressions
from trading.strategies.base import LifecycleStatus
from trading.strategies.builtin import BuyAndHold, SmaCross
from trading.strategies.config_strategy import load_strategy
from trading.strategies.lifecycle import Evidence, evaluate_promotion
from trading.strategies.registry import default_registry

app = typer.Typer(add_completion=False, help="AI algorithmic trading platform")
console = Console()

_STRATEGIES = ("buy_and_hold", "sma_cross")


def _instrument(symbol: str) -> Instrument:
    iid = InstrumentId.parse(symbol if ":" in symbol else f"NSE:{symbol}")
    exchange = EXCHANGES[iid.exchange]
    return Instrument(
        id=iid,
        name=iid.symbol,
        instrument_class=InstrumentClass.EQUITY,
        currency=exchange.currency,
    )


def _provider(source: str, *, adjustment: str = "SPLIT_ONLY") -> HistoricalDataProvider:
    if source == "store":
        settings = Settings()
        return StoreBackedProvider(
            ParquetBarStore(settings.data_dir),
            action_store=CorporateActionStore(settings.data_dir),
            adjustment=AdjustmentMode(adjustment),
        )
    if source == "fixture":
        return FixtureProvider()
    if source == "yfinance":
        # Deferred so a `fixture` run never needs yfinance installed or reachable.
        from trading.data.yfinance_provider import YFinanceProvider  # noqa: PLC0415

        return YFinanceProvider()
    raise typer.BadParameter(f"Unknown source {source!r}. Use 'store', 'fixture' or 'yfinance'.")


@app.command()
def status() -> None:
    """Show how this instance is configured."""
    settings = Settings()
    table = Table(title="Configuration", show_header=False, title_justify="left")
    table.add_column(style="bold cyan")
    table.add_column()
    for key, value in settings.describe().items():
        table.add_row(key, str(value))
    console.print(table)

    profile = settings.compliance
    console.print(f"\n[bold]Compliance — {profile.regulator} ({profile.version})[/bold]")
    for note in profile.notes:
        console.print(f"  • {note}")
    console.print(
        "\n[yellow]Live trading is not implemented. TRADING_MODE=LIVE is refused "
        "at startup and stays refused until Phase 12.[/yellow]"
    )


@app.command()
def indicators(name: Annotated[str, typer.Option(help="One indicator, or all")] = "") -> None:
    """Document the available technical indicators."""
    console.print(ind.describe(name or None))


@app.command()
def check_data(
    symbol: str = "RELIANCE",
    source: str = "fixture",
    start: str = "2024-01-01",
    end: str = "2026-08-21",
) -> None:
    """Load bars and run the full validation gate — understand data before trusting it."""
    instrument = _instrument(symbol)
    provider = _provider(source)
    start_date, end_date = dt.date.fromisoformat(start), dt.date.fromisoformat(end)

    console.print(
        f"[bold]{instrument.id}[/bold] via [cyan]{provider.info.provider_id}[/cyan] "
        f"(tier: {provider.info.tier})"
    )
    if provider.info.tier is not DataTier.PRODUCTION:
        for caveat in provider.info.caveats:
            console.print(f"  [yellow]![/yellow] {caveat}")

    bars = provider.get_bars(instrument.id, start_date, end_date)
    if bars.empty:
        console.print("[red]No data returned.[/red]")
        raise typer.Exit(1)

    calendar = calendar_for(EXCHANGES[instrument.id.exchange].calendar_code)
    report = validate_ohlcv(
        bars,
        symbol=str(instrument.id),
        expected_sessions=calendar.sessions_between(start_date, end_date),
    )
    console.print(f"\n{report.summary()}")
    colours = {Severity.ERROR: "red", Severity.WARNING: "yellow", Severity.INFO: "dim"}
    for issue in report.issues:
        console.print(f"  [{colours[issue.severity]}]{issue}[/{colours[issue.severity]}]")
    if not report.is_usable:
        raise typer.Exit(1)


@app.command()
def ingest(
    symbol: str = "RELIANCE",
    source: str = "fixture",
    start: str = "2015-01-01",
    end: str = "2026-08-21",
    chunk_days: int = 365,
    pause: float = 0.0,
    force: bool = False,
) -> None:
    """Backfill bars into the local store. Resumable and safe to re-run."""
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)

    instrument = _instrument(symbol)
    provider = _provider(source)
    store = ParquetBarStore(settings.data_dir)

    console.print(
        f"Backfilling [bold]{instrument.id}[/bold] from [cyan]{provider.info.provider_id}[/cyan] "
        f"(tier: {provider.info.tier})"
    )
    report = BackfillService(
        provider, store, chunk_days=chunk_days, request_pause_seconds=pause
    ).backfill(
        instrument.id,
        dt.date.fromisoformat(start),
        dt.date.fromisoformat(end),
        force=force,
    )

    console.print(f"\n{report.summary()}")
    for chunk in report.chunks:
        colour = {"written": "green", "quarantined": "red", "error": "red"}.get(chunk.status, "dim")
        console.print(
            f"  [{colour}]{chunk.start} → {chunk.end}  {chunk.status:<16}"
            f"{chunk.rows:>5} rows[/{colour}] {chunk.detail[:70]}"
        )

    coverage = store.coverage(instrument.id)
    if coverage:
        console.print(
            f"\n[bold]Store now holds[/bold] {coverage.rows} bars, "
            f"{coverage.first_timestamp:%Y-%m-%d} to {coverage.last_timestamp:%Y-%m-%d}, "
            f"across {coverage.revisions} revision(s)"
        )
    if report.quarantined:
        raise typer.Exit(1)


@app.command()
def catalog() -> None:
    """What data is actually in the local store."""
    frame = ParquetBarStore(Settings().data_dir).catalog()
    if frame.empty:
        console.print("[yellow]Store is empty. Run `trading ingest` first.[/yellow]")
        return

    table = Table(title="Local data catalog", title_justify="left")
    for column in ("Instrument", "Provider", "Bars", "From", "To", "Revisions"):
        table.add_column(column, justify="right" if column in ("Bars", "Revisions") else "left")
    for _, row in frame.iterrows():
        table.add_row(
            str(row["instrument_id"]),
            str(row["provider"]),
            f"{int(row['rows']):,}",
            f"{pd.Timestamp(row['first']):%Y-%m-%d}",
            f"{pd.Timestamp(row['last']):%Y-%m-%d}",
            str(int(row["revisions"])),
        )
    console.print(table)


@app.command()
def actions(
    symbol: str = "RELIANCE",
    add_split: str = "",
    add_dividend: str = "",
) -> None:
    """List or record corporate actions.

    Examples:
        trading actions --symbol RELIANCE
        trading actions --symbol RELIANCE --add-split 2024-10-28:4
        trading actions --symbol RELIANCE --add-dividend 2025-08-14:10.50
    """
    instrument = _instrument(symbol)
    store = CorporateActionStore(Settings().data_dir)
    existing = store.load(instrument.id)

    if add_split or add_dividend:
        raw = add_split or add_dividend
        try:
            date_part, _, value = raw.partition(":")
            ex_date = dt.date.fromisoformat(date_part)
            value_decimal = Decimal(value)
        except (ValueError, ArithmeticError) as exc:
            raise typer.BadParameter(f"Expected YYYY-MM-DD:VALUE, got {raw!r}") from exc

        action = (
            CorporateAction(
                instrument.id, ActionType.SPLIT, ex_date, ratio=value_decimal, source="manual"
            )
            if add_split
            else CorporateAction(
                instrument.id, ActionType.DIVIDEND, ex_date, amount=value_decimal, source="manual"
            )
        )
        existing = sorted([*existing, action], key=lambda a: a.ex_date)
        store.save(instrument.id, existing)
        console.print(f"[green]Recorded[/green] {action}")

    if not existing:
        console.print(f"No corporate actions recorded for {instrument.id}.")
        return

    console.print(f"[bold]{instrument.id}[/bold] — {len(existing)} action(s)")
    for action in existing:
        console.print(f"  {action}  [dim]({action.source})[/dim]")
    console.print(
        "\n[dim]Raw prices are stored immutably; adjusted series are computed on read, "
        "so recording an action never rewrites history on disk.[/dim]"
    )


@app.command()
def strategies(config_dir: str = "configs/strategies") -> None:
    """List every registered strategy, from Python and from YAML."""
    registry = default_registry(config_dir)
    if not len(registry):
        console.print("[yellow]No strategies registered.[/yellow]")
        return

    table = Table(title=f"{len(registry)} registered strategies", title_justify="left")
    for column in ("ID", "Ver", "Status", "Universe", "Hash", "Origin"):
        table.add_column(column)
    for row in registry.summary():
        table.add_row(
            row["id"],
            row["version"],
            row["status"],
            row["universe"],
            row["content_hash"],
            row["origin"].replace("configs/strategies/", ""),
        )
    console.print(table)
    console.print(
        "\n[dim]Hash covers parameters, universe and timeframe. Change any of them and "
        "you have a different strategy, with a different backtest identity.[/dim]"
    )


@app.command()
def strategy_language() -> None:
    """What a YAML strategy rule may contain — and what it deliberately may not."""
    console.print(expressions.describe_language())


@app.command()
def validate_strategy(path: str, source: str = "fixture", symbol: str = "") -> None:
    """Check a YAML strategy: does it compile, and are its rules causal?"""
    try:
        strategy = load_strategy(path)
    except Exception as exc:
        console.print(f"[red]Invalid:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Compiles[/green] — {strategy.spec.name} v{strategy.spec.version}")
    console.print(f"  content hash : {strategy.spec.content_hash}")
    console.print(f"  entry rule   : {strategy.config.entry.when}")
    console.print(f"  exit rule    : {strategy.config.exit.when}")

    instrument = _instrument(symbol or str(strategy.spec.universe[0]))
    bars = _provider(source).get_bars(instrument.id, dt.date(2020, 1, 1), dt.date(2026, 8, 21))
    if bars.empty:
        console.print("[yellow]No bars available, so rules were not exercised.[/yellow]")
        return

    console.print("\n[bold]Causality[/bold] [dim](does a rule read the future?)[/dim]")
    all_causal = True
    for spec in strategy.feature_specs():
        causal = verify_causality(spec, bars)
        all_causal &= causal
        mark = "[green]PASS[/green]" if causal else "[red]FAIL — reads the future[/red]"
        console.print(f"  {spec.name:<8} {mark}")

    fired = strategy.explain(bars)
    console.print(
        f"\n[bold]Over {len(bars)} bars[/bold]: entry fired {int(fired['entry'].sum())}x, "
        f"exit fired {int(fired['exit'].sum())}x"
    )
    if int(fired["entry"].sum()) == 0:
        console.print(
            "[yellow]The entry rule never fired. Check it against "
            "`trading strategy-language` — the rolling_max trap is a common cause.[/yellow]"
        )
    if not all_causal:
        raise typer.Exit(1)


@app.command()
def lifecycle(
    strategy_id: str = "sma_cross",
    to_status: str = "PROMISING",
    from_status: str = "BACKTESTING",
    data_tier: str = "PROTOTYPE",
) -> None:
    """Ask whether a strategy may be promoted, and see exactly what blocks it."""
    decision = evaluate_promotion(
        strategy_id,
        LifecycleStatus(from_status),
        LifecycleStatus(to_status),
        Evidence(data_tier=DataTier(data_tier)),
    )
    colour = "green" if decision.approved else "red"
    console.print(f"[{colour}]{decision.summary()}[/{colour}]\n")
    for result in decision.results:
        tone = {"PASS": "green", "FAIL": "red", "UNAVAILABLE": "yellow"}[result.status]
        console.print(f"  [{tone}]{result}[/{tone}]")
    console.print(
        "\n[dim]UNAVAILABLE blocks promotion. A gate that passes because nobody "
        "implemented its check manufactures confidence.[/dim]"
    )


@app.command()
def canary(
    symbol: str = "RELIANCE",
    strategy: str = "sma_cross",
    source: str = "fixture",
    start: str = "2015-01-01",
    end: str = "2026-08-21",
    trials: int = 10,
) -> None:
    """Run the leakage canary: does the strategy profit on structureless data?"""
    settings = Settings()
    configure_logging("WARNING", settings.log_format)

    instrument = _instrument(symbol)
    provider = _provider(source)
    start_date, end_date = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    bars = provider.get_bars(instrument.id, start_date, end_date)
    if bars.empty:
        console.print("[red]No bars available.[/red]")
        raise typer.Exit(1)

    compliance = profile_for(instrument.market)

    def run_on(frame: pd.DataFrame) -> float:
        class Fixed:
            info = provider.info

            def get_bars(self, *args: object, **kwargs: object) -> pd.DataFrame:
                return frame

        outcome = WalkingSkeletonRunner(
            provider=Fixed(),
            instrument=instrument,
            strategy=(
                default_registry().build(strategy)
                if strategy not in _STRATEGIES
                else SmaCross(instrument.id, 50, 200, Decimal("0.20"))
            ),
            cost_model=IndiaDeliveryEquityCosts(),
            risk_engine=RiskEngine(RiskLimits(), compliance, kill_switch_path=Path("KILL")),
            compliance=compliance,
            starting_capital=Money(Decimal("400000"), instrument.currency),
            mode=TradingMode.BACKTEST,
        ).run(start_date, end_date)
        return float(outcome.metrics.get("net_return", 0))

    console.print(f"Shuffling {len(bars)} bars {trials} times — this takes a moment…")
    result = run_leakage_canary(run_on, bars, trials=trials)

    colour = "red" if result.leaking else ("yellow" if result.implausible_magnitude else "green")
    console.print(f"\n[{colour}]{result.summary()}[/{colour}]")
    console.print(
        "\n[dim]This proves a class of leak is absent, not that the strategy is correct. "
        "A leak whose edge depends on real autocorrelation can evade it — see "
        "docs/backtesting-guide.md.[/dim]"
    )


@app.command()
def risk_rules(profile: str = "") -> None:
    """Every pre-trade rule, in evaluation order, with what each one is for."""
    limits = load_risk_profile(profile) if profile else RiskLimits()
    engine = RiskEngine(limits, profile_for(Settings().market), kill_switch_path=Path("KILL"))

    table = Table(
        title=f"{len(engine.rules)} pre-trade rules — profile '{limits.profile_id}'",
        title_justify="left",
    )
    table.add_column("Rule", style="bold cyan")
    table.add_column("Purpose")
    for rule in engine.describe_rules():
        table.add_row(rule["rule_id"], rule["doc"] or "—")
    console.print(table)

    monitors = Table(title="Continuous monitors", title_justify="left")
    monitors.add_column("Monitor", style="bold cyan")
    monitors.add_column("Halts at")
    for monitor in engine.monitors.monitors:
        doc = (type(monitor).__doc__ or "").strip().split("\n")[0]
        monitors.add_row(monitor.monitor_id, doc or "—")
    console.print()
    console.print(monitors)
    console.print(
        "\n[dim]A rule that raises is a rejection, never a skip. A halt outranks "
        "every rule and never clears itself.[/dim]"
    )


@app.command()
def risk_profile(path: str = "configs/risk/default.yaml") -> None:
    """Show a risk profile and what each limit is protecting against."""
    limits = load_risk_profile(path)
    table = Table(
        title=f"Risk profile '{limits.profile_id}' v{limits.version}", title_justify="left"
    )
    table.add_column("Limit", style="bold cyan")
    table.add_column("Value", justify="right")
    for key, value in limits.model_dump().items():
        if key in ("profile_id", "version") or value in ((), None):
            continue
        table.add_row(key, str(value))
    console.print(table)
    console.print(
        "\n[dim]Recorded in every run manifest. The same strategy under a 10% and a "
        "25% position cap is two different strategies.[/dim]"
    )


@app.command()
def halt_drill(profile: str = "configs/risk/default.yaml") -> None:
    """Exercise the halt path. An undrilled kill switch is an assumption, not a control."""
    limits = load_risk_profile(profile)
    engine = RiskEngine(limits, profile_for(Settings().market), kill_switch_path=Path("KILL"))
    now = dt.datetime.now(dt.UTC)

    console.print(f"Initial state: [green]{engine.halt.summary()}[/green]")
    engine.engage_halt(HaltLevel.HARD, "drill: operator-initiated", now)
    console.print(f"After engage:  [red]{engine.halt.summary()}[/red]")
    console.print(f"  blocks new positions : {engine.halt.level.blocks_new_positions}")
    console.print(f"  blocks everything    : {engine.halt.level.blocks_everything}")
    console.print(
        "  [yellow]does NOT auto-liquidate[/yellow] — force-closing into the chaos that "
        "caused the halt is frequently worse than holding"
    )
    engine.release_halt(acknowledged_by="halt-drill")
    console.print(f"After release: [green]{engine.halt.summary()}[/green]")
    console.print(
        "\n[dim]A halt never clears itself: whatever tripped it needs a human to look, "
        "and an automatic resume would re-enter the situation that caused it.[/dim]"
    )


@app.command()
def backtest(
    symbol: str = "RELIANCE",
    strategy: str = "sma_cross",
    source: str = "store",
    start: str = "2022-01-01",
    end: str = "2026-08-21",
    capital: float = 400_000,
    weight: float = 0.20,
    show_decisions: int = 10,
    adjustment: str = "SPLIT_ONLY",
    risk_profile_path: str = "",
    slippage: str = "spread",
    max_participation: float = 0.05,
    trials: int = 1,
    full_report: bool = False,
) -> None:
    """Run the walking skeleton end to end.

    ``--source store`` reads the local bitemporal store with corporate actions
    applied, which is the path a real backtest takes. ``fixture`` and
    ``yfinance`` bypass the store and go straight to the provider.
    """
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)

    instrument = _instrument(symbol)
    target = Decimal(str(weight))
    if strategy == "buy_and_hold":
        strat: Any = BuyAndHold(instrument.id, target_weight=target)
    elif strategy == "sma_cross":
        strat = SmaCross(instrument.id, fast=50, slow=200, target_weight=target)
    else:
        # Anything else is looked up in the registry, so a YAML strategy runs
        # through exactly the same path as a Python one.
        try:
            strat = default_registry().build(strategy)
        except KeyError as exc:
            raise typer.BadParameter(
                f"Unknown strategy {strategy!r}. Built in: {_STRATEGIES}. "
                f"Run `trading strategies` to see what is registered."
            ) from exc
    compliance = profile_for(instrument.market)

    result = WalkingSkeletonRunner(
        provider=_provider(source, adjustment=adjustment),
        instrument=instrument,
        strategy=strat,
        cost_model=IndiaDeliveryEquityCosts(),
        risk_engine=RiskEngine(
            load_risk_profile(risk_profile_path) if risk_profile_path else RiskLimits(),
            compliance,
            kill_switch_path=Path("KILL"),
        ),
        compliance=compliance,
        starting_capital=Money(Decimal(str(capital)), instrument.currency),
        mode=TradingMode.BACKTEST,
        fill_model=RealisticFillModel(
            slippage_model=_slippage(slippage),
            max_participation=Decimal(str(max_participation)),
        ),
        trials=trials,
    ).run(dt.date.fromisoformat(start), dt.date.fromisoformat(end))

    _render(result, strat.spec.name, show_decisions)
    if full_report:
        _render_full(result)


_SLIPPAGE_MODELS = {
    "none": NoSlippage,
    "fixed": FixedBpsSlippage,
    "spread": SpreadSlippage,
    "impact": SquareRootImpact,
}


def _slippage(name: str) -> Any:
    try:
        return _SLIPPAGE_MODELS[name]()
    except KeyError as exc:
        raise typer.BadParameter(
            f"Unknown slippage model {name!r}. Choose from {sorted(_SLIPPAGE_MODELS)}."
        ) from exc


def _render_full(result: RunResult) -> None:
    """The full metric set, plus the flags worth looking at before believing it."""
    report = build_report(result.metrics, result.equity_curve, result.trades, result.manifest)

    table = Table(title="Performance", title_justify="left")
    table.add_column("Section", style="dim")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    last_section = ""
    for section, label, value in report.summary_lines():
        table.add_row(section if section != last_section else "", label, value)
        last_section = section
    console.print()
    console.print(table)

    if not report.monthly_returns.empty:
        monthly = Table(
            title="Monthly returns (%)", title_justify="left", padding=(0, 1), box=box.SIMPLE
        )
        monthly.add_column("Yr", style="bold cyan")
        for month in range(1, 13):
            monthly.add_column(f"{month:02d}", justify="right")
        for year, row in report.monthly_returns.iterrows():
            cells = []
            for month in range(1, 13):
                value = row.get(month)
                if value is None or pd.isna(value):
                    cells.append("·")
                else:
                    colour = "green" if value >= 0 else "red"
                    cells.append(f"[{colour}]{value * 100:+.1f}[/{colour}]")
            monthly.add_row(str(year), *cells)
        console.print()
        console.print(monthly)

    flags = report.verdict()
    console.print()
    if flags:
        console.print("[bold yellow]Worth looking at before believing this[/bold yellow]")
        for flag in flags:
            console.print(f"  [yellow]•[/yellow] {flag}")
    else:
        console.print("[green]No automated flags raised.[/green]")
    console.print(
        "\n[dim]Flags are prompts to look harder, not conclusions. Every one has an "
        "innocent explanation; several together rarely do.[/dim]"
    )


def _render(result: RunResult, strategy_name: str, show_decisions: int) -> None:
    console.print()
    summary = Table(title=f"{strategy_name} — walking skeleton", title_justify="left")
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Starting capital", str(result.starting_capital.settled()))
    summary.add_row("Final equity", str(result.final_equity.settled()))
    summary.add_row("Gross return", f"{result.gross_return:+.2%}")
    summary.add_row("Net return", f"{result.net_return:+.2%}")
    summary.add_row("Total costs", str(result.total_costs.settled()))
    summary.add_row("Cost drag", f"{result.cost_drag:.2%}")
    summary.add_row("Fills", str(len(result.fills)))
    summary.add_row("Sharpe", f"{result.metrics.get('sharpe', 0):.2f}")
    summary.add_row("Max drawdown", f"{result.metrics.get('max_drawdown', 0):.2%}")
    summary.add_row(
        "Deflated Sharpe",
        f"{result.metrics.get('deflated_sharpe', 0):.3f} "
        f"({result.metrics.get('trials', 1)} trial(s) claimed)",
    )
    summary.add_row("Unfilled / partial", str(len(result.unfilled)))
    summary.add_row("Risk rejections", str(len(result.rejections)))
    summary.add_row(
        "Halt", result.halt_summary if not result.halted else f"[red]{result.halt_summary}[/red]"
    )
    console.print(summary)

    if show_decisions:
        console.print(f"\n[bold]Last {show_decisions} decisions[/bold]")
        for record in result.decisions[-show_decisions:]:
            console.print(
                f"  [dim]{record.timestamp:%Y-%m-%d}[/dim] "
                f"[bold]{record.kind:<14}[/bold] {record.detail}"
            )

    console.print("\n[bold]Run manifest[/bold] [dim](provenance — makes this reproducible)[/dim]")
    for key, value in result.manifest.items():
        console.print(f"  [cyan]{key:<22}[/cyan] {value}")

    console.print(
        "\n[yellow]This is the Phase 1 walking skeleton, not a backtest you "
        "should trust.[/yellow]\n[dim]No slippage model, no spread, no partial fills, "
        "no volume cap, and no out-of-sample validation. Those arrive in Phases 5 "
        "and 6.[/dim]"
    )


if __name__ == "__main__":
    app()

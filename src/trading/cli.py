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
from trading.strategies.lifecycle import (
    CriterionStatus,
    Evidence,
    evaluate_promotion,
    evidence_from_validation,
)
from trading.strategies.registry import default_registry
from trading.validation.engine_adapter import EngineEvaluator
from trading.validation.experiments import ExperimentLedger
from trading.validation.harness import Objective
from trading.validation.montecarlo import (
    MonteCarloResult,
    block_bootstrap_returns,
    resample_trades,
)
from trading.validation.purged_cv import PurgedCvConfig, PurgedCvResult, run_purged_cv
from trading.validation.report import ValidationReport
from trading.validation.sensitivity import GridPoint, ParameterGrid, SensitivitySurface
from trading.validation.splits import Window, window_from
from trading.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    run_walk_forward,
)

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


# ── validation (Phase 6) ────────────────────────────────────────────────────
def _evaluator(
    *,
    symbol: str,
    config: str,
    source: str,
    capital: float,
    adjustment: str,
    slippage: str,
    objective: str,
    min_trades: int,
    ledger_path: str,
    kind: str,
) -> tuple[EngineEvaluator, ExperimentLedger]:
    """Assemble the engine-backed evaluator every validation command shares."""
    path = Path(config)
    if not path.exists():
        raise typer.BadParameter(
            f"{config} does not exist. Validation sweeps need a parameterised YAML "
            f"strategy — see configs/strategies/sma_cross_param.yaml."
        )
    instrument = _instrument(symbol)
    ledger = ExperimentLedger(ledger_path)
    return (
        EngineEvaluator(
            provider=_provider(source, adjustment=adjustment),
            instrument=instrument,
            build_strategy=lambda p: load_strategy(path, overrides=p),
            cost_model=IndiaDeliveryEquityCosts(),
            starting_capital=Money(Decimal(str(capital)), instrument.currency),
            fill_model=RealisticFillModel(slippage_model=_slippage(slippage)),
            objective=Objective(metric=objective, min_trades=min_trades),
            ledger=ledger,
            ledger_kind=kind,
        ),
        ledger,
    )


def _parse_axis(spec: str) -> list[float]:
    """Parse ``10,20,30`` or ``10:60:10`` (start:stop:step, stop inclusive)."""
    text = spec.strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise typer.BadParameter(f"{spec!r} must be start:stop:step or a comma list")
        start, stop, step = (Decimal(p) for p in parts)
        if step <= 0:
            raise typer.BadParameter(f"{spec!r} has a non-positive step")
        values: list[float] = []
        current = start
        while current <= stop:
            values.append(float(current))
            current += step
        return values
    return [float(Decimal(p)) for p in text.split(",") if p.strip()]


def _grid_from(specs: list[str]) -> ParameterGrid:
    """Build a grid from repeated ``--axis name=values`` options."""
    axes: dict[str, list[int | float | Decimal]] = {}
    for spec in specs:
        if "=" not in spec:
            raise typer.BadParameter(f"{spec!r} must be name=values, e.g. fast=10,20,30")
        name, values = spec.split("=", 1)
        parsed = _parse_axis(values)
        # Integral values become ints so an indicator period is a period, not 20.0.
        axes[name.strip()] = [int(v) if float(v).is_integer() else v for v in parsed]
    return ParameterGrid(axes)


def _warmup_for(params: dict[str, Any], floor: int) -> int:
    """Derive warmup from the largest period in the parameter set.

    A sweep that raises an indicator period above the configured warmup leaves the
    strategy evaluating a NaN average: no trades, and a flat surface that reads as
    a perfect plateau. Deriving warmup from the parameters removes that trap
    instead of documenting it.
    """
    periods = [int(v) for v in params.values() if isinstance(v, int) and v > 1]
    return max([floor, *(p + 1 for p in periods)])


def _expand(grid: ParameterGrid, warmup_floor: int) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for params in grid.points():
        point = dict(params)
        point["warmup_bars"] = _warmup_for(point, warmup_floor)
        points.append(point)
    return points


def _render_rows(rows: list[tuple[str, str, str]], title: str) -> None:
    table = Table(title=title, title_justify="left", box=box.SIMPLE)
    table.add_column("Section", style="dim")
    table.add_column("Metric", style="bold cyan")
    table.add_column("Value", justify="right")
    last = ""
    for section, label, value in rows:
        table.add_row(section if section != last else "", label, value)
        last = section
    console.print(table)


@app.command()
def walk_forward(
    symbol: str = "RELIANCE",
    config: str = "configs/strategies/sma_cross_param.yaml",
    axis: Annotated[list[str] | None, typer.Option(help="name=values, repeatable")] = None,
    source: str = "fixture",
    start: str = "2015-01-01",
    end: str = "2026-08-21",
    capital: float = 400_000,
    train_bars: int = 756,
    test_bars: int = 252,
    anchored: bool = False,
    objective: str = "sharpe",
    min_trades: int = 2,
    slippage: str = "spread",
    adjustment: str = "SPLIT_ONLY",
    ledger_path: str = "data/experiments.sqlite",
) -> None:
    """Optimise on each training window, test on the window that follows it.

    Only the concatenated test segments are quotable; the in-sample column is
    there to be compared against, not reported.
    """
    settings = Settings()
    configure_logging("WARNING", settings.log_format)

    grid = _grid_from(axis or ["fast=20,50", "slow=100,200"])
    evaluator, ledger = _evaluator(
        symbol=symbol,
        config=config,
        source=source,
        capital=capital,
        adjustment=adjustment,
        slippage=slippage,
        objective=objective,
        min_trades=min_trades,
        ledger_path=ledger_path,
        kind="walk_forward",
    )
    index = evaluator.load(dt.date.fromisoformat(start), dt.date.fromisoformat(end))
    points = _expand(grid, warmup_floor=2)
    console.print(
        f"{len(index)} bars, {len(points)} parameter sets, "
        f"train {train_bars} / test {test_bars} — this takes a moment…"
    )

    result = run_walk_forward(
        evaluator,
        index,
        points,
        WalkForwardConfig(
            train_bars=train_bars,
            test_bars=test_bars,
            warmup_bars=max(_warmup_for(p, 2) for p in points),
            anchored=anchored,
            objective=Objective(metric=objective, min_trades=min_trades),
        ),
    )

    console.print()
    for fold in result.folds:
        colour = "green" if fold.scored else "yellow"
        console.print(f"  [{colour}]{fold.label()}[/{colour}]")
    console.print()
    _render_rows(
        [("Walk-forward", label, value) for label, value in result.summary_lines()],
        f"Walk-forward — {config} on {symbol}",
    )
    colour = "green" if result.passed else "red"
    console.print(f"[{colour}]{result.verdict()}[/{colour}]")
    console.print(
        f"[dim]Configuration: {result.config.describe()}\n"
        f"Ledger now holds {ledger.trial_count(Path(config).stem)} distinct trials "
        f"for this strategy.[/dim]"
    )


@app.command()
def sensitivity(
    symbol: str = "RELIANCE",
    config: str = "configs/strategies/sma_cross_param.yaml",
    axis: Annotated[list[str] | None, typer.Option(help="name=values, repeatable")] = None,
    source: str = "fixture",
    start: str = "2015-01-01",
    end: str = "2022-12-31",
    capital: float = 400_000,
    objective: str = "sharpe",
    min_trades: int = 2,
    slippage: str = "spread",
    adjustment: str = "SPLIT_ONLY",
    ledger_path: str = "data/experiments.sqlite",
) -> None:
    """Sweep the parameters and report whether the surface is a plateau or a spike.

    Run this on the *training* period. Sweeping the out-of-sample window and
    reporting the best point is tuning on the test set with extra steps — so the
    default end date here stops well before the default backtest end.
    """
    settings = Settings()
    configure_logging("WARNING", settings.log_format)

    grid = _grid_from(axis or ["fast=10:60:10"])
    evaluator, ledger = _evaluator(
        symbol=symbol,
        config=config,
        source=source,
        capital=capital,
        adjustment=adjustment,
        slippage=slippage,
        objective=objective,
        min_trades=min_trades,
        ledger_path=ledger_path,
        kind="sensitivity",
    )
    index = evaluator.load(dt.date.fromisoformat(start), dt.date.fromisoformat(end))
    console.print(f"{len(index)} bars, sweeping {grid.size} points ({grid.describe()})…")

    warmup = max(_warmup_for(dict(p), 2) for p in grid.points())
    window = window_from(index, 0, len(index))
    measured = window_from(index, min(warmup, len(index) - 1), len(index))
    surface = _sweep(evaluator, grid, window, measured, Objective(objective, min_trades=min_trades))

    console.print()
    for name in grid.names:
        rows = [(str(v), f"{s:.3f}" if s > -1e308 else "—") for v, s in surface.axis_profile(name)]
        table = Table(title=f"{objective} along {name}", title_justify="left", box=box.SIMPLE)
        table.add_column(name, style="bold cyan")
        table.add_column(objective, justify="right")
        for value, score in rows:
            table.add_row(value, score)
        console.print(table)

    _render_rows(
        [("Sensitivity", label, value) for label, value in surface.summary_lines()],
        f"Parameter surface — {config} on {symbol}",
    )
    colour = "green" if surface.is_plateau else "red"
    console.print(f"[{colour}]{surface.verdict()}[/{colour}]")
    console.print(
        f"[dim]Ledger now holds {ledger.trial_count(Path(config).stem)} distinct trials "
        f"for this strategy. Every one of them deflates the Sharpe of whichever "
        f"point you pick.[/dim]"
    )


def _sweep(
    evaluator: EngineEvaluator,
    grid: ParameterGrid,
    window: Window,
    measured: Window,
    objective: Objective,
) -> SensitivitySurface:
    """Sweep with the warmup derived per point, which ``run_sensitivity`` cannot do."""
    points: list[GridPoint] = []
    for params in grid.points():
        point = dict(params)
        point["warmup_bars"] = _warmup_for(point, 2)
        outcome = evaluator(point, window, measured)
        # Keyed on the swept axes only: `warmup_bars` is derived, so including it
        # would make every point its own island and break the neighbourhood test.
        points.append(
            GridPoint(params=dict(params), score=objective.score(outcome), outcome=outcome)
        )
    return SensitivitySurface(grid=grid, points=tuple(points), objective=objective)


@app.command()
def experiments(
    strategy: str = "",
    ledger_path: str = "data/experiments.sqlite",
    limit: int = 10,
) -> None:
    """What has been tried. The denominator the deflated Sharpe needs."""
    ledger = ExperimentLedger(ledger_path)
    rows = ledger.strategies()
    if not rows:
        console.print(
            "[yellow]No experiments recorded.[/yellow] Run `trading sensitivity` or "
            "`trading walk-forward` — they record every evaluation as they go."
        )
        return

    table = Table(title=f"Experiment ledger — {ledger_path}", title_justify="left", box=box.SIMPLE)
    for column in ("strategy", "instrument", "trials", "evaluations", "best", "last seen"):
        table.add_column(column, justify="right" if column in ("trials", "evaluations") else "left")
    for row in rows:
        best = row["best_objective"]
        table.add_row(
            str(row["strategy_id"]),
            str(row["instrument"]),
            str(row["trials"]),
            str(row["evaluations"]),
            f"{best:.3f}" if best is not None else "—",
            str(row["last_seen"])[:19],
        )
    console.print(table)

    if strategy:
        top = ledger.best(strategy, limit=limit)
        detail = Table(title=f"Top {len(top)} by objective", title_justify="left", box=box.SIMPLE)
        for column in ("kind", "parameters", "objective", "window", "tier"):
            detail.add_column(column)
        for row in top:
            detail.add_row(
                str(row["kind"]),
                str(row["params_json"]),
                f"{row['objective_value']:.3f}",
                f"{row['start_date']!s} → {row['end_date']!s}",
                str(row["data_tier"]),
            )
        console.print(detail)
        console.print(
            "[dim]The top of this list is selected, so its ratio is the maximum of "
            f"{ledger.trial_count(strategy)} trials. Read the two together or not "
            "at all.[/dim]"
        )


@app.command()
def validate(
    symbol: str = "RELIANCE",
    config: str = "configs/strategies/sma_cross_param.yaml",
    axis: Annotated[list[str] | None, typer.Option(help="name=values, repeatable")] = None,
    source: str = "fixture",
    start: str = "2015-01-01",
    end: str = "2026-08-21",
    sweep_end: str = "",
    capital: float = 400_000,
    train_bars: int = 756,
    test_bars: int = 252,
    objective: str = "sharpe",
    min_trades: int = 2,
    slippage: str = "spread",
    adjustment: str = "SPLIT_ONLY",
    risk_profile_path: str = "",
    paths: int = 2_000,
    seed: int = 0,
    cv_folds: int = 5,
    skip_cv: bool = False,
    ledger_path: str = "data/experiments.sqlite",
) -> None:
    """Run the whole Phase 6 battery and answer the ``VALIDATED`` lifecycle gate.

    Walk-forward, a parameter sweep on the training period, purged
    cross-validation, and two Monte Carlo resamplings — then the gate, with the
    evidence each criterion came from.

    Expect this to fail. It is built to.
    """
    settings = Settings()
    configure_logging("WARNING", settings.log_format)

    limits = load_risk_profile(risk_profile_path) if risk_profile_path else RiskLimits()
    grid = _grid_from(axis or ["fast=20,30,40,50,60", "slow=100,150,200"])
    evaluator, ledger = _evaluator(
        symbol=symbol,
        config=config,
        source=source,
        capital=capital,
        adjustment=adjustment,
        slippage=slippage,
        objective=objective,
        min_trades=min_trades,
        ledger_path=ledger_path,
        kind="validate",
    )
    start_date, end_date = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    index = evaluator.load(start_date, end_date)
    points = _expand(grid, warmup_floor=2)
    warmup = max(_warmup_for(p, 2) for p in points)
    obj = Objective(metric=objective, min_trades=min_trades)

    console.print(
        f"[bold]Validating {config} on {symbol}[/bold]\n"
        f"{len(index)} bars {index[0].date()}..{index[-1].date()}, "
        f"{len(points)} parameter sets, tier {evaluator.data_tier}"
    )

    # ── 1. walk-forward ─────────────────────────────────────────────────────
    console.print("\n[bold]1/4 walk-forward[/bold] — optimise, then trade forward…")
    evaluator.ledger_kind = "walk_forward"
    wf = run_walk_forward(
        evaluator,
        index,
        points,
        WalkForwardConfig(
            train_bars=train_bars,
            test_bars=test_bars,
            warmup_bars=warmup,
            objective=obj,
        ),
    )
    for fold in wf.folds:
        console.print(f"  [{'green' if fold.scored else 'yellow'}]{fold.label()}[/]")

    # ── 2. sensitivity, on the training period only ─────────────────────────
    console.print("\n[bold]2/4 parameter surface[/bold] — plateau or spike…")
    surface = _validate_sweep(
        evaluator, grid, index, obj, sweep_end=sweep_end, train_bars=train_bars, warmup=warmup
    )

    # ── 3. purged cross-validation ──────────────────────────────────────────
    console.print("\n[bold]3/4 purged cross-validation[/bold] — every regime as a test fold…")
    cv = _validate_cv(evaluator, index, points, obj, folds=cv_folds, warmup=warmup, skip=skip_cv)

    # ── 4. Monte Carlo ──────────────────────────────────────────────────────
    console.print("\n[bold]4/4 Monte Carlo[/bold] — the drawdown you did not happen to see…")
    trade_mc, path_mc = _validate_monte_carlo(
        wf, capital=capital, paths=paths, seed=seed, limit=float(limits.max_drawdown)
    )

    # ── the report, and the gate ────────────────────────────────────────────
    spec = load_strategy(Path(config)).spec
    instrument_id = str(_instrument(symbol).id)
    report = ValidationReport(
        strategy_id=spec.id,
        strategy_version=spec.version,
        instrument=instrument_id,
        data_tier=evaluator.data_tier,
        walk_forward=wf,
        trial_count=ledger.trial_count(spec.id, instrument_id),
        drawdown_limit=limits.max_drawdown,
        purged_cv=cv,
        sensitivity=surface,
        trade_monte_carlo=trade_mc,
        path_monte_carlo=path_mc,
        starting_capital=Decimal(str(capital)),
    )
    _render_validation(report, symbol)


def _validate_sweep(
    evaluator: EngineEvaluator,
    grid: ParameterGrid,
    index: pd.DatetimeIndex,
    objective: Objective,
    *,
    sweep_end: str,
    train_bars: int,
    warmup: int,
) -> SensitivitySurface:
    """Sweep the parameters over the training period only.

    The window stops at the end of the first training block by default. Sweeping
    the out-of-sample period and reporting the best point would be tuning on the
    test set, which is the failure the rest of this command exists to detect.
    """
    evaluator.ledger_kind = "sensitivity"
    stop = (
        dt.date.fromisoformat(sweep_end)
        if sweep_end
        else index[min(train_bars, len(index) - 1)].date()
    )
    # The bar index is UTC-aware; a date is not. Comparing them raises rather
    # than silently mis-slicing, which is the right behaviour and has to be
    # honoured here rather than worked around downstream.
    cutoff = pd.Timestamp(stop, tz=index.tz) if index.tz is not None else pd.Timestamp(stop)
    stop_index = max(int(index.searchsorted(cutoff)), warmup + 2)
    window = window_from(index, 0, min(stop_index, len(index)))
    measured = window_from(index, min(warmup, window.stop_index - 1), window.stop_index)
    surface = _sweep(evaluator, grid, window, measured, objective)
    console.print(f"  swept {window.label()} — {surface.verdict()}")
    return surface


def _validate_cv(
    evaluator: EngineEvaluator,
    index: pd.DatetimeIndex,
    points: list[dict[str, Any]],
    objective: Objective,
    *,
    folds: int,
    warmup: int,
    skip: bool,
) -> PurgedCvResult | None:
    if skip:
        console.print("  [yellow]skipped[/yellow]")
        return None
    evaluator.ledger_kind = "purged_cv"
    result = run_purged_cv(
        evaluator,
        index,
        points,
        PurgedCvConfig(
            n_splits=folds,
            purge_bars=20,
            embargo_bars=10,
            warmup_bars=warmup,
            objective=objective,
        ),
    )
    for fold in result.folds:
        console.print(f"  [{'green' if fold.scored else 'yellow'}]{fold.label()}[/]")
    return result


def _validate_monte_carlo(
    wf: WalkForwardResult,
    *,
    capital: float,
    paths: int,
    seed: int,
    limit: float,
) -> tuple[MonteCarloResult | None, MonteCarloResult | None]:
    """Resample the out-of-sample record two ways, or say why neither was possible."""
    trade_mc = path_mc = None
    pnl = wf.oos_trade_pnl
    if len(pnl) >= 2:
        trade_mc = resample_trades(
            pnl, starting_capital=capital, paths=paths, seed=seed, ruin_threshold=limit
        )
        console.print(f"  trades: {trade_mc.verdict(limit)}")
    else:
        console.print(
            f"  [yellow]only {len(pnl)} out-of-sample trades — cannot resample a "
            f"distribution from that[/yellow]"
        )

    oos = wf.oos_returns
    if len(oos) >= 40:
        path_mc = block_bootstrap_returns(
            oos, block_bars=20, paths=paths, seed=seed, ruin_threshold=limit
        )
        console.print(f"  path:   {path_mc.verdict(limit)}")
    else:
        console.print(
            f"  [yellow]only {len(oos)} out-of-sample bars — too few to bootstrap[/yellow]"
        )
    return trade_mc, path_mc


def _render_validation(report: ValidationReport, symbol: str) -> None:
    """Print the report, then put it through the lifecycle gate it answers."""
    console.print()
    _render_rows(report.summary_lines(), f"Validation report — {report.strategy_id} on {symbol}")
    colour = "green" if report.passed else "red"
    console.print(f"[{colour}]{report.verdict()}[/{colour}]")

    decision = evaluate_promotion(
        report.strategy_id,
        LifecycleStatus.PROMISING,
        LifecycleStatus.VALIDATED,
        evidence_from_validation(report),
    )
    console.print(f"\n[bold]Lifecycle gate:[/bold] {decision.summary()}")
    for result in decision.results:
        mark = "green" if result.status is CriterionStatus.PASS else "red"
        console.print(f"  [{mark}]{result}[/{mark}]")

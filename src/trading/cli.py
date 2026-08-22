"""Command line interface."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

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
from trading.observability.logging import configure_logging
from trading.risk.engine import RiskEngine, RiskLimits
from trading.strategies.builtin import BuyAndHold, SmaCross

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
) -> None:
    """Run the walking skeleton end to end.

    ``--source store`` reads the local bitemporal store with corporate actions
    applied, which is the path a real backtest takes. ``fixture`` and
    ``yfinance`` bypass the store and go straight to the provider.
    """
    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)

    if strategy not in _STRATEGIES:
        raise typer.BadParameter(f"Unknown strategy {strategy!r}. Choose from {_STRATEGIES}.")

    instrument = _instrument(symbol)
    target = Decimal(str(weight))
    strat = (
        BuyAndHold(instrument.id, target_weight=target)
        if strategy == "buy_and_hold"
        else SmaCross(instrument.id, fast=50, slow=200, target_weight=target)
    )
    compliance = profile_for(instrument.market)

    result = WalkingSkeletonRunner(
        provider=_provider(source, adjustment=adjustment),
        instrument=instrument,
        strategy=strat,
        cost_model=IndiaDeliveryEquityCosts(),
        risk_engine=RiskEngine(RiskLimits(), compliance, kill_switch_path=Path("KILL")),
        compliance=compliance,
        starting_capital=Money(Decimal(str(capital)), instrument.currency),
        mode=TradingMode.BACKTEST,
    ).run(dt.date.fromisoformat(start), dt.date.fromisoformat(end))

    _render(result, strat.spec.name, show_decisions)


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
    summary.add_row("Risk rejections", str(len(result.rejections)))
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

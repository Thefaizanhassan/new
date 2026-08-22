"""Command line interface."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from trading.calendars.base import calendar_for
from trading.config.compliance import profile_for
from trading.config.settings import Settings
from trading.core.instrument import EXCHANGES, Instrument, InstrumentId
from trading.core.types import InstrumentClass, Money, TradingMode
from trading.costs.india import IndiaDeliveryEquityCosts
from trading.data.fixture import FixtureProvider
from trading.data.provider import DataTier, HistoricalDataProvider
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


def _provider(source: str) -> HistoricalDataProvider:
    if source == "fixture":
        return FixtureProvider()
    if source == "yfinance":
        # Deferred so a `fixture` run never needs yfinance installed or reachable.
        from trading.data.yfinance_provider import YFinanceProvider  # noqa: PLC0415

        return YFinanceProvider()
    raise typer.BadParameter(f"Unknown source {source!r}. Use 'fixture' or 'yfinance'.")


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
def backtest(
    symbol: str = "RELIANCE",
    strategy: str = "sma_cross",
    source: str = "fixture",
    start: str = "2022-01-01",
    end: str = "2026-08-21",
    capital: float = 400_000,
    weight: float = 0.20,
    show_decisions: int = 10,
) -> None:
    """Run the Phase 1 walking skeleton end to end."""
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
        provider=_provider(source),
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

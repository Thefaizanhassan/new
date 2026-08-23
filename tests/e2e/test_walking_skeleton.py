"""End-to-end: bars -> validate -> strategy -> risk -> order -> fill -> equity."""

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from trading.config.compliance import INDIA_SEBI
from trading.core.instrument import Instrument, InstrumentId
from trading.core.types import Currency, InstrumentClass, Money, TradingMode
from trading.costs.india import IndiaDeliveryEquityCosts
from trading.data.fixture import FixtureProvider
from trading.engine.runner import WalkingSkeletonRunner
from trading.risk.engine import RiskEngine, RiskLimits
from trading.strategies.builtin import BuyAndHold, SmaCross
from trading.strategies.config_strategy import load_strategy

START, END = dt.date(2022, 1, 1), dt.date(2026, 8, 21)


@pytest.fixture
def instrument() -> Instrument:
    return Instrument(
        id=InstrumentId("NSE", "RELIANCE"),
        name="Reliance",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
    )


def build(instrument, strategy, *, limits=None, kill_switch=None):
    return WalkingSkeletonRunner(
        provider=FixtureProvider(),
        instrument=instrument,
        strategy=strategy,
        cost_model=IndiaDeliveryEquityCosts(),
        risk_engine=RiskEngine(
            limits or RiskLimits(),
            INDIA_SEBI,
            kill_switch_path=kill_switch or Path("/nonexistent/KILL"),
        ),
        compliance=INDIA_SEBI,
        starting_capital=Money.inr("400000"),
        mode=TradingMode.BACKTEST,
    )


def test_buy_and_hold_trades_once_and_holds(instrument):
    result = build(instrument, BuyAndHold(instrument.id, Decimal("0.20"))).run(START, END)
    assert len(result.fills) == 1
    assert result.data_quality.is_usable


def test_sma_cross_trades_and_pays_more_in_costs_than_buy_and_hold(instrument):
    """Turnover costs money. This is the lesson the cost model exists to teach."""
    hold = build(instrument, BuyAndHold(instrument.id, Decimal("0.20"))).run(START, END)
    cross = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    assert len(cross.fills) > len(hold.fills)
    assert cross.cost_drag > hold.cost_drag


def test_net_return_is_always_below_gross_return(instrument):
    result = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    assert result.net_return < result.gross_return
    assert result.total_costs.amount > 0


def test_the_run_is_reproducible(instrument):
    a = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    b = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    assert a.final_equity == b.final_equity
    assert len(a.fills) == len(b.fills)
    assert a.manifest["content_hash"] == b.manifest["content_hash"]


def test_manifest_records_everything_needed_to_explain_a_changed_result(instrument):
    manifest = build(instrument, BuyAndHold(instrument.id)).run(START, END).manifest
    for key in (
        "strategy_id",
        "content_hash",
        "provider_id",
        "tier",
        "cost_model_id",
        "cost_model_version",
        "compliance_profile",
        "instrument",
        "start",
        "end",
        "bars",
        "fill_convention",
    ):
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["tier"] == "SYNTHETIC"


def test_fills_happen_at_the_next_bars_open_never_the_signal_bars_close(instrument):
    """Same-bar fills are the largest single source of fake alpha."""
    result = build(instrument, BuyAndHold(instrument.id, Decimal("0.20"))).run(START, END)
    bars = FixtureProvider().get_bars(instrument.id, START, END)
    fill = result.fills[0]
    assert float(fill.price) == pytest.approx(float(bars.loc[fill.timestamp, "open"]))


def test_kill_switch_file_blocks_every_order(instrument, tmp_path):
    kill = tmp_path / "KILL"
    kill.write_text("halted for testing")
    result = build(instrument, BuyAndHold(instrument.id, Decimal("0.20")), kill_switch=kill).run(
        START, END
    )
    assert not result.fills
    assert result.rejections
    assert "kill_switch" in result.rejections[0].detail
    assert result.final_equity == result.starting_capital


def test_position_limit_rejects_an_oversized_target(instrument):
    result = build(
        instrument,
        BuyAndHold(instrument.id, Decimal("0.95")),
        limits=RiskLimits(max_position_weight=Decimal("0.10")),
    ).run(START, END)
    assert not result.fills
    assert any("max_position_weight" in r.detail for r in result.rejections)


def test_research_mode_refuses_to_place_orders(instrument):
    runner = build(instrument, BuyAndHold(instrument.id, Decimal("0.20")))
    runner.mode = TradingMode.RESEARCH
    result = runner.run(START, END)
    assert not result.fills
    assert any("trading_mode" in r.detail for r in result.rejections)


def test_the_decision_chain_is_reconstructable_for_every_fill(instrument):
    """Phase 0 §16: signal -> risk verdict -> order -> fill, all present."""
    result = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    kinds = [d.kind for d in result.decisions]
    for expected in ("SIGNAL", "RISK_APPROVED", "ORDER", "FILL"):
        assert expected in kinds
    assert kinds.count("FILL") == len(result.fills)
    signals = [d for d in result.decisions if d.kind == "SIGNAL"]
    assert all(d.detail for d in signals), "every signal must carry a human-readable reason"
    assert any(d.evidence for d in signals), "signals must carry the values that drove them"


def test_equity_curve_covers_every_bar_and_tracks_costs(instrument):
    result = build(instrument, SmaCross(instrument.id, 50, 200, Decimal("0.20"))).run(START, END)
    curve = result.equity_curve
    assert len(curve) == int(result.manifest["bars"])
    assert curve["cumulative_costs"].is_monotonic_increasing
    assert curve["cumulative_costs"].iloc[-1] == pytest.approx(float(result.total_costs.amount))


def test_strategy_never_receives_future_bars(instrument, monkeypatch):
    """The look-ahead guard, asserted rather than assumed."""
    seen: list[tuple] = []
    strategy = BuyAndHold(instrument.id, Decimal("0.20"))
    original = strategy.on_bar

    def spy(ctx, inst):
        history = ctx.history(inst.id)
        if len(history):
            seen.append((ctx.now, history.index.max().to_pydatetime()))
        return original(ctx, inst)

    monkeypatch.setattr(strategy, "on_bar", spy)
    build(instrument, strategy).run(START, END)

    assert seen, "strategy was never called"
    for now, latest_bar in seen:
        assert latest_bar <= now, f"strategy saw a bar at {latest_bar} while now was {now}"


# ── config strategies through the same engine ──────────────────────────────
def test_a_yaml_strategy_runs_through_exactly_the_same_pipeline(instrument):
    """Phase 0 §10.7: a config strategy is a real strategy, not a special case."""
    strategy = load_strategy("configs/strategies/sma_cross_reliance.yaml")
    result = build(instrument, strategy).run(START, END)

    assert result.manifest["strategy_id"] == "sma_cross_reliance"
    assert result.manifest["content_hash"]
    assert result.data_quality.is_usable
    kinds = [d.kind for d in result.decisions]
    assert "RISK_APPROVED" in kinds


def test_precomputing_features_does_not_change_the_result(instrument):
    """The optimisation is only legitimate if it is behaviour-preserving.

    Precomputing over the full history and slicing yields exactly what
    recomputing on truncated history does — because every feature is causal.
    """
    with_features = build(
        instrument, load_strategy("configs/strategies/sma_cross_reliance.yaml")
    ).run(START, END)

    runner = build(instrument, load_strategy("configs/strategies/sma_cross_reliance.yaml"))
    runner._feature_engine = None
    without_features = runner.run(START, END)

    assert with_features.final_equity == without_features.final_equity
    assert len(with_features.fills) == len(without_features.fills)
    assert [f.price for f in with_features.fills] == [f.price for f in without_features.fills]


def test_the_manifest_records_which_features_were_precomputed(instrument):
    result = build(
        instrument, load_strategy("configs/strategies/mean_reversion_reliance.yaml")
    ).run(START, END)
    assert result.manifest["features_precomputed"] == "_entry,_exit"


def test_a_python_strategy_without_declared_features_still_runs(instrument):
    """Feature declaration is optional — the fallback path must keep working."""
    result = build(instrument, BuyAndHold(instrument.id, Decimal("0.20"))).run(START, END)
    assert result.manifest["features_precomputed"] == "none"
    assert len(result.fills) == 1

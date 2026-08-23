"""The walking skeleton: one vertical slice, end to end.

Phase 0 §19 argues for building the thinnest possible end-to-end path before
building any component properly, so integration pain surfaces in week one
rather than month four.  This is that path:

    bars -> validate -> strategy -> intent -> sizing -> RISK -> order
         -> simulated fill (with real Indian costs) -> position -> equity

Two conventions here are the ones that keep a backtest honest:

* **Signal on bar N's close, fill at bar N+1's open.** At the moment bar N's
  close is known the market is shut. Same-bar fills are the single largest
  source of fake alpha (Phase 0 §11.2).
* **Every decision is an event**, including the rejections. A blocked trade is
  as informative as an executed one.

This is not the Phase 5 backtesting engine. There is no slippage model, no
partial fills, no volume cap and no walk-forward machinery yet. It exists to
prove the wiring, and it says so in its own output.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from trading.config.compliance import ComplianceProfile
from trading.core.fill import Fill
from trading.core.instrument import Instrument
from trading.core.intent import Flat, Intent, TargetQty, TargetWeight
from trading.core.order import Order, OrderRequest, OrderState
from trading.core.position import Position
from trading.core.types import Money, Side, TradingMode
from trading.costs.model import CostContext, CostModel, SessionCostState
from trading.data.provider import HistoricalDataProvider
from trading.data.validation import DataQualityReport, validate_ohlcv
from trading.features.engine import FeatureEngine
from trading.observability.logging import get_logger
from trading.risk.context import OrderRecord, PortfolioView, RiskContext
from trading.risk.engine import RiskEngine
from trading.risk.monitors import MonitorReading, MonitorResult
from trading.strategies.base import Strategy, StrategyContext

__all__ = ["RunResult", "WalkingSkeletonRunner"]

log = get_logger(__name__)


@dataclass
class DecisionRecord:
    """One row of the audit chain from Phase 0 §16."""

    timestamp: dt.datetime
    kind: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "kind": self.kind,
            "detail": self.detail,
            **{f"ev_{k}": v for k, v in self.evidence.items()},
        }


@dataclass
class RunResult:
    manifest: dict[str, str]
    equity_curve: pd.DataFrame
    fills: list[Fill]
    decisions: list[DecisionRecord]
    data_quality: DataQualityReport
    starting_capital: Money
    final_equity: Money
    total_costs: Money
    halt_summary: str = "NORMAL"
    monitor_events: list[MonitorResult] = field(default_factory=list)

    @property
    def gross_return(self) -> Decimal:
        gross = self.final_equity + self.total_costs - self.starting_capital
        return gross.ratio_to(self.starting_capital)

    @property
    def net_return(self) -> Decimal:
        return (self.final_equity - self.starting_capital).ratio_to(self.starting_capital)

    @property
    def cost_drag(self) -> Decimal:
        """How much of the gross return the costs consumed."""
        return self.total_costs.ratio_to(self.starting_capital)

    @property
    def rejections(self) -> list[DecisionRecord]:
        return [d for d in self.decisions if d.kind == "RISK_REJECTED"]

    @property
    def halted(self) -> bool:
        return self.halt_summary != "NORMAL"


class WalkingSkeletonRunner:
    def __init__(
        self,
        *,
        provider: HistoricalDataProvider,
        instrument: Instrument,
        strategy: Strategy,
        cost_model: CostModel,
        risk_engine: RiskEngine,
        compliance: ComplianceProfile,
        starting_capital: Money,
        mode: TradingMode = TradingMode.BACKTEST,
        feature_engine: FeatureEngine | None = None,
    ) -> None:
        self.provider = provider
        self.instrument = instrument
        self.strategy = strategy
        self.cost_model = cost_model
        self.risk_engine = risk_engine
        self.compliance = compliance
        self.starting_capital = starting_capital
        self.mode = mode
        # A strategy may declare the features it needs so they can be computed
        # once over the whole history instead of recomputed on every bar. This
        # is only safe because every feature in the expression language is
        # causal — verified by `features.engine.verify_causality`.
        self._feature_engine = feature_engine or self._engine_from(strategy)

    @staticmethod
    def _engine_from(strategy: Strategy) -> FeatureEngine | None:
        specs = getattr(strategy, "feature_specs", None)
        if not callable(specs):
            return None
        declared = specs()
        return FeatureEngine(declared) if declared else None

    def run(self, start: dt.date, end: dt.date) -> RunResult:
        bars = self.provider.get_bars(self.instrument.id, start, end)
        quality = validate_ohlcv(bars, symbol=str(self.instrument.id))
        if not quality.is_usable:
            raise ValueError(
                f"Refusing to run on quarantined data — {quality.summary()}\n  "
                + "\n  ".join(str(i) for i in quality.errors)
            )

        currency = self.instrument.currency
        cash = self.starting_capital
        position = Position(instrument=self.instrument)
        fills: list[Fill] = []
        decisions: list[DecisionRecord] = []
        equity_rows: list[dict[str, Any]] = []
        total_costs = Money.zero(currency)
        pending: Order | None = None
        session = SessionCostState(session_date=start)
        recent_orders: list[OrderRecord] = []
        day_start_equity = self.starting_capital
        peak_equity = self.starting_capital
        current_date = start
        monitor_events: list[MonitorResult] = []

        warmup = self.strategy.warmup_bars()
        # Computed once over the full history, then sliced per bar. Slicing a
        # causal feature to `now` yields exactly what computing it on truncated
        # history would, which is what makes this safe rather than look-ahead.
        features = self._feature_engine.compute(bars) if self._feature_engine is not None else None
        # Correlation and liquidity inputs. Both are sliced to `now` before use,
        # so neither can leak a value the engine could not have known.
        returns = bars[["close"]].pct_change().rename(columns={"close": str(self.instrument.id)})
        advs = bars["volume"].rolling(20, min_periods=5).mean().shift(1)

        for i, (timestamp, bar) in enumerate(bars.iterrows()):
            now, open_price, close_price = self._bar_values(timestamp, bar)

            # ── 1. execute any order raised on the previous bar's close ─────
            if pending is not None:
                if session.session_date != now.date():
                    session = SessionCostState(session_date=now.date())
                fill = self._execute(pending, open_price, now, session)
                position, cash, total_costs = self._apply_fill(
                    fill,
                    position,
                    cash,
                    total_costs,
                    fills=fills,
                    decisions=decisions,
                    now=now,
                )
                pending = None

            equity = cash + position.market_value(close_price)
            if now.date() != current_date:
                # A daily loss limit measures the day, so it resets with it.
                current_date = now.date()
                day_start_equity = equity
            peak_equity = equity if equity > peak_equity else peak_equity

            portfolio = self._portfolio_view(
                equity=equity,
                cash=cash,
                position=position,
                close_price=close_price,
                day_start_equity=day_start_equity,
                peak_equity=peak_equity,
            )
            monitor_events.extend(self._run_monitors(portfolio, now, decisions))

            equity_rows.append(
                self._equity_row(
                    timestamp=timestamp,
                    cash=cash,
                    position=position,
                    close_price=close_price,
                    equity=equity,
                    total_costs=total_costs,
                )
            )

            if i < warmup or i == len(bars) - 1:
                continue
            if self.risk_engine.halted:
                # A halt outranks every rule; there is nothing to decide.
                continue

            # ── 2. strategy sees only history up to and including this bar ──
            ctx = StrategyContext(
                now=now,
                history={self.instrument.id: bars.iloc[: i + 1]},
                positions={self.instrument.id: position},
                equity=equity,
                features=(
                    {self.instrument.id: features.iloc[: i + 1]} if features is not None else None
                ),
            )
            intents = self.strategy.on_bar(ctx, self.instrument)
            if not intents:
                continue

            pending = self._decide(
                intents=intents,
                portfolio=portfolio,
                close_price=close_price,
                now=now,
                decisions=decisions,
                recent_orders=recent_orders,
                returns=returns.iloc[: i + 1] if returns is not None else None,
                adv=advs.iloc[i] if advs is not None else None,
            )

        final_price = Decimal(str(bars["close"].iloc[-1]))
        final_equity = cash + position.market_value(final_price)

        return RunResult(
            manifest=self._manifest(start, end, len(bars)),
            equity_curve=pd.DataFrame(equity_rows).set_index("timestamp"),
            fills=fills,
            decisions=decisions,
            data_quality=quality,
            starting_capital=self.starting_capital,
            final_equity=final_equity,
            total_costs=total_costs,
            halt_summary=self.risk_engine.halt.summary(),
            monitor_events=monitor_events,
        )

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bar_values(timestamp: Any, bar: Any) -> tuple[dt.datetime, Decimal, Decimal]:
        """Pull the three values the loop needs, converting to Decimal at the edge.

        This is the pandas/Decimal boundary: floats stop here and everything
        downstream is exact.
        """
        return (
            timestamp.to_pydatetime(),
            Decimal(str(bar["open"])),
            Decimal(str(bar["close"])),
        )

    @staticmethod
    def _apply_fill(
        fill: Fill,
        position: Position,
        cash: Money,
        total_costs: Money,
        *,
        fills: list[Fill],
        decisions: list[DecisionRecord],
        now: dt.datetime,
    ) -> tuple[Position, Money, Money]:
        """Book a fill into the ledger and record it in the decision chain."""
        position = position.apply(fill)
        cash = cash + fill.cash_delta
        total_costs = total_costs + fill.costs.total
        fills.append(fill)
        decisions.append(
            DecisionRecord(
                now,
                "FILL",
                f"{fill.side} {fill.quantity} @ {fill.price:.2f}",
                {"costs": str(fill.costs.total), "cash_after": str(cash.settled())},
            )
        )
        return position, cash, total_costs

    def _portfolio_view(
        self,
        *,
        equity: Money,
        cash: Money,
        position: Position,
        close_price: Decimal,
        day_start_equity: Money,
        peak_equity: Money,
    ) -> PortfolioView:
        return PortfolioView(
            equity=equity,
            cash=cash,
            positions={self.instrument.id: position},
            marks={self.instrument.id: close_price},
            day_start_equity=day_start_equity,
            peak_equity=peak_equity,
        )

    def _run_monitors(
        self,
        portfolio: PortfolioView,
        now: dt.datetime,
        decisions: list[DecisionRecord],
    ) -> list[MonitorResult]:
        """Evaluate portfolio-level monitors and record anything that fires."""
        before = self.risk_engine.halt.level
        results = self.risk_engine.check_monitors(
            MonitorReading(
                portfolio=portfolio, now=now, is_live_like=self.mode.name in ("PAPER", "LIVE")
            )
        )
        fired = [r for r in results if r.triggered]
        for result in fired:
            decisions.append(
                DecisionRecord(now, "MONITOR", str(result), {"monitor": result.monitor_id})
            )
        if self.risk_engine.halt.level > before:
            decisions.append(
                DecisionRecord(
                    now,
                    "HALT",
                    self.risk_engine.halt.summary(),
                    {"level": str(self.risk_engine.halt.level)},
                )
            )
        return fired

    def _decide(
        self,
        *,
        intents: list[Intent],
        portfolio: PortfolioView,
        close_price: Decimal,
        now: dt.datetime,
        decisions: list[DecisionRecord],
        recent_orders: list[OrderRecord],
        returns: pd.DataFrame | None,
        adv: Any,
    ) -> Order | None:
        """Size each intent, put it through risk, and return an order if approved.

        The skeleton carries one pending order at a time, so the first approved
        intent wins. Rejections are recorded before returning — a blocked trade
        is as informative as an executed one (Phase 0 §3.10).
        """
        position = portfolio.positions[self.instrument.id]
        for intent in intents:
            request = self._size(intent, portfolio.equity, close_price, position)
            if request is None:
                continue
            decisions.append(DecisionRecord(now, "SIGNAL", intent.reason, dict(intent.evidence)))

            decision = self.risk_engine.evaluate(
                RiskContext(
                    request=request,
                    reference_price=close_price,
                    portfolio=portfolio,
                    now=now,
                    mode=self.mode,
                    returns=returns,
                    average_daily_volume=(
                        Decimal(str(adv)) if adv is not None and pd.notna(adv) else None
                    ),
                    recent_orders=tuple(recent_orders[-50:]),
                )
            )
            if not decision.approved:
                decisions.append(
                    DecisionRecord(
                        now,
                        "RISK_REJECTED",
                        decision.rejection_reason,
                        {"strategy": intent.strategy_id},
                    )
                )
                continue

            decisions.append(
                DecisionRecord(
                    now,
                    "RISK_APPROVED",
                    f"{len(decision.results)} rules passed",
                    {"decision_id": decision.decision_id},
                )
            )
            order = Order.create(request, decision, now)
            recent_orders.append(
                OrderRecord(
                    client_order_id=order.client_order_id,
                    instrument_id=self.instrument.id,
                    side=str(request.side),
                    quantity=request.quantity,
                    submitted_at=now,
                    strategy_id=request.strategy_id,
                )
            )
            decisions.append(
                DecisionRecord(
                    now,
                    "ORDER",
                    f"{request.side} {request.quantity} {self.instrument.id.symbol}",
                    {"client_order_id": order.client_order_id},
                )
            )
            return order
        return None

    @staticmethod
    def _equity_row(
        *,
        timestamp: Any,
        cash: Money,
        position: Position,
        close_price: Decimal,
        equity: Money,
        total_costs: Money,
    ) -> dict[str, Any]:
        return {
            "timestamp": timestamp,
            "cash": float(cash.amount),
            "position_qty": float(position.quantity),
            "position_value": float(position.market_value(close_price).amount),
            "equity": float(equity.amount),
            "cumulative_costs": float(total_costs.amount),
        }

    @staticmethod
    def _fill_record(fill: Fill, now: dt.datetime, cash_after: Money) -> DecisionRecord:
        return DecisionRecord(
            now,
            "FILL",
            f"{fill.side} {fill.quantity} @ {fill.price:.2f}",
            {"costs": str(fill.costs.total), "cash_after": str(cash_after.settled())},
        )

    def _size(
        self, intent: Intent, equity: Money, price: Decimal, position: Position
    ) -> OrderRequest | None:
        """Turn a declarative target into a concrete order request.

        Sizing lives here, not in the strategy, so multiple strategies cannot
        independently stack exposure (Phase 0 §2.4).
        """
        multiplier = self.instrument.multiplier
        match intent.target:
            case Flat():
                target_qty = Decimal(0)
            case TargetWeight(weight=w):
                target_qty = (equity.amount * w) / (price * multiplier)
            case TargetQty(quantity=q):
                target_qty = q
            case _:
                return None

        delta = self.instrument.round_to_lot(target_qty - position.quantity)
        if delta == 0:
            return None

        return OrderRequest(
            instrument=self.instrument,
            side=Side.BUY if delta > 0 else Side.SELL,
            quantity=abs(delta),
            strategy_id=intent.strategy_id,
            reason=intent.reason,
        )

    def _execute(
        self, order: Order, price: Decimal, now: dt.datetime, session: SessionCostState
    ) -> Fill:
        """Fill at the next bar's open. No slippage model yet — Phase 5."""
        request = order.request
        costs = self.cost_model.compute(
            CostContext(
                instrument=request.instrument,
                side=request.side,
                quantity=request.quantity,
                price=price,
                timestamp=now,
                session=session,
            )
        )
        order.with_state(OrderState.FILLED, filled_quantity=request.quantity)
        return Fill.create(
            client_order_id=order.client_order_id,
            instrument=request.instrument,
            side=request.side,
            quantity=request.quantity,
            price=price,
            costs=costs,
            timestamp=now,
            strategy_id=request.strategy_id,
        )

    def _manifest(self, start: dt.date, end: dt.date, bars: int) -> dict[str, str]:
        """Provenance. Without it a backtest is an anecdote (Phase 0 §16.3)."""
        # A store-backed provider can hash exactly the rows this run saw, which
        # is what makes "did the data move or did the code?" answerable.
        dataset_version = "unversioned"
        versioner = getattr(self.provider, "dataset_version", None)
        if callable(versioner):
            dataset_version = str(versioner(self.instrument.id, start, end))

        return {
            "dataset_version": dataset_version,
            **self.strategy.spec.manifest_entry(),
            **self.provider.info.manifest_entry(),
            "cost_model_id": self.cost_model.model_id,
            "cost_model_version": self.cost_model.version,
            "compliance_profile": self.compliance.profile_id,
            "risk_profile_id": self.risk_engine.limits.profile_id,
            "risk_profile_version": self.risk_engine.limits.version,
            "risk_rules": str(len(self.risk_engine.rules)),
            "mode": str(self.mode),
            "instrument": str(self.instrument.id),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bars": str(bars),
            "starting_capital": str(self.starting_capital.amount),
            "fill_convention": "signal at bar N close, fill at bar N+1 open",
            "features_precomputed": (
                ",".join(self._feature_engine.names) if self._feature_engine else "none"
            ),
            "engine": "walking_skeleton_phase1",
        }

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
from trading.observability.logging import get_logger
from trading.risk.engine import RiskEngine, RiskInputs
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
    ) -> None:
        self.provider = provider
        self.instrument = instrument
        self.strategy = strategy
        self.cost_model = cost_model
        self.risk_engine = risk_engine
        self.compliance = compliance
        self.starting_capital = starting_capital
        self.mode = mode

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

        warmup = self.strategy.warmup_bars()

        for i, (timestamp, bar) in enumerate(bars.iterrows()):
            now, open_price, close_price = self._bar_values(timestamp, bar)

            # ── 1. execute any order raised on the previous bar's close ─────
            if pending is not None:
                if session.session_date != now.date():
                    session = SessionCostState(session_date=now.date())
                fill = self._execute(pending, open_price, now, session)
                position = position.apply(fill)
                cash = cash + fill.cash_delta
                total_costs = total_costs + fill.costs.total
                fills.append(fill)
                decisions.append(self._fill_record(fill, now, cash))
                pending = None

            equity = cash + position.market_value(close_price)
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

            # ── 2. strategy sees only history up to and including this bar ──
            ctx = StrategyContext(
                now=now,
                history={self.instrument.id: bars.iloc[: i + 1]},
                positions={self.instrument.id: position},
                equity=equity,
            )
            intents = self.strategy.on_bar(ctx, self.instrument)
            if not intents:
                continue

            for intent in intents:
                request = self._size(intent, equity, close_price, position)
                if request is None:
                    continue
                decisions.append(
                    DecisionRecord(now, "SIGNAL", intent.reason, dict(intent.evidence))
                )

                gross = position.market_value(close_price).abs()
                decision = self.risk_engine.evaluate(
                    RiskInputs(
                        request=request,
                        reference_price=close_price,
                        equity=equity,
                        cash=cash,
                        gross_exposure_value=gross,
                        open_positions=0 if position.is_flat else 1,
                        mode=self.mode,
                        now=now,
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
                    log.info(
                        "risk_rejected",
                        instrument=str(self.instrument.id),
                        reason=decision.rejection_reason,
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
                pending = Order.create(request, decision, now)
                decisions.append(
                    DecisionRecord(
                        now,
                        "ORDER",
                        f"{request.side} {request.quantity} {self.instrument.id.symbol}",
                        {"client_order_id": pending.client_order_id},
                    )
                )
                break  # the skeleton carries one pending order at a time

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

    def _decide(
        self,
        *,
        intents: list[Intent],
        position: Position,
        equity: Money,
        cash: Money,
        close_price: Decimal,
        now: dt.datetime,
        decisions: list[DecisionRecord],
    ) -> Order | None:
        """Size each intent, put it through risk, and return an order if approved.

        The skeleton carries one pending order at a time, so the first approved
        intent wins. Rejections are recorded before returning — a blocked trade
        is as informative as an executed one (Phase 0 §3.10).
        """
        for intent in intents:
            request = self._size(intent, equity, close_price, position)
            if request is None:
                continue
            decisions.append(DecisionRecord(now, "SIGNAL", intent.reason, dict(intent.evidence)))

            decision = self.risk_engine.evaluate(
                RiskInputs(
                    request=request,
                    reference_price=close_price,
                    equity=equity,
                    cash=cash,
                    gross_exposure_value=position.market_value(close_price).abs(),
                    open_positions=0 if position.is_flat else 1,
                    mode=self.mode,
                    now=now,
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
                log.info(
                    "risk_rejected",
                    instrument=str(self.instrument.id),
                    reason=decision.rejection_reason,
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
        return {
            **self.strategy.spec.manifest_entry(),
            **self.provider.info.manifest_entry(),
            "cost_model_id": self.cost_model.model_id,
            "cost_model_version": self.cost_model.version,
            "compliance_profile": self.compliance.profile_id,
            "mode": str(self.mode),
            "instrument": str(self.instrument.id),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "bars": str(bars),
            "starting_capital": str(self.starting_capital.amount),
            "fill_convention": "signal at bar N close, fill at bar N+1 open",
            "engine": "walking_skeleton_phase1",
        }

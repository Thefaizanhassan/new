"""Phase 6 — validation.

The layer that decides whether a backtest result is evidence or an artefact of
how hard it was looked for.  Nothing here makes a strategy better; everything
here makes a claim about a strategy harder to make.

Read the modules in this order:

* :mod:`~trading.validation.splits` — how history is cut so that "out of sample"
  means it.
* :mod:`~trading.validation.harness` — the contract validation asks a backtest to
  satisfy, and what "better" means during a sweep.
* :mod:`~trading.validation.walkforward` — the only check here that simulates what
  an operator would have experienced.
* :mod:`~trading.validation.purged_cv` — a second opinion that tests every regime,
  and is not evidence of tradeability.
* :mod:`~trading.validation.sensitivity` — plateau versus spike; the cheapest and
  most convincing overfitting test there is.
* :mod:`~trading.validation.montecarlo` — the drawdown distribution, because the
  historical path is a sample of one.
* :mod:`~trading.validation.experiments` — an append-only count of what was tried,
  so the deflated Sharpe has an honest denominator.
* :mod:`~trading.validation.report` — the aggregate the lifecycle gate reads.
* :mod:`~trading.validation.engine_adapter` — the only module that knows about the
  backtest engine.
"""

from trading.validation.experiments import ExperimentLedger, ExperimentRecord
from trading.validation.harness import Objective, SegmentEvaluator, SegmentOutcome
from trading.validation.montecarlo import (
    MonteCarloResult,
    block_bootstrap_returns,
    resample_trades,
)
from trading.validation.purged_cv import PurgedCvConfig, PurgedCvResult, run_purged_cv
from trading.validation.report import ValidationReport, ValidationThresholds
from trading.validation.sensitivity import ParameterGrid, SensitivitySurface, run_sensitivity
from trading.validation.splits import Split, Window, purged_kfold_splits, walk_forward_splits
from trading.validation.walkforward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardThresholds,
    run_walk_forward,
)

__all__ = [
    "ExperimentLedger",
    "ExperimentRecord",
    "MonteCarloResult",
    "Objective",
    "ParameterGrid",
    "PurgedCvConfig",
    "PurgedCvResult",
    "SegmentEvaluator",
    "SegmentOutcome",
    "SensitivitySurface",
    "Split",
    "ValidationReport",
    "ValidationThresholds",
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardThresholds",
    "Window",
    "block_bootstrap_returns",
    "purged_kfold_splits",
    "resample_trades",
    "run_purged_cv",
    "run_sensitivity",
    "run_walk_forward",
    "walk_forward_splits",
]

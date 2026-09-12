"""The experiment ledger.

Its job is to make the trial count something the machine knows rather than
something the reporter claims. The tests that matter are the ones about
canonicalisation — two spellings of the same parameters counting once — and about
the ledger having no way to forget an unflattering run.
"""

import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from trading.validation.experiments import ExperimentLedger, ExperimentRecord, params_hash


@pytest.fixture
def ledger(tmp_path: Path) -> ExperimentLedger:
    return ExperimentLedger(tmp_path / "experiments.sqlite")


def record(**overrides) -> ExperimentRecord:
    base = {
        "kind": "sensitivity",
        "strategy_id": "sma_cross_param",
        "strategy_version": "1.0.0",
        "instrument": "NSE:RELIANCE",
        "start_date": "2020-01-01",
        "end_date": "2024-01-01",
        "params": {"fast": 50, "slow": 200},
        "data_tier": "PROTOTYPE",
    }
    return ExperimentRecord(**{**base, **overrides})


# ── canonicalisation ────────────────────────────────────────────────────────
def test_key_order_does_not_change_the_fingerprint():
    assert params_hash({"fast": 10, "slow": 50}) == params_hash({"slow": 50, "fast": 10})


def test_equal_numbers_written_differently_are_one_trial():
    """``10``, ``10.0`` and ``Decimal("10")`` are the same parameter to a strategy.

    Counting them separately sounds conservative but makes the ledger's uniqueness
    claim false, and inflates one strategy's count relative to another's.
    """
    assert params_hash({"f": 10}) == params_hash({"f": 10.0}) == params_hash({"f": Decimal("10")})
    assert params_hash({"w": 0.2}) == params_hash({"w": Decimal("0.20")})


def test_booleans_do_not_collapse_into_integers():
    """``bool`` is an ``int`` subclass; ``True`` must not canonicalise to ``1``."""
    assert params_hash({"f": True}) != params_hash({"f": 1})
    assert params_hash({"f": False}) != params_hash({"f": 0})


def test_different_values_are_different_trials():
    assert params_hash({"f": 10}) != params_hash({"f": 11})
    assert params_hash({"f": 10}) != params_hash({"f": 10, "s": 20})


# ── counting ────────────────────────────────────────────────────────────────
def test_distinct_parameters_are_counted_not_repeats(ledger):
    """Re-running the same parameters is one trial; the deflation should not
    punish a reproducibility check."""
    for fast in (10, 20, 30):
        ledger.record(record(params={"fast": fast, "slow": 200}))
    ledger.record(record(params={"slow": 200, "fast": 10.0}))

    assert ledger.trial_count("sma_cross_param") == 3
    assert ledger.evaluation_count("sma_cross_param") == 4


def test_counts_are_scoped_by_instrument(ledger):
    ledger.record(record(params={"fast": 10}, instrument="NSE:RELIANCE"))
    ledger.record(record(params={"fast": 20}, instrument="NSE:TCS"))
    assert ledger.trial_count("sma_cross_param") == 2
    assert ledger.trial_count("sma_cross_param", "NSE:RELIANCE") == 1


def test_an_unknown_strategy_counts_zero_rather_than_raising(ledger):
    assert ledger.trial_count("never_run") == 0
    assert ledger.evaluation_count("never_run") == 0


def test_tiers_used_lets_the_gate_find_out_without_being_told(ledger):
    ledger.record(record(data_tier="SYNTHETIC"))
    ledger.record(record(params={"fast": 20}, data_tier="PROTOTYPE"))
    assert ledger.tiers_used("sma_cross_param") == {"SYNTHETIC", "PROTOTYPE"}


# ── append-only ─────────────────────────────────────────────────────────────
def test_the_ledger_exposes_no_way_to_remove_a_run(ledger):
    """A failed experiment is evidence. The natural thing to delete is exactly the
    run that makes the winner look lucky, so there is no delete."""
    for name in ("delete", "remove", "update", "purge", "clear", "reset"):
        assert not hasattr(ledger, name), f"ExperimentLedger.{name} must not exist"


def test_rows_survive_reopening_the_file(tmp_path):
    path = tmp_path / "experiments.sqlite"
    ExperimentLedger(path).record(record())
    assert ExperimentLedger(path).evaluation_count("sma_cross_param") == 1


def test_every_row_keeps_enough_provenance_to_distrust_it(ledger):
    ledger.record(
        record(
            dataset_version="abc123",
            objective_name="sharpe",
            objective_value=1.25,
            metrics={"sharpe": 1.25, "net_return": 0.3},
            manifest={"content_hash": "deadbeef"},
            note="fold 3",
        )
    )
    with sqlite3.connect(ledger.path) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM experiments").fetchone())
    assert row["dataset_version"] == "abc123"
    assert row["data_tier"] == "PROTOTYPE"
    assert json.loads(row["manifest_json"])["content_hash"] == "deadbeef"
    assert json.loads(row["metrics_json"])["sharpe"] == 1.25
    assert row["recorded_at"].endswith("+00:00"), "timestamps must be unambiguous"


# ── reading back ────────────────────────────────────────────────────────────
def test_best_is_ordered_and_excludes_unscored_rows(ledger):
    ledger.record(record(params={"fast": 10}, objective_value=0.5, objective_name="sharpe"))
    ledger.record(record(params={"fast": 20}, objective_value=1.5, objective_name="sharpe"))
    ledger.record(record(params={"fast": 30}, objective_value=None))
    top = ledger.best("sma_cross_param")
    assert [r["objective_value"] for r in top] == [1.5, 0.5]


def test_strategies_summary_reports_trials_and_evaluations(ledger):
    ledger.record_many(
        [record(params={"fast": f}) for f in (10, 20)] + [record(params={"fast": 10})]
    )
    (row,) = ledger.strategies()
    assert row["trials"] == 2
    assert row["evaluations"] == 3
    assert row["strategy_id"] == "sma_cross_param"


def test_an_empty_ledger_summarises_to_nothing(ledger):
    assert ledger.strategies() == []

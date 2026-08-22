"""Bars, clocks, FX and intents."""

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from trading.core.bar import Bar, BarValidationError
from trading.core.clock import SimulatedClock, SystemClock, ensure_utc
from trading.core.fx import FxConverter, FxRate, MissingRateError
from trading.core.instrument import EXCHANGES, Instrument, InstrumentId
from trading.core.intent import Flat, Intent, TargetQty, TargetWeight
from trading.core.types import Currency, InstrumentClass, Market, Money, Timeframe


# ── clock ──────────────────────────────────────────────────────────────────
def test_naive_datetimes_are_rejected_everywhere():
    with pytest.raises(ValueError, match="timezone-aware"):
        ensure_utc(datetime(2026, 8, 21, 10, 0))


def test_simulated_clock_refuses_to_move_backwards():
    clock = SimulatedClock(datetime(2026, 8, 21, tzinfo=UTC))
    clock.advance_to(datetime(2026, 8, 22, tzinfo=UTC))
    with pytest.raises(ValueError, match="backwards"):
        clock.advance_to(datetime(2026, 8, 21, tzinfo=UTC))


def test_system_clock_is_utc_aware():
    assert SystemClock().now().tzinfo is UTC


# ── bar ────────────────────────────────────────────────────────────────────
def _bar(**kw) -> Bar:
    defaults = {
        "instrument_id": InstrumentId("NSE", "RELIANCE"),
        "timestamp": datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
        "timeframe": Timeframe.DAY_1,
        "open": Decimal(1400),
        "high": Decimal(1420),
        "low": Decimal(1395),
        "close": Decimal(1410),
        "volume": Decimal(5_000_000),
    }
    return Bar(**{**defaults, **kw})


def test_valid_bar_computes_typical_price_and_turnover():
    bar = _bar()
    assert bar.range == Decimal(25)
    assert bar.turnover > 0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("high", Decimal(1380), "high"),
        ("low", Decimal(1450), "low"),
        ("close", Decimal(0), "non-positive"),
        ("volume", Decimal(-1), "negative volume"),
    ],
)
def test_impossible_bars_are_rejected_at_construction(field, value, match):
    with pytest.raises(BarValidationError, match=match):
        _bar(**{field: value})


def test_bar_normalizes_timestamp_to_utc():
    bar = _bar(timestamp=datetime(2026, 8, 21, 15, 30, tzinfo=ZoneInfo("Asia/Kolkata")))
    assert bar.timestamp.tzinfo is UTC
    assert bar.timestamp.hour == 10  # 15:30 IST is 10:00 UTC — the NSE close


# ── fx ─────────────────────────────────────────────────────────────────────
def test_fx_conversion_returns_the_rate_it_used():
    rate = FxRate(Currency.USD, Currency.INR, Decimal("87.4"), datetime.now(UTC), "fixture")
    converted, used = FxConverter([rate]).convert(Money.usd(100), Currency.INR)
    assert converted == Money.inr(Decimal("8740.0"))
    assert used.source == "fixture"


def test_fx_inverts_automatically():
    rate = FxRate(Currency.USD, Currency.INR, Decimal("100"), datetime.now(UTC), "fixture")
    back, used = FxConverter([rate]).convert(Money.inr(100), Currency.USD)
    assert back == Money.usd(1)
    assert "inverted" in used.source


def test_missing_rate_fails_closed_rather_than_guessing():
    with pytest.raises(MissingRateError, match="Refusing to convert"):
        FxConverter().convert(Money.usd(100), Currency.INR)


def test_identity_conversion_needs_no_loaded_rate():
    converted, used = FxConverter().convert(Money.inr(100), Currency.INR)
    assert converted == Money.inr(100)
    assert used.source == "identity"


def test_non_positive_fx_rate_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        FxRate(Currency.USD, Currency.INR, Decimal(0), datetime.now(UTC), "bad")


# ── instrument ─────────────────────────────────────────────────────────────
def test_instrument_id_round_trips_through_its_string_form():
    assert InstrumentId.parse("nse:reliance") == InstrumentId("NSE", "RELIANCE")
    assert str(InstrumentId("NSE", "RELIANCE")) == "NSE:RELIANCE"


def test_instrument_id_rejects_an_unqualified_symbol():
    with pytest.raises(ValueError, match="EXCHANGE:SYMBOL"):
        InstrumentId.parse("RELIANCE")


def test_derivative_without_expiry_is_rejected():
    with pytest.raises(ValueError, match="expiry"):
        Instrument(
            id=InstrumentId("NSE", "NIFTY26AUG25000CE"),
            name="Nifty call",
            instrument_class=InstrumentClass.OPTION,
            currency=Currency.INR,
        )


def test_rounding_to_tick_and_lot(reliance):
    assert reliance.round_to_tick(Decimal("1402.37")) == Decimal("1402.35")
    lotted = Instrument(
        id=InstrumentId("NSE", "NIFTY"),
        name="Nifty",
        instrument_class=InstrumentClass.EQUITY,
        currency=Currency.INR,
        lot_size=Decimal(75),
    )
    assert lotted.round_to_lot(Decimal(200)) == Decimal(150)
    assert lotted.round_to_lot(Decimal(50)) == Decimal(0)


def test_both_indian_exchanges_share_the_same_calendar():
    assert EXCHANGES["NSE"].calendar_code == EXCHANGES["BSE"].calendar_code == "XBOM"
    assert EXCHANGES["NSE"].market is Market.INDIA


# ── intent ─────────────────────────────────────────────────────────────────
def test_target_weight_above_full_equity_is_rejected():
    with pytest.raises(ValueError, match="exceeds 100%"):
        TargetWeight(Decimal("1.5"))


def test_confidence_must_be_a_probability():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Intent(InstrumentId("NSE", "X"), Flat(), "s", "r", confidence=Decimal("1.4"))


def test_every_target_form_normalizes_to_a_weight():
    equity, price, mult = Decimal(100_000), Decimal(1000), Decimal(1)

    def intent_for(target):
        return Intent(InstrumentId("NSE", "X"), target, "s", "r")

    assert intent_for(Flat()).target_weight(equity, price, mult) == 0
    assert intent_for(TargetWeight(Decimal("0.04"))).target_weight(equity, price, mult) == Decimal(
        "0.04"
    )
    assert intent_for(TargetQty(Decimal(4))).target_weight(equity, price, mult) == Decimal("0.04")

"""Configuration guards and jurisdiction profiles."""

import datetime as dt
from decimal import Decimal

import pytest

from trading.calendars.base import calendar_for
from trading.config.compliance import INDIA_SEBI, US_RETAIL, profile_for
from trading.config.settings import Settings
from trading.core.instrument import NSE
from trading.core.types import Currency, Market, TradingMode


def test_default_mode_is_research():
    assert Settings().mode is TradingMode.RESEARCH


def test_live_mode_is_refused_because_its_safety_gates_do_not_exist():
    with pytest.raises(ValueError, match="live trading is not implemented"):
        Settings(mode=TradingMode.LIVE)


def test_base_currency_must_match_the_configured_market():
    with pytest.raises(ValueError, match="does not match market"):
        Settings(market=Market.INDIA, base_currency=Currency.USD)


def test_negative_capital_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        Settings(starting_capital=Decimal(-1))


def test_india_profile_encodes_the_sebi_constraints():
    p = profile_for(Market.INDIA)
    assert p is INDIA_SEBI
    assert p.max_orders_per_second == 10
    assert p.requires_static_ip
    assert p.requires_daily_session_logout
    assert p.requires_algo_id_tagging
    assert p.requires_domestic_hosting
    assert p.version == "2026.04.01"


def test_us_profile_encodes_the_pdt_rule():
    p = profile_for(Market.US)
    assert p is US_RETAIL
    assert p.pattern_day_trader_rule
    assert p.pdt_min_equity == Decimal(25_000)
    assert p.pdt_max_day_trades == 3


def test_settings_expose_the_compliance_profile_for_the_run_manifest():
    assert Settings(market=Market.INDIA).compliance.profile_id == "india_sebi_retail_algo"


# ── calendar ───────────────────────────────────────────────────────────────
def test_nse_calendar_knows_indian_public_holidays():
    cal = calendar_for(NSE.calendar_code)
    assert not cal.is_session(dt.date(2026, 1, 26))  # Republic Day
    assert not cal.is_session(dt.date(2026, 8, 15))  # Independence Day
    assert not cal.is_session(dt.date(2026, 8, 22))  # a Saturday


def test_nse_session_close_is_1530_ist_expressed_in_utc():
    cal = calendar_for(NSE.calendar_code)
    sessions = cal.sessions_between(dt.date(2026, 8, 17), dt.date(2026, 8, 21))
    assert len(sessions) == 5
    close = cal.session_close_utc(dt.date(2026, 8, 21))
    assert (close.hour, close.minute) == (10, 0)  # 15:30 IST

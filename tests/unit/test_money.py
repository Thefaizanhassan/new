"""Money must be exact and currency-safe. Phase 0 §5.1."""

from decimal import Decimal

import pytest

from trading.core.types import Currency, CurrencyMismatchError, Money


def test_decimal_arithmetic_is_exact_where_float_is_not():
    total = sum((Money.inr("0.1") for _ in range(10)), Money.zero(Currency.INR))
    assert total.amount == Decimal("1.0")

    float_total = 0.0
    for _ in range(10):
        float_total += 0.1
    assert float_total != 1.0, "the hazard Decimal exists to avoid"


def test_combining_currencies_raises_rather_than_guessing_a_rate():
    with pytest.raises(CurrencyMismatchError):
        Money.inr(100) + Money.usd(100)


def test_settled_rounds_to_minor_units_only_at_settlement():
    m = Money.inr("100.005")
    assert m.amount == Decimal("100.005")  # precision retained internally
    assert m.settled().amount == Decimal("100.01")


def test_ratio_to_zero_is_zero_not_an_exception():
    assert Money.inr(50).ratio_to(Money.zero(Currency.INR)) == Decimal(0)


def test_comparison_and_display():
    assert Money.inr(10) < Money.inr(20)
    assert str(Money.inr("123456.789")) == "₹123,456.79"
    assert str(Money.usd("1234.5")) == "$1,234.50"

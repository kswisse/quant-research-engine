"""Arbitrage detection regression tests.

Covers same-market YES+NO arbitrage detection for binary prediction markets.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_arbitrage.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.arbitrage import (
    detect_same_market_arbitrage,
)
from quant_engine.order_book import OrderBookLevel, OrderBookSnapshot

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN_CLOCK = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
SOURCE_TIME = datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC)


def _make_book(
    asks: list[tuple[float, float]],
    provider: str = "test",
    instrument: str = "YES",
) -> OrderBookSnapshot:
    """Create a test order book with specified ask levels."""
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in asks]
    return OrderBookSnapshot(
        provider=provider,
        provider_instrument_id=instrument,
        source_timestamp=SOURCE_TIME,
        ingestion_timestamp=FROZEN_CLOCK,
        bids=(),
        asks=tuple(ask_levels),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASIC PROFITABLE CASE
# ══════════════════════════════════════════════════════════════════════════════


class TestBasicProfitableCase:
    """Simple arbitrage detection with clear profit."""

    def test_basic_arbitrage_detected(self) -> None:
        """YES ask = 0.45, NO ask = 0.50, sum = 0.95 < 1.0."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)

        assert result is not None
        assert result.executable_size == 100.0
        assert result.total_cost == pytest.approx(95.0)
        assert result.guaranteed_payoff == 100.0
        assert result.gross_profit == pytest.approx(5.0)
        assert result.gross_return == pytest.approx(5.0 / 95.0)
        assert result.fully_executable is True

    def test_profitability_boundary(self) -> None:
        """First level profitable, second level unprofitable."""
        yes_book = _make_book(asks=[(0.45, 100.0), (0.48, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0), (0.52, 100.0)], instrument="NO")

        # For 100 pairs: YES cost = 45, NO cost = 50, total = 95, profit = 5
        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.executable_size == 100.0

        # For 200 pairs: YES cost = 45 + 48 = 93, NO cost = 50 + 52 = 102
        # total = 195, payoff = 200, profit = 5 (still profitable!)
        result2 = detect_same_market_arbitrage(yes_book, no_book, 200.0)
        assert result2 is not None
        assert result2.executable_size == 200.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. NO ARBITRAGE
# ══════════════════════════════════════════════════════════════════════════════


class TestNoArbitrage:
    """Cases where no arbitrage opportunity exists."""

    def test_fair_pricing(self) -> None:
        """YES ask = 0.50, NO ask = 0.50, sum = 1.00."""
        yes_book = _make_book(asks=[(0.50, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None

    def test_negative_edge(self) -> None:
        """YES ask = 0.52, NO ask = 0.50, sum = 1.02 > 1.0."""
        yes_book = _make_book(asks=[(0.52, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None

    def test_zero_profit_excluded(self) -> None:
        """Sum exactly 1.00 — zero profit excluded."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.55, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 3. PARTIAL LIQUIDITY
# ══════════════════════════════════════════════════════════════════════════════


class TestPartialLiquidity:
    """Cases where one side has less liquidity than the other."""

    def test_yes_less_liquidity(self) -> None:
        """YES has 50 contracts, NO has 100. Only 50 pairs executable."""
        yes_book = _make_book(asks=[(0.45, 50.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)

        assert result is not None
        assert result.executable_size == 50.0
        assert result.fully_executable is False

    def test_no_less_liquidity(self) -> None:
        """YES has 100 contracts, NO has 50. Only 50 pairs executable."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 50.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)

        assert result is not None
        assert result.executable_size == 50.0
        assert result.fully_executable is False

    def test_requested_size_cap(self) -> None:
        """Profitable liquidity > requested_size. Only requested_size reported."""
        yes_book = _make_book(asks=[(0.45, 200.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 200.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)

        assert result is not None
        assert result.executable_size == 100.0
        assert result.fully_executable is True


# ══════════════════════════════════════════════════════════════════════════════
# 4. MULTI-LEVEL DEPTH
# ══════════════════════════════════════════════════════════════════════════════


class TestMultiLevelDepth:
    """Cases with multiple price levels on both sides."""

    def test_multi_level_both_sides(self) -> None:
        """Both YES and NO have multiple levels."""
        yes_book = _make_book(asks=[(0.45, 100.0), (0.48, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0), (0.51, 100.0)], instrument="NO")

        # 100 pairs: YES cost = 45, NO cost = 50, total = 95
        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.executable_size == 100.0
        assert result.total_cost == pytest.approx(95.0)

    def test_multi_level_depth_analysis(self) -> None:
        """Deeper levels have worse prices — verify economics."""
        yes_book = _make_book(asks=[(0.40, 100.0), (0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0), (0.55, 100.0)], instrument="NO")

        # 100 pairs: YES cost = 40, NO cost = 50, total = 90
        result100 = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result100 is not None
        assert result100.total_cost == pytest.approx(90.0)
        assert result100.gross_profit == pytest.approx(10.0)

        # 200 pairs: YES cost = 40 + 45 = 85, NO cost = 50 + 55 = 105
        # total = 190, payoff = 200, profit = 10
        result200 = detect_same_market_arbitrage(yes_book, no_book, 200.0)
        assert result200 is not None
        assert result200.total_cost == pytest.approx(190.0)
        assert result200.gross_profit == pytest.approx(10.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. EMPTY BOOKS
# ══════════════════════════════════════════════════════════════════════════════


class TestEmptyBooks:
    """Empty book behavior."""

    def test_empty_yes_asks(self) -> None:
        """Empty YES asks — no arbitrage."""
        yes_book = _make_book(asks=[], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None

    def test_empty_no_asks(self) -> None:
        """Empty NO asks — no arbitrage."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None

    def test_both_empty(self) -> None:
        """Both books empty — no arbitrage."""
        yes_book = _make_book(asks=[], instrument="YES")
        no_book = _make_book(asks=[], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 6. INPUT VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


class TestInputValidation:
    """Invalid input handling."""

    def test_zero_requested_size(self) -> None:
        """Zero requested_size raises ValueError."""
        yes_book = _make_book(asks=[(0.45, 100.0)])
        no_book = _make_book(asks=[(0.50, 100.0)])

        with pytest.raises(ValueError, match="requested_size"):
            detect_same_market_arbitrage(yes_book, no_book, 0.0)

    def test_negative_requested_size(self) -> None:
        """Negative requested_size raises ValueError."""
        yes_book = _make_book(asks=[(0.45, 100.0)])
        no_book = _make_book(asks=[(0.50, 100.0)])

        with pytest.raises(ValueError, match="requested_size"):
            detect_same_market_arbitrage(yes_book, no_book, -10.0)

    def test_nan_requested_size(self) -> None:
        """NaN requested_size raises ValueError."""
        yes_book = _make_book(asks=[(0.45, 100.0)])
        no_book = _make_book(asks=[(0.50, 100.0)])

        with pytest.raises(ValueError, match="requested_size"):
            detect_same_market_arbitrage(yes_book, no_book, float("nan"))

    def test_inf_requested_size(self) -> None:
        """Infinity requested_size raises ValueError."""
        yes_book = _make_book(asks=[(0.45, 100.0)])
        no_book = _make_book(asks=[(0.50, 100.0)])

        with pytest.raises(ValueError, match="requested_size"):
            detect_same_market_arbitrage(yes_book, no_book, float("inf"))


# ══════════════════════════════════════════════════════════════════════════════
# 7. PROVIDER MISMATCH
# ══════════════════════════════════════════════════════════════════════════════


class TestProviderMismatch:
    """Different providers are rejected."""

    def test_different_providers_rejected(self) -> None:
        """YES and NO from different providers raises ValueError."""
        yes_book = _make_book(asks=[(0.45, 100.0)], provider="polymarket", instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], provider="kalshi", instrument="NO")

        with pytest.raises(ValueError, match="same provider"):
            detect_same_market_arbitrage(yes_book, no_book, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# 8. ACCOUNTING INVARIANTS
# ══════════════════════════════════════════════════════════════════════════════


class TestAccountingInvariants:
    """Verify accounting invariants for all detected opportunities."""

    def test_executable_size_positive(self) -> None:
        """executable_size > 0 when opportunity exists."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.executable_size > 0

    def test_total_cost_positive(self) -> None:
        """total_cost > 0 when opportunity exists."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.total_cost > 0

    def test_guaranteed_payoff_equals_executable(self) -> None:
        """guaranteed_payoff == executable_size."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.guaranteed_payoff == result.executable_size

    def test_gross_profit_positive(self) -> None:
        """gross_profit > 0 when opportunity exists."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.gross_profit > 0

    def test_gross_return_formula(self) -> None:
        """gross_return == gross_profit / total_cost."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.gross_return == pytest.approx(result.gross_profit / result.total_cost)

    def test_executable_size_leq_requested(self) -> None:
        """executable_size <= requested_size."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.executable_size <= result.requested_size

    def test_total_cost_less_than_payoff(self) -> None:
        """total_cost < guaranteed_payoff (profitability condition)."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.total_cost < result.guaranteed_payoff


# ══════════════════════════════════════════════════════════════════════════════
# 9. EXECUTION DETAILS
# ══════════════════════════════════════════════════════════════════════════════


class TestExecutionDetails:
    """Verify execution results are correct."""

    def test_yes_execution_is_buy(self) -> None:
        """YES execution is a buy order."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.yes_execution.side == "buy"

    def test_no_execution_is_buy(self) -> None:
        """NO execution is a buy order."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.no_execution.side == "buy"

    def test_yes_vwap_equals_price(self) -> None:
        """YES VWAP equals the ask price for single level."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.yes_execution.vwap == pytest.approx(0.45)

    def test_no_vwap_equals_price(self) -> None:
        """NO VWAP equals the ask price for single level."""
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")

        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.no_execution.vwap == pytest.approx(0.50)


# ══════════════════════════════════════════════════════════════════════════════
# 10. PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-based tests for invariants."""

    @given(
        yes_price=st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
        no_price=st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_profitable_when_sum_below_one(
        self, yes_price: float, no_price: float, size: float
    ) -> None:
        """When YES + NO < 1.0, arbitrage is detected."""
        yes_book = _make_book(asks=[(yes_price, size)])
        no_book = _make_book(asks=[(no_price, size)])

        if yes_price + no_price < 1.0:
            result = detect_same_market_arbitrage(yes_book, no_book, size)
            assert result is not None
            assert result.gross_profit > 0

    @given(
        yes_price=st.floats(min_value=0.51, max_value=0.99, allow_nan=False),
        no_price=st.floats(min_value=0.51, max_value=0.99, allow_nan=False),
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_no_profit_when_sum_above_one(
        self, yes_price: float, no_price: float, size: float
    ) -> None:
        """When YES + NO > 1.0, no arbitrage."""
        yes_book = _make_book(asks=[(yes_price, size)])
        no_book = _make_book(asks=[(no_price, size)])

        result = detect_same_market_arbitrage(yes_book, no_book, size)
        assert result is None

"""Cross-venue arbitrage detection and cost evaluation tests.

Covers cross-venue prediction market arbitrage detection and cost evaluation.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_cross_venue_arbitrage.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    evaluate_cross_venue_costs,
)
from quant_engine.arbitrage.cross_venue import (
    CrossVenueOpportunity,
    detect_cross_venue_arbitrage,
)
from quant_engine.order_book import OrderBookLevel, OrderBookSnapshot

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN_CLOCK = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
SOURCE_TIME = datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC)


def _make_book(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    provider: str = "test",
    instrument: str = "YES",
) -> OrderBookSnapshot:
    """Create a test order book with specified bid/ask levels."""
    bid_levels = [OrderBookLevel(price=p, size=s) for p, s in (bids or [])]
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in (asks or [])]
    return OrderBookSnapshot(
        provider=provider,
        provider_instrument_id=instrument,
        source_timestamp=SOURCE_TIME,
        ingestion_timestamp=FROZEN_CLOCK,
        bids=tuple(bid_levels),
        asks=tuple(ask_levels),
    )


def _detect_cross_venue(
    buy_asks: list[tuple[float, float]],
    sell_bids: list[tuple[float, float]],
    size: float = 100.0,
    buy_provider: str = "venue_a",
    sell_provider: str = "venue_b",
) -> CrossVenueOpportunity:
    """Helper to detect cross-venue arbitrage opportunity."""
    buy_book = _make_book(asks=buy_asks, provider=buy_provider, instrument="YES")
    sell_book = _make_book(bids=sell_bids, provider=sell_provider, instrument="YES")
    result = detect_cross_venue_arbitrage(buy_book, sell_book, size)
    assert result is not None, "Expected cross-venue arbitrage opportunity"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1. DIRECTION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDirection:
    """Test arbitrage detection in both directions."""

    def test_a_cheaper_than_b(self) -> None:
        """Venue A ask < Venue B bid — buy A, sell B."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )

        assert result.buy_venue == "venue_a"
        assert result.sell_venue == "venue_b"
        assert result.gross_spread == pytest.approx(5.0)

    def test_b_cheaper_than_a(self) -> None:
        """Venue B ask < Venue A bid — buy B, sell A."""
        result = _detect_cross_venue(
            buy_asks=[(0.35, 100.0)],
            sell_bids=[(0.50, 100.0)],
            buy_provider="venue_b",
            sell_provider="venue_a",
        )

        assert result.buy_venue == "venue_b"
        assert result.sell_venue == "venue_a"
        assert result.gross_spread == pytest.approx(15.0)

    def test_no_arbitrage(self) -> None:
        """No profitable spread — no opportunity."""
        buy_book = _make_book(asks=[(0.45, 100.0)], provider="venue_a")
        sell_book = _make_book(bids=[(0.40, 100.0)], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None

    def test_equal_prices(self) -> None:
        """Equal prices — no opportunity."""
        buy_book = _make_book(asks=[(0.45, 100.0)], provider="venue_a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None

    def test_zero_gross_edge(self) -> None:
        """Zero spread — no opportunity."""
        buy_book = _make_book(asks=[(0.45, 100.0)], provider="venue_a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. DEPTH TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestDepth:
    """Test depth-aware execution."""

    def test_single_level_books(self) -> None:
        """Single level on both sides."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )

        assert result.executable_size == 100.0
        assert result.gross_spread == pytest.approx(5.0)

    def test_multi_level_buy_book(self) -> None:
        """Multiple levels on buy side."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 50.0), (0.42, 50.0)],
            sell_bids=[(0.45, 100.0)],
        )

        # First 50 at 0.40, next 50 at 0.42
        assert result.executable_size == 100.0
        # Cost = 50*0.40 + 50*0.42 = 20 + 21 = 41
        # Proceeds = 100*0.45 = 45
        assert result.gross_spread == pytest.approx(4.0)

    def test_multi_level_sell_book(self) -> None:
        """Multiple levels on sell side."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 50.0), (0.43, 50.0)],
        )

        # First 50 at 0.45, next 50 at 0.43
        assert result.executable_size == 100.0
        # Cost = 100*0.40 = 40
        # Proceeds = 50*0.45 + 50*0.43 = 22.5 + 21.5 = 44
        assert result.gross_spread == pytest.approx(4.0)

    def test_asymmetric_liquidity(self) -> None:
        """Different liquidity on each side."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 50.0)],
            sell_bids=[(0.45, 100.0)],
        )

        assert result.executable_size == 50.0

    def test_partial_execution(self) -> None:
        """Requested size exceeds available liquidity."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 30.0)],
            sell_bids=[(0.45, 80.0)],
            size=100.0,
        )

        assert result.executable_size == 30.0
        assert result.fully_executable is False

    def test_fully_executable(self) -> None:
        """Requested size fully executable."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 200.0)],
            sell_bids=[(0.45, 200.0)],
            size=100.0,
        )

        assert result.executable_size == 100.0
        assert result.fully_executable is True


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPOSURE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestExposure:
    """Test unpaired exposure calculations."""

    def test_buy_overfill(self) -> None:
        """Buy has more liquidity than sell — detector limits to sell's liquidity."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 200.0)],
            sell_bids=[(0.45, 100.0)],
        )

        assert result.executable_size == 100.0
        assert result.buy_execution.filled_size == 100.0
        assert result.sell_execution.filled_size == 100.0

    def test_sell_overfill(self) -> None:
        """Sell has more liquidity than buy — detector limits to buy's liquidity."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 200.0)],
        )

        assert result.executable_size == 100.0
        assert result.buy_execution.filled_size == 100.0
        assert result.sell_execution.filled_size == 100.0

    def test_paired_size_invariants(self) -> None:
        """Verify paired size invariants."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 50.0)],
            sell_bids=[(0.45, 100.0)],
        )

        assert result.executable_size <= result.buy_execution.filled_size
        assert result.executable_size <= result.sell_execution.filled_size

    def test_execution_identity(self) -> None:
        """executable_size == filled on both legs (detector ensures equal fills)."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 75.0)],
            sell_bids=[(0.45, 125.0)],
        )

        # Detector recomputes to ensure equal fills
        assert result.buy_execution.filled_size == result.executable_size
        assert result.sell_execution.filled_size == result.executable_size


# ══════════════════════════════════════════════════════════════════════════════
# 4. ECONOMICS TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEconomics:
    """Test gross profit and return calculations."""

    def test_gross_profit(self) -> None:
        """Gross profit equals proceeds minus cost."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )

        # Cost = 100*0.40 = 40
        # Proceeds = 100*0.45 = 45
        assert result.gross_spread == pytest.approx(5.0)

    def test_gross_return(self) -> None:
        """Gross return equals spread divided by cost."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )

        # Return = 5.0 / 40.0 = 0.125
        assert result.gross_return == pytest.approx(0.125)

    def test_zero_buy_cost_return(self) -> None:
        """Zero buy cost results in None return."""
        result = _detect_cross_venue(
            buy_asks=[(0.0, 100.0)],
            sell_bids=[(0.05, 100.0)],
        )

        # Cost = 0, proceeds = 5, spread = 5
        # Return should be None (division by zero)
        assert result.gross_return is None


# ══════════════════════════════════════════════════════════════════════════════
# 5. COST MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCostModel:
    """Test cost model integration."""

    def test_fees_reduce_profit(self) -> None:
        """Buy and sell fees reduce net profit."""
        opp = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )
        model = ArbitrageCostModel(buy_fee_rate=0.02, sell_fee_rate=0.03)

        result = evaluate_cross_venue_costs(opp, model)

        # Buy fee = 40 * 0.02 = 0.8
        # Sell fee = 45 * 0.03 = 1.35
        assert result.buy_fee == pytest.approx(0.8)
        assert result.sell_fee == pytest.approx(1.35)
        assert result.net_spread < result.gross_spread

    def test_fixed_cost_reduces_profit(self) -> None:
        """Fixed cost reduces net profit."""
        opp = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )
        model = ArbitrageCostModel(fixed_cost=1.0)

        result = evaluate_cross_venue_costs(opp, model)

        assert result.fixed_cost == 1.0
        assert result.net_spread == pytest.approx(result.gross_spread - 1.0)

    def test_transfer_cost_reduces_profit(self) -> None:
        """Transfer cost reduces net profit."""
        opp = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )
        model = ArbitrageCostModel(transfer_cost=0.5)

        result = evaluate_cross_venue_costs(opp, model)

        assert result.transfer_cost == 0.5
        assert result.net_spread == pytest.approx(result.gross_spread - 0.5)

    def test_costs_can_turn_positive_gross_to_negative_net(self) -> None:
        """High costs can make net profit negative."""
        opp = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.42, 100.0)],
        )
        # Spread = 2.0, but fees + transfer = 3.0
        model = ArbitrageCostModel(
            buy_fee_rate=0.02,
            sell_fee_rate=0.02,
            transfer_cost=2.0,
        )

        result = evaluate_cross_venue_costs(opp, model)

        assert result.gross_spread > 0
        assert result.net_spread < 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestIdentity:
    """Test market identity validation."""

    def test_same_canonical_market_accepted(self) -> None:
        """Same outcome from different venues is accepted."""
        result = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
            buy_provider="venue_a",
            sell_provider="venue_b",
        )

        assert result.outcome_label == "outcome"

    def test_same_provider_rejected(self) -> None:
        """Same provider for both legs raises ValueError."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="same_venue")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="same_venue")

        with pytest.raises(ValueError, match="different providers"):
            detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    """Test input validation."""

    def test_zero_requested_size(self) -> None:
        """Zero requested size raises ValueError."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="b")

        with pytest.raises(ValueError, match="requested_size"):
            detect_cross_venue_arbitrage(buy_book, sell_book, 0.0)

    def test_negative_requested_size(self) -> None:
        """Negative requested size raises ValueError."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="b")

        with pytest.raises(ValueError, match="requested_size"):
            detect_cross_venue_arbitrage(buy_book, sell_book, -10.0)

    def test_nan_requested_size(self) -> None:
        """NaN requested size raises ValueError."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="b")

        with pytest.raises(ValueError, match="requested_size"):
            detect_cross_venue_arbitrage(buy_book, sell_book, float("nan"))

    def test_inf_requested_size(self) -> None:
        """Infinity requested size raises ValueError."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="b")

        with pytest.raises(ValueError, match="requested_size"):
            detect_cross_venue_arbitrage(buy_book, sell_book, float("inf"))


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-based tests for invariants."""

    @given(
        buy_price=st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
        sell_price=st.floats(min_value=0.51, max_value=0.99, allow_nan=False),
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_profitable_when_buy_below_sell(
        self, buy_price: float, sell_price: float, size: float
    ) -> None:
        """When buy < sell, arbitrage is detected."""
        opp = _detect_cross_venue(
            buy_asks=[(buy_price, size)],
            sell_bids=[(sell_price, size)],
            size=size,
        )

        assert opp.gross_spread > 0

    @given(
        buy_fee=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
        sell_fee=st.floats(min_value=0.0, max_value=0.1, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_net_spread_leq_gross_spread(
        self, buy_fee: float, sell_fee: float
    ) -> None:
        """Net spread <= gross spread for non-negative costs."""
        opp = _detect_cross_venue(
            buy_asks=[(0.40, 100.0)],
            sell_bids=[(0.45, 100.0)],
        )
        model = ArbitrageCostModel(buy_fee_rate=buy_fee, sell_fee_rate=sell_fee)

        result = evaluate_cross_venue_costs(opp, model)

        assert result.net_spread <= result.gross_spread + 1e-10


# ══════════════════════════════════════════════════════════════════════════════
# 9. EMPTY BOOKS
# ══════════════════════════════════════════════════════════════════════════════


class TestEmptyBooks:
    """Test empty order book behavior."""

    def test_empty_buy_asks(self) -> None:
        """Empty buy book asks — no opportunity."""
        buy_book = _make_book(asks=[], provider="venue_a")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None

    def test_empty_sell_bids(self) -> None:
        """Empty sell book bids — no opportunity."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="venue_a")
        sell_book = _make_book(bids=[], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None

    def test_both_empty(self) -> None:
        """Both books empty — no opportunity."""
        buy_book = _make_book(asks=[], provider="venue_a")
        sell_book = _make_book(bids=[], provider="venue_b")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 10. COST MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


class TestCostModelValidation:
    """Test cross-venue cost model validation."""

    def test_negative_buy_fee_rate_rejected(self) -> None:
        """Negative buy fee rate raises ValueError."""
        with pytest.raises(ValueError, match="buy_fee_rate must be >= 0"):
            ArbitrageCostModel(buy_fee_rate=-0.01)

    def test_negative_sell_fee_rate_rejected(self) -> None:
        """Negative sell fee rate raises ValueError."""
        with pytest.raises(ValueError, match="sell_fee_rate must be >= 0"):
            ArbitrageCostModel(sell_fee_rate=-0.01)

    def test_buy_fee_rate_above_one_rejected(self) -> None:
        """Buy fee rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="buy_fee_rate must be <= 1.0"):
            ArbitrageCostModel(buy_fee_rate=1.5)

    def test_sell_fee_rate_above_one_rejected(self) -> None:
        """Sell fee rate > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="sell_fee_rate must be <= 1.0"):
            ArbitrageCostModel(sell_fee_rate=1.5)

    def test_buy_fee_rate_at_boundary(self) -> None:
        """Buy fee rate exactly 1.0 is valid."""
        model = ArbitrageCostModel(buy_fee_rate=1.0)
        assert model.buy_fee_rate == 1.0

    def test_sell_fee_rate_at_boundary(self) -> None:
        """Sell fee rate exactly 1.0 is valid."""
        model = ArbitrageCostModel(sell_fee_rate=1.0)
        assert model.sell_fee_rate == 1.0

    def test_negative_transfer_cost_rejected(self) -> None:
        """Negative transfer cost raises ValueError."""
        with pytest.raises(ValueError, match="transfer_cost must be >= 0"):
            ArbitrageCostModel(transfer_cost=-1.0)

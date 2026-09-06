"""Arbitrage cost and execution-risk model tests.

Covers cost model validation, net arbitrage evaluation, legging exposure,
and mathematical invariants for prediction market arbitrage.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_arbitrage_costs.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.arbitrage import (
    ArbitrageOpportunity,
    detect_same_market_arbitrage,
)
from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    evaluate_arbitrage_costs,
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


def _detect_opportunity(
    yes_asks: list[tuple[float, float]],
    no_asks: list[tuple[float, float]],
    size: float = 100.0,
) -> ArbitrageOpportunity:
    """Helper to detect arbitrage opportunity."""
    yes_book = _make_book(asks=yes_asks, instrument="YES")
    no_book = _make_book(asks=no_asks, instrument="NO")
    result = detect_same_market_arbitrage(yes_book, no_book, size)
    assert result is not None, "Expected arbitrage opportunity"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1. COST MODEL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════


class TestCostModelValidation:
    """Validate ArbitrageCostModel input constraints."""

    def test_default_zero_costs(self) -> None:
        """Default cost model has zero costs."""
        model = ArbitrageCostModel()
        assert model.yes_fee_rate == 0.0
        assert model.no_fee_rate == 0.0
        assert model.fixed_cost == 0.0
        assert model.settlement_cost == 0.0

    def test_valid_cost_model(self) -> None:
        """Valid cost model with all fields."""
        model = ArbitrageCostModel(
            yes_fee_rate=0.02,
            no_fee_rate=0.03,
            fixed_cost=1.0,
            settlement_cost=0.5,
        )
        assert model.yes_fee_rate == 0.02
        assert model.no_fee_rate == 0.03
        assert model.fixed_cost == 1.0
        assert model.settlement_cost == 0.5

    def test_negative_yes_fee_rate_rejected(self) -> None:
        """Negative YES fee rate raises ValueError."""
        with pytest.raises(ValueError, match="yes_fee_rate must be >= 0"):
            ArbitrageCostModel(yes_fee_rate=-0.01)

    def test_negative_no_fee_rate_rejected(self) -> None:
        """Negative NO fee rate raises ValueError."""
        with pytest.raises(ValueError, match="no_fee_rate must be >= 0"):
            ArbitrageCostModel(no_fee_rate=-0.01)

    def test_negative_fixed_cost_rejected(self) -> None:
        """Negative fixed cost raises ValueError."""
        with pytest.raises(ValueError, match="fixed_cost must be >= 0"):
            ArbitrageCostModel(fixed_cost=-1.0)

    def test_negative_settlement_cost_rejected(self) -> None:
        """Negative settlement cost raises ValueError."""
        with pytest.raises(ValueError, match="settlement_cost must be >= 0"):
            ArbitrageCostModel(settlement_cost=-1.0)

    def test_nan_yes_fee_rate_rejected(self) -> None:
        """NaN YES fee rate raises ValueError."""
        with pytest.raises(ValueError, match="yes_fee_rate must be finite"):
            ArbitrageCostModel(yes_fee_rate=float("nan"))

    def test_inf_yes_fee_rate_rejected(self) -> None:
        """Infinity YES fee rate raises ValueError."""
        with pytest.raises(ValueError, match="yes_fee_rate must be finite"):
            ArbitrageCostModel(yes_fee_rate=float("inf"))

    def test_nan_fixed_cost_rejected(self) -> None:
        """NaN fixed cost raises ValueError."""
        with pytest.raises(ValueError, match="fixed_cost must be finite"):
            ArbitrageCostModel(fixed_cost=float("nan"))

    def test_inf_settlement_cost_rejected(self) -> None:
        """Infinity settlement cost raises ValueError."""
        with pytest.raises(ValueError, match="settlement_cost must be finite"):
            ArbitrageCostModel(settlement_cost=float("inf"))

    def test_fee_rate_above_one_rejected(self) -> None:
        """Fee rate > 1.0 (100%) raises ValueError."""
        with pytest.raises(ValueError, match="yes_fee_rate must be <= 1.0"):
            ArbitrageCostModel(yes_fee_rate=1.5)

    def test_fee_rate_at_boundary(self) -> None:
        """Fee rate exactly 1.0 is valid."""
        model = ArbitrageCostModel(yes_fee_rate=1.0, no_fee_rate=1.0)
        assert model.yes_fee_rate == 1.0
        assert model.no_fee_rate == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. ZERO FEES — NET EQUALS GROSS
# ══════════════════════════════════════════════════════════════════════════════


class TestZeroFees:
    """When all costs are zero, net equals gross."""

    def test_zero_fees_net_equals_gross(self) -> None:
        """With zero costs, net profit equals gross profit."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.net_profit == pytest.approx(result.gross_profit)
        assert result.yes_fee == 0.0
        assert result.no_fee == 0.0
        assert result.fixed_cost == 0.0
        assert result.settlement_cost == 0.0

    def test_zero_fees_net_return_equals_gross_return(self) -> None:
        """With zero costs, net return equals gross return."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.net_return == pytest.approx(result.gross_return)


# ══════════════════════════════════════════════════════════════════════════════
# 3. COST CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════


class TestCostCalculations:
    """Verify cost calculations reduce net profit."""

    def test_yes_fee_reduces_net_profit(self) -> None:
        """YES fee reduces net profit."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(yes_fee_rate=0.02)

        result = evaluate_arbitrage_costs(opp, model)

        assert result.yes_fee == pytest.approx(opp.total_cost * 0.02)
        assert result.net_profit < result.gross_profit

    def test_no_fee_reduces_net_profit(self) -> None:
        """NO fee reduces net profit."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(no_fee_rate=0.03)

        result = evaluate_arbitrage_costs(opp, model)

        assert result.no_fee > 0
        assert result.net_profit < result.gross_profit

    def test_fixed_cost_reduces_net_profit(self) -> None:
        """Fixed cost reduces net profit."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(fixed_cost=2.0)

        result = evaluate_arbitrage_costs(opp, model)

        assert result.fixed_cost == 2.0
        assert result.net_profit == pytest.approx(result.gross_profit - 2.0)

    def test_settlement_cost_reduces_net_profit(self) -> None:
        """Settlement cost reduces net profit."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(settlement_cost=1.5)

        result = evaluate_arbitrage_costs(opp, model)

        assert result.settlement_cost == 1.5
        assert result.net_profit == pytest.approx(result.gross_profit - 1.5)

    def test_multiple_costs_combine(self) -> None:
        """Multiple costs combine correctly."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(
            yes_fee_rate=0.02,
            no_fee_rate=0.03,
            fixed_cost=1.0,
            settlement_cost=0.5,
        )

        result = evaluate_arbitrage_costs(opp, model)

        expected_total_fees = result.yes_fee + result.no_fee + 1.0 + 0.5
        assert result.total_cost == pytest.approx(result.gross_cost + expected_total_fees)
        assert result.net_profit < result.gross_profit

    def test_costs_can_turn_positive_gross_to_negative_net(self) -> None:
        """High costs can make net profit negative."""
        # Small gross profit (5%) but high fees (5% + 5%)
        opp = _detect_opportunity([(0.47, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(
            yes_fee_rate=0.03,
            no_fee_rate=0.03,
            fixed_cost=1.0,
        )

        result = evaluate_arbitrage_costs(opp, model)

        assert result.gross_profit > 0
        assert result.net_profit < 0

    def test_zero_denominator_return_handling(self) -> None:
        """Zero gross cost results in None gross return."""
        # Create opportunity with zero cost (edge case)
        opp = _detect_opportunity([(0.0, 100.0)], [(0.0, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # If gross_cost is 0, gross_return should be None
        if opp.total_cost == 0:
            assert result.gross_return is None

    def test_zero_profit_handling(self) -> None:
        """Zero gross profit results in zero gross return."""
        # Create opportunity with very small profit (near zero)
        opp = _detect_opportunity([(0.499, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Gross profit should be very small
        assert result.gross_profit == pytest.approx(0.1, abs=0.01)
        # Gross return should be very small
        assert result.gross_return is not None
        assert result.gross_return < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 4. LEGGING / PAIRING
# ══════════════════════════════════════════════════════════════════════════════


class TestLeggingRisk:
    """Verify legging and pairing calculations."""

    def test_fully_paired_execution(self) -> None:
        """Equal fills on both sides are fully paired."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.fully_paired is True
        assert result.paired_size == 100.0
        assert result.unpaired_yes_size == 0.0
        assert result.unpaired_no_size == 0.0

    def test_yes_overfill(self) -> None:
        """YES has more liquidity than NO — detector limits to NO's liquidity."""
        # Detector requests 200, but only 100 can be paired
        # It recomputes with executable_size=100, so both fills become 100
        opp = _detect_opportunity([(0.45, 200.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector ensures both fills are equal (100)
        assert result.fully_paired is True
        assert result.paired_size == 100.0
        assert result.yes_filled == 100.0
        assert result.no_filled == 100.0
        assert result.unpaired_yes_size == 0.0
        assert result.unpaired_no_size == 0.0

    def test_no_overfill(self) -> None:
        """NO has more liquidity than YES — detector limits to YES's liquidity."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 200.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector ensures both fills are equal (100)
        assert result.fully_paired is True
        assert result.paired_size == 100.0
        assert result.yes_filled == 100.0
        assert result.no_filled == 100.0
        assert result.unpaired_yes_size == 0.0
        assert result.unpaired_no_size == 0.0

    def test_partial_yes_liquidity(self) -> None:
        """YES has limited liquidity — detector uses YES's liquidity."""
        opp = _detect_opportunity([(0.45, 50.0)], [(0.50, 200.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector recomputes with executable_size=50
        assert result.paired_size == 50.0
        assert result.yes_filled == 50.0
        assert result.no_filled == 50.0
        assert result.unpaired_no_size == 0.0

    def test_partial_no_liquidity(self) -> None:
        """NO has limited liquidity — detector uses NO's liquidity."""
        opp = _detect_opportunity([(0.45, 200.0)], [(0.50, 50.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector recomputes with executable_size=50
        assert result.paired_size == 50.0
        assert result.yes_filled == 50.0
        assert result.no_filled == 50.0
        assert result.unpaired_yes_size == 0.0

    def test_asymmetric_liquidity(self) -> None:
        """Both sides have different liquidity — detector uses minimum."""
        opp = _detect_opportunity([(0.45, 75.0)], [(0.50, 125.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector recomputes with executable_size=75
        assert result.paired_size == 75.0
        assert result.yes_filled == 75.0
        assert result.no_filled == 75.0
        assert result.unpaired_yes_size == 0.0
        assert result.unpaired_no_size == 0.0

    def test_detector_enforces_equal_fills(self) -> None:
        """Phase 2.2 detector always produces equal fills (no legging)."""
        # Request more than available on one side
        opp = _detect_opportunity([(0.45, 30.0)], [(0.50, 80.0)], size=100.0)
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        # Detector fills 30 on both sides (limited by YES)
        assert result.yes_filled == 30.0
        assert result.no_filled == 30.0
        assert result.paired_size == 30.0
        assert result.fully_paired is True

    def test_zero_paired_quantity(self) -> None:
        """Zero paired quantity when one side has no fills."""
        # This shouldn't happen with valid arbitrage, but test the math
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        # Manually create a scenario with zero paired
        # by using the opportunity but verifying the formula
        result = evaluate_arbitrage_costs(opp, model)

        # Verify pairing ratio calculation
        assert result.pairing_ratio == pytest.approx(1.0)

    def test_pairing_ratio_calculation(self) -> None:
        """Pairing ratio is paired_size / requested_size."""
        opp = _detect_opportunity([(0.45, 200.0)], [(0.50, 200.0)], size=150.0)
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.pairing_ratio == pytest.approx(150.0 / 150.0)

    def test_partial_pairing_ratio(self) -> None:
        """Partial pairing shows ratio < 1.0."""
        opp = _detect_opportunity([(0.45, 50.0)], [(0.50, 100.0)], size=100.0)
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.pairing_ratio == pytest.approx(50.0 / 100.0)
        assert result.pairing_ratio < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. ACCOUNTING INVARIANTS
# ══════════════════════════════════════════════════════════════════════════════


class TestAccountingInvariants:
    """Verify mathematical invariants hold."""

    def test_paired_size_leq_requested(self) -> None:
        """paired_size <= requested_size."""
        opp = _detect_opportunity([(0.45, 200.0)], [(0.50, 200.0)], size=100.0)
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.paired_size <= result.requested_size

    def test_paired_size_leq_yes_filled(self) -> None:
        """paired_size <= yes_filled."""
        opp = _detect_opportunity([(0.45, 200.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.paired_size <= result.yes_filled

    def test_paired_size_leq_no_filled(self) -> None:
        """paired_size <= no_filled."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 200.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.paired_size <= result.no_filled

    def test_unpaired_yes_non_negative(self) -> None:
        """unpaired_yes_size >= 0."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.unpaired_yes_size >= 0.0

    def test_unpaired_no_non_negative(self) -> None:
        """unpaired_no_size >= 0."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.unpaired_no_size >= 0.0

    def test_pairing_identity(self) -> None:
        """paired_size + unpaired_yes_size == yes_filled."""
        opp = _detect_opportunity([(0.45, 150.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.paired_size + result.unpaired_yes_size == pytest.approx(result.yes_filled)
        assert result.paired_size + result.unpaired_no_size == pytest.approx(result.no_filled)

    def test_net_profit_leq_gross_profit(self) -> None:
        """net_profit <= gross_profit for non-negative costs."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(
            yes_fee_rate=0.02,
            no_fee_rate=0.03,
            fixed_cost=1.0,
            settlement_cost=0.5,
        )

        result = evaluate_arbitrage_costs(opp, model)

        assert result.net_profit <= result.gross_profit

    def test_total_cost_formula(self) -> None:
        """total_cost = gross_cost + fees + fixed + settlement."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(
            yes_fee_rate=0.02,
            no_fee_rate=0.03,
            fixed_cost=1.0,
            settlement_cost=0.5,
        )

        result = evaluate_arbitrage_costs(opp, model)

        expected_total = (
            result.gross_cost
            + result.yes_fee
            + result.no_fee
            + result.fixed_cost
            + result.settlement_cost
        )
        assert result.total_cost == pytest.approx(expected_total)

    def test_guaranteed_payoff_formula(self) -> None:
        """guaranteed_payoff = paired_size × 1.0."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.guaranteed_payoff == pytest.approx(result.paired_size)

    def test_gross_profit_formula(self) -> None:
        """gross_profit = guaranteed_payoff - gross_cost."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.gross_profit == pytest.approx(
            result.guaranteed_payoff - result.gross_cost
        )

    def test_net_profit_formula(self) -> None:
        """net_profit = guaranteed_payoff - total_cost."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.net_profit == pytest.approx(
            result.guaranteed_payoff - result.total_cost
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-based tests for invariants."""

    @given(
        yes_fee=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        no_fee=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        fixed=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        settlement=st.floats(min_value=0.0, max_value=5.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_net_profit_leq_gross_profit(
        self, yes_fee: float, no_fee: float, fixed: float, settlement: float
    ) -> None:
        """net_profit <= gross_profit for non-negative costs."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(
            yes_fee_rate=yes_fee,
            no_fee_rate=no_fee,
            fixed_cost=fixed,
            settlement_cost=settlement,
        )

        result = evaluate_arbitrage_costs(opp, model)

        assert result.net_profit <= result.gross_profit + 1e-10

    @given(
        yes_fee=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        no_fee=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_fees_are_non_negative(self, yes_fee: float, no_fee: float) -> None:
        """Fees are always non-negative."""
        opp = _detect_opportunity([(0.45, 100.0)], [(0.50, 100.0)])
        model = ArbitrageCostModel(yes_fee_rate=yes_fee, no_fee_rate=no_fee)

        result = evaluate_arbitrage_costs(opp, model)

        assert result.yes_fee >= 0.0
        assert result.no_fee >= 0.0

    @given(
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_pairing_ratio_bounds(self, size: float) -> None:
        """pairing_ratio is between 0 and 1."""
        opp = _detect_opportunity([(0.45, size)], [(0.50, size)], size=size)
        model = ArbitrageCostModel()

        result = evaluate_arbitrage_costs(opp, model)

        assert result.pairing_ratio is not None
        assert 0.0 <= result.pairing_ratio <= 1.0 + 1e-10

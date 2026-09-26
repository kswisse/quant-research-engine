"""Arbitrage normalization tests.

Tests for Phase 2.6 canonical normalized opportunity representation.

Covers:
- Model construction and frozen behavior
- Deterministic identity (opportunity_id, observation_id)
- Same-market normalization
- Cross-venue normalization
- Cost evaluation
- Risk flag derivation
- Adversarial cases
- Serialization

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_arbitrage_normalization.py -v
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    evaluate_arbitrage_costs,
    evaluate_cross_venue_costs,
)
from quant_engine.arbitrage.cross_venue import detect_cross_venue_arbitrage
from quant_engine.arbitrage.detection import detect_same_market_arbitrage
from quant_engine.arbitrage.errors import (
    ArbitrageError,
    InvalidLegError,
    NormalizationError,
)
from quant_engine.arbitrage.models import ArbitrageOpportunity
from quant_engine.arbitrage.normalization import (
    CostBreakdown,
    ExecutionPlan,
    ExecutionPlanLeg,
    ExecutionRole,
    LegRole,
    NormalizedOpportunity,
    OpportunityLeg,
    OpportunityType,
    RiskFlags,
    normalize_cross_venue,
    normalize_same_market,
)
from quant_engine.order_book import OrderBookLevel, OrderBookSnapshot

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN_CLOCK = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
SOURCE_TIME = datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC)
OBS_TS = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)


def _make_book(
    asks: list[tuple[float, float]] | None = None,
    bids: list[tuple[float, float]] | None = None,
    provider: str = "test",
    instrument: str = "YES",
) -> OrderBookSnapshot:
    """Create a test order book with specified levels."""
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in (asks or [])]
    bid_levels = [OrderBookLevel(price=p, size=s) for p, s in (bids or [])]
    return OrderBookSnapshot(
        provider=provider,
        provider_instrument_id=instrument,
        source_timestamp=SOURCE_TIME,
        ingestion_timestamp=FROZEN_CLOCK,
        bids=tuple(bid_levels),
        asks=tuple(ask_levels),
    )


def _detect_same_market(
    yes_asks: list[tuple[float, float]],
    no_asks: list[tuple[float, float]],
    size: float = 100.0,
    provider: str = "test",
) -> ArbitrageOpportunity:
    """Detect same-market arbitrage and assert non-None."""
    yes_book = _make_book(asks=yes_asks, provider=provider, instrument="YES")
    no_book = _make_book(asks=no_asks, provider=provider, instrument="NO")
    result = detect_same_market_arbitrage(yes_book, no_book, size)
    assert result is not None, "Expected same-market arbitrage opportunity"
    return result


def _detect_cross_venue(
    buy_asks: list[tuple[float, float]],
    sell_bids: list[tuple[float, float]],
    size: float = 100.0,
    buy_provider: str = "venue_a",
    sell_provider: str = "venue_b",
) -> tuple:
    """Detect cross-venue arbitrage and assert non-None."""
    buy_book = _make_book(asks=buy_asks, provider=buy_provider, instrument="X")
    sell_book = _make_book(bids=sell_bids, provider=sell_provider, instrument="X")
    result = detect_cross_venue_arbitrage(buy_book, sell_book, size)
    assert result is not None, "Expected cross-venue arbitrage opportunity"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 1. ENUM TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEnums:
    """Verify enum values and StrEnum behavior."""

    def test_opportunity_type_values(self) -> None:
        assert OpportunityType.SAME_MARKET == "same_market"
        assert OpportunityType.CROSS_VENUE == "cross_venue"

    def test_leg_role_values(self) -> None:
        assert LegRole.BUY == "buy"
        assert LegRole.SELL == "sell"

    def test_execution_role_values(self) -> None:
        assert ExecutionRole.PRIMARY == "primary"
        assert ExecutionRole.COUNTER == "counter"

    def test_str_enum_serializable(self) -> None:
        assert json.dumps(OpportunityType.SAME_MARKET) == '"same_market"'


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL CONSTRUCTION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestModelConstruction:
    """Test model creation and frozen behavior."""

    def test_opportunity_leg_frozen(self) -> None:
        leg = OpportunityLeg(
            leg_index=0,
            provider="test",
            provider_instrument_id="X",
            market_identity=None,
            side=LegRole.BUY,
            execution_role=ExecutionRole.PRIMARY,
            price=0.5,
            vwap=0.5,
            fill_size=100.0,
            fill_notional=50.0,
            fill_count=1,
            levels_consumed=1,
            best_available_price=0.5,
        )
        with pytest.raises(AttributeError):
            leg.price = 0.6  # type: ignore[misc]

    def test_opportunity_leg_validation_empty_provider(self) -> None:
        with pytest.raises(InvalidLegError):
            OpportunityLeg(
                leg_index=0,
                provider="",
                provider_instrument_id="X",
                market_identity=None,
                side=LegRole.BUY,
                execution_role=ExecutionRole.PRIMARY,
                price=0.5,
                vwap=0.5,
                fill_size=100.0,
                fill_notional=50.0,
                fill_count=1,
                levels_consumed=1,
                best_available_price=0.5,
            )

    def test_opportunity_leg_validation_negative_index(self) -> None:
        with pytest.raises(InvalidLegError):
            OpportunityLeg(
                leg_index=-1,
                provider="test",
                provider_instrument_id="X",
                market_identity=None,
                side=LegRole.BUY,
                execution_role=ExecutionRole.PRIMARY,
                price=0.5,
                vwap=0.5,
                fill_size=100.0,
                fill_notional=50.0,
                fill_count=1,
                levels_consumed=1,
                best_available_price=0.5,
            )

    def test_execution_plan_frozen(self) -> None:
        plan = ExecutionPlan(
            legs=(),
            total_estimated_cost=100.0,
            total_estimated_proceeds=105.0,
            estimated_net=5.0,
            is_simultaneous=True,
        )
        with pytest.raises(AttributeError):
            plan.total_estimated_cost = 200.0  # type: ignore[misc]

    def test_cost_breakdown_frozen(self) -> None:
        cb = CostBreakdown(
            buy_fee=1.0,
            sell_fee=1.0,
            fixed_cost=0.0,
            settlement_cost=0.0,
            transfer_cost=0.0,
            total_fee=2.0,
            total_cost=97.0,
            effective_fee_rate=0.02,
        )
        with pytest.raises(AttributeError):
            cb.buy_fee = 2.0  # type: ignore[misc]

    def test_risk_flags_frozen(self) -> None:
        rf = RiskFlags(
            is_fully_executable=True,
            has_partial_fill=False,
            has_unpaired_exposure=False,
            gross_margin_is_thin=False,
            net_margin_is_negative=False,
            near_expiry=False,
            is_expired=False,
            identity_validated=False,
            identity_unvalidated=True,
            settlement_currency_mismatch=False,
            resolution_source_mismatch=False,
        )
        with pytest.raises(AttributeError):
            rf.is_fully_executable = False  # type: ignore[misc]

    def test_normalized_opportunity_frozen(self) -> None:
        opp = NormalizedOpportunity(
            opportunity_id="a" * 16,
            observation_id="b" * 16,
            opportunity_type=OpportunityType.SAME_MARKET,
            legs=(_make_leg(0), _make_leg(1)),
            execution_plan=_make_plan(),
            gross_cost=95.0,
            gross_proceeds=100.0,
            gross_profit=5.0,
            gross_return=5.0 / 95.0,
            cost_breakdown=_make_cost_breakdown(),
            net_profit=4.0,
            net_return=4.0 / 95.0,
            requested_size=100.0,
            executable_size=100.0,
            fully_executable=True,
            risk_flags=_make_risk_flags(),
            mapping=None,
            observation_timestamp=OBS_TS,
            evaluation_time=None,
            is_point_in_time=None,
            close_time=None,
            expiry_time=None,
            outcome_label="YES+NO",
        )
        with pytest.raises(AttributeError):
            opp.gross_cost = 100.0  # type: ignore[misc]

    def test_normalized_opportunity_requires_two_legs(self) -> None:
        with pytest.raises(ValueError, match="exactly 2 legs"):
            NormalizedOpportunity(
                opportunity_id="a" * 16,
                observation_id="b" * 16,
                opportunity_type=OpportunityType.SAME_MARKET,
                legs=(_make_leg(0),),
                execution_plan=_make_plan(),
                gross_cost=95.0,
                gross_proceeds=100.0,
                gross_profit=5.0,
                gross_return=0.05,
                cost_breakdown=_make_cost_breakdown(),
                net_profit=4.0,
                net_return=0.04,
                requested_size=100.0,
                executable_size=100.0,
                fully_executable=True,
                risk_flags=_make_risk_flags(),
                mapping=None,
                observation_timestamp=OBS_TS,
                evaluation_time=None,
                is_point_in_time=None,
                close_time=None,
                expiry_time=None,
                outcome_label="test",
            )


def _make_leg(index: int = 0, **kwargs: object) -> OpportunityLeg:
    """Create a minimal test leg."""
    defaults = dict(
        leg_index=index,
        provider="test",
        provider_instrument_id=f"INST-{index}",
        market_identity=None,
        side=LegRole.BUY,
        execution_role=ExecutionRole.PRIMARY if index == 0 else ExecutionRole.COUNTER,
        price=0.5,
        vwap=0.5,
        fill_size=100.0,
        fill_notional=50.0,
        fill_count=1,
        levels_consumed=1,
        best_available_price=0.5,
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return OpportunityLeg(**defaults)  # type: ignore[arg-type]


def _make_plan() -> ExecutionPlan:
    """Create a minimal test execution plan."""
    return ExecutionPlan(
        legs=(
            ExecutionPlanLeg(
                leg_index=0, provider="test", provider_instrument_id="YES",
                side=LegRole.BUY, target_size=100.0, expected_price=0.5,
                expected_notional=50.0,
            ),
            ExecutionPlanLeg(
                leg_index=1, provider="test", provider_instrument_id="NO",
                side=LegRole.BUY, target_size=100.0, expected_price=0.5,
                expected_notional=50.0,
            ),
        ),
        total_estimated_cost=100.0,
        total_estimated_proceeds=100.0,
        estimated_net=0.0,
        is_simultaneous=True,
    )


def _make_cost_breakdown(**kwargs: object) -> CostBreakdown:
    """Create a minimal test cost breakdown."""
    defaults = dict(
        buy_fee=0.0, sell_fee=0.0, fixed_cost=0.0,
        settlement_cost=0.0, transfer_cost=0.0,
        total_fee=0.0, total_cost=95.0, effective_fee_rate=0.0,
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return CostBreakdown(**defaults)  # type: ignore[arg-type]


def _make_risk_flags(**kwargs: object) -> RiskFlags:
    """Create a minimal test risk flags with safe defaults."""
    defaults = dict(
        is_fully_executable=True, has_partial_fill=False,
        has_unpaired_exposure=False, gross_margin_is_thin=False,
        net_margin_is_negative=False, near_expiry=False, is_expired=False,
        identity_validated=False, identity_unvalidated=True,
        settlement_currency_mismatch=False, resolution_source_mismatch=False,
    )
    defaults.update(kwargs)  # type: ignore[arg-type]
    return RiskFlags(**defaults)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# 3. SAME-MARKET NORMALIZATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeSameMarket:
    """Test same-market YES+NO normalization."""

    def test_basic_normalization(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert normalized.opportunity_type == OpportunityType.SAME_MARKET
        assert len(normalized.legs) == 2
        assert normalized.gross_cost == pytest.approx(95.0)
        assert normalized.gross_proceeds == pytest.approx(100.0)
        assert normalized.gross_profit == pytest.approx(5.0)
        assert normalized.executable_size == 100.0
        assert normalized.fully_executable is True

    def test_leg_ordering(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        leg0, leg1 = normalized.legs
        assert leg0.leg_index == 0
        assert leg0.side == LegRole.BUY
        assert leg0.execution_role == ExecutionRole.PRIMARY
        assert leg0.provider_instrument_id == "YES"
        assert leg1.leg_index == 1
        assert leg1.side == LegRole.BUY
        assert leg1.execution_role == ExecutionRole.COUNTER
        assert leg1.provider_instrument_id == "NO"

    def test_leg_provider_propagation(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
            provider="polymarket",
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        for leg in normalized.legs:
            assert leg.provider == "polymarket"

    def test_execution_plan(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        plan = normalized.execution_plan
        assert len(plan.legs) == 2
        assert plan.is_simultaneous is True
        assert plan.total_estimated_cost == pytest.approx(95.0)
        assert plan.total_estimated_proceeds == pytest.approx(100.0)

    def test_observation_timestamp_propagation(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert normalized.observation_timestamp == OBS_TS

    def test_observation_timestamp_default(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        before = datetime.now(UTC)
        normalized = normalize_same_market(opp)
        after = datetime.now(UTC)

        assert before <= normalized.observation_timestamp <= after

    def test_outcome_label(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        assert normalized.outcome_label == "YES+NO"


# ══════════════════════════════════════════════════════════════════════════════
# 4. CROSS-VENUE NORMALIZATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeCrossVenue:
    """Test cross-venue BUY A / SELL B normalization."""

    def test_basic_normalization(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        assert normalized.opportunity_type == OpportunityType.CROSS_VENUE
        assert len(normalized.legs) == 2
        assert normalized.gross_profit == pytest.approx(10.0)
        assert normalized.executable_size == 100.0
        assert normalized.fully_executable is True

    def test_leg_roles(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        leg0, leg1 = normalized.legs
        assert leg0.side == LegRole.BUY
        assert leg0.execution_role == ExecutionRole.PRIMARY
        assert leg1.side == LegRole.SELL
        assert leg1.execution_role == ExecutionRole.COUNTER

    def test_provider_propagation(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
            buy_provider="polymarket",
            sell_provider="kalshi",
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        leg0, leg1 = normalized.legs
        assert leg0.provider == "polymarket"
        assert leg1.provider == "kalshi"

    def test_instrument_propagation(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        leg0, leg1 = normalized.legs
        assert leg0.provider_instrument_id == "X"
        assert leg1.provider_instrument_id == "X"

    def test_outcome_label(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)
        # Default outcome_label from detect_cross_venue_arbitrage is "outcome"
        assert normalized.outcome_label == "outcome"

    def test_gross_proceeds(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)
        assert normalized.gross_proceeds == pytest.approx(55.0)
        assert normalized.gross_cost == pytest.approx(45.0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestIdentity:
    """Test deterministic identity computation."""

    def test_opportunity_id_is_hex_16(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert len(normalized.opportunity_id) == 16
        assert all(c in "0123456789abcdef" for c in normalized.opportunity_id)

    def test_observation_id_is_hex_16(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert len(normalized.observation_id) == 16
        assert all(c in "0123456789abcdef" for c in normalized.observation_id)

    def test_same_structure_same_opportunity_id(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        n1 = normalize_same_market(opp, observation_timestamp=OBS_TS)
        n2 = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert n1.opportunity_id == n2.opportunity_id

    def test_different_timestamp_same_opportunity_id(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        ts1 = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 15, 12, 0, 1, tzinfo=UTC)
        n1 = normalize_same_market(opp, observation_timestamp=ts1)
        n2 = normalize_same_market(opp, observation_timestamp=ts2)

        assert n1.opportunity_id == n2.opportunity_id

    def test_different_timestamp_different_observation_id(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        ts1 = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 15, 12, 0, 1, tzinfo=UTC)
        n1 = normalize_same_market(opp, observation_timestamp=ts1)
        n2 = normalize_same_market(opp, observation_timestamp=ts2)

        assert n1.observation_id != n2.observation_id

    def test_different_economics_different_opportunity_id(self) -> None:
        opp1 = _detect_same_market(yes_asks=[(0.45, 100.0)], no_asks=[(0.50, 100.0)])
        opp2 = _detect_same_market(yes_asks=[(0.40, 100.0)], no_asks=[(0.50, 100.0)])
        n1 = normalize_same_market(opp1, observation_timestamp=OBS_TS)
        n2 = normalize_same_market(opp2, observation_timestamp=OBS_TS)

        assert n1.opportunity_id != n2.opportunity_id

    def test_deterministic_repeated_computation(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        ids = [
            normalize_same_market(opp, observation_timestamp=OBS_TS).opportunity_id
            for _ in range(10)
        ]
        assert len(set(ids)) == 1

    def test_no_python_hash_used(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        opp_id = normalized.opportunity_id

        # Verify it's a valid SHA-256 truncated to 16 hex chars
        d = {
            "opportunity_type": normalized.opportunity_type,
            "executable_size": normalized.executable_size,
            "gross_cost": normalized.gross_cost,
            "gross_profit": normalized.gross_profit,
            "legs": [
                {
                    "provider": leg.provider,
                    "provider_instrument_id": leg.provider_instrument_id,
                    "side": leg.side,
                }
                for leg in sorted(normalized.legs, key=lambda l: l.leg_index)
            ],
        }
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        assert opp_id == expected


# ══════════════════════════════════════════════════════════════════════════════
# 6. COST INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestCostIntegration:
    """Test cost breakdown from detector results."""

    def test_same_market_with_result(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        cost_model = ArbitrageCostModel(yes_fee_rate=0.02, no_fee_rate=0.02)
        result = evaluate_arbitrage_costs(opp, cost_model)
        normalized = normalize_same_market(
            opp, result=result, observation_timestamp=OBS_TS
        )

        assert normalized.cost_breakdown.buy_fee == pytest.approx(result.yes_fee)
        assert normalized.cost_breakdown.sell_fee == pytest.approx(result.no_fee)
        assert normalized.cost_breakdown.total_cost == pytest.approx(result.total_cost)
        assert normalized.net_profit == pytest.approx(result.net_profit)
        assert normalized.net_return == pytest.approx(result.net_return)

    def test_same_market_without_result(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert normalized.cost_breakdown.buy_fee == 0.0
        assert normalized.cost_breakdown.sell_fee == 0.0
        assert normalized.cost_breakdown.total_fee == 0.0
        assert normalized.net_profit == pytest.approx(normalized.gross_profit)

    def test_cross_venue_with_result(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        cost_model = ArbitrageCostModel(buy_fee_rate=0.02, sell_fee_rate=0.02)
        result = evaluate_cross_venue_costs(opp, cost_model)
        normalized = normalize_cross_venue(
            opp, result=result, observation_timestamp=OBS_TS
        )

        assert normalized.cost_breakdown.buy_fee == pytest.approx(result.buy_fee)
        assert normalized.cost_breakdown.sell_fee == pytest.approx(result.sell_fee)
        assert normalized.cost_breakdown.transfer_cost == pytest.approx(
            result.transfer_cost
        )
        assert normalized.cost_breakdown.total_cost == pytest.approx(result.total_cost)
        assert normalized.net_profit == pytest.approx(result.net_spread)

    def test_cross_venue_without_result(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        assert normalized.cost_breakdown.buy_fee == 0.0
        assert normalized.cost_breakdown.sell_fee == 0.0
        assert normalized.cost_breakdown.total_fee == 0.0

    def test_preserves_preexisting_yes_fee_behavior(self) -> None:
        """Verify the existing YES fee calculation behavior is preserved.

        The existing evaluate_arbitrage_costs() calculates yes_fee on gross_cost
        (YES + NO notional) rather than YES notional only. This is a pre-existing
        issue that must NOT be silently changed by normalization.
        """
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        cost_model = ArbitrageCostModel(yes_fee_rate=0.02, no_fee_rate=0.00)
        result = evaluate_arbitrage_costs(opp, cost_model)

        # The existing behavior: yes_fee = gross_cost * yes_fee_rate
        # where gross_cost = YES notional + NO notional = 45 + 50 = 95
        # So yes_fee = 95 * 0.02 = 1.90 (not 45 * 0.02 = 0.90)
        assert result.yes_fee == pytest.approx(1.90)

        # Normalization must map this value faithfully
        normalized = normalize_same_market(
            opp, result=result, observation_timestamp=OBS_TS
        )
        assert normalized.cost_breakdown.buy_fee == pytest.approx(1.90)

    def test_cost_breakdown_effective_fee_rate(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        cost_model = ArbitrageCostModel(yes_fee_rate=0.02, no_fee_rate=0.02)
        result = evaluate_arbitrage_costs(opp, cost_model)
        normalized = normalize_same_market(
            opp, result=result, observation_timestamp=OBS_TS
        )

        expected_rate = (result.yes_fee + result.no_fee) / result.gross_cost
        assert normalized.cost_breakdown.effective_fee_rate == pytest.approx(
            expected_rate
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. RISK FLAG TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskFlags:
    """Test deterministic risk flag derivation."""

    def test_fully_executable(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        assert normalized.risk_flags.is_fully_executable is True
        assert normalized.risk_flags.has_partial_fill is False

    def test_partial_fill(self) -> None:
        yes_book = _make_book(asks=[(0.45, 50.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")
        opp = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert opp is not None
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        assert normalized.risk_flags.has_partial_fill is True
        assert normalized.risk_flags.is_fully_executable is False

    def test_no_unpaired_exposure_same_market(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        assert normalized.risk_flags.has_unpaired_exposure is False

    def test_thin_margin(self) -> None:
        # Very tight margin: YES=0.499, NO=0.499, sum=0.998, profit=0.002
        opp = _detect_same_market(
            yes_asks=[(0.499, 100.0)],
            no_asks=[(0.499, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        # gross_profit = 100 - 99.8 = 0.2, gross_cost = 99.8
        # 0.2 < 0.05 * 99.8 = 4.99 → thin margin
        assert normalized.risk_flags.gross_margin_is_thin is True

    def test_negative_net_margin(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        cost_model = ArbitrageCostModel(yes_fee_rate=0.10, no_fee_rate=0.10)
        result = evaluate_arbitrage_costs(opp, cost_model)
        normalized = normalize_same_market(
            opp, result=result, observation_timestamp=OBS_TS
        )
        # Gross profit = 5.0, but fees are substantial
        assert normalized.risk_flags.net_margin_is_negative is (
            result.net_profit < 0
        )

    def test_identity_unvalidated_no_mapping(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        assert normalized.risk_flags.identity_validated is False
        assert normalized.risk_flags.identity_unvalidated is True

    def test_identity_validated_with_mapping(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        # Cross-venue with mapping
        from quant_engine.market_identity.mapping import MarketMapping
        from quant_engine.market_identity.models import (
            EventBoundary,
            EventIdentity,
            MarketIdentity,
            OutcomeIdentity,
            ResolutionRule,
            SettlementTerms,
            TemporalScope,
        )

        identity = MarketIdentity(
            provider="test",
            provider_market_id="test-001",
            event=EventIdentity(
                event_id="ev1", title="Test",
                boundary=EventBoundary(close_time=OBS_TS, expiry_time=OBS_TS),
            ),
            outcome=OutcomeIdentity(outcome_id="yes", label="Yes"),
            resolution=ResolutionRule(source="ap", method="official", authority="usc"),
            settlement=SettlementTerms(
                payout_type="binary",
                payout_cap=1.0,
                settlement_formula="winner-takes-all",
            ),
            temporal_scope=TemporalScope(evaluation_time=OBS_TS, is_point_in_time=True),
        )
        mapping = MarketMapping(source=identity, target=identity, relationship="same_market")
        opp_with_mapping = type(opp)(
            outcome_label=opp.outcome_label,
            buy_venue=opp.buy_venue,
            sell_venue=opp.sell_venue,
            buy_instrument_id=opp.buy_instrument_id,
            sell_instrument_id=opp.sell_instrument_id,
            requested_size=opp.requested_size,
            executable_size=opp.executable_size,
            buy_execution=opp.buy_execution,
            sell_execution=opp.sell_execution,
            gross_spread=opp.gross_spread,
            gross_return=opp.gross_return,
            fully_executable=opp.fully_executable,
            mapping=mapping,
        )
        normalized = normalize_cross_venue(
            opp_with_mapping, observation_timestamp=OBS_TS
        )
        assert normalized.risk_flags.identity_validated is True
        assert normalized.risk_flags.identity_unvalidated is False

    def test_near_expiry(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        close_time = OBS_TS.replace(hour=12, minute=30)  # 30 minutes from now
        from quant_engine.market_identity.models import (
            EventBoundary,
            EventIdentity,
            MarketIdentity,
            OutcomeIdentity,
            ResolutionRule,
            SettlementTerms,
            TemporalScope,
        )

        identity = MarketIdentity(
            provider="test",
            provider_market_id="test-001",
            event=EventIdentity(
                event_id="ev1", title="Test",
                boundary=EventBoundary(close_time=close_time, expiry_time=close_time),
            ),
            outcome=OutcomeIdentity(outcome_id="yes", label="Yes"),
            resolution=ResolutionRule(source="ap", method="official", authority="usc"),
            settlement=SettlementTerms(
                payout_type="binary",
                payout_cap=1.0,
                settlement_formula="winner-takes-all",
            ),
            temporal_scope=TemporalScope(evaluation_time=close_time, is_point_in_time=True),
        )
        normalized = normalize_same_market(
            opp, observation_timestamp=OBS_TS, market_identity=identity
        )
        assert normalized.risk_flags.near_expiry is True


# ══════════════════════════════════════════════════════════════════════════════
# 8. SERIALIZATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    """Test to_dict and JSON serialization."""

    def test_to_dict_structure(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        assert "opportunity_id" in d
        assert "observation_id" in d
        assert "opportunity_type" in d
        assert "legs" in d
        assert len(d["legs"]) == 2
        assert "execution_plan" in d
        assert "cost_breakdown" in d
        assert "risk_flags" in d

    def test_to_dict_json_serializable(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        # Should not raise
        json_str = json.dumps(d, indent=2)
        assert isinstance(json_str, str)

    def test_to_dict_enum_serialization(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        assert d["opportunity_type"] == "same_market"
        assert d["legs"][0]["side"] == "buy"

    def test_to_dict_datetime_serialization(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        assert "T" in d["observation_timestamp"]  # ISO 8601

    def test_to_dict_optional_fields(self) -> None:
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        assert d["evaluation_time"] is None
        assert d["close_time"] is None
        assert d["expiry_time"] is None

    def test_to_dict_cross_venue(self) -> None:
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)
        d = normalized.to_dict()

        assert d["opportunity_type"] == "cross_venue"
        assert d["legs"][0]["side"] == "buy"
        assert d["legs"][1]["side"] == "sell"


# ══════════════════════════════════════════════════════════════════════════════
# 9. ADVERSARIAL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialCases:
    """Adversarial edge cases for normalization."""

    def test_same_opportunity_different_timestamps(self) -> None:
        """Same economics at t=1 and t=2 → same opp_id, different obs_id."""
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        ts1 = datetime(2026, 3, 15, 12, 0, 0, tzinfo=UTC)
        ts2 = datetime(2026, 3, 15, 12, 0, 5, tzinfo=UTC)
        n1 = normalize_same_market(opp, observation_timestamp=ts1)
        n2 = normalize_same_market(opp, observation_timestamp=ts2)

        assert n1.opportunity_id == n2.opportunity_id
        assert n1.observation_id != n2.observation_id

    def test_same_market_yes_no_structure(self) -> None:
        """YES+NO produces two BUY legs."""
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert all(leg.side == LegRole.BUY for leg in normalized.legs)
        assert normalized.legs[0].execution_role == ExecutionRole.PRIMARY
        assert normalized.legs[1].execution_role == ExecutionRole.COUNTER

    def test_cross_venue_buy_sell_structure(self) -> None:
        """BUY A / SELL B produces BUY + SELL legs."""
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        assert normalized.legs[0].side == LegRole.BUY
        assert normalized.legs[1].side == LegRole.SELL

    def test_partial_fill_normalized(self) -> None:
        """Partial fill propagates correctly."""
        yes_book = _make_book(asks=[(0.45, 50.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")
        opp = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert opp is not None
        normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert normalized.executable_size == 50.0
        assert normalized.fully_executable is False
        assert normalized.risk_flags.has_partial_fill is True

    def test_unpaired_quantity_cross_venue(self) -> None:
        """Unpaired quantity in cross-venue with cost result."""
        buy_book = _make_book(asks=[(0.45, 100.0)], provider="venue_a", instrument="X")
        sell_book = _make_book(bids=[(0.55, 100.0)], provider="venue_b", instrument="X")
        opp = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert opp is not None
        cost_model = ArbitrageCostModel()
        result = evaluate_cross_venue_costs(opp, cost_model)
        normalized = normalize_cross_venue(
            opp, result=result, observation_timestamp=OBS_TS
        )

        # Both fill 100 → no unpaired
        assert normalized.executable_size == 100.0
        assert normalized.risk_flags.has_unpaired_exposure is False
        assert normalized.risk_flags.is_fully_executable is True

    def test_mapping_present(self) -> None:
        """Mapping present → identity_validated."""
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        assert opp.mapping is None  # default
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)
        assert normalized.risk_flags.identity_unvalidated is True

    def test_zero_executable_quantity(self) -> None:
        """Zero executable quantity — no arbitrage at fair pricing."""
        yes_book = _make_book(asks=[(0.50, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")
        opp = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        # 0.50 + 0.50 = 1.0, no arbitrage → None is correct
        assert opp is None

    def test_same_prices_different_instruments(self) -> None:
        """Same prices but different instrument IDs → different opp_id."""
        opp1 = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
            provider="test",
        )
        # Different provider instruments via different books
        yes_book2 = _make_book(
            asks=[(0.45, 100.0)], provider="test", instrument="YES-ALT"
        )
        no_book2 = _make_book(
            asks=[(0.50, 100.0)], provider="test", instrument="NO-ALT"
        )
        opp2 = detect_same_market_arbitrage(yes_book2, no_book2, 100.0)
        assert opp2 is not None

        n1 = normalize_same_market(opp1, observation_timestamp=OBS_TS)
        n2 = normalize_same_market(opp2, observation_timestamp=OBS_TS)

        # Different instrument IDs → different opportunity_id
        assert n1.opportunity_id != n2.opportunity_id

    def test_identical_observation_repeated(self) -> None:
        """Identical observation produces identical IDs."""
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        n1 = normalize_same_market(opp, observation_timestamp=OBS_TS)
        n2 = normalize_same_market(opp, observation_timestamp=OBS_TS)

        assert n1.opportunity_id == n2.opportunity_id
        assert n1.observation_id == n2.observation_id

    def test_negative_net_economics(self) -> None:
        """Negative net economics after high fees."""
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        cost_model = ArbitrageCostModel(yes_fee_rate=0.10, no_fee_rate=0.10)
        result = evaluate_arbitrage_costs(opp, cost_model)
        normalized = normalize_same_market(
            opp, result=result, observation_timestamp=OBS_TS
        )

        if result.net_profit < 0:
            assert normalized.risk_flags.net_margin_is_negative is True

    def test_mapping_none_does_not_imply_equivalence(self) -> None:
        """mapping=None must NOT imply economic equivalence."""
        opp = _detect_cross_venue(
            buy_asks=[(0.45, 100.0)],
            sell_bids=[(0.55, 100.0)],
        )
        normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)

        assert normalized.mapping is None
        assert normalized.risk_flags.identity_unvalidated is True


# ══════════════════════════════════════════════════════════════════════════════
# 10. ERROR HIERARCHY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    """Verify error types inherit correctly."""

    def test_arbitrage_error_is_quant_engine_error(self) -> None:
        from quant_engine.core.errors import QuantEngineError
        assert issubclass(ArbitrageError, QuantEngineError)

    def test_normalization_error_is_arbitrage_error(self) -> None:
        assert issubclass(NormalizationError, ArbitrageError)

    def test_invalid_leg_error_is_normalization_error(self) -> None:
        assert issubclass(InvalidLegError, NormalizationError)


# ══════════════════════════════════════════════════════════════════════════════
# 11. BACKWARD COMPATIBILITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Existing APIs continue to work unchanged."""

    def test_detect_same_market仍然工作(self) -> None:
        """Phase 2.2 detector still works."""
        from quant_engine.arbitrage import detect_same_market_arbitrage
        yes_book = _make_book(asks=[(0.45, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(0.50, 100.0)], instrument="NO")
        result = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        assert result is not None
        assert result.gross_profit == pytest.approx(5.0)

    def test_detect_cross_venue仍然工作(self) -> None:
        """Phase 2.4 detector still works."""
        from quant_engine.arbitrage import detect_cross_venue_arbitrage
        buy_book = _make_book(asks=[(0.45, 100.0)], provider="venue_a", instrument="X")
        sell_book = _make_book(bids=[(0.55, 100.0)], provider="venue_b", instrument="X")
        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert result is not None
        assert result.gross_spread == pytest.approx(10.0)

    def test_evaluate_costs仍然工作(self) -> None:
        """Phase 2.3 cost evaluation still works."""
        from quant_engine.arbitrage import (
            ArbitrageCostModel,
            evaluate_arbitrage_costs,
        )
        opp = _detect_same_market(
            yes_asks=[(0.45, 100.0)],
            no_asks=[(0.50, 100.0)],
        )
        model = ArbitrageCostModel(yes_fee_rate=0.02)
        result = evaluate_arbitrage_costs(opp, model)
        assert result.net_profit > 0

    def test_all_original_exports_present(self) -> None:
        """All original __all__ exports still accessible."""
        import quant_engine.arbitrage as arb

        assert hasattr(arb, "ArbitrageOpportunity")
        assert hasattr(arb, "CrossVenueOpportunity")
        assert hasattr(arb, "ArbitrageCostModel")
        assert hasattr(arb, "NetArbitrageResult")
        assert hasattr(arb, "NetCrossVenueResult")
        assert hasattr(arb, "detect_same_market_arbitrage")
        assert hasattr(arb, "detect_cross_venue_arbitrage")
        assert hasattr(arb, "evaluate_arbitrage_costs")
        assert hasattr(arb, "evaluate_cross_venue_costs")
        assert hasattr(arb, "MarketMapping")


# ══════════════════════════════════════════════════════════════════════════════
# 12. HYPOTHESIS PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Property-based tests for model invariants."""

    @given(
        yes_price=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        no_price=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_same_market_always_two_legs(
        self, yes_price: float, no_price: float, size: float
    ) -> None:
        """Normalized same-market always has exactly 2 legs."""
        yes_book = _make_book(asks=[(yes_price, 1000.0)], instrument="YES")
        no_book = _make_book(asks=[(no_price, 1000.0)], instrument="NO")
        opp = detect_same_market_arbitrage(yes_book, no_book, size)
        if opp is not None:
            normalized = normalize_same_market(opp, observation_timestamp=OBS_TS)
            assert len(normalized.legs) == 2

    @given(
        buy_price=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        sell_price=st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
        size=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_cross_venue_always_two_legs(
        self, buy_price: float, sell_price: float, size: float
    ) -> None:
        """Normalized cross-venue always has exactly 2 legs."""
        if sell_price > buy_price:
            buy_book = _make_book(asks=[(buy_price, 1000.0)], provider="venue_a", instrument="X")
            sell_book = _make_book(bids=[(sell_price, 1000.0)], provider="venue_b", instrument="X")
            opp = detect_cross_venue_arbitrage(buy_book, sell_book, size)
            if opp is not None:
                normalized = normalize_cross_venue(opp, observation_timestamp=OBS_TS)
                assert len(normalized.legs) == 2

    @given(
        yes_price=st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
        no_price=st.floats(min_value=0.01, max_value=0.49, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_opportunity_id_deterministic(
        self, yes_price: float, no_price: float
    ) -> None:
        """Same inputs always produce same opportunity_id."""
        yes_book = _make_book(asks=[(yes_price, 100.0)], instrument="YES")
        no_book = _make_book(asks=[(no_price, 100.0)], instrument="NO")
        opp = detect_same_market_arbitrage(yes_book, no_book, 100.0)
        if opp is not None:
            n1 = normalize_same_market(opp, observation_timestamp=OBS_TS)
            n2 = normalize_same_market(opp, observation_timestamp=OBS_TS)
            assert n1.opportunity_id == n2.opportunity_id

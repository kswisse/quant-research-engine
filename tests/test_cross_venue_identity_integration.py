"""Integration tests for MarketMapping → detector → cost pipeline.

Phase 2.5 — Market Identity & Resolution Semantics.
Tests that a validated compatible mapping is carried through correctly
from MarketMapping construction through detection to cost evaluation.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_cross_venue_identity_integration.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_engine.arbitrage.costs import (
    ArbitrageCostModel,
    evaluate_cross_venue_costs,
)
from quant_engine.arbitrage.cross_venue import (
    detect_cross_venue_arbitrage,
)
from quant_engine.market_identity.errors import (
    IncompatibleOutcomeError,
    IncompatibleResolutionError,
    IncompatibleResolutionStateError,
    IncompatibleSettlementError,
    MarketIdentityError,
)
from quant_engine.market_identity.mapping import MarketMapping
from quant_engine.market_identity.models import (
    EventBoundary,
    EventIdentity,
    MarketIdentity,
    OutcomeIdentity,
    ResolutionRule,
    ResolutionState,
    SettlementTerms,
    TemporalScope,
)
from quant_engine.order_book import OrderBookLevel, OrderBookSnapshot

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

SOURCE_TIME = datetime(2026, 3, 15, 9, 59, 58, tzinfo=UTC)
FROZEN_CLOCK = datetime(2026, 3, 15, 10, 0, 0, tzinfo=UTC)
ELECTION_CLOSE = datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
ELECTION_EXPIRY = datetime(2026, 11, 4, 6, 0, 0, tzinfo=UTC)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _make_book(
    bids: list[tuple[float, float]] | None = None,
    asks: list[tuple[float, float]] | None = None,
    provider: str = "test",
    instrument: str = "YES",
) -> OrderBookSnapshot:
    """Create an OrderBookSnapshot for testing."""
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


def _make_identity(
    *,
    provider: str = "polymarket",
    provider_market_id: str = "test-001",
    event_id: str = "event-1",
    outcome_id: str = "yes",
    resolution_source: str = "associated-press",
    resolution_state: ResolutionState = ResolutionState.UNRESOLVED,
    settlement_formula: str = "winner-takes-all",
    settlement_payout_cap: float = 1.0,
) -> MarketIdentity:
    """Create a MarketIdentity for testing."""
    return MarketIdentity(
        provider=provider,
        provider_market_id=provider_market_id,
        event=EventIdentity(
            event_id=event_id,
            title="Test Event",
            boundary=EventBoundary(
                close_time=ELECTION_CLOSE,
                expiry_time=ELECTION_EXPIRY,
            ),
        ),
        outcome=OutcomeIdentity(
            outcome_id=outcome_id,
            label="Yes",
        ),
        resolution=ResolutionRule(
            source=resolution_source,
            method="official",
            authority="usc",
        ),
        resolution_state=resolution_state,
        settlement=SettlementTerms(
            payout_type="binary",
            payout_cap=settlement_payout_cap,
            settlement_formula=settlement_formula,
        ),
        temporal_scope=TemporalScope(
            evaluation_time=ELECTION_CLOSE,
            is_point_in_time=True,
        ),
    )


def _make_mapping(
    *,
    buy_provider: str = "polymarket",
    sell_provider: str = "kalshi",
    event_id: str = "event-1",
    outcome_id: str = "yes",
    resolution_state: ResolutionState = ResolutionState.UNRESOLVED,
) -> MarketMapping:
    """Create a validated MarketMapping for testing."""
    source = _make_identity(
        provider=buy_provider,
        provider_market_id=f"{buy_provider}-001",
        event_id=event_id,
        outcome_id=outcome_id,
        resolution_state=resolution_state,
    )
    target = _make_identity(
        provider=sell_provider,
        provider_market_id=f"{sell_provider}-001",
        event_id=event_id,
        outcome_id=outcome_id,
        resolution_state=resolution_state,
    )
    return MarketMapping(source=source, target=target, relationship="same_market")


# ══════════════════════════════════════════════════════════════════════════════
# 1. MAPPING → DETECTOR INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════


class TestMappingDetectorIntegration:
    """Test MarketMapping → detect_cross_venue_arbitrage() integration."""

    def test_validated_mapping_carried_through(self) -> None:
        """Validated mapping is carried through to CrossVenueOpportunity."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")
        mapping = _make_mapping()

        result = detect_cross_venue_arbitrage(
            buy_book, sell_book, 100.0, mapping=mapping
        )

        assert result is not None
        assert result.mapping is not None
        assert result.mapping.relationship == "same_market"
        assert result.mapping.source.provider == "polymarket"
        assert result.mapping.target.provider == "kalshi"

    def test_mapping_none_preserves_backward_compatibility(self) -> None:
        """Detection without mapping (backward compatibility) still works."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)

        assert result is not None
        assert result.mapping is None

    def test_absent_mapping_not_semantic_equivalence(self) -> None:
        """mapping=None does not imply economic equivalence."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")

        result = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)

        # Result exists (opportunity detected), but no identity claim is made
        assert result is not None
        assert result.mapping is None

    def test_no_opportunity_with_mapping(self) -> None:
        """No opportunity detected even with valid mapping."""
        buy_book = _make_book(asks=[(0.50, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")
        mapping = _make_mapping()

        result = detect_cross_venue_arbitrage(
            buy_book, sell_book, 100.0, mapping=mapping
        )

        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAPPING → COST EVALUATION INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════


class TestMappingCostEvaluationIntegration:
    """Test MarketMapping → cost evaluation pipeline."""

    def test_mapping_carried_to_net_result(self) -> None:
        """Mapping is carried through to NetCrossVenueResult."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")
        mapping = _make_mapping()
        cost_model = ArbitrageCostModel(buy_fee_rate=0.02, sell_fee_rate=0.03)

        opportunity = detect_cross_venue_arbitrage(
            buy_book, sell_book, 100.0, mapping=mapping
        )
        assert opportunity is not None

        result = evaluate_cross_venue_costs(opportunity, cost_model)

        assert result.mapping is not None
        assert result.mapping.relationship == "same_market"

    def test_cost_evaluation_without_mapping(self) -> None:
        """Cost evaluation works without mapping (backward compatible)."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")
        cost_model = ArbitrageCostModel(buy_fee_rate=0.02, sell_fee_rate=0.03)

        opportunity = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert opportunity is not None

        result = evaluate_cross_venue_costs(opportunity, cost_model)

        assert result.mapping is None

    def test_phase_24_economics_unchanged(self) -> None:
        """Phase 2.4 economics remain unchanged with mapping=None."""
        buy_book = _make_book(asks=[(0.40, 100.0)], provider="polymarket", instrument="0xAAA")
        sell_book = _make_book(bids=[(0.45, 100.0)], provider="kalshi", instrument="0xBBB")
        cost_model = ArbitrageCostModel(buy_fee_rate=0.02, sell_fee_rate=0.03)

        opportunity = detect_cross_venue_arbitrage(buy_book, sell_book, 100.0)
        assert opportunity is not None

        result = evaluate_cross_venue_costs(opportunity, cost_model)

        # Verify economics are unchanged from Phase 2.4
        assert result.gross_spread == 5.0  # (0.45 - 0.40) * 100
        assert result.buy_fee == pytest.approx(0.80)  # 0.40 * 100 * 0.02
        assert result.sell_fee == pytest.approx(1.35)  # 0.45 * 100 * 0.03


# ══════════════════════════════════════════════════════════════════════════════
# 3. INCOMPATIBLE MAPPING REJECTION
# ══════════════════════════════════════════════════════════════════════════════


class TestIncompatibleMappingRejection:
    """Test that incompatible mappings are rejected at construction time."""

    def test_different_events_rejected(self) -> None:
        """Different events cannot be mapped as same_market."""
        source = _make_identity(event_id="event-a")
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            event_id="event-b",
        )
        with pytest.raises(MarketIdentityError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_different_outcomes_rejected(self) -> None:
        """Different outcomes cannot be mapped as same_market."""
        source = _make_identity(outcome_id="yes")
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            outcome_id="no",
        )
        with pytest.raises(IncompatibleOutcomeError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_different_resolution_sources_rejected(self) -> None:
        """Different resolution sources reject same_market mapping."""
        source = _make_identity(resolution_source="associated-press")
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_source="reuters",
        )
        with pytest.raises(IncompatibleResolutionError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_void_vs_resolved_rejected(self) -> None:
        """VOID vs RESOLVED raises IncompatibleResolutionStateError."""
        source = _make_identity(resolution_state=ResolutionState.VOID)
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_state=ResolutionState.RESOLVED,
        )
        with pytest.raises(IncompatibleResolutionStateError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_different_settlement_formulas_rejected(self) -> None:
        """Different settlement formulas reject same_market mapping."""
        source = _make_identity(settlement_formula="winner-takes-all")
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            settlement_formula="proportional",
        )
        with pytest.raises(IncompatibleSettlementError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_invalid_relationship_rejected(self) -> None:
        """Invalid relationship type is rejected."""
        source = _make_identity()
        target = _make_identity(provider="kalshi", provider_market_id="kalshi-001")
        with pytest.raises(MarketIdentityError):
            MarketMapping(source=source, target=target, relationship="maybe_same")


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESOLUTION STATE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


class TestResolutionStatePipeline:
    """Test resolution state through the full pipeline."""

    def test_void_market_cannot_be_arbitraged(self) -> None:
        """VOID market cannot be paired with RESOLVED market."""
        source = _make_identity(resolution_state=ResolutionState.VOID)
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_state=ResolutionState.RESOLVED,
        )
        with pytest.raises(IncompatibleResolutionStateError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_unresolved_to_resolved_allowed(self) -> None:
        """UNRESOLVED → RESOLVED is allowed (market may have resolved on one venue)."""
        source = _make_identity(resolution_state=ResolutionState.UNRESOLVED)
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_state=ResolutionState.RESOLVED,
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

    def test_both_resolved_compatible(self) -> None:
        """Both RESOLVED markets are compatible."""
        source = _make_identity(resolution_state=ResolutionState.RESOLVED)
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_state=ResolutionState.RESOLVED,
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

    def test_void_opposite_outcome_allowed(self) -> None:
        """VOID market can be mapped as opposite_outcome (no validation)."""
        source = _make_identity(resolution_state=ResolutionState.VOID)
        target = _make_identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            resolution_state=ResolutionState.VOID,
        )
        mapping = MarketMapping(source=source, target=target, relationship="opposite_outcome")
        assert mapping.relationship == "opposite_outcome"

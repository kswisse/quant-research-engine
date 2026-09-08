"""Adversarial edge-case tests for market identity validation.

Phase 2.5 — Market Identity & Resolution Semantics.
Tests that contracts which appear similar but are economically different
are correctly rejected as non-equivalent.

These are the critical false-positive tests — if the mapping validator
accepts these as equivalent, a false arbitrage signal would be generated.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data

Run: pytest tests/test_market_identity_adversarial.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_engine.market_identity.errors import (
    IncompatibleOutcomeError,
    IncompatibleResolutionError,
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
    SettlementTerms,
    TemporalScope,
)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
ELECTION_CLOSE = datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
ELECTION_EXPIRY = datetime(2026, 11, 4, 6, 0, 0, tzinfo=UTC)
BTC_NOON = datetime(2026, 9, 15, 12, 0, 0, tzinfo=UTC)
BTC_END_OF_DAY = datetime(2026, 9, 15, 16, 0, 0, tzinfo=UTC)


def _identity(
    *,
    provider: str = "polymarket",
    provider_market_id: str = "test-001",
    event_id: str = "event-1",
    event_title: str = "Test Event",
    outcome_id: str = "yes",
    outcome_label: str = "Yes",
    outcome_description: str = "",
    resolution_source: str = "associated-press",
    resolution_method: str = "official",
    resolution_authority: str = "usc",
    settlement_formula: str = "winner-takes-all",
    settlement_payout_cap: float = 1.0,
    close_time: datetime = ELECTION_CLOSE,
    expiry_time: datetime = ELECTION_EXPIRY,
    evaluation_time: datetime = ELECTION_CLOSE,
    is_point_in_time: bool = True,
    start_time: datetime | None = None,
) -> MarketIdentity:
    """Construct a MarketIdentity with fine-grained control over every field."""
    return MarketIdentity(
        provider=provider,
        provider_market_id=provider_market_id,
        event=EventIdentity(
            event_id=event_id,
            title=event_title,
            boundary=EventBoundary(
                close_time=close_time,
                expiry_time=expiry_time,
            ),
        ),
        outcome=OutcomeIdentity(
            outcome_id=outcome_id,
            label=outcome_label,
            description=outcome_description,
        ),
        resolution=ResolutionRule(
            source=resolution_source,
            method=resolution_method,
            authority=resolution_authority,
        ),
        settlement=SettlementTerms(
            payout_type="binary",
            payout_cap=settlement_payout_cap,
            settlement_formula=settlement_formula,
        ),
        temporal_scope=TemporalScope(
            evaluation_time=evaluation_time,
            is_point_in_time=is_point_in_time,
            start_time=start_time,
        ),
    )


def _same_market(
    source: MarketIdentity,
    target: MarketIdentity,
    relationship: str = "same_market",
) -> MarketMapping:
    """Try to create a same_market mapping; returns the mapping or raises."""
    return MarketMapping(source=source, target=target, relationship=relationship)


# ══════════════════════════════════════════════════════════════════════════════
# 1. ELECTION: WIN vs MAJORITY
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialElectionWinVsMajority:
    """'Will candidate X win the election?' vs
    'Will candidate X receive more than 50% of the vote?'

    These are economically different:
    - Win = most votes (plurality), even if < 50%
    - Majority = > 50% of votes
    A candidate can win with 45% in a multi-candidate race but NOT have majority.
    """

    def test_must_not_validate_as_equivalent(self) -> None:
        """Win and majority are NOT the same market."""
        source = _identity(
            event_id="election-2026",
            event_title="2026 US Presidential Election",
            outcome_id="candidate-x-wins",
            outcome_label="Candidate X wins the election",
            outcome_description=(
                "Candidate X receives the most votes and wins the presidency"
            ),
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-001",
            event_id="election-2026",
            event_title="2026 US Presidential Election",
            outcome_id="candidate-x-majority",
            outcome_label="Candidate X receives more than 50%",
            outcome_description=(
                "Candidate X receives more than 50% of the popular vote"
            ),
        )

        with pytest.raises((IncompatibleOutcomeError, MarketIdentityError)):
            _same_market(source, target)


class TestAdversarialWinVsPlurality:
    """'Will candidate X win?' vs 'Will candidate X get the most votes?'

    These are semantically equivalent (both = plurality), but they have
    different outcome_ids ('candidate-x-wins' vs 'candidate-x-most-votes').
    The validator compares outcome_ids — semantic NLP matching is out of scope.
    If these should be mappable, they need the same outcome_id.
    """

    def test_different_outcome_ids_are_rejected(self) -> None:
        """Different outcome_ids are rejected — even if semantically equivalent.

        To map these, use the same outcome_id on both sides, or use
        a separate semantic matching layer above the identity validator.
        """
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            outcome_label="Candidate X wins",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-002",
            event_id="election-2026",
            outcome_id="candidate-x-most-votes",
            outcome_label="Candidate X gets the most votes",
        )

        with pytest.raises(IncompatibleOutcomeError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 2. BTC: POINT-IN-TIME vs PATH-DEPENDENT
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialPointInTimeVsPathDependent:
    """'Will BTC be above $100K at 12:00 UTC?' vs
    'Will BTC ever reach $100K before the end of the day?'

    These are economically different:
    - Point-in-time: snapshot at a specific moment
    - Path-dependent: touch at any point during the window
    BTC could touch $100K at 11:00 then close at $99K — path-dependent wins,
    point-in-time loses.
    """

    def test_must_not_validate_as_equivalent(self) -> None:
        """Point-in-time and path-dependent are NOT the same market."""
        source = _identity(
            event_id="btc-price-2026-09-15",
            event_title="BTC Price",
            outcome_id="btc-above-100k-noon",
            outcome_label="BTC above $100K at 12:00 UTC",
            evaluation_time=BTC_NOON,
            is_point_in_time=True,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-2026-09-15-001",
            event_id="btc-price-2026-09-15",
            event_title="BTC Price",
            outcome_id="btc-ever-reaches-100k",
            outcome_label="BTC ever reaches $100K before end of day",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=FROZEN,
        )

        with pytest.raises((IncompatibleOutcomeError, MarketIdentityError)):
            _same_market(source, target)


class TestAdversarialDifferentTemporalScope:
    """Both point-in-time but at different timestamps."""

    def test_different_timestamps_not_equivalent(self) -> None:
        """Same event, same outcome, but evaluated at different times."""
        source = _identity(
            event_id="btc-price-2026-09-15",
            outcome_id="btc-above-100k",
            evaluation_time=BTC_NOON,
            is_point_in_time=True,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-2026-09-15-002",
            event_id="btc-price-2026-09-15",
            outcome_id="btc-above-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=True,
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DIFFERENT RESOLUTION SOURCE
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentResolutionSource:
    """Same event + same outcome + different resolution source.

    Resolution source matters because:
    - AP might call a race differently than Reuters
    - Different sources have different accuracy and timing
    - Conflicting resolution = conflicting settlement
    """

    def test_different_sources_rejected(self) -> None:
        """Different resolution sources reject same_market mapping."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="associated-press",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-003",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="reuters",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)

    def test_different_methods_rejected(self) -> None:
        """Different resolution methods reject same_market mapping."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_method="official-certification",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-004",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_method="media-projection",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)

    def test_different_authority_rejected(self) -> None:
        """Different resolution authority rejects same_market mapping."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_authority="usc",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-005",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_authority="state-election-board",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)

    def test_same_source_different_provider_accepted(self) -> None:
        """Same resolution source from different providers is accepted."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="associated-press",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-006",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="associated-press",
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"


# ══════════════════════════════════════════════════════════════════════════════
# 4. DIFFERENT OUTCOME
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentOutcome:
    """Same event + different outcome — must reject as same_market."""

    def test_different_outcome_ids_rejected(self) -> None:
        """Different outcome_ids reject same_market mapping."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            outcome_label="Candidate X wins",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-007",
            event_id="election-2026",
            outcome_id="candidate-y-wins",
            outcome_label="Candidate Y wins",
        )

        with pytest.raises(IncompatibleOutcomeError):
            _same_market(source, target)

    def test_same_outcome_id_different_labels_accepted(self) -> None:
        """Same outcome_id with different labels is accepted.

        outcome_id is the semantic identity key — same outcome_id = same
        semantic identity regardless of label text. Labels are human-readable
        decorations, not identity components. If the outcome_ids match,
        the markets refer to the same outcome even if the labels differ.
        """
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x",
            outcome_label="Candidate X wins the presidency",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-008",
            event_id="election-2026",
            outcome_id="candidate-x",
            outcome_label="Candidate X wins California",
        )

        # Same outcome_id = same semantic identity — accepted
        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPPOSITE OUTCOME
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialOppositeOutcome:
    """Same event + opposite outcome.

    These are NOT same_market — they are complementary.
    A proper mapping should use relationship="opposite_outcome".
    """

    def test_opposite_outcome_not_same_market(self) -> None:
        """Yes/No on same event is not same_market."""
        source = _identity(
            event_id="election-2026",
            outcome_id="yes",
            outcome_label="Yes",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-009",
            event_id="election-2026",
            outcome_id="no",
            outcome_label="No",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)

    def test_opposite_outcome_can_be_mapped_explicitly(self) -> None:
        """Opposite outcomes CAN be mapped with explicit relationship."""
        source = _identity(
            event_id="election-2026",
            outcome_id="yes",
            outcome_label="Yes",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-010",
            event_id="election-2026",
            outcome_id="no",
            outcome_label="No",
        )

        mapping = _same_market(source, target, relationship="opposite_outcome")
        assert mapping.relationship == "opposite_outcome"

    def test_complementary_outcomes_not_same_market(self) -> None:
        """Complementary outcomes (not binary opposites) are not same_market."""
        source = _identity(
            event_id="btc-100k",
            event_title="BTC 100K",
            outcome_id="above-100k",
            outcome_label="BTC above $100K",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-001",
            event_id="btc-100k",
            event_title="BTC 100K",
            outcome_id="below-100k",
            outcome_label="BTC below $100K",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DIFFERENT EVENTS WITH SIMILAR DESCRIPTIONS
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialSimilarDescriptions:
    """Events with similar titles but different event_ids are NOT equivalent."""

    def test_similar_titles_different_events_rejected(self) -> None:
        """'2026 Election' vs '2026 Midterm Election' are different events."""
        source = _identity(
            event_id="election-2026-presidential",
            event_title="2026 Presidential Election",
            outcome_id="candidate-x-wins",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-midterm",
            event_id="election-2026-midterm",
            event_title="2026 Midterm Election",
            outcome_id="candidate-x-wins",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)

    def test_same_title_different_event_id_rejected(self) -> None:
        """Even identical titles but different event_ids are different events."""
        source = _identity(
            event_id="event-abc",
            event_title="Will it rain tomorrow?",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-rain-001",
            event_id="event-xyz",
            event_title="Will it rain tomorrow?",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 7. DIFFERENT EXPIRY TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentExpiry:
    """Different expiry timestamps on same event + outcome.

    A market expiring at midnight vs 6am the next day can resolve
    differently if the outcome changes overnight.
    """

    def test_different_expiry_rejected(self) -> None:
        """Different expiry times reject same_market mapping."""
        source = _identity(
            event_id="btc-100k",
            outcome_id="btc-above-100k",
            close_time=BTC_NOON,
            expiry_time=BTC_NOON,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-002",
            event_id="btc-100k",
            outcome_id="btc-above-100k",
            close_time=BTC_NOON,
            expiry_time=BTC_END_OF_DAY,
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 8. DIFFERENT RESOLUTION TIMESTAMPS
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentResolutionTimestamp:
    """Markets resolving at different times for the same underlying event."""

    def test_different_resolution_time_rejected(self) -> None:
        """Different evaluation_times reject same_market mapping."""
        source = _identity(
            event_id="btc-100k",
            outcome_id="btc-above-100k",
            evaluation_time=BTC_NOON,
            is_point_in_time=True,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-003",
            event_id="btc-100k",
            outcome_id="btc-above-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=True,
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 8b. DIFFERENT START TIME (PATH-DEPENDENT)
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentStartTime:
    """Both path-dependent but different start times.

    Example:
        Market A: "BTC ever reaches $100K between 09:00 UTC and 12:00 UTC"
        Market B: "BTC ever reaches $100K between 00:00 UTC and 12:00 UTC"

    Same evaluation_time, same is_point_in_time=False, but different
    start_time. BTC could touch $100K at 08:00 — Market B wins, Market A
    loses. These MUST NOT validate as equivalent.
    """

    def test_different_start_times_not_equivalent(self) -> None:
        """Path-dependent markets with different start_times are rejected."""
        source = _identity(
            event_id="btc-100k",
            outcome_id="btc-ever-reaches-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=FROZEN,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-001",
            event_id="btc-100k",
            outcome_id="btc-ever-reaches-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=BTC_NOON,
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)

    def test_same_start_times_compatible(self) -> None:
        """Path-dependent markets with identical start_times are accepted."""
        source = _identity(
            event_id="btc-100k",
            outcome_id="btc-ever-reaches-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=FROZEN,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-002",
            event_id="btc-100k",
            outcome_id="btc-ever-reaches-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=FROZEN,
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"


# ══════════════════════════════════════════════════════════════════════════════
# 9. DIFFERENT RESOLUTION AUTHORITIES
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialDifferentAuthority:
    """Different resolution authorities for the same event + outcome.

    Even if source and method match, different authorities can produce
    conflicting results.
    """

    def test_different_authority_rejected(self) -> None:
        """Different authorities reject same_market mapping."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_authority="federal-election-commission",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-011",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_authority="state-election-board",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 10. VOID / CANCELLED OUTCOMES
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialVoidOutcome:
    """Markets where the outcome is void or cancelled."""

    def test_void_outcome_not_equivalent_to_active(self) -> None:
        """A voided market is not equivalent to an active market."""
        source = _identity(
            event_id="event-1",
            outcome_id="yes",
            outcome_label="Yes",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-void-001",
            event_id="event-1",
            outcome_id="void",
            outcome_label="Void / Cancelled",
        )

        with pytest.raises((IncompatibleOutcomeError, MarketIdentityError)):
            _same_market(source, target)


class TestAdversarialUnresolvedMarket:
    """Markets that are unresolved vs resolved."""

    def test_unresolved_not_equivalent_to_resolved(self) -> None:
        """An unresolved market is not the same as a resolved one."""
        source = _identity(
            event_id="event-1",
            outcome_id="yes",
            resolution_source="pending",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-resolved-001",
            event_id="event-1",
            outcome_id="yes",
            resolution_source="associated-press",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 11. AMBIGUOUS SETTLEMENT SEMANTICS
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialAmbiguousSettlement:
    """Different settlement formulas for the same event + outcome.

    Settlement formula determines how payouts are calculated.
    """

    def test_different_settlement_formula_rejected(self) -> None:
        """winner-takes-all vs proportional payout are different."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            settlement_formula="winner-takes-all",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-012",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            settlement_formula="proportional",
        )

        with pytest.raises(IncompatibleSettlementError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 12. INCOMPATIBLE PAYOUT STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialIncompatiblePayout:
    """Different payout types or caps for the same event + outcome."""

    def test_different_payout_cap_rejected(self) -> None:
        """Different payout caps mean different risk/reward profiles."""
        source = _identity(
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            settlement_payout_cap=1.0,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-2026-013",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            settlement_payout_cap=10.0,
        )

        with pytest.raises(IncompatibleSettlementError):
            _same_market(source, target)

    def test_different_payout_type_rejected(self) -> None:
        """Different payout types (binary vs range) are different."""
        source = _identity(
            event_id="btc-100k",
            outcome_id="btc-above-100k",
        )
        # Override settlement for range-type payout
        source_with_range = MarketIdentity(
            provider=source.provider,
            provider_market_id=source.provider_market_id,
            event=source.event,
            outcome=source.outcome,
            resolution=source.resolution,
            settlement=SettlementTerms(
                payout_type="range",
                payout_cap=1.0,
                settlement_formula="winner-takes-all",
            ),
            temporal_scope=source.temporal_scope,
        )

        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-004",
            event_id="btc-100k",
            outcome_id="btc-above-100k",
        )

        with pytest.raises(IncompatibleSettlementError):
            _same_market(source_with_range, target)


# ══════════════════════════════════════════════════════════════════════════════
# 13. PROVIDER-SPECIFIC vs CANONICAL IDENTITY
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialProviderVsCanonical:
    """Provider-specific identifiers should not override canonical identity.

    Two markets with different provider_market_ids but identical semantic
    content SHOULD be mappable (same_market).
    """

    def test_different_provider_ids_same_semantics_accepted(self) -> None:
        """Different provider IDs with identical semantics are same_market."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-0xABC123",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="KXLS-2026-ELEC-PRES-001",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"

    def test_same_provider_different_ids_same_semantics_accepted(self) -> None:
        """Same provider, different market IDs, identical semantics."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-0xABC123",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )
        target = _identity(
            provider="polymarket",
            provider_market_id="poly-0xDEF456",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"


# ══════════════════════════════════════════════════════════════════════════════
# 14. CROSS-CUTTING: MULTIPLE MISMATCHES
# ══════════════════════════════════════════════════════════════════════════════


class TestAdversarialMultipleMismatches:
    """Markets that differ on multiple dimensions simultaneously."""

    def test_multiple_mismatches_rejected(self) -> None:
        """Different event + different outcome + different source — rejected."""
        source = _identity(
            event_id="election-2026",
            event_title="2026 Presidential Election",
            outcome_id="candidate-x-wins",
            resolution_source="associated-press",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k",
            event_id="btc-100k",
            event_title="BTC 100K",
            outcome_id="btc-above-100k",
            resolution_source="coindesk",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 15. INTEGRITY: FALSE ARBITRAGE PREVENTION
# ══════════════════════════════════════════════════════════════════════════════


class TestFalseArbitragePrevention:
    """End-to-end adversarial scenarios where false arbitrage would be
    generated if the mapping validator accepted these as equivalent."""

    def test_false_arb_win_vs_majority(self) -> None:
        """Arbitrageur sees 'Candidate X wins' on Polymarket and
        'Candidate X > 50%' on Kalshi with similar prices.
        These are NOT the same market — no arbitrage exists."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-election-001",
            event_id="election-2026",
            event_title="2026 Presidential Election",
            outcome_id="candidate-x-wins",
            outcome_label="Candidate X wins",
            resolution_source="associated-press",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-001",
            event_id="election-2026",
            event_title="2026 Presidential Election",
            outcome_id="candidate-x-majority",
            outcome_label="Candidate X > 50%",
            resolution_source="associated-press",
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)

    def test_false_arb_point_vs_path(self) -> None:
        """Arbitrageur sees BTC-100K at 12:00 on Polymarket and
        BTC-100K-ever on Kalshi with overlapping prices.
        These are NOT the same market — no arbitrage exists."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-btc-100k-noon",
            event_id="btc-100k-2026-09-15",
            event_title="BTC 100K",
            outcome_id="btc-above-100k-noon",
            evaluation_time=BTC_NOON,
            is_point_in_time=True,
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-btc-100k-ever",
            event_id="btc-100k-2026-09-15",
            event_title="BTC 100K",
            outcome_id="btc-ever-reaches-100k",
            evaluation_time=BTC_END_OF_DAY,
            is_point_in_time=False,
            start_time=FROZEN,
        )

        with pytest.raises(MarketIdentityError):
            _same_market(source, target)

    def test_false_arb_different_resolution(self) -> None:
        """Arbitrageur sees same event + outcome but different resolution sources.
        These could resolve differently — no arbitrage exists."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-election-ap",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="associated-press",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-election-reuters",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
            resolution_source="reuters",
        )

        with pytest.raises(IncompatibleResolutionError):
            _same_market(source, target)


# ══════════════════════════════════════════════════════════════════════════════
# 16. VALID MAPPINGS (POSITIVE CASES)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidMappings:
    """Confirm that genuinely equivalent markets ARE accepted."""

    def test_identical_markets_accepted(self) -> None:
        """Identical markets from different providers are same_market."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-001",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-001",
            event_id="election-2026",
            outcome_id="candidate-x-wins",
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"

    def test_identical_markets_same_provider_accepted(self) -> None:
        """Identical markets from same provider (different IDs) are same_market."""
        source = _identity(
            provider="polymarket",
            provider_market_id="poly-001",
            event_id="btc-100k",
            outcome_id="btc-above-100k",
        )
        target = _identity(
            provider="polymarket",
            provider_market_id="poly-002",
            event_id="btc-100k",
            outcome_id="btc-above-100k",
        )

        mapping = _same_market(source, target)
        assert mapping.relationship == "same_market"

    def test_opposite_outcome_explicit_mapping(self) -> None:
        """Opposite outcomes can be explicitly mapped."""
        source = _identity(
            event_id="election-2026",
            outcome_id="yes",
        )
        target = _identity(
            provider="kalshi",
            provider_market_id="kalshi-no",
            event_id="election-2026",
            outcome_id="no",
        )

        mapping = _same_market(source, target, relationship="opposite_outcome")
        assert mapping.relationship == "opposite_outcome"

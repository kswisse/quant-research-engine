"""Market identity model tests.

Phase 2.5 — Market Identity & Resolution Semantics.
Tests the core identity models, error hierarchy, deterministic IDs,
serialization, equality, immutability, and schema versioning.

All tests are deterministic:
- No network calls
- No shared state between tests
- Every test owns its data
- No external service dependencies

Run: pytest tests/test_market_identity_models.py -v
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FROZEN_CLOCK = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
ELECTION_DAY = datetime(2026, 11, 3, 23, 59, 59, tzinfo=UTC)
BTC_DEADLINE = datetime(2026, 9, 15, 16, 0, 0, tzinfo=UTC)


def _make_event_identity(
    *,
    event_id: str = "election-2026-us-presidential",
    title: str = "2026 US Presidential Election",
    category: str = "politics",
    boundary: EventBoundary | None = None,
) -> EventIdentity:
    """Create a deterministic EventIdentity for tests."""
    return EventIdentity(
        event_id=event_id,
        title=title,
        category=category,
        boundary=boundary or EventBoundary(
            close_time=ELECTION_DAY,
            expiry_time=ELECTION_DAY + timedelta(hours=1),
        ),
    )


def _make_outcome_identity(
    *,
    outcome_id: str = "candidate-x-wins",
    label: str = "Candidate X wins",
    description: str = "Candidate X receives the most votes and wins the election",
) -> OutcomeIdentity:
    """Create a deterministic OutcomeIdentity for tests."""
    return OutcomeIdentity(
        outcome_id=outcome_id,
        label=label,
        description=description,
    )


def _make_resolution_rule(
    *,
    source: str = "associated-press",
    method: str = "official-certification",
    authority: str = "usc",
    is_definitive: bool = True,
) -> ResolutionRule:
    """Create a deterministic ResolutionRule for tests."""
    return ResolutionRule(
        source=source,
        method=method,
        authority=authority,
        is_definitive=is_definitive,
    )


def _make_settlement_terms(
    *,
    payout_type: str = "binary",
    payout_cap: float = 1.0,
    settlement_formula: str = "winner-takes-all",
    settlement_currency: str = "USD",
) -> SettlementTerms:
    """Create deterministic SettlementTerms for tests."""
    return SettlementTerms(
        payout_type=payout_type,
        payout_cap=payout_cap,
        settlement_formula=settlement_formula,
        settlement_currency=settlement_currency,
    )


def _make_market_identity(
    *,
    provider: str = "polymarket",
    provider_market_id: str = "polymarket-election-2026-001",
    event: EventIdentity | None = None,
    outcome: OutcomeIdentity | None = None,
    resolution: ResolutionRule | None = None,
    resolution_state: ResolutionState = ResolutionState.UNRESOLVED,
    settlement: SettlementTerms | None = None,
    temporal_scope: TemporalScope | None = None,
) -> MarketIdentity:
    """Create a complete MarketIdentity for tests."""
    return MarketIdentity(
        provider=provider,
        provider_market_id=provider_market_id,
        event=event or _make_event_identity(),
        outcome=outcome or _make_outcome_identity(),
        resolution=resolution or _make_resolution_rule(),
        resolution_state=resolution_state,
        settlement=settlement or _make_settlement_terms(),
        temporal_scope=temporal_scope or TemporalScope(
            evaluation_time=ELECTION_DAY,
            is_point_in_time=True,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 1. ERROR HIERARCHY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorHierarchy:
    """Test error type hierarchy and catchability."""

    def test_market_identity_error_is_base_exception(self) -> None:
        """MarketIdentityError is the base for all market identity errors."""
        assert issubclass(MarketIdentityError, Exception)

    def test_incompatible_outcome_error_inherits_market_identity_error(self) -> None:
        """IncompatibleOutcomeError inherits from MarketIdentityError."""
        assert issubclass(IncompatibleOutcomeError, MarketIdentityError)

    def test_incompatible_resolution_error_inherits_market_identity_error(self) -> None:
        """IncompatibleResolutionError inherits from MarketIdentityError."""
        assert issubclass(IncompatibleResolutionError, MarketIdentityError)

    def test_incompatible_settlement_error_inherits_market_identity_error(self) -> None:
        """IncompatibleSettlementError inherits from MarketIdentityError."""
        assert issubclass(IncompatibleSettlementError, MarketIdentityError)

    def test_all_errors_catchable_as_base(self) -> None:
        """All specific errors are catchable as MarketIdentityError."""
        for exc_cls in (
            IncompatibleOutcomeError,
            IncompatibleResolutionError,
            IncompatibleSettlementError,
        ):
            with pytest.raises(MarketIdentityError):
                raise exc_cls("test message")

    def test_error_preserves_message(self) -> None:
        """Error messages are preserved through the hierarchy."""
        msg = "resolution source mismatch: ap vs reuters"
        err = IncompatibleResolutionError(msg)
        assert str(err) == msg

    def test_error_with_context(self) -> None:
        """Errors carry their message through the constructor."""
        err = IncompatibleOutcomeError(
            "outcome mismatch: expected candidate-x-wins, got candidate-y-wins",
        )
        assert "outcome mismatch" in str(err)
        assert "candidate-x-wins" in str(err)


# ══════════════════════════════════════════════════════════════════════════════
# 2. EVENT IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestEventIdentity:
    """Test EventIdentity model construction and invariants."""

    def test_construction(self) -> None:
        """EventIdentity constructs with valid data."""
        event = _make_event_identity()
        assert event.event_id == "election-2026-us-presidential"
        assert event.title == "2026 US Presidential Election"
        assert event.category == "politics"

    def test_deterministic_id(self) -> None:
        """Same fields produce the same event_id property."""
        event1 = _make_event_identity()
        event2 = _make_event_identity()
        assert event1.event_id == event2.event_id

    def test_frozen(self) -> None:
        """EventIdentity is immutable (frozen)."""
        event = _make_event_identity()
        with pytest.raises(Exception):  # noqa: B017
            event.event_id = "changed"  # type: ignore[misc]

    def test_optional_category(self) -> None:
        """EventIdentity allows empty/missing category."""
        event = EventIdentity(
            event_id="test-event",
            title="Test Event",
            boundary=EventBoundary(
                close_time=FROZEN_CLOCK,
                expiry_time=FROZEN_CLOCK + timedelta(hours=1),
            ),
        )
        assert event.category == ""

    def test_boundary_required(self) -> None:
        """EventIdentity requires a boundary."""
        with pytest.raises(Exception):  # noqa: B017
            EventIdentity(event_id="test", title="Test")  # type: ignore[call-arg]

    def test_serialization_roundtrip(self) -> None:
        """EventIdentity survives serialization roundtrip."""
        event = _make_event_identity()
        d = event.model_dump()
        restored = EventIdentity.model_validate(d)
        assert restored.event_id == event.event_id
        assert restored.title == event.title

    def test_json_roundtrip(self) -> None:
        """EventIdentity survives JSON serialization roundtrip."""
        event = _make_event_identity()
        json_str = event.model_dump_json()
        restored = EventIdentity.model_validate_json(json_str)
        assert restored == event

    def test_different_events_different_ids(self) -> None:
        """Different event data produces different event_ids."""
        event1 = _make_event_identity(event_id="event-a", title="Event A")
        event2 = _make_event_identity(event_id="event-b", title="Event B")
        # event_id is provided directly, so they differ
        assert event1.event_id != event2.event_id

    def test_schema_version_present(self) -> None:
        """EventIdentity has a schema_version field."""
        event = _make_event_identity()
        assert hasattr(event, "schema_version")
        assert event.schema_version == "1"


class TestEventBoundary:
    """Test EventBoundary model."""

    def test_construction(self) -> None:
        """EventBoundary constructs with valid times."""
        boundary = EventBoundary(
            close_time=FROZEN_CLOCK,
            expiry_time=FROZEN_CLOCK + timedelta(hours=1),
        )
        assert boundary.close_time == FROZEN_CLOCK
        assert boundary.expiry_time == FROZEN_CLOCK + timedelta(hours=1)

    def test_close_before_expiry(self) -> None:
        """EventBoundary accepts close_time before expiry_time."""
        boundary = EventBoundary(
            close_time=FROZEN_CLOCK,
            expiry_time=FROZEN_CLOCK + timedelta(hours=1),
        )
        assert boundary.close_time < boundary.expiry_time

    def test_close_equals_expiry(self) -> None:
        """EventBoundary accepts close_time == expiry_time."""
        boundary = EventBoundary(
            close_time=FROZEN_CLOCK,
            expiry_time=FROZEN_CLOCK,
        )
        assert boundary.close_time == boundary.expiry_time

    def test_frozen(self) -> None:
        """EventBoundary is immutable."""
        boundary = EventBoundary(
            close_time=FROZEN_CLOCK,
            expiry_time=FROZEN_CLOCK + timedelta(hours=1),
        )
        with pytest.raises(Exception):  # noqa: B017
            boundary.close_time = FROZEN_CLOCK + timedelta(days=1)  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTCOME IDENTITY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestOutcomeIdentity:
    """Test OutcomeIdentity model construction and invariants."""

    def test_construction(self) -> None:
        """OutcomeIdentity constructs with valid data."""
        outcome = _make_outcome_identity()
        assert outcome.outcome_id == "candidate-x-wins"
        assert outcome.label == "Candidate X wins"

    def test_frozen(self) -> None:
        """OutcomeIdentity is immutable."""
        outcome = _make_outcome_identity()
        with pytest.raises(Exception):  # noqa: B017
            outcome.label = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        """OutcomeIdentity survives serialization roundtrip."""
        outcome = _make_outcome_identity()
        d = outcome.model_dump()
        restored = OutcomeIdentity.model_validate(d)
        assert restored == outcome

    def test_description_optional(self) -> None:
        """OutcomeIdentity allows empty description."""
        outcome = OutcomeIdentity(
            outcome_id="yes",
            label="Yes",
        )
        assert outcome.description == ""

    def test_different_outcomes_different_ids(self) -> None:
        """Different outcome_id produces different identities."""
        o1 = OutcomeIdentity(outcome_id="yes", label="Yes")
        o2 = OutcomeIdentity(outcome_id="no", label="No")
        assert o1.outcome_id != o2.outcome_id


# ══════════════════════════════════════════════════════════════════════════════
# 4. RESOLUTION RULE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestResolutionRule:
    """Test ResolutionRule model construction and invariants."""

    def test_construction(self) -> None:
        """ResolutionRule constructs with valid data."""
        rule = _make_resolution_rule()
        assert rule.source == "associated-press"
        assert rule.method == "official-certification"
        assert rule.is_definitive is True

    def test_frozen(self) -> None:
        """ResolutionRule is immutable."""
        rule = _make_resolution_rule()
        with pytest.raises(Exception):  # noqa: B017
            rule.source = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        """ResolutionRule survives serialization roundtrip."""
        rule = _make_resolution_rule()
        d = rule.model_dump()
        restored = ResolutionRule.model_validate(d)
        assert restored == rule

    def test_source_required(self) -> None:
        """ResolutionRule requires a non-empty source."""
        with pytest.raises(Exception):  # noqa: B017
            ResolutionRule(source="", method="test", authority="test")  # type: ignore[call-arg]

    def test_non_definitive_source(self) -> None:
        """ResolutionRule allows non-definitive sources."""
        rule = ResolutionRule(
            source="community-poll",
            method="crowd-vote",
            authority="community",
            is_definitive=False,
        )
        assert rule.is_definitive is False


# ══════════════════════════════════════════════════════════════════════════════
# 5. SETTLEMENT TERMS TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestSettlementTerms:
    """Test SettlementTerms model construction and invariants."""

    def test_construction(self) -> None:
        """SettlementTerms constructs with valid data."""
        terms = _make_settlement_terms()
        assert terms.payout_type == "binary"
        assert terms.payout_cap == 1.0
        assert terms.settlement_formula == "winner-takes-all"

    def test_frozen(self) -> None:
        """SettlementTerms is immutable."""
        terms = _make_settlement_terms()
        with pytest.raises(Exception):  # noqa: B017
            terms.payout_cap = 2.0  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        """SettlementTerms survives serialization roundtrip."""
        terms = _make_settlement_terms()
        d = terms.model_dump()
        restored = SettlementTerms.model_validate(d)
        assert restored == terms

    def test_payout_cap_positive(self) -> None:
        """SettlementTerms requires positive payout_cap."""
        with pytest.raises(Exception):  # noqa: B017
            SettlementTerms(payout_type="binary", payout_cap=-1.0)

    def test_payout_cap_zero_rejected(self) -> None:
        """SettlementTerms rejects zero payout_cap."""
        with pytest.raises(Exception):  # noqa: B017
            SettlementTerms(payout_type="binary", payout_cap=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TEMPORAL SCOPE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestTemporalScope:
    """Test TemporalScope model construction and invariants."""

    def test_point_in_time(self) -> None:
        """TemporalScope for a point-in-time evaluation."""
        scope = TemporalScope(
            evaluation_time=ELECTION_DAY,
            is_point_in_time=True,
        )
        assert scope.is_point_in_time is True
        assert scope.evaluation_time == ELECTION_DAY

    def test_path_dependent(self) -> None:
        """TemporalScope for a path-dependent condition."""
        scope = TemporalScope(
            evaluation_time=BTC_DEADLINE,
            is_point_in_time=False,
            start_time=FROZEN_CLOCK,
        )
        assert scope.is_point_in_time is False
        assert scope.start_time == FROZEN_CLOCK

    def test_frozen(self) -> None:
        """TemporalScope is immutable."""
        scope = TemporalScope(
            evaluation_time=ELECTION_DAY,
            is_point_in_time=True,
        )
        with pytest.raises(Exception):  # noqa: B017
            scope.is_point_in_time = False  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# 7. MARKET IDENTITY COMPOSITE MODEL TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketIdentity:
    """Test MarketIdentity composite model."""

    def test_construction(self) -> None:
        """MarketIdentity constructs with all components."""
        identity = _make_market_identity()
        assert identity.provider == "polymarket"
        assert identity.provider_market_id == "polymarket-election-2026-001"
        assert identity.event.event_id == "election-2026-us-presidential"

    def test_frozen(self) -> None:
        """MarketIdentity is immutable."""
        identity = _make_market_identity()
        with pytest.raises(Exception):  # noqa: B017
            identity.provider = "changed"  # type: ignore[misc]

    def test_deterministic_identity(self) -> None:
        """Same components produce the same canonical_identity hash."""
        id1 = _make_market_identity()
        id2 = _make_market_identity()
        assert id1.canonical_identity == id2.canonical_identity

    def test_different_providers_same_canonical_identity(self) -> None:
        """Different providers with identical semantics produce the same canonical_identity.

        Provider-specific identifiers are intentionally EXCLUDED from the
        canonical hash — two markets with identical semantic content from
        different providers are the same market. This is the same design
        as snapshot_id in order_book.models.
        """
        id1 = _make_market_identity(provider="polymarket")
        id2 = _make_market_identity(provider="kalshi")
        assert id1.canonical_identity == id2.canonical_identity

    def test_different_outcomes_different_identity(self) -> None:
        """Different outcomes produce different canonical_identity."""
        id1 = _make_market_identity(
            outcome=OutcomeIdentity(outcome_id="yes", label="Yes")
        )
        id2 = _make_market_identity(
            outcome=OutcomeIdentity(outcome_id="no", label="No")
        )
        assert id1.canonical_identity != id2.canonical_identity

    def test_canonical_identity_is_hex(self) -> None:
        """canonical_identity is a hex string (SHA-256 derived)."""
        identity = _make_market_identity()
        # Should be valid hex
        int(identity.canonical_identity, 16)

    def test_serialization_roundtrip(self) -> None:
        """MarketIdentity survives serialization roundtrip."""
        identity = _make_market_identity()
        d = identity.model_dump()
        restored = MarketIdentity.model_validate(d)
        assert restored.canonical_identity == identity.canonical_identity

    def test_json_roundtrip(self) -> None:
        """MarketIdentity survives JSON serialization roundtrip."""
        identity = _make_market_identity()
        json_str = identity.model_dump_json()
        restored = MarketIdentity.model_validate_json(json_str)
        assert restored == identity

    def test_schema_version(self) -> None:
        """MarketIdentity has a schema_version field."""
        identity = _make_market_identity()
        assert identity.schema_version == "1"

    def test_provider_specific_identity(self) -> None:
        """MarketIdentity preserves provider-specific identifiers."""
        identity = _make_market_identity(
            provider="polymarket",
            provider_market_id="poly-0x123abc",
        )
        assert identity.provider == "polymarket"
        assert identity.provider_market_id == "poly-0x123abc"

    def test_to_dict(self) -> None:
        """MarketIdentity can be serialized to dict."""
        identity = _make_market_identity()
        d = identity.model_dump()
        assert isinstance(d, dict)
        assert d["provider"] == "polymarket"
        assert "event" in d
        assert "outcome" in d
        assert "resolution" in d
        assert "settlement" in d

    def test_from_dict(self) -> None:
        """MarketIdentity can be deserialized from dict."""
        identity = _make_market_identity()
        d = identity.model_dump()
        restored = MarketIdentity.model_validate(d)
        assert restored == identity


# ══════════════════════════════════════════════════════════════════════════════
# 8. MARKET MAPPING VALIDATION TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestMarketMapping:
    """Test MarketMapping validation rules."""

    def test_valid_explicit_mapping(self) -> None:
        """Explicit mapping between identical markets succeeds."""
        source = _make_market_identity(provider="polymarket")
        target = _make_market_identity(provider="kalshi")
        mapping = MarketMapping(
            source=source,
            target=target,
            relationship="same_market",
        )
        assert mapping.relationship == "same_market"
        assert mapping.mapping_id is not None

    def test_same_event_same_outcome(self) -> None:
        """Same event + same outcome validates as same_market."""
        source = _make_market_identity(
            event=EventIdentity(
                event_id="election-2026",
                title="Election",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY + timedelta(hours=1),
                ),
            ),
            outcome=OutcomeIdentity(outcome_id="candidate-x", label="X wins"),
        )
        target = _make_market_identity(
            provider="kalshi",
            event=EventIdentity(
                event_id="election-2026",
                title="Election",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY + timedelta(hours=1),
                ),
            ),
            outcome=OutcomeIdentity(outcome_id="candidate-x", label="X wins"),
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

    def test_same_event_opposite_outcome(self) -> None:
        """Same event + opposite outcome can be mapped with explicit relationship."""
        source = _make_market_identity(
            outcome=OutcomeIdentity(outcome_id="yes", label="Yes"),
        )
        target = _make_market_identity(
            provider="kalshi",
            outcome=OutcomeIdentity(outcome_id="no", label="No"),
        )
        mapping = MarketMapping(
            source=source,
            target=target,
            relationship="opposite_outcome",
        )
        assert mapping.relationship == "opposite_outcome"

    def test_different_events_rejected(self) -> None:
        """Different events cannot be mapped as same_market."""
        source = _make_market_identity(
            event=EventIdentity(
                event_id="event-a",
                title="Event A",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY + timedelta(hours=1),
                ),
            ),
        )
        target = _make_market_identity(
            provider="kalshi",
            event=EventIdentity(
                event_id="event-b",
                title="Event B",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY + timedelta(hours=1),
                ),
            ),
        )
        with pytest.raises(MarketIdentityError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_resolution_source_mismatch_rejected(self) -> None:
        """Different resolution sources reject same_market mapping."""
        source = _make_market_identity(
            resolution=ResolutionRule(
                source="associated-press",
                method="official",
                authority="usc",
            ),
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution=ResolutionRule(
                source="reuters",
                method="official",
                authority="usc",
            ),
        )
        with pytest.raises(IncompatibleResolutionError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_expiry_mismatch_rejected(self) -> None:
        """Different expiry times reject same_market mapping."""
        source = _make_market_identity(
            event=EventIdentity(
                event_id="election-2026",
                title="Election",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY,
                ),
            ),
        )
        target = _make_market_identity(
            provider="kalshi",
            event=EventIdentity(
                event_id="election-2026",
                title="Election",
                boundary=EventBoundary(
                    close_time=ELECTION_DAY,
                    expiry_time=ELECTION_DAY + timedelta(hours=6),
                ),
            ),
        )
        with pytest.raises(MarketIdentityError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_settlement_mismatch_rejected(self) -> None:
        """Different settlement formulas reject same_market mapping."""
        source = _make_market_identity(
            settlement=SettlementTerms(
                payout_type="binary",
                payout_cap=1.0,
                settlement_formula="winner-takes-all",
            ),
        )
        target = _make_market_identity(
            provider="kalshi",
            settlement=SettlementTerms(
                payout_type="binary",
                payout_cap=1.0,
                settlement_formula="proportional",
            ),
        )
        with pytest.raises(IncompatibleSettlementError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_ambiguous_semantics_rejected(self) -> None:
        """Ambiguous relationship is rejected."""
        source = _make_market_identity()
        target = _make_market_identity(provider="kalshi")
        with pytest.raises(MarketIdentityError):
            MarketMapping(source=source, target=target, relationship="maybe_same")

    def test_deterministic_mapping_id(self) -> None:
        """Same source/target always produces the same mapping_id."""
        source = _make_market_identity()
        target = _make_market_identity(provider="kalshi")
        m1 = MarketMapping(source=source, target=target, relationship="same_market")
        m2 = MarketMapping(source=source, target=target, relationship="same_market")
        assert m1.mapping_id == m2.mapping_id

    def test_mapping_id_differs_for_different_relationships(self) -> None:
        """Different relationships produce different mapping_ids."""
        source = _make_market_identity()
        target = _make_market_identity(provider="kalshi")
        m1 = MarketMapping(source=source, target=target, relationship="same_market")
        m2 = MarketMapping(source=source, target=target, relationship="opposite_outcome")
        assert m1.mapping_id != m2.mapping_id

    def test_mapping_frozen(self) -> None:
        """MarketMapping is immutable."""
        source = _make_market_identity()
        target = _make_market_identity(provider="kalshi")
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        with pytest.raises(Exception):  # noqa: B017
            mapping.relationship = "changed"  # type: ignore[misc]

    def test_mapping_serialization_roundtrip(self) -> None:
        """MarketMapping survives serialization roundtrip."""
        source = _make_market_identity()
        target = _make_market_identity(provider="kalshi")
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        d = mapping.model_dump()
        restored = MarketMapping.model_validate(d)
        assert restored.mapping_id == mapping.mapping_id


# ══════════════════════════════════════════════════════════════════════════════
# 9. PROPERTY-BASED TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestPropertyBased:
    """Hypothesis-based tests for model invariants."""

    @given(
        provider=st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        market_id=st.text(min_size=1, max_size=100).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_provider_and_market_id_preserved(
        self, provider: str, market_id: str
    ) -> None:
        """Provider and market_id survive construction roundtrip."""
        identity = _make_market_identity(
            provider=provider.strip(),
            provider_market_id=market_id.strip(),
        )
        assert identity.provider == provider.strip()
        assert identity.provider_market_id == market_id.strip()

    @given(
        label=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    )
    @settings(max_examples=50)
    def test_outcome_label_preserved(self, label: str) -> None:
        """Outcome label is preserved through construction."""
        outcome = OutcomeIdentity(
            outcome_id="test",
            label=label.strip(),
        )
        assert outcome.label == label.strip()

    @given(
        formula=st.sampled_from([
            "winner-takes-all",
            "proportional",
            "first-to-x",
            "market-close",
        ]),
    )
    @settings(max_examples=20)
    def test_settlement_formula_preserved(self, formula: str) -> None:
        """Settlement formula is preserved through construction."""
        terms = SettlementTerms(
            payout_type="binary",
            payout_cap=1.0,
            settlement_formula=formula,
        )
        assert terms.settlement_formula == formula

    def test_canonical_identity_deterministic_over_construction(self) -> None:
        """Canonical identity is deterministic across multiple constructions."""
        identities = [_make_market_identity() for _ in range(10)]
        unique_ids = {i.canonical_identity for i in identities}
        assert len(unique_ids) == 1, (
            f"Expected 1 unique canonical_identity, got {len(unique_ids)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 10. RESOLUTION STATE TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestResolutionState:
    """Test ResolutionState enum and MarketIdentity integration."""

    def test_resolved_state(self) -> None:
        """ResolutionState.RESOLVED is valid."""
        assert ResolutionState.RESOLVED.value == "resolved"

    def test_unresolved_state(self) -> None:
        """ResolutionState.UNRESOLVED is valid."""
        assert ResolutionState.UNRESOLVED.value == "unresolved"

    def test_void_state(self) -> None:
        """ResolutionState.VOID is valid."""
        assert ResolutionState.VOID.value == "void"

    def test_default_state_is_unresolved(self) -> None:
        """MarketIdentity defaults to UNRESOLVED when not specified."""
        identity = _make_market_identity()
        assert identity.resolution_state == ResolutionState.UNRESOLVED

    def test_explicit_resolved_state(self) -> None:
        """MarketIdentity accepts explicit RESOLVED state."""
        identity = _make_market_identity(resolution_state=ResolutionState.RESOLVED)
        assert identity.resolution_state == ResolutionState.RESOLVED

    def test_explicit_void_state(self) -> None:
        """MarketIdentity accepts explicit VOID state."""
        identity = _make_market_identity(resolution_state=ResolutionState.VOID)
        assert identity.resolution_state == ResolutionState.VOID

    def test_resolution_state_affects_canonical_identity(self) -> None:
        """Different resolution states produce different canonical_identity."""
        resolved = _make_market_identity(resolution_state=ResolutionState.RESOLVED)
        unresolved = _make_market_identity(resolution_state=ResolutionState.UNRESOLVED)
        void = _make_market_identity(resolution_state=ResolutionState.VOID)
        assert resolved.canonical_identity != unresolved.canonical_identity
        assert resolved.canonical_identity != void.canonical_identity
        assert unresolved.canonical_identity != void.canonical_identity

    def test_resolution_state_frozen(self) -> None:
        """ResolutionState in MarketIdentity is immutable."""
        identity = _make_market_identity(resolution_state=ResolutionState.RESOLVED)
        with pytest.raises(Exception):  # noqa: B017
            identity.resolution_state = ResolutionState.VOID  # type: ignore[misc]

    def test_resolution_state_serialization_roundtrip(self) -> None:
        """ResolutionState survives serialization roundtrip."""
        identity = _make_market_identity(resolution_state=ResolutionState.RESOLVED)
        d = identity.model_dump()
        restored = MarketIdentity.model_validate(d)
        assert restored.resolution_state == ResolutionState.RESOLVED

    def test_resolution_state_json_roundtrip(self) -> None:
        """ResolutionState survives JSON serialization roundtrip."""
        identity = _make_market_identity(resolution_state=ResolutionState.VOID)
        json_str = identity.model_dump_json()
        restored = MarketIdentity.model_validate_json(json_str)
        assert restored.resolution_state == ResolutionState.VOID


class TestResolutionStateMappingValidation:
    """Test MarketMapping validation with resolution states."""

    def test_same_resolution_state_compatible(self) -> None:
        """Same resolution states are compatible."""
        source = _make_market_identity(
            provider="polymarket",
            resolution_state=ResolutionState.RESOLVED,
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution_state=ResolutionState.RESOLVED,
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

    def test_both_unresolved_compatible(self) -> None:
        """Both markets unresolved are compatible."""
        source = _make_market_identity(
            provider="polymarket",
            resolution_state=ResolutionState.UNRESOLVED,
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution_state=ResolutionState.UNRESOLVED,
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

    def test_void_vs_resolved_incompatible(self) -> None:
        """VOID vs RESOLVED raises IncompatibleResolutionStateError."""
        source = _make_market_identity(
            provider="polymarket",
            resolution_state=ResolutionState.VOID,
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution_state=ResolutionState.RESOLVED,
        )
        with pytest.raises(IncompatibleResolutionStateError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_void_vs_unresolved_incompatible(self) -> None:
        """VOID vs UNRESOLVED raises IncompatibleResolutionStateError."""
        source = _make_market_identity(
            provider="polymarket",
            resolution_state=ResolutionState.VOID,
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution_state=ResolutionState.UNRESOLVED,
        )
        with pytest.raises(IncompatibleResolutionStateError):
            MarketMapping(source=source, target=target, relationship="same_market")

    def test_unresolved_vs_resolved_compatible(self) -> None:
        """UNRESOLVED vs RESOLVED is allowed (market may have resolved on one venue)."""
        source = _make_market_identity(
            provider="polymarket",
            resolution_state=ResolutionState.UNRESOLVED,
        )
        target = _make_market_identity(
            provider="kalshi",
            resolution_state=ResolutionState.RESOLVED,
        )
        mapping = MarketMapping(source=source, target=target, relationship="same_market")
        assert mapping.relationship == "same_market"

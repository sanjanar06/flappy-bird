import json
from datetime import date
from pathlib import Path

from flappy_bird.decision import analyze_offers, assign_choice_roles
from flappy_bird.models import (
    ChoiceRole,
    ConnectionLabel,
    InputSource,
    ProtectionStatus,
    TripRequest,
)
from flappy_bird.providers.ignav import IgnavObservation, normalize_ignav_response

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"


def _scenario() -> tuple[TripRequest, IgnavObservation]:
    request = TripRequest(
        origin="ORD",
        destination="COK",
        departure_date=date(2026, 11, 10),
        checked_bags=2,
        field_sources={
            "origin": InputSource.FORM,
            "destination": InputSource.FORM,
            "departure_date": InputSource.FORM,
            "checked_bags": InputSource.FORM,
        },
    )
    fixture = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
    return request, fixture


def test_ignav_fixture_normalizes_provider_evidence() -> None:
    request, fixture = _scenario()

    offers = normalize_ignav_response(fixture.response, request, fixture.retrieved_at)

    assert len(offers) == 3
    assert [offer.provider for offer in offers] == ["ignav"] * 3
    assert [offer.total_price for offer in offers] == [104800, 104800, 148700]
    assert [offer.baggage_allowance.pieces for offer in offers] == [2, 2, 2]
    assert [offer.protection_status for offer in offers] == [
        ProtectionStatus.UNKNOWN,
        ProtectionStatus.SELF_TRANSFER,
        ProtectionStatus.SELF_TRANSFER,
    ]
    assert offers[0].segments[0].departure_at.tzinfo.key == "America/Chicago"
    assert offers[0].segments[1].arrival_at.tzinfo.key == "Asia/Kolkata"
    assert offers[1].segments[1].operating_carrier == "Akasa Air"


def test_analysis_preserves_unknowns_and_connection_tradeoffs() -> None:
    request, fixture = _scenario()
    offers = normalize_ignav_response(fixture.response, request, fixture.retrieved_at)

    analyses = analyze_offers(offers, requested_checked_bags=2)

    first, second, third = [analyses[offer.provider_offer_id] for offer in offers]
    assert first.connection_labels == frozenset({ConnectionLabel.PROTECTION_UNKNOWN})
    assert first.total_duration_minutes == 1150
    assert first.layover_minutes == 115
    assert first.has_operating_carrier_change is False
    assert first.total_price_for_requested_bags == 104800

    assert second.connection_labels == frozenset({ConnectionLabel.SELF_TRANSFER})
    assert second.total_duration_minutes == 1560
    assert second.layover_minutes == 505
    assert second.has_operating_carrier_change is True

    assert third.connection_labels == frozenset({ConnectionLabel.SELF_TRANSFER})
    assert third.total_duration_minutes == 1300
    assert third.layover_minutes == 265
    assert third.has_operating_carrier_change is True


def test_choice_roles_include_ties_and_keep_three_distinct_choices() -> None:
    request, fixture = _scenario()
    offers = normalize_ignav_response(fixture.response, request, fixture.retrieved_at)
    analyses = analyze_offers(offers, requested_checked_bags=2)

    assignments = assign_choice_roles(offers, analyses)

    assert len(assignments) == 3
    assert len({assignment.offer_id for assignment in assignments}) == 3
    assert assignments[0].roles == frozenset(
        {
            ChoiceRole.LOWEST_APPARENT_PRICE,
            ChoiceRole.LOWEST_CONNECTION_RISK,
            ChoiceRole.SHORTEST_TOTAL_TRAVEL_TIME,
        }
    )
    assert assignments[1].roles == frozenset({ChoiceRole.LOWEST_APPARENT_PRICE})
    assert assignments[2].roles == frozenset()

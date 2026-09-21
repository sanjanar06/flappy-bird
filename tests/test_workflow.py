import json
from datetime import date
from pathlib import Path

from flappy_bird.models import ChoiceRole, InputSource, TripRequest
from flappy_bird.providers.fixture import FixtureIgnavProvider
from flappy_bird.providers.ignav import IgnavObservation
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"


def test_fixture_graph_reaches_every_node_and_returns_three_choices() -> None:
    fixture = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
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
    graph = build_flight_decision_graph(FixtureIgnavProvider(fixture))

    result = FlightDecisionState.model_validate(graph.invoke({"request": request}))

    assert result.provider_observation == fixture
    assert len(result.normalized_offers) == 3
    assert set(result.offer_analyses) == {
        offer.provider_offer_id for offer in result.normalized_offers
    }
    assert len(result.choice_assignments) == 3
    assert len({choice.offer_id for choice in result.choice_assignments}) == 3
    assert result.choice_assignments[0].roles == frozenset(
        {
            ChoiceRole.LOWEST_APPARENT_PRICE,
            ChoiceRole.LOWEST_CONNECTION_RISK,
            ChoiceRole.SHORTEST_TOTAL_TRAVEL_TIME,
        }
    )
    assert result.choice_assignments[1].roles == frozenset(
        {ChoiceRole.LOWEST_APPARENT_PRICE}
    )
    assert result.choice_assignments[2].roles == frozenset()
    assert [event.node for event in result.trace] == [
        "retrieve_offers",
        "normalize_offers",
        "analyze_offers",
        "construct_choice_set",
    ]
    assert [event.writes for event in result.trace] == [
        ("provider_observation",),
        ("normalized_offers",),
        ("offer_analyses",),
        ("choice_assignments",),
    ]

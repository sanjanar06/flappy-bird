import json
from datetime import date
from pathlib import Path

from flappy_bird.models import InputSource, TripRequest
from flappy_bird.providers.base import CompositeFlightProvider
from flappy_bird.providers.ignav import (
    IgnavObservation,
    normalize_ignav_response,
)
from flappy_bird.providers.serpapi import SerpApiProviderError
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"


def _request() -> TripRequest:
    return TripRequest(
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


class SuccessfulProvider:
    name = "ignav"

    def retrieve(self, request):
        observation = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
        return normalize_ignav_response(
            observation.response,
            request,
            observation.retrieved_at,
        )


class UnavailableProvider:
    name = "serpapi"

    def retrieve(self, _request):
        raise SerpApiProviderError("SerpApi request failed before receiving a response")


def test_composite_graph_keeps_partial_provider_results_and_trace() -> None:
    graph = build_flight_decision_graph(
        CompositeFlightProvider([SuccessfulProvider(), UnavailableProvider()])
    )

    state = FlightDecisionState.model_validate(graph.invoke({"request": _request()}))

    assert len(state.normalized_offers) == 3
    assert len(state.choice_assignments) == 3
    assert state.provider_failures[0].provider == "serpapi"
    assert "reconcile_offers" in [event.node for event in state.trace]


def test_composite_graph_handles_no_results_after_all_providers_succeed() -> None:
    class EmptyProvider:
        name = "empty"

        def retrieve(self, _request):
            return []

    state = FlightDecisionState.model_validate(
        build_flight_decision_graph(CompositeFlightProvider([EmptyProvider()])).invoke(
            {"request": _request()}
        )
    )

    assert state.provider_failures == []
    assert state.normalized_offers == []
    assert state.choice_assignments == []

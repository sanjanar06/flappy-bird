import json
from pathlib import Path

from fastapi.testclient import TestClient

from flappy_bird.api import create_app
from flappy_bird.providers.ignav import IgnavObservation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"


class FixtureProvider:
    def __init__(self) -> None:
        self.observation = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))

    def retrieve(self, _request):
        return self.observation


def test_api_uses_shared_graph_and_returns_traceable_choice_data() -> None:
    client = TestClient(create_app(FixtureProvider()))

    response = client.post(
        "/trips/search",
        json={
            "origin": "ORD",
            "destination": "COK",
            "departure_date": "2026-11-10",
            "checked_bags": 2,
            "field_sources": {
                "origin": "form",
                "destination": "form",
                "departure_date": "form",
                "checked_bags": "form",
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["choice_assignments"]) == 3
    assert [event["node"] for event in payload["trace"]] == [
        "retrieve_offers",
        "normalize_offers",
        "reconcile_offers",
        "analyze_offers",
        "construct_choice_set",
    ]


def test_api_rejects_invalid_trip_request() -> None:
    response = TestClient(create_app(FixtureProvider())).post(
        "/trips/search", json={"origin": "ORD"}
    )

    assert response.status_code == 422

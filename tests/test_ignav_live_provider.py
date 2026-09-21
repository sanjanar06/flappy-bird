import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pytest

from flappy_bird.models import InputSource, TripRequest
from flappy_bird.providers.ignav import (
    IgnavObservation,
    IgnavProviderError,
    LiveIgnavProvider,
)
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"
RETRIEVED_AT = datetime(2026, 9, 20, 16, 30, tzinfo=UTC)


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


def test_live_provider_sends_supported_request_and_validates_response() -> None:
    captured_fixture = IgnavObservation.model_validate(
        json.loads(FIXTURE_PATH.read_text())
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://ignav.com/api/fares/one-way"
        assert request.headers["X-Api-Key"] == "test-api-key"
        assert json.loads(request.content) == {
            "origin": "ORD",
            "destination": "COK",
            "departure_date": "2026-11-10",
            "adults": 1,
            "cabin_class": "economy",
            "max_stops": 1,
            "min_checked_bags": 2,
            "allow_self_transfer": True,
            "market": "US",
        }
        return httpx.Response(
            200,
            json=captured_fixture.response.model_dump(mode="json"),
        )

    with httpx.Client(
        base_url="https://ignav.com",
        timeout=20,
        transport=httpx.MockTransport(handler),
    ) as client:
        provider = LiveIgnavProvider(
            api_key="test-api-key",
            client=client,
            clock=lambda: RETRIEVED_AT,
        )

        observation = provider.retrieve(_request())

    assert observation.provider == "ignav"
    assert observation.retrieved_at == RETRIEVED_AT
    assert observation.response == captured_fixture.response


def test_live_provider_reads_key_from_environment_without_exposing_it() -> None:
    with httpx.Client(base_url="https://ignav.com") as client:
        provider = LiveIgnavProvider.from_environment(
            client=client,
            environ={"IGNAV_API_KEY": "environment-api-key"},
        )

    assert "environment-api-key" not in repr(provider)


def test_live_provider_reports_http_failure_without_response_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "error": {
                    "type": "auth_error",
                    "code": "invalid_api_key",
                    "message": "sensitive provider detail",
                }
            },
        )

    with httpx.Client(
        base_url="https://ignav.com",
        transport=httpx.MockTransport(handler),
    ) as client:
        provider = LiveIgnavProvider(api_key="invalid", client=client)

        with pytest.raises(IgnavProviderError) as raised:
            provider.retrieve(_request())

    assert raised.value.status_code == 401
    assert str(raised.value) == "Ignav request failed with HTTP 401"
    assert "sensitive provider detail" not in str(raised.value)


def test_mocked_live_provider_drives_the_complete_decision_graph() -> None:
    captured_fixture = IgnavObservation.model_validate(
        json.loads(FIXTURE_PATH.read_text())
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=captured_fixture.response.model_dump(mode="json"),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = LiveIgnavProvider(
            api_key="test-api-key",
            client=client,
            clock=lambda: RETRIEVED_AT,
        )
        graph = build_flight_decision_graph(provider)

        result = FlightDecisionState.model_validate(graph.invoke({"request": _request()}))

    assert len(result.choice_assignments) == 3
    assert [event.node for event in result.trace] == [
        "retrieve_offers",
        "normalize_offers",
        "analyze_offers",
        "construct_choice_set",
    ]


def test_graph_stops_with_explicit_failure_when_live_provider_rejects_request() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = LiveIgnavProvider(api_key="invalid", client=client)
        graph = build_flight_decision_graph(provider)

        result = FlightDecisionState.model_validate(graph.invoke({"request": _request()}))

    assert result.provider_observation is None
    assert len(result.provider_failures) == 1
    assert result.provider_failures[0].provider == "ignav"
    assert result.provider_failures[0].status_code == 401
    assert result.normalized_offers == []
    assert result.choice_assignments == []
    assert [event.node for event in result.trace] == ["retrieve_offers"]
    assert result.trace[0].writes == ("provider_failures",)

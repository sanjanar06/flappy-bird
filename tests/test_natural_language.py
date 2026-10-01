"""Fixture-based checks for natural-language intake and provider gating."""

import json
from io import StringIO
from pathlib import Path

import httpx
import pytest

from flappy_bird.cli import main
from flappy_bird.models import InputSource, TripRequest
from flappy_bird.natural_language import ExtractedTrip, IntakeError, build_request_from_message
from flappy_bird.providers.groq_intake import GroqTripInterpreter
from flappy_bird.providers.ignav import IgnavObservation

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"
MESSAGE = "Find a one-way economy flight from ORD to COK on November 10, 2026 with two checked bags"


def _extracted(**updates: object) -> ExtractedTrip:
    values: dict[str, object] = {
        "origin_text": "ORD",
        "destination_text": "COK",
        "departure_date_text": "November 10, 2026",
        "checked_bags_text": "two checked bags",
        "max_stops_text": None,
        "unsupported_constraints": [],
    }
    values.update(updates)
    return ExtractedTrip.model_validate(values)


class FixtureInterpreter:
    def __init__(self, extracted: ExtractedTrip):
        self.extracted = extracted

    def extract(self, _message: str) -> ExtractedTrip:
        return self.extracted


class CountingProvider:
    def __init__(self):
        self.calls = 0
        self.request: TripRequest | None = None

    def retrieve(self, request: TripRequest) -> IgnavObservation:
        self.calls += 1
        self.request = request
        return IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))


def test_natural_language_cli_reaches_existing_graph_with_provenance() -> None:
    provider = CountingProvider()
    stdout = StringIO()
    stderr = StringIO()

    code = main(
        ["ask", MESSAGE],
        provider=provider,
        interpreter=FixtureInterpreter(_extracted()),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 0
    assert provider.calls == 1
    assert provider.request is not None
    assert provider.request.origin == "ORD"
    assert provider.request.destination == "COK"
    assert provider.request.checked_bags == 2
    assert provider.request.field_sources["origin"] is InputSource.NATURAL_LANGUAGE
    assert provider.request.field_sources["max_stops"] is InputSource.DEFAULT
    assert "3 flight choices" in stdout.getvalue()
    assert stderr.getvalue() == ""


@pytest.mark.parametrize(
    ("message", "extracted", "expected"),
    [
        (MESSAGE, _extracted(origin_text="Chicago"), "did not match your message"),
        (
            "Find flights from Chicago to COK on November 10, 2026 with two checked bags",
            _extracted(origin_text="Chicago"),
            "Which departure airport",
        ),
        (MESSAGE, _extracted(departure_date_text=None), "Include the year"),
        (MESSAGE, _extracted(checked_bags_text=None), "How many checked bags"),
        (
            MESSAGE + " and no self-transfers",
            _extracted(),
            "cannot enforce",
        ),
        (
            MESSAGE + " and under $1,000",
            _extracted(),
            "cannot enforce",
        ),
    ],
)
def test_incomplete_or_unsupported_intake_never_calls_provider(
    message: str, extracted: ExtractedTrip, expected: str
) -> None:
    provider = CountingProvider()
    stderr = StringIO()

    code = main(
        ["ask", message],
        provider=provider,
        interpreter=FixtureInterpreter(extracted),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert code == 2
    assert expected in stderr.getvalue()
    assert provider.calls == 0


def test_validated_spans_do_not_infer_missing_year_or_bag_type() -> None:
    with pytest.raises(IntakeError, match="date"):
        build_request_from_message(
            "ORD to COK November 10 with two checked bags",
            _extracted(departure_date_text=None),
        )
    with pytest.raises(IntakeError, match="checked bags"):
        build_request_from_message(
            "ORD to COK November 10, 2026 with two bags",
            _extracted(checked_bags_text=None),
        )


def test_groq_interpreter_requests_strict_spans_without_provider_data() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][1] == {"role": "user", "content": MESSAGE}
        assert body["response_format"]["json_schema"]["strict"] is True
        assert "offer" not in body["messages"][1]["content"].casefold()
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _extracted().model_dump_json()}}]},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        interpreter = GroqTripInterpreter(api_key="test-key", client=client)
        result = interpreter.extract(MESSAGE)

    assert result == _extracted()

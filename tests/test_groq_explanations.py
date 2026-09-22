import json

import httpx
import pytest

from flappy_bird.explanations import (
    ChoiceEvidence,
    ChoiceExplanation,
    ExplanationError,
    SegmentEvidence,
)
from flappy_bird.models import ChoiceRole, ConnectionLabel
from flappy_bird.providers.groq import GroqChoiceExplainer


def _evidence() -> list[ChoiceEvidence]:
    return [
        ChoiceEvidence(
            offer_id="offer-1",
            provider="ignav",
            roles=[ChoiceRole.LOWEST_CONNECTION_RISK],
            apparent_price_display="$1,048.00 USD",
            total_duration_display="19h 10m",
            stops=1,
            layover_display="1h 55m",
            connection_airports=["AUH"],
            connection_labels=[ConnectionLabel.PROTECTION_UNKNOWN],
            has_airport_change=False,
            has_operating_carrier_change=False,
            segments=[
                SegmentEvidence(
                    flight_number="EY10",
                    marketing_carrier="EY",
                    operating_carrier="Etihad Airways",
                    origin="ORD",
                    destination="AUH",
                )
            ],
        )
    ]


def test_groq_explainer_sends_only_prepared_evidence_and_validates_output() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.groq.com/openai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-groq-key"
        body = json.loads(request.content)
        assert body["model"] == "openai/gpt-oss-20b"
        assert body["response_format"]["json_schema"]["strict"] is True
        supplied_evidence = json.loads(body["messages"][1]["content"])
        assert supplied_evidence == [item.model_dump(mode="json") for item in _evidence()]
        assert "retrieved_at" not in body["messages"][1]["content"]
        model_output = {
            "explanations": [
                {
                    "offer_id": "offer-1",
                    "summary": "A one-stop itinerary with the shortest connection-risk profile.",
                    "tradeoff": "Connection protection is unknown despite the lower risk label.",
                }
            ]
        }
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(model_output)}}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        explainer = GroqChoiceExplainer(api_key="test-groq-key", client=client)

        explanations = explainer.explain(_evidence())

    assert explanations == [
        ChoiceExplanation(
            offer_id="offer-1",
            summary="A one-stop itinerary with the shortest connection-risk profile.",
            tradeoff="Connection protection is unknown despite the lower risk label.",
        )
    ]


def test_groq_explainer_rejects_changed_offer_identity() -> None:
    model_output = {
        "explanations": [
            {
                "offer_id": "invented-offer",
                "summary": "Summary",
                "tradeoff": "Tradeoff",
            }
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(model_output)}}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        explainer = GroqChoiceExplainer(api_key="test-groq-key", client=client)

        with pytest.raises(ExplanationError, match="offer IDs"):
            explainer.explain(_evidence())


def test_groq_explainer_rejects_omitted_self_transfer_warning() -> None:
    evidence = [
        _evidence()[0].model_copy(
            update={"connection_labels": [ConnectionLabel.SELF_TRANSFER]}
        )
    ]
    model_output = {
        "explanations": [
            {
                "offer_id": "offer-1",
                "summary": "A one-stop itinerary.",
                "tradeoff": "The layover is 1h 55m.",
            }
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(model_output)}}
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        explainer = GroqChoiceExplainer(api_key="test-groq-key", client=client)

        with pytest.raises(ExplanationError, match="self-transfer"):
            explainer.explain(evidence)

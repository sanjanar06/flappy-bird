"""Groq adapter for grounded, structured choice explanations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ValidationError

from flappy_bird.explanations import (
    ChoiceEvidence,
    ChoiceExplanation,
    ChoiceExplanationBatch,
    ExplanationError,
)
from flappy_bird.models import ConnectionLabel

GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
SYSTEM_PROMPT = """You explain flight choices using only the supplied JSON evidence.
Do not invent facts, prices, protection, baggage, reliability, or recommendations.
Do not select a best flight. Treat roles as descriptive and leave the choice to the traveler.
Copy supplied display values exactly; do not calculate or reformat prices or durations.
Refer to airports only by the supplied IATA codes; never expand them to airport or city names.
Do not repeat internal offer IDs in summary or tradeoff text.
If self_transfer is present, explicitly say "self-transfer" in the tradeoff.
If protection_unknown is present, explicitly say "protection is unknown" in the tradeoff.
Keep each summary and tradeoff concise. Return one explanation for every offer_id."""


class _GroqMessage(BaseModel):
    content: str


class _GroqChoice(BaseModel):
    message: _GroqMessage


class _GroqResponse(BaseModel):
    choices: list[_GroqChoice]


class GroqChoiceExplainer:
    """Call Groq with strict structured output and validate offer identity."""

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Groq API key cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("Groq timeout must be positive")
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        *,
        client: httpx.Client,
        environ: Mapping[str, str] | None = None,
    ) -> GroqChoiceExplainer:
        environment = os.environ if environ is None else environ
        api_key = environment.get("GROQ_API_KEY")
        if api_key is None:
            raise ValueError("GROQ_API_KEY is not configured")
        return cls(api_key=api_key, client=client)

    def explain(self, evidence: list[ChoiceEvidence]) -> list[ChoiceExplanation]:
        if not evidence:
            return []
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        [item.model_dump(mode="json") for item in evidence]
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "flight_choice_explanations",
                    "strict": True,
                    "schema": ChoiceExplanationBatch.model_json_schema(),
                },
            },
        }
        try:
            response = self._client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            groq_response = _GroqResponse.model_validate(response.json())
            if not groq_response.choices:
                raise ExplanationError("Groq returned no explanation choice")
            batch = ChoiceExplanationBatch.model_validate_json(
                groq_response.choices[0].message.content
            )
        except httpx.HTTPStatusError as error:
            raise ExplanationError(
                f"Groq explanation failed with HTTP {error.response.status_code}"
            ) from error
        except httpx.RequestError as error:
            raise ExplanationError(
                "Groq explanation failed before receiving a response"
            ) from error
        except (ValidationError, ValueError, TypeError) as error:
            raise ExplanationError("Groq returned an invalid explanation response") from error

        expected_ids = {item.offer_id for item in evidence}
        returned_ids = [item.offer_id for item in batch.explanations]
        if len(returned_ids) != len(set(returned_ids)) or set(returned_ids) != expected_ids:
            raise ExplanationError("Groq explanation offer IDs do not match the choices")
        explanations_by_id = {
            explanation.offer_id: explanation for explanation in batch.explanations
        }
        ordered_explanations = [
            explanations_by_id[item.offer_id] for item in evidence
        ]
        _validate_grounding_requirements(evidence, ordered_explanations)
        return ordered_explanations


def _validate_grounding_requirements(
    evidence: list[ChoiceEvidence],
    explanations: list[ChoiceExplanation],
) -> None:
    for item, explanation in zip(evidence, explanations, strict=True):
        prose = f"{explanation.summary} {explanation.tradeoff}"
        normalized_tradeoff = explanation.tradeoff.casefold()
        if item.offer_id in prose:
            raise ExplanationError("Groq explanation exposed an internal offer ID")
        if (
            ConnectionLabel.SELF_TRANSFER in item.connection_labels
            and "self-transfer" not in normalized_tradeoff
        ):
            raise ExplanationError(
                "Groq explanation omitted a required self-transfer warning"
            )
        if ConnectionLabel.PROTECTION_UNKNOWN in item.connection_labels and not (
            "protection" in normalized_tradeoff
            and "unknown" in normalized_tradeoff
        ):
            raise ExplanationError(
                "Groq explanation omitted an unknown-protection warning"
            )

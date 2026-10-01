"""Groq extracts exact spans from a natural-language flight request."""

from __future__ import annotations

import os
from collections.abc import Mapping

import httpx
from pydantic import BaseModel, ValidationError

from flappy_bird.natural_language import ExtractedTrip, IntakeError
from flappy_bird.providers.groq import GROQ_CHAT_COMPLETIONS_URL, GROQ_MODEL

SYSTEM_PROMPT = """Extract travel-request facts as exact substrings from the user's message.
Return null for anything absent. Never infer an airport code from a city name.
Never infer a year from context.
origin_text and destination_text are the location names or codes actually written by the user.
departure_date_text must include the year exactly as written.
checked_bags_text must explicitly say checked bags; 'bags' alone is ambiguous.
max_stops_text must be an exact nonstop or maximum-stops phrase when present.
This version supports one adult, one-way economy, and zero or one stop; do not flag
these as unsupported. Copy any other stated hard constraint or preference into
unsupported_constraints, including restrictions on self-transfers, airport changes,
layovers, price, airline, other cabins, multiple passengers, or round trips.
Do not treat permission for self-transfers or airport changes as a constraint.
Do not add defaults. Return only the specified JSON schema."""


class _GroqMessage(BaseModel):
    content: str


class _GroqChoice(BaseModel):
    message: _GroqMessage


class _GroqResponse(BaseModel):
    choices: list[_GroqChoice]


class GroqTripInterpreter:
    """Make one structured extraction call; validation remains local."""

    def __init__(self, *, api_key: str, client: httpx.Client, timeout_seconds: float = 20.0):
        if not api_key.strip():
            raise ValueError("Groq API key cannot be empty")
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls, *, client: httpx.Client, environ: Mapping[str, str] | None = None
    ) -> GroqTripInterpreter:
        environment = os.environ if environ is None else environ
        api_key = environment.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        return cls(api_key=api_key, client=client)

    def extract(self, message: str) -> ExtractedTrip:
        try:
            response = self._client.post(
                GROQ_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": message},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "flight_trip_extraction",
                            "strict": True,
                            "schema": ExtractedTrip.model_json_schema(),
                        },
                    },
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            result = _GroqResponse.model_validate(response.json())
            if not result.choices:
                raise IntakeError("Groq returned no trip interpretation.")
            return ExtractedTrip.model_validate_json(result.choices[0].message.content)
        except httpx.HTTPStatusError as error:
            raise IntakeError(
                f"Groq trip interpretation failed with HTTP {error.response.status_code}."
            ) from error
        except httpx.RequestError as error:
            raise IntakeError(
                "Groq trip interpretation failed before receiving a response."
            ) from error
        except (ValidationError, ValueError, TypeError) as error:
            raise IntakeError("Groq returned an invalid trip interpretation.") from error

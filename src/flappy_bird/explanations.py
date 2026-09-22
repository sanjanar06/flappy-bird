"""Prepared evidence and validated output for optional LLM explanations."""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from flappy_bird.models import (
    ChoiceAssignment,
    ChoiceRole,
    ConnectionLabel,
    FlightOffer,
    OfferAnalysis,
)


class SegmentEvidence(BaseModel):
    """Normalized segment facts that an explainer may mention."""

    model_config = ConfigDict(extra="forbid")

    flight_number: str
    marketing_carrier: str
    operating_carrier: str | None
    origin: str
    destination: str


class ChoiceEvidence(BaseModel):
    """Complete, provider-neutral evidence available to the LLM for one choice."""

    model_config = ConfigDict(extra="forbid")

    offer_id: str
    provider: str
    roles: list[ChoiceRole]
    apparent_price_display: str
    total_duration_display: str
    stops: int
    layover_display: str | None
    connection_airports: list[str]
    connection_labels: list[ConnectionLabel]
    has_airport_change: bool
    has_operating_carrier_change: bool
    segments: list[SegmentEvidence]


class ChoiceExplanation(BaseModel):
    """Grounded prose for one deterministic choice."""

    model_config = ConfigDict(extra="forbid")

    offer_id: str
    summary: str = Field(min_length=1, max_length=300)
    tradeoff: str = Field(min_length=1, max_length=300)


class ChoiceExplanationBatch(BaseModel):
    """Structured model response for the displayed choice set."""

    model_config = ConfigDict(extra="forbid")

    explanations: list[ChoiceExplanation] = Field(min_length=1)


class ExplanationError(RuntimeError):
    """Sanitized failure raised by an explanation provider."""


class ChoiceExplainer(Protocol):
    def explain(self, evidence: list[ChoiceEvidence]) -> list[ChoiceExplanation]:
        """Explain prepared facts without changing deterministic decisions."""


def prepare_choice_evidence(
    offers: list[FlightOffer],
    analyses: dict[str, OfferAnalysis],
    assignments: list[ChoiceAssignment],
) -> list[ChoiceEvidence]:
    """Build the only data boundary exposed to an explanation model."""

    offers_by_id = {offer.provider_offer_id: offer for offer in offers}
    evidence: list[ChoiceEvidence] = []
    for assignment in assignments:
        offer = offers_by_id[assignment.offer_id]
        analysis = analyses[assignment.offer_id]
        evidence.append(
            ChoiceEvidence(
                offer_id=assignment.offer_id,
                provider=offer.provider,
                roles=sorted(assignment.roles, key=lambda role: role.value),
                apparent_price_display=_format_price(
                    analysis.total_price_for_requested_bags,
                    offer.currency,
                ),
                total_duration_display=_format_duration(
                    analysis.total_duration_minutes
                ),
                stops=analysis.stops,
                layover_display=(
                    _format_duration(analysis.layover_minutes)
                    if analysis.layover_minutes is not None
                    else None
                ),
                connection_airports=_connection_airports(offer),
                connection_labels=sorted(
                    analysis.connection_labels,
                    key=lambda label: label.value,
                ),
                has_airport_change=analysis.has_airport_change,
                has_operating_carrier_change=analysis.has_operating_carrier_change,
                segments=[
                    SegmentEvidence(
                        flight_number=segment.flight_number,
                        marketing_carrier=segment.marketing_carrier,
                        operating_carrier=segment.operating_carrier,
                        origin=segment.origin,
                        destination=segment.destination,
                    )
                    for segment in offer.segments
                ],
            )
        )
    return evidence


def _format_price(amount: int | None, currency: str) -> str:
    if amount is None:
        return "UNKNOWN"
    major_units = Decimal(amount) / Decimal(100)
    if currency == "USD":
        return f"${major_units:,.2f} USD"
    return f"{major_units:,.2f} {currency}"


def _format_duration(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m"


def _connection_airports(offer: FlightOffer) -> list[str]:
    if offer.stops == 0:
        return []
    airports = [offer.segments[0].destination]
    second_departure = offer.segments[1].origin
    if second_departure != airports[0]:
        airports.append(second_departure)
    return airports

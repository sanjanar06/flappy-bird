"""Typed contracts for the first one-way flight-search slice.

These models deliberately describe facts, not ranking policy. Later workflow nodes can
use them regardless of whether a request arrived from a form, natural language, or MCP.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class InputSource(StrEnum):
    """Where a request field came from."""

    FORM = "form"
    NATURAL_LANGUAGE = "natural_language"
    DEFAULT = "default"
    PRIOR_INTERACTION = "prior_interaction"


class CabinClass(StrEnum):
    ECONOMY = "economy"


class TripType(StrEnum):
    ONE_WAY = "one_way"


class ProtectionStatus(StrEnum):
    PROTECTED = "protected"
    SELF_TRANSFER = "self_transfer"
    UNKNOWN = "unknown"


class TripRequest(BaseModel):
    """The provider-neutral request for the initial P0 search.

    The first vertical slice intentionally supports only one-way economy searches for
    one adult. Later slices can widen these constraints without changing the request's
    meaning or losing source provenance.
    """

    origin: str = Field(description="IATA origin airport code")
    destination: str = Field(description="IATA destination airport code")
    departure_date: date
    trip_type: TripType = TripType.ONE_WAY
    passengers: int = Field(default=1, ge=1, le=1)
    cabin: CabinClass = CabinClass.ECONOMY
    max_stops: int = Field(default=1, ge=0, le=1)
    preference_text: str | None = Field(
        default=None,
        description="Optional user wording such as 'avoid very short connections'.",
    )
    field_sources: dict[str, InputSource] = Field(default_factory=dict)

    @field_validator("origin", "destination")
    @classmethod
    def validate_iata_code(cls, airport: str) -> str:
        code = airport.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("must be a three-letter IATA airport code")
        return code

    @model_validator(mode="after")
    def validate_route_and_provenance(self) -> TripRequest:
        if self.origin == self.destination:
            raise ValueError("origin and destination must be different")

        required_fields = {"origin", "destination", "departure_date"}
        missing_sources = required_fields - self.field_sources.keys()
        if missing_sources:
            missing = ", ".join(sorted(missing_sources))
            raise ValueError(f"field_sources must identify the source of: {missing}")
        return self


class FlightSegment(BaseModel):
    """One marketed flight segment after provider normalization."""

    origin: str
    destination: str
    departure_at: datetime
    arrival_at: datetime
    marketing_carrier: str
    flight_number: str

    @field_validator("origin", "destination")
    @classmethod
    def normalize_airport(cls, airport: str) -> str:
        code = airport.strip().upper()
        if len(code) != 3 or not code.isalpha():
            raise ValueError("must be a three-letter IATA airport code")
        return code

    @model_validator(mode="after")
    def validate_time_order(self) -> FlightSegment:
        if self.arrival_at <= self.departure_at:
            raise ValueError("arrival_at must be after departure_at")
        return self


class FlightOffer(BaseModel):
    """A normalized one-way offer from one provider observation.

    `protection_status` intentionally defaults to UNKNOWN. A later risk rule may
    label an offer as protected or self-transfer only when provider evidence permits.
    """

    provider: str
    provider_offer_id: str
    total_price: int = Field(ge=0, description="Minor currency units, e.g. cents")
    currency: str = Field(min_length=3, max_length=3)
    segments: list[FlightSegment] = Field(min_length=1, max_length=2)
    protection_status: ProtectionStatus = ProtectionStatus.UNKNOWN
    retrieved_at: datetime

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, currency: str) -> str:
        return currency.strip().upper()

    @property
    def stops(self) -> int:
        """Number of connections, derived instead of trusted from a provider field."""

        return len(self.segments) - 1

"""Typed contracts for the first one-way flight-search slice.

These models deliberately describe facts, not ranking policy. Later workflow nodes can
use them regardless of whether a request arrived from a form, natural language, or MCP.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
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


class ConnectionLabel(StrEnum):
    """Evidence-backed connection characteristics shown to the traveler."""

    PROTECTED_CONNECTION = "protected_connection"
    SELF_TRANSFER = "self_transfer"
    PROTECTION_UNKNOWN = "protection_unknown"
    AIRPORT_CHANGE = "airport_change"


class ChoiceRole(StrEnum):
    """Advisory roles that an offer may win in the displayed choice set."""

    LOWEST_APPARENT_PRICE = "lowest_apparent_price"
    LOWEST_CONNECTION_RISK = "lowest_connection_risk"
    SHORTEST_TOTAL_TRAVEL_TIME = "shortest_total_travel_time"


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
    checked_bags: int | None = Field(
        default=None,
        ge=0,
        description="Traveler-supplied checked-bag count; unknown when omitted.",
    )
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

        if self.checked_bags is not None and "checked_bags" not in self.field_sources:
            raise ValueError("field_sources must identify the source of: checked_bags")
        return self


class CheckedBaggageAllowance(BaseModel):
    """Included checked-baggage facts normalized from provider evidence."""

    pieces: int | None = Field(default=None, ge=0)
    weight_per_piece_kg: Decimal | None = Field(default=None, gt=0)
    total_weight_kg: Decimal | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def require_provider_fact(self) -> CheckedBaggageAllowance:
        if all(
            value is None
            for value in (self.pieces, self.weight_per_piece_kg, self.total_weight_kg)
        ):
            raise ValueError("at least one baggage-allowance fact is required")
        if self.pieces == 0 and self.weight_per_piece_kg is not None:
            raise ValueError("weight_per_piece_kg requires at least one included piece")
        return self


class CheckedBaggagePricing(BaseModel):
    """Exact provider pricing for a requested number of checked bags."""

    requested_pieces: int = Field(ge=1)
    fee_total: int = Field(ge=0, description="Minor currency units")
    currency: str = Field(min_length=3, max_length=3)
    fees_by_piece: list[int] | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, currency: str) -> str:
        return currency.strip().upper()

    @model_validator(mode="after")
    def validate_fee_breakdown(self) -> CheckedBaggagePricing:
        if self.fees_by_piece is None:
            return self
        if len(self.fees_by_piece) != self.requested_pieces:
            raise ValueError("fees_by_piece must contain one entry per requested piece")
        if any(fee < 0 for fee in self.fees_by_piece):
            raise ValueError("fees_by_piece cannot contain negative amounts")
        if sum(self.fees_by_piece) != self.fee_total:
            raise ValueError("fees_by_piece must sum to fee_total")
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
        if self.departure_at.utcoffset() is None or self.arrival_at.utcoffset() is None:
            raise ValueError("segment timestamps must include UTC offsets")
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
    baggage_allowance: CheckedBaggageAllowance | None = None
    checked_baggage_pricing: CheckedBaggagePricing | None = None
    retrieved_at: datetime

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, currency: str) -> str:
        return currency.strip().upper()

    @model_validator(mode="after")
    def validate_itinerary_and_baggage_currency(self) -> FlightOffer:
        for previous, following in zip(self.segments, self.segments[1:], strict=False):
            if following.departure_at <= previous.arrival_at:
                raise ValueError("each connecting segment must depart after the prior arrival")

        if (
            self.checked_baggage_pricing is not None
            and self.checked_baggage_pricing.currency != self.currency
        ):
            raise ValueError("baggage pricing must use the offer currency in P0")
        return self

    @property
    def stops(self) -> int:
        """Number of connections, derived instead of trusted from a provider field."""

        return len(self.segments) - 1

    @property
    def total_duration(self) -> timedelta:
        """Elapsed journey time, including the connection."""

        return self.segments[-1].arrival_at - self.segments[0].departure_at

    @property
    def layover_duration(self) -> timedelta | None:
        """Elapsed connection time for the P0 zero-or-one-stop itinerary."""

        if self.stops == 0:
            return None
        return self.segments[1].departure_at - self.segments[0].arrival_at

    @property
    def has_airport_change(self) -> bool:
        """Whether consecutive segments use different connection airports."""

        return self.stops == 1 and self.segments[0].destination != self.segments[1].origin

    @property
    def connection_labels(self) -> frozenset[ConnectionLabel]:
        """Deterministic factual labels; these are advisory, not a safety verdict."""

        if self.stops == 0:
            return frozenset()

        protection_label = {
            ProtectionStatus.PROTECTED: ConnectionLabel.PROTECTED_CONNECTION,
            ProtectionStatus.SELF_TRANSFER: ConnectionLabel.SELF_TRANSFER,
            ProtectionStatus.UNKNOWN: ConnectionLabel.PROTECTION_UNKNOWN,
        }[self.protection_status]
        labels = {protection_label}
        if self.has_airport_change:
            labels.add(ConnectionLabel.AIRPORT_CHANGE)
        return frozenset(labels)

    def total_with_checked_bags(self, requested_pieces: int) -> int | None:
        """Return an exact baggage-adjusted total, or UNKNOWN as ``None``."""

        if requested_pieces < 0:
            raise ValueError("requested_pieces cannot be negative")
        if requested_pieces == 0:
            return self.total_price
        if (
            self.baggage_allowance is not None
            and self.baggage_allowance.pieces is not None
            and self.baggage_allowance.pieces >= requested_pieces
        ):
            return self.total_price
        if (
            self.checked_baggage_pricing is not None
            and self.checked_baggage_pricing.requested_pieces == requested_pieces
        ):
            return self.total_price + self.checked_baggage_pricing.fee_total
        return None

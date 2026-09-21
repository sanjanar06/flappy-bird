"""Ignav response contracts and deterministic normalization."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, Field

from flappy_bird.models import (
    CheckedBaggageAllowance,
    FlightOffer,
    FlightSegment,
    ProtectionStatus,
    TripRequest,
)

IGNAV_ONE_WAY_URL = "https://ignav.com/api/fares/one-way"


class IgnavPrice(BaseModel):
    amount: Decimal = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    status: str


class IgnavSegment(BaseModel):
    marketing_carrier_code: str
    flight_number: str
    operating_carrier_name: str | None = None
    departure_airport: str
    departure_time_local: datetime
    departure_timezone: str
    departure_time_utc: datetime
    arrival_airport: str
    arrival_time_local: datetime
    arrival_timezone: str
    arrival_time_utc: datetime
    duration_minutes: int = Field(gt=0)
    aircraft: str | None = None


class IgnavDirection(BaseModel):
    carrier: str
    duration_minutes: int = Field(gt=0)
    segments: list[IgnavSegment] = Field(min_length=1, max_length=2)


class IgnavBags(BaseModel):
    carry_on: int | None = Field(default=None, ge=0)
    checked: int | None = Field(default=None, ge=0)


class IgnavItinerary(BaseModel):
    price: IgnavPrice
    outbound: IgnavDirection
    cabin_class: str
    bags: IgnavBags | None = None
    requires_self_transfer: bool
    ignav_id: str


class IgnavOneWayResponse(BaseModel):
    origin: str
    destination: str
    departure_date: date
    itineraries: list[IgnavItinerary]


class IgnavObservation(BaseModel):
    """One validated provider response plus retrieval provenance."""

    provider: Literal["ignav"]
    retrieved_at: datetime
    response: IgnavOneWayResponse


class IgnavOneWayRequest(BaseModel):
    """Supported subset of Ignav's one-way fare request."""

    origin: str
    destination: str
    departure_date: date
    adults: int = Field(ge=1, le=9)
    cabin_class: str
    max_stops: int = Field(ge=0, le=2)
    min_checked_bags: int | None = Field(default=None, ge=0)
    allow_self_transfer: Literal[True] = True
    market: Literal["US"] = "US"


class IgnavProviderError(RuntimeError):
    """Safe provider failure with machine-readable status when available."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LiveIgnavProvider:
    """Synchronous HTTP adapter for Ignav's one-way fare endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client,
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Ignav API key cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("Ignav timeout must be positive")
        self._api_key = api_key
        self._client = client
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        *,
        client: httpx.Client,
        environ: Mapping[str, str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> LiveIgnavProvider:
        environment = os.environ if environ is None else environ
        api_key = environment.get("IGNAV_API_KEY")
        if api_key is None:
            raise ValueError("IGNAV_API_KEY is not configured")
        return cls(api_key=api_key, client=client, clock=clock)

    def retrieve(self, request: TripRequest) -> IgnavObservation:
        provider_request = IgnavOneWayRequest(
            origin=request.origin,
            destination=request.destination,
            departure_date=request.departure_date,
            adults=request.passengers,
            cabin_class=request.cabin.value,
            max_stops=request.max_stops,
            min_checked_bags=request.checked_bags,
            allow_self_transfer=True,
            market="US",
        )
        try:
            response = self._client.post(
                IGNAV_ONE_WAY_URL,
                headers={"X-Api-Key": self._api_key},
                json=provider_request.model_dump(mode="json", exclude_none=True),
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            parsed = IgnavOneWayResponse.model_validate(response.json())
        except httpx.HTTPStatusError as error:
            raise IgnavProviderError(
                f"Ignav request failed with HTTP {error.response.status_code}",
                status_code=error.response.status_code,
            ) from error
        except httpx.RequestError as error:
            raise IgnavProviderError("Ignav request failed before receiving a response") from error
        except (ValueError, TypeError) as error:
            raise IgnavProviderError("Ignav returned an invalid response") from error

        retrieved_at = self._clock()
        if retrieved_at.utcoffset() is None:
            raise ValueError("Ignav retrieval clock must return a timezone-aware timestamp")
        return IgnavObservation(
            provider="ignav",
            retrieved_at=retrieved_at,
            response=parsed,
        )


def normalize_ignav_response(
    response: IgnavOneWayResponse,
    request: TripRequest,
    retrieved_at: datetime,
) -> list[FlightOffer]:
    """Normalize valid Ignav itineraries without guessing missing facts."""

    if (response.origin, response.destination, response.departure_date) != (
        request.origin,
        request.destination,
        request.departure_date,
    ):
        raise ValueError("Ignav response does not match the requested route and date")
    if retrieved_at.utcoffset() is None:
        raise ValueError("retrieved_at must include a UTC offset")

    offers: list[FlightOffer] = []
    for itinerary in response.itineraries:
        if itinerary.cabin_class != request.cabin.value:
            continue
        if len(itinerary.outbound.segments) - 1 > request.max_stops:
            continue

        segments = [_normalize_segment(segment) for segment in itinerary.outbound.segments]
        if segments[0].origin != request.origin or segments[-1].destination != request.destination:
            continue
        if segments[0].departure_at.date() != request.departure_date:
            continue

        checked_bags = itinerary.bags.checked if itinerary.bags is not None else None
        baggage_allowance = (
            CheckedBaggageAllowance(pieces=checked_bags) if checked_bags is not None else None
        )
        protection_status = (
            ProtectionStatus.SELF_TRANSFER
            if itinerary.requires_self_transfer
            else ProtectionStatus.UNKNOWN
        )
        offer = FlightOffer(
            provider="ignav",
            provider_offer_id=itinerary.ignav_id,
            total_price=_usd_minor_units(itinerary.price),
            currency=itinerary.price.currency,
            segments=segments,
            protection_status=protection_status,
            baggage_allowance=baggage_allowance,
            retrieved_at=retrieved_at,
        )
        if _whole_minutes(offer.total_duration) != itinerary.outbound.duration_minutes:
            raise ValueError(f"Ignav duration mismatch for offer {itinerary.ignav_id}")
        offers.append(offer)

    return offers


def _normalize_segment(segment: IgnavSegment) -> FlightSegment:
    departure_at = _localize_and_verify(
        segment.departure_time_local,
        segment.departure_timezone,
        segment.departure_time_utc,
    )
    arrival_at = _localize_and_verify(
        segment.arrival_time_local,
        segment.arrival_timezone,
        segment.arrival_time_utc,
    )
    normalized = FlightSegment(
        origin=segment.departure_airport,
        destination=segment.arrival_airport,
        departure_at=departure_at,
        arrival_at=arrival_at,
        marketing_carrier=segment.marketing_carrier_code,
        operating_carrier=segment.operating_carrier_name,
        flight_number=f"{segment.marketing_carrier_code}{segment.flight_number}",
    )
    if _whole_minutes(normalized.arrival_at - normalized.departure_at) != segment.duration_minutes:
        raise ValueError(
            f"Ignav segment duration mismatch for {normalized.flight_number}"
        )
    return normalized


def _localize_and_verify(local: datetime, timezone_name: str, utc: datetime) -> datetime:
    if local.utcoffset() is not None:
        raise ValueError("Ignav local timestamps must not contain an offset")
    if utc.utcoffset() is None:
        raise ValueError("Ignav UTC timestamps must contain an offset")
    try:
        localized = local.replace(tzinfo=ZoneInfo(timezone_name))
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"unknown Ignav timezone: {timezone_name}") from error
    if localized.timestamp() != utc.timestamp():
        raise ValueError(f"Ignav local and UTC timestamps disagree for {timezone_name}")
    return localized


def _usd_minor_units(price: IgnavPrice) -> int:
    if price.currency.upper() != "USD":
        raise ValueError("P0 Ignav normalization currently supports USD only")
    minor_units = price.amount * 100
    if minor_units != minor_units.to_integral_value():
        raise ValueError("Ignav USD price has more than two decimal places")
    return int(minor_units)


def _whole_minutes(duration: timedelta) -> int:
    total_seconds = duration.total_seconds()
    if total_seconds % 60:
        raise ValueError("flight durations must resolve to whole minutes")
    return int(total_seconds // 60)

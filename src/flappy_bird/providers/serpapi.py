"""SerpApi Google Flights adapter with explicit timezone resolution."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from flappy_bird.models import FlightOffer, FlightSegment, ProtectionStatus, TripRequest

SERPAPI_URL = "https://serpapi.com/search.json"


class SerpApiProviderError(RuntimeError):
    """Safe error raised for request, response, or normalization failures."""


class _Airport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    time: datetime


class _RawSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    departure_airport: _Airport
    arrival_airport: _Airport
    airline: str
    flight_number: str
    plane_and_crew_by: str | None = None


class _RawItinerary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    flights: list[_RawSegment] = Field(min_length=1, max_length=2)
    price: int = Field(ge=0)


class _RawResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    best_flights: list[_RawItinerary] = Field(default_factory=list)
    other_flights: list[_RawItinerary] = Field(default_factory=list)


class SerpApiGoogleFlightsProvider:
    """Retrieve and normalize Google Flights results through SerpApi.

    SerpApi times are local wall-clock values without UTC offsets. The caller must
    supply airport IANA timezone mappings; missing mappings fail closed.
    """

    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client,
        airport_timezones: Mapping[str, str],
        clock: Callable[[], datetime] | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("SerpApi key cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("SerpApi timeout must be positive")
        self._api_key = api_key
        self._client = client
        self._airport_timezones = {code.upper(): zone for code, zone in airport_timezones.items()}
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(
        cls,
        *,
        client: httpx.Client,
        airport_timezones: Mapping[str, str],
        environ: Mapping[str, str] | None = None,
    ) -> SerpApiGoogleFlightsProvider:
        environment = os.environ if environ is None else environ
        api_key = environment.get("SERPAPI_API_KEY")
        if api_key is None:
            raise ValueError("SERPAPI_API_KEY is not configured")
        return cls(api_key=api_key, client=client, airport_timezones=airport_timezones)

    def retrieve(self, request: TripRequest) -> list[FlightOffer]:
        try:
            response = self._client.get(
                SERPAPI_URL,
                params={
                    "engine": "google_flights",
                    "api_key": self._api_key,
                    "departure_id": request.origin,
                    "arrival_id": request.destination,
                    "outbound_date": request.departure_date.isoformat(),
                    "type": 2,
                    "travel_class": 1,
                    "adults": request.passengers,
                    "stops": 1 if request.max_stops == 0 else 2,
                    "currency": "USD",
                    "gl": "us",
                    "hl": "en",
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            raw = _RawResponse.model_validate(response.json())
            retrieved_at = self._clock()
            if retrieved_at.utcoffset() is None:
                raise ValueError("retrieval timestamp must be timezone-aware")
            itineraries = [*raw.best_flights, *raw.other_flights]
            offers = [self._normalize(item, request, retrieved_at) for item in itineraries]
            return [offer for offer in offers if offer is not None]
        except httpx.HTTPStatusError as error:
            raise SerpApiProviderError(
                f"SerpApi request failed with HTTP {error.response.status_code}"
            ) from error
        except httpx.RequestError as error:
            raise SerpApiProviderError(
                "SerpApi request failed before receiving a response"
            ) from error
        except (ValidationError, ValueError, TypeError, KeyError) as error:
            raise SerpApiProviderError(
                "SerpApi returned an invalid or incomplete response"
            ) from error

    def _normalize(
        self,
        itinerary: _RawItinerary,
        request: TripRequest,
        retrieved_at: datetime,
    ) -> FlightOffer | None:
        if len(itinerary.flights) - 1 > request.max_stops:
            return None
        segments = [self._normalize_segment(item) for item in itinerary.flights]
        if segments[0].origin != request.origin or segments[-1].destination != request.destination:
            return None
        if segments[0].departure_at.date() != request.departure_date:
            return None
        signature = "|".join(
            f"{item.marketing_carrier}:{item.flight_number}:{item.origin}:{item.destination}:"
            f"{item.departure_at.isoformat()}"
            for item in segments
        )
        offer_id = hashlib.sha256(signature.encode()).hexdigest()[:20]
        return FlightOffer(
            provider="serpapi",
            provider_offer_id=f"serpapi-{offer_id}",
            total_price=int(Decimal(itinerary.price) * 100),
            currency="USD",
            segments=segments,
            protection_status=ProtectionStatus.UNKNOWN,
            retrieved_at=retrieved_at,
        )

    def _normalize_segment(self, raw: _RawSegment) -> FlightSegment:
        origin = raw.departure_airport.id.upper()
        destination = raw.arrival_airport.id.upper()
        departure = self._localize(raw.departure_airport.time, origin)
        arrival = self._localize(raw.arrival_airport.time, destination)
        carrier_code = raw.flight_number.strip().split()[0]
        flight_digits = raw.flight_number.strip().split(maxsplit=1)
        if len(flight_digits) != 2:
            raise ValueError("flight number does not include a carrier code and number")
        return FlightSegment(
            origin=origin,
            destination=destination,
            departure_at=departure,
            arrival_at=arrival,
            marketing_carrier=carrier_code,
            operating_carrier=raw.plane_and_crew_by,
            flight_number=flight_digits[1],
        )

    def _localize(self, value: datetime, airport: str) -> datetime:
        zone_name = self._airport_timezones.get(airport)
        if zone_name is None:
            raise ValueError(f"timezone is unknown for airport {airport}")
        if value.utcoffset() is not None:
            return value
        try:
            return value.replace(tzinfo=ZoneInfo(zone_name))
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"invalid timezone mapping for airport {airport}") from error

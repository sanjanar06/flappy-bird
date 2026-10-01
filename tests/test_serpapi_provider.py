from datetime import UTC, date, datetime

import httpx
import pytest

from flappy_bird.models import InputSource, TripRequest
from flappy_bird.providers.serpapi import SerpApiGoogleFlightsProvider, SerpApiProviderError


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


def _payload() -> dict:
    return {
        "best_flights": [
            {
                "price": 812,
                "flights": [
                    {
                        "departure_airport": {"id": "ORD", "time": "2026-11-10 08:00"},
                        "arrival_airport": {"id": "COK", "time": "2026-11-12 03:30"},
                        "airline": "United",
                        "flight_number": "UA 123",
                    }
                ],
            }
        ]
    }


def test_serpapi_adapter_normalizes_provider_facts_and_keeps_unknowns() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_payload())
        )
    )
    provider = SerpApiGoogleFlightsProvider(
        api_key="test-key",
        client=client,
        airport_timezones={"ORD": "America/Chicago", "COK": "Asia/Kolkata"},
        clock=lambda: datetime(2026, 9, 26, tzinfo=UTC),
    )

    offers = provider.retrieve(_request())

    assert len(offers) == 1
    offer = offers[0]
    assert offer.provider == "serpapi"
    assert offer.total_price == 81_200
    assert offer.segments[0].departure_at.isoformat() == "2026-11-10T08:00:00-06:00"
    assert offer.protection_status.value == "unknown"
    assert offer.total_with_checked_bags(2) is None


def test_serpapi_adapter_fails_closed_when_timezone_is_not_known() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_payload())
        )
    )
    provider = SerpApiGoogleFlightsProvider(
        api_key="test-key",
        client=client,
        airport_timezones={"ORD": "America/Chicago"},
    )

    with pytest.raises(SerpApiProviderError, match="invalid or incomplete"):
        provider.retrieve(_request())

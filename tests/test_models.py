from datetime import date, datetime

import pytest
from pydantic import ValidationError

from flappy_bird.models import (
    FlightOffer,
    FlightSegment,
    InputSource,
    ProtectionStatus,
    TripRequest,
)


def test_trip_request_normalizes_airports_and_keeps_provenance() -> None:
    request = TripRequest(
        origin=" ord ",
        destination="cok",
        departure_date=date(2026, 12, 12),
        field_sources={
            "origin": InputSource.FORM,
            "destination": InputSource.NATURAL_LANGUAGE,
            "departure_date": InputSource.FORM,
        },
    )

    assert request.origin == "ORD"
    assert request.destination == "COK"
    assert request.max_stops == 1
    assert request.field_sources["destination"] is InputSource.NATURAL_LANGUAGE


def test_trip_request_requires_provenance_for_required_fields() -> None:
    with pytest.raises(ValidationError, match="field_sources"):
        TripRequest(
            origin="ORD",
            destination="COK",
            departure_date=date(2026, 12, 12),
            field_sources={"origin": InputSource.FORM},
        )


def test_normalized_offer_derives_stops_and_preserves_unknown_protection() -> None:
    offer = FlightOffer(
        provider="fixture",
        provider_offer_id="fixture-001",
        total_price=61200,
        currency="usd",
        segments=[
            FlightSegment(
                origin="ORD",
                destination="AUH",
                departure_at=datetime(2026, 12, 12, 13, 10),
                arrival_at=datetime(2026, 12, 13, 12, 30),
                marketing_carrier="EY",
                flight_number="EY10",
            ),
            FlightSegment(
                origin="AUH",
                destination="COK",
                departure_at=datetime(2026, 12, 13, 14, 25),
                arrival_at=datetime(2026, 12, 13, 19, 50),
                marketing_carrier="EY",
                flight_number="EY334",
            ),
        ],
        retrieved_at=datetime(2026, 9, 19, 12, 0),
    )

    assert offer.stops == 1
    assert offer.currency == "USD"
    assert offer.protection_status is ProtectionStatus.UNKNOWN

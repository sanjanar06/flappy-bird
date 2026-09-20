from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from flappy_bird.models import (
    CheckedBaggageAllowance,
    CheckedBaggagePricing,
    ConnectionLabel,
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
        checked_bags=2,
        field_sources={
            "origin": InputSource.FORM,
            "destination": InputSource.NATURAL_LANGUAGE,
            "departure_date": InputSource.FORM,
            "checked_bags": InputSource.NATURAL_LANGUAGE,
        },
    )

    assert request.origin == "ORD"
    assert request.destination == "COK"
    assert request.max_stops == 1
    assert request.checked_bags == 2
    assert request.field_sources["destination"] is InputSource.NATURAL_LANGUAGE


def test_trip_request_requires_provenance_for_required_fields() -> None:
    with pytest.raises(ValidationError, match="field_sources"):
        TripRequest(
            origin="ORD",
            destination="COK",
            departure_date=date(2026, 12, 12),
            field_sources={"origin": InputSource.FORM},
        )


def test_trip_request_requires_provenance_for_supplied_checked_bags() -> None:
    with pytest.raises(ValidationError, match="checked_bags"):
        TripRequest(
            origin="ORD",
            destination="COK",
            departure_date=date(2026, 11, 10),
            checked_bags=2,
            field_sources={
                "origin": InputSource.FORM,
                "destination": InputSource.FORM,
                "departure_date": InputSource.FORM,
            },
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
                departure_at=datetime(2026, 12, 12, 13, 10, tzinfo=UTC),
                arrival_at=datetime(2026, 12, 13, 2, 30, tzinfo=UTC),
                marketing_carrier="EY",
                flight_number="EY10",
            ),
            FlightSegment(
                origin="AUH",
                destination="COK",
                departure_at=datetime(2026, 12, 13, 4, 25, tzinfo=UTC),
                arrival_at=datetime(2026, 12, 13, 8, 50, tzinfo=UTC),
                marketing_carrier="EY",
                flight_number="EY334",
            ),
        ],
        baggage_allowance=CheckedBaggageAllowance(
            pieces=2,
            weight_per_piece_kg=Decimal("23"),
        ),
        retrieved_at=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert offer.stops == 1
    assert offer.currency == "USD"
    assert offer.protection_status is ProtectionStatus.UNKNOWN
    assert offer.layover_duration == timedelta(hours=1, minutes=55)
    assert offer.total_duration == timedelta(hours=19, minutes=40)
    assert offer.has_airport_change is False
    assert offer.connection_labels == frozenset({ConnectionLabel.PROTECTION_UNKNOWN})
    assert offer.total_with_checked_bags(2) == 61200


def test_offer_preserves_exact_baggage_fees_without_estimating_missing_fees() -> None:
    segment = FlightSegment(
        origin="ORD",
        destination="COK",
        departure_at=datetime(2026, 11, 10, 10, 0, tzinfo=UTC),
        arrival_at=datetime(2026, 11, 11, 6, 0, tzinfo=UTC),
        marketing_carrier="ZZ",
        flight_number="ZZ100",
    )
    priced_offer = FlightOffer(
        provider="fixture",
        provider_offer_id="priced-bags",
        total_price=60000,
        currency="USD",
        segments=[segment],
        checked_baggage_pricing=CheckedBaggagePricing(
            requested_pieces=2,
            fee_total=18000,
            currency="USD",
            fees_by_piece=[8000, 10000],
        ),
        retrieved_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )
    unknown_offer = priced_offer.model_copy(
        update={"provider_offer_id": "unknown-bags", "checked_baggage_pricing": None}
    )

    assert priced_offer.total_with_checked_bags(2) == 78000
    assert unknown_offer.total_with_checked_bags(2) is None


def test_offer_derives_airport_change_and_self_transfer_labels() -> None:
    offer = FlightOffer(
        provider="fixture",
        provider_offer_id="airport-change",
        total_price=55000,
        currency="USD",
        segments=[
            FlightSegment(
                origin="ORD",
                destination="LHR",
                departure_at=datetime(2026, 11, 10, 8, 0, tzinfo=UTC),
                arrival_at=datetime(2026, 11, 10, 16, 0, tzinfo=UTC),
                marketing_carrier="ZZ",
                flight_number="ZZ100",
            ),
            FlightSegment(
                origin="LGW",
                destination="COK",
                departure_at=datetime(2026, 11, 10, 21, 0, tzinfo=UTC),
                arrival_at=datetime(2026, 11, 11, 7, 0, tzinfo=UTC),
                marketing_carrier="YY",
                flight_number="YY200",
            ),
        ],
        protection_status=ProtectionStatus.SELF_TRANSFER,
        retrieved_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
    )

    assert offer.has_airport_change is True
    assert offer.connection_labels == frozenset(
        {ConnectionLabel.SELF_TRANSFER, ConnectionLabel.AIRPORT_CHANGE}
    )


def test_segment_rejects_timestamps_without_utc_offsets() -> None:
    with pytest.raises(ValidationError, match="UTC offsets"):
        FlightSegment(
            origin="ORD",
            destination="COK",
            departure_at=datetime(2026, 11, 10, 10, 0),
            arrival_at=datetime(2026, 11, 11, 6, 0),
            marketing_carrier="ZZ",
            flight_number="ZZ100",
        )

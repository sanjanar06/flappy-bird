from datetime import UTC, datetime

from flappy_bird.models import FlightOffer, FlightSegment, ProtectionStatus
from flappy_bird.reconciliation import reconcile_offers


def _offer(provider: str, offer_id: str, price: int, protection: ProtectionStatus):
    return FlightOffer(
        provider=provider,
        provider_offer_id=offer_id,
        total_price=price,
        currency="USD",
        segments=[
            FlightSegment(
                origin="ORD",
                destination="COK",
                departure_at=datetime(2026, 11, 10, 8, tzinfo=UTC),
                arrival_at=datetime(2026, 11, 12, 3, 30, tzinfo=UTC),
                marketing_carrier="UA",
                flight_number="123",
            )
        ],
        protection_status=protection,
        retrieved_at=datetime(2026, 9, 26, tzinfo=UTC),
    )


def test_reconciliation_keeps_provider_observations_and_flags_price_conflict() -> None:
    offers = [
        _offer("ignav", "ignav-1", 81200, ProtectionStatus.UNKNOWN),
        _offer("serpapi", "serpapi-1", 82900, ProtectionStatus.UNKNOWN),
    ]

    [itinerary] = reconcile_offers(offers)

    assert itinerary.price_conflict is True
    assert itinerary.protection_conflict is False
    assert [item.offer.provider for item in itinerary.observations] == ["ignav", "serpapi"]
    assert [item.offer.total_price for item in itinerary.observations] == [81200, 82900]


def test_reconciliation_keeps_different_flights_separate() -> None:
    first = _offer("ignav", "one", 81200, ProtectionStatus.UNKNOWN)
    second = _offer("serpapi", "two", 81200, ProtectionStatus.UNKNOWN).model_copy(
        update={
            "segments": [first.segments[0].model_copy(update={"flight_number": "999"})]
        }
    )

    assert len(reconcile_offers([first, second])) == 2

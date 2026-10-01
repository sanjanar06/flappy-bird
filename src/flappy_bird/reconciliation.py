"""Provider-neutral itinerary matching that preserves every provider observation."""

from __future__ import annotations

from collections import OrderedDict
from datetime import UTC

from pydantic import BaseModel, Field

from flappy_bird.models import FlightOffer, ProtectionStatus


class ProviderOfferObservation(BaseModel):
    """One provider's intact, normalized observation of an itinerary."""

    offer: FlightOffer


class ReconciledItinerary(BaseModel):
    """Matched itinerary plus all source observations and explicit conflicts."""

    itinerary_key: str
    observations: list[ProviderOfferObservation] = Field(min_length=1)
    price_conflict: bool
    protection_conflict: bool

    @property
    def representative(self) -> FlightOffer:
        """A stable representative for analysis; source observations remain attached."""

        return self.observations[0].offer


def reconcile_offers(offers: list[FlightOffer]) -> list[ReconciledItinerary]:
    """Group identical itineraries while surfacing price/protection disagreement.

    Matching requires the same carriers, flight numbers, airports, and offset-aware
    segment timestamps. Similar routes are not merged. Provider observations are
    retained verbatim and never averaged or overwritten.
    """

    groups: OrderedDict[str, list[FlightOffer]] = OrderedDict()
    for offer in offers:
        key = _itinerary_key(offer)
        groups.setdefault(key, []).append(offer)

    reconciled: list[ReconciledItinerary] = []
    for key, observations in groups.items():
        prices = {(offer.currency, offer.total_price) for offer in observations}
        protections = {offer.protection_status for offer in observations}
        reconciled.append(
            ReconciledItinerary(
                itinerary_key=key,
                observations=[ProviderOfferObservation(offer=offer) for offer in observations],
                price_conflict=len(prices) > 1,
                protection_conflict=(
                    len(protections) > 1
                    and ProtectionStatus.UNKNOWN not in protections
                ),
            )
        )
    return reconciled


def _itinerary_key(offer: FlightOffer) -> str:
    segments = ";".join(
        ":".join(
            (
                segment.marketing_carrier,
                segment.flight_number,
                segment.origin,
                segment.destination,
                segment.departure_at.astimezone(UTC).isoformat(),
                segment.arrival_at.astimezone(UTC).isoformat(),
            )
        )
        for segment in offer.segments
    )
    return segments

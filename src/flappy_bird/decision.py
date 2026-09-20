"""Deterministic offer analysis and advisory choice-role assignment."""

from __future__ import annotations

from datetime import timedelta

from flappy_bird.models import (
    ChoiceAssignment,
    ChoiceRole,
    ConnectionLabel,
    FlightOffer,
    OfferAnalysis,
)


def analyze_offers(
    offers: list[FlightOffer], requested_checked_bags: int
) -> dict[str, OfferAnalysis]:
    """Derive traceable facts without mutating normalized provider observations."""

    analyses: dict[str, OfferAnalysis] = {}
    for offer in offers:
        if offer.provider_offer_id in analyses:
            raise ValueError(f"duplicate offer id: {offer.provider_offer_id}")
        operating_carriers = {
            segment.operating_carrier or segment.marketing_carrier
            for segment in offer.segments
        }
        analyses[offer.provider_offer_id] = OfferAnalysis(
            offer_id=offer.provider_offer_id,
            connection_labels=offer.connection_labels,
            stops=offer.stops,
            total_duration_minutes=_whole_minutes(offer.total_duration),
            layover_minutes=(
                _whole_minutes(offer.layover_duration)
                if offer.layover_duration is not None
                else None
            ),
            has_airport_change=offer.has_airport_change,
            has_operating_carrier_change=len(operating_carriers) > 1,
            total_price_for_requested_bags=offer.total_with_checked_bags(
                requested_checked_bags
            ),
        )
    return analyses


def assign_choice_roles(
    offers: list[FlightOffer], analyses: dict[str, OfferAnalysis]
) -> list[ChoiceAssignment]:
    """Assign every applicable role, including ties, in provider order."""

    if not offers:
        return []
    missing = {offer.provider_offer_id for offer in offers} - analyses.keys()
    if missing:
        raise ValueError(f"missing analyses for offers: {', '.join(sorted(missing))}")

    priced = [
        analysis
        for analysis in analyses.values()
        if analysis.total_price_for_requested_bags is not None
    ]
    lowest_price = (
        min(analysis.total_price_for_requested_bags for analysis in priced)
        if priced
        else None
    )
    shortest_duration = min(
        analyses[offer.provider_offer_id].total_duration_minutes for offer in offers
    )
    lowest_risk = min(_connection_risk_key(analyses[offer.provider_offer_id]) for offer in offers)

    assignments: list[ChoiceAssignment] = []
    for offer in offers:
        analysis = analyses[offer.provider_offer_id]
        roles: set[ChoiceRole] = set()
        if (
            lowest_price is not None
            and analysis.total_price_for_requested_bags == lowest_price
        ):
            roles.add(ChoiceRole.LOWEST_APPARENT_PRICE)
        if analysis.total_duration_minutes == shortest_duration:
            roles.add(ChoiceRole.SHORTEST_TOTAL_TRAVEL_TIME)
        if _connection_risk_key(analysis) == lowest_risk:
            roles.add(ChoiceRole.LOWEST_CONNECTION_RISK)
        assignments.append(
            ChoiceAssignment(offer_id=offer.provider_offer_id, roles=frozenset(roles))
        )
    return assignments


def _connection_risk_key(analysis: OfferAnalysis) -> tuple[int, bool, bool]:
    if analysis.stops == 0:
        protection_rank = 0
    elif ConnectionLabel.PROTECTED_CONNECTION in analysis.connection_labels:
        protection_rank = 0
    elif ConnectionLabel.PROTECTION_UNKNOWN in analysis.connection_labels:
        protection_rank = 1
    else:
        protection_rank = 2
    return (
        protection_rank,
        analysis.has_airport_change,
        analysis.has_operating_carrier_change,
    )


def _whole_minutes(duration: timedelta) -> int:
    total_seconds = duration.total_seconds()
    if total_seconds % 60:
        raise ValueError("flight durations must resolve to whole minutes")
    return int(total_seconds // 60)

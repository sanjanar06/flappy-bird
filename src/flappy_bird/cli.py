"""Terminal entry point for the live flight-decision workflow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from typing import TextIO

import httpx
from dotenv import load_dotenv

from flappy_bird.models import (
    ChoiceRole,
    ConnectionLabel,
    InputSource,
    TripRequest,
)
from flappy_bird.providers.fixture import IgnavObservationProvider
from flappy_bird.providers.ignav import LiveIgnavProvider
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph

ROLE_LABELS = {
    ChoiceRole.LOWEST_APPARENT_PRICE: "Lowest apparent price",
    ChoiceRole.LOWEST_CONNECTION_RISK: "Lowest connection risk",
    ChoiceRole.SHORTEST_TOTAL_TRAVEL_TIME: "Shortest total travel time",
}
ROLE_ORDER = tuple(ROLE_LABELS)
CONNECTION_LABELS = {
    ConnectionLabel.PROTECTED_CONNECTION: "Protected connection",
    ConnectionLabel.SELF_TRANSFER: "Self-transfer",
    ConnectionLabel.PROTECTION_UNKNOWN: "Protection unknown",
    ConnectionLabel.AIRPORT_CHANGE: "Airport change",
}


def main(
    argv: Sequence[str] | None = None,
    *,
    provider: IgnavObservationProvider | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse CLI input and run the same graph used by other interfaces."""

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = _build_parser().parse_args(argv)
    request = _build_request(args)

    if provider is not None:
        return _run_search(request, provider, output, error_output)

    load_dotenv()
    try:
        with httpx.Client() as client:
            live_provider = LiveIgnavProvider.from_environment(client=client)
            return _run_search(request, live_provider, output, error_output)
    except ValueError as error:
        print(f"Configuration error: {error}", file=error_output)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flappy-bird",
        description="Compare evidence-backed flight choices without selecting for the traveler.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    search = subparsers.add_parser("search", help="Search and compare one-way flights")
    search.add_argument("--origin", required=True, help="Three-letter origin IATA code")
    search.add_argument("--destination", required=True, help="Three-letter destination IATA code")
    search.add_argument(
        "--date",
        required=True,
        type=_parse_date,
        help="Departure date: YYYY-MM-DD",
    )
    search.add_argument("--checked-bags", required=True, type=int, choices=range(0, 10))
    search.add_argument("--max-stops", type=int, choices=(0, 1), default=None)
    return parser


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must use YYYY-MM-DD") from error


def _build_request(args: argparse.Namespace) -> TripRequest:
    max_stops = 1 if args.max_stops is None else args.max_stops
    return TripRequest(
        origin=args.origin,
        destination=args.destination,
        departure_date=args.date,
        checked_bags=args.checked_bags,
        max_stops=max_stops,
        field_sources={
            "origin": InputSource.COMMAND_LINE,
            "destination": InputSource.COMMAND_LINE,
            "departure_date": InputSource.COMMAND_LINE,
            "checked_bags": InputSource.COMMAND_LINE,
            "max_stops": (
                InputSource.DEFAULT
                if args.max_stops is None
                else InputSource.COMMAND_LINE
            ),
            "passengers": InputSource.DEFAULT,
            "cabin": InputSource.DEFAULT,
        },
    )


def _run_search(
    request: TripRequest,
    provider: IgnavObservationProvider,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    graph = build_flight_decision_graph(provider)
    try:
        state = FlightDecisionState.model_validate(graph.invoke({"request": request}))
    except ValueError as error:
        print(f"Decision workflow failed: {error}", file=stderr)
        return 2

    if state.provider_failures:
        failure = state.provider_failures[0]
        print(f"Flight search failed: {failure.message}", file=stderr)
        return 2

    print(_render_choices(state, request.checked_bags), file=stdout)
    return 0


def _render_choices(state: FlightDecisionState, checked_bags: int | None) -> str:
    offers = {offer.provider_offer_id: offer for offer in state.normalized_offers}
    lines = [f"{len(state.choice_assignments)} flight choices"]
    for index, assignment in enumerate(state.choice_assignments, start=1):
        offer = offers[assignment.offer_id]
        analysis = state.offer_analyses[assignment.offer_id]
        roles = [ROLE_LABELS[role] for role in ROLE_ORDER if role in assignment.roles]
        labels = [
            label
            for connection_label, label in CONNECTION_LABELS.items()
            if connection_label in analysis.connection_labels
        ]
        lines.extend(
            [
                "",
                f"Choice {index}",
                f"  Roles: {', '.join(roles) if roles else 'None'}",
                "  Apparent total"
                f" for {checked_bags} checked bags: "
                f"{_format_price(analysis.total_price_for_requested_bags, offer.currency)}",
                f"  Duration: {_format_duration(analysis.total_duration_minutes)}",
                f"  Stops: {analysis.stops}",
                f"  Connection labels: {', '.join(labels) if labels else 'None'}",
                f"  Provider: {offer.provider}",
            ]
        )
    return "\n".join(lines)


def _format_price(amount: int | None, currency: str) -> str:
    if amount is None:
        return "UNKNOWN"
    major_units = Decimal(amount) / Decimal(100)
    if currency == "USD":
        return f"${major_units:,.2f} USD"
    return f"{major_units:,.2f} {currency}"


def _format_duration(total_minutes: int) -> str:
    hours, minutes = divmod(total_minutes, 60)
    return f"{hours}h {minutes}m"

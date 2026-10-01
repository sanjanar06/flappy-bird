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

from flappy_bird.explanations import ChoiceExplainer
from flappy_bird.models import (
    ChoiceRole,
    ConnectionLabel,
    InputSource,
    TripRequest,
)
from flappy_bird.natural_language import (
    IntakeError,
    TripInterpreter,
    build_request_from_message,
)
from flappy_bird.providers.fixture import IgnavObservationProvider
from flappy_bird.providers.groq import GroqChoiceExplainer
from flappy_bird.providers.groq_intake import GroqTripInterpreter
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
    explainer: ChoiceExplainer | None = None,
    interpreter: TripInterpreter | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Parse CLI input and run the same graph used by other interfaces."""

    output = stdout or sys.stdout
    error_output = stderr or sys.stderr
    args = _build_parser().parse_args(argv)
    if args.command == "search":
        request = _build_request(args)
        if provider is not None:
            return _run_search(request, provider, explainer, output, error_output)
    elif interpreter is not None:
        try:
            request = build_request_from_message(args.message, interpreter.extract(args.message))
        except IntakeError as error:
            print(f"Trip clarification: {error}", file=error_output)
            return 2
        if provider is not None:
            return _run_search(request, provider, explainer, output, error_output)

    load_dotenv()
    try:
        with httpx.Client() as client:
            if args.command == "ask":
                live_interpreter = GroqTripInterpreter.from_environment(client=client)
                try:
                    request = build_request_from_message(
                        args.message, live_interpreter.extract(args.message)
                    )
                except IntakeError as error:
                    print(f"Trip clarification: {error}", file=error_output)
                    return 2
            live_provider = provider or LiveIgnavProvider.from_environment(client=client)
            if explainer is not None:
                live_explainer = explainer
            else:
                try:
                    live_explainer = GroqChoiceExplainer.from_environment(client=client)
                except ValueError:
                    live_explainer = None
            return _run_search(
                request,
                live_provider,
                live_explainer,
                output,
                error_output,
            )
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
    ask = subparsers.add_parser("ask", help="Search from a natural-language trip request")
    ask.add_argument("message", help="Include airport codes, date with year, and checked bags")
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
    explainer: ChoiceExplainer | None,
    stdout: TextIO,
    stderr: TextIO,
) -> int:
    graph = build_flight_decision_graph(provider, explainer)
    try:
        state = FlightDecisionState.model_validate(graph.invoke({"request": request}))
    except ValueError as error:
        print(f"Decision workflow failed: {error}", file=stderr)
        return 2

    if state.provider_failures and not state.normalized_offers:
        failure = state.provider_failures[0]
        print(f"Flight search failed: {failure.message}", file=stderr)
        return 2

    print(_render_choices(state, request.checked_bags), file=stdout)
    if state.provider_failures:
        failed_providers = ", ".join(
            sorted({failure.provider for failure in state.provider_failures})
        )
        print(f"Partial flight search; unavailable providers: {failed_providers}", file=stderr)
    if state.explanation_failure is not None:
        print(
            f"AI explanations unavailable: {state.explanation_failure}",
            file=stderr,
        )
    return 0


def _render_choices(state: FlightDecisionState, checked_bags: int | None) -> str:
    offers = {offer.provider_offer_id: offer for offer in state.normalized_offers}
    explanations = {
        explanation.offer_id: explanation
        for explanation in state.choice_explanations
    }
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
        choice_lines = [
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
        for segment_index, segment in enumerate(offer.segments, start=1):
            carrier = segment.marketing_carrier
            if segment.operating_carrier and segment.operating_carrier != carrier:
                carrier += f" (operated by {segment.operating_carrier})"
            choice_lines.extend(
                [
                    f"  Segment {segment_index}: {carrier} {segment.flight_number}",
                    f"    Depart: {segment.origin} {_format_local_time(segment.departure_at)}",
                    f"    Arrive: {segment.destination} {_format_local_time(segment.arrival_at)}",
                ]
            )
            if segment_index < len(offer.segments):
                next_segment = offer.segments[segment_index]
                layover = next_segment.departure_at - segment.arrival_at
                connection = f"    Layover: {_format_duration(int(layover.total_seconds() // 60))}"
                if segment.destination != next_segment.origin:
                    connection += f"; airport change {segment.destination} → {next_segment.origin}"
                choice_lines.append(connection)
        explanation = explanations.get(assignment.offer_id)
        if explanation is not None:
            choice_lines.extend(
                [
                    f"  Explanation: {explanation.summary}",
                    f"  Tradeoff: {explanation.tradeoff}",
                ]
            )
        lines.extend(choice_lines)
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


def _format_local_time(value) -> str:
    """Keep each provider's local wall-clock time and UTC offset visible."""

    return value.strftime("%Y-%m-%d %H:%M %z")

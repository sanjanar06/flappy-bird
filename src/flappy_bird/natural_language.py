"""Validate model-extracted request snippets before a provider search."""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from flappy_bird.models import InputSource, TripRequest


class ExtractedTrip(BaseModel):
    """Exact spans copied from a traveler's message, or null when absent."""

    model_config = ConfigDict(extra="forbid")

    origin_text: str | None
    destination_text: str | None
    departure_date_text: str | None
    checked_bags_text: str | None
    max_stops_text: str | None
    unsupported_constraints: list[str]


class TripInterpreter(Protocol):
    def extract(self, message: str) -> ExtractedTrip:
        """Identify request spans without deciding whether they are valid."""


class IntakeError(ValueError):
    """A safe clarification or unsupported-request message for the traveler."""


def build_request_from_message(message: str, extracted: ExtractedTrip) -> TripRequest:
    """Convert verified source spans to the existing provider-neutral request."""

    if not message.strip():
        raise IntakeError("Please describe the trip you want to search.")
    for span in (
        extracted.origin_text,
        extracted.destination_text,
        extracted.departure_date_text,
        extracted.checked_bags_text,
        extracted.max_stops_text,
        *extracted.unsupported_constraints,
    ):
        if span is not None and (not span.strip() or span.casefold() not in message.casefold()):
            raise IntakeError(
                "The extracted request did not match your message. Please rephrase it."
            )

    unsupported = list(extracted.unsupported_constraints)
    for pattern in (
        r"\b(?:no|avoid|exclude|without)\s+"
        r"(?:self[- ]transfers?|airport changes?|long layovers?)\b",
        r"\b(?:only|must be)\s+protected\b",
        r"\b(?:business|first|premium)\s+class\b",
        r"\bround[- ]trip\b",
        r"\b(?:two|three|[2-9])\s+(?:adults?|passengers?)\b",
        r"\b(?:under|below|less than|up to|max(?:imum)?)\s*(?:\$|USD\s*)\d[\d,]*\b",
        r"\b\d+(?:\.\d+)?\s?kg\b",
    ):
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match and match.group() not in unsupported:
            unsupported.append(match.group())
    if unsupported:
        raise IntakeError(
            "This first version cannot enforce: "
            + ", ".join(unsupported)
            + ". Please revise the request before searching."
        )

    origin = _airport(extracted.origin_text, "departure")
    destination = _airport(extracted.destination_text, "arrival")
    departure_date = _departure_date(extracted.departure_date_text)
    checked_bags = _checked_bags(extracted.checked_bags_text)
    max_stops = _max_stops(extracted.max_stops_text)
    sources = {
        "origin": InputSource.NATURAL_LANGUAGE,
        "destination": InputSource.NATURAL_LANGUAGE,
        "departure_date": InputSource.NATURAL_LANGUAGE,
        "checked_bags": InputSource.NATURAL_LANGUAGE,
        "passengers": InputSource.DEFAULT,
        "cabin": InputSource.DEFAULT,
        "max_stops": (
            InputSource.NATURAL_LANGUAGE
            if extracted.max_stops_text is not None
            else InputSource.DEFAULT
        ),
    }
    try:
        return TripRequest(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            checked_bags=checked_bags,
            max_stops=max_stops,
            field_sources=sources,
        )
    except ValueError as error:
        raise IntakeError("Please check the route and trip details in your request.") from error


def _airport(span: str | None, direction: str) -> str:
    if span is None:
        raise IntakeError(f"Which {direction} airport? Include its three-letter IATA code.")
    codes = (
        [span.upper()]
        if re.fullmatch(r"[a-zA-Z]{3}", span)
        else re.findall(r"\b[A-Z]{3}\b", span)
    )
    if len(codes) != 1:
        raise IntakeError(f"Which {direction} airport? Include its three-letter IATA code.")
    return codes[0]


def _departure_date(span: str | None) -> date:
    if span is None:
        raise IntakeError("What is the departure date? Include the year.")
    for format_string in (
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(span, format_string).date()
        except ValueError:
            continue
    raise IntakeError("What is the departure date? Use a date with a year, such as 2026-11-10.")


def _checked_bags(span: str | None) -> int:
    if span is None:
        raise IntakeError("How many checked bags should the search include?")
    match = re.fullmatch(
        r"(no|zero|one|two|three|four|five|six|seven|eight|nine|[0-9])\s+checked\s+bags?",
        span.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise IntakeError("How many checked bags? Say, for example, 'two checked bags'.")
    count = match.group(1).casefold()
    words = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
             "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
    return words[count] if count in words else int(count)


def _max_stops(span: str | None) -> int:
    if span is None:
        return 1
    value = span.strip().casefold()
    if value in {"nonstop", "non-stop", "no stops", "zero stops", "0 stops"}:
        return 0
    if re.fullmatch(r"(?:at most|up to|max(?:imum)?)\s+(?:one|1)\s+stops?", value):
        return 1
    raise IntakeError("This first version supports nonstop or at most one stop.")

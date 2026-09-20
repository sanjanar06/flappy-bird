"""Offline provider used to exercise the workflow with captured observations."""

from dataclasses import dataclass
from typing import Protocol

from flappy_bird.models import TripRequest
from flappy_bird.providers.ignav import IgnavFixture


class IgnavObservationProvider(Protocol):
    """Retrieval boundary for providers that return an Ignav observation."""

    def retrieve(self, request: TripRequest) -> IgnavFixture:
        """Return one provider observation for the requested search."""


@dataclass(frozen=True)
class FixtureIgnavProvider:
    """Return a sanitized observation without network access."""

    fixture: IgnavFixture

    def retrieve(self, _request: TripRequest) -> IgnavFixture:
        return self.fixture

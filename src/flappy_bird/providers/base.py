"""Provider-neutral normalized flight search contracts and composition."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol

from pydantic import BaseModel

from flappy_bird.models import FlightOffer, TripRequest
from flappy_bird.providers.ignav import IgnavProviderError
from flappy_bird.providers.serpapi import SerpApiProviderError


class FlightProvider(Protocol):
    name: str

    def retrieve(self, request: TripRequest) -> list[FlightOffer]:
        """Return normalized offers from this provider."""


class ProviderSearchFailure(BaseModel):
    provider: str
    message: str
    status_code: int | None = None


class ProviderSearchBatch(BaseModel):
    offers: list[FlightOffer]
    failures: list[ProviderSearchFailure]


class IgnavFlightProviderAdapter:
    """Adapt the legacy raw Ignav observation interface to normalized offers."""

    name = "ignav"

    def __init__(self, provider) -> None:
        self._provider = provider

    def retrieve(self, request: TripRequest) -> list[FlightOffer]:
        from flappy_bird.providers.ignav import normalize_ignav_response

        observation = self._provider.retrieve(request)
        return normalize_ignav_response(
            observation.response,
            request,
            observation.retrieved_at,
        )


class CompositeFlightProvider:
    """Query normalized providers concurrently and retain partial failures."""

    name = "composite"

    def __init__(self, providers: list[FlightProvider]) -> None:
        if not providers:
            raise ValueError("at least one flight provider is required")
        self._providers = providers

    def retrieve(self, request: TripRequest) -> ProviderSearchBatch:
        offers: list[FlightOffer] = []
        failures: list[ProviderSearchFailure] = []
        with ThreadPoolExecutor(max_workers=len(self._providers)) as executor:
            futures = {
                executor.submit(provider.retrieve, request): provider
                for provider in self._providers
            }
            for future in as_completed(futures):
                provider = futures[future]
                name = getattr(provider, "name", provider.__class__.__name__.lower())
                try:
                    offers.extend(future.result())
                except (IgnavProviderError, SerpApiProviderError) as error:
                    failures.append(
                        ProviderSearchFailure(
                            provider=name,
                            message=str(error),
                            status_code=getattr(error, "status_code", None),
                        )
                    )
        return ProviderSearchBatch(offers=offers, failures=failures)

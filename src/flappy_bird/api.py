"""FastAPI boundary for the shared flight decision workflow."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from flappy_bird.explanations import ChoiceExplainer
from flappy_bird.models import TripRequest
from flappy_bird.providers.fixture import IgnavObservationProvider
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph


def create_app(
    provider: IgnavObservationProvider,
    explainer: ChoiceExplainer | None = None,
) -> FastAPI:
    """Build an API app around injected adapters; tests can use deterministic fixtures."""

    app = FastAPI(title="Flappy Bird Flight Decision API", version="0.1.0")
    graph = build_flight_decision_graph(provider, explainer)

    @app.post("/trips/search", response_model=FlightDecisionState)
    def search(request: TripRequest) -> FlightDecisionState:
        try:
            state = FlightDecisionState.model_validate(graph.invoke({"request": request}))
        except ValueError as error:
            raise HTTPException(
                status_code=502,
                detail="Flight decision workflow failed",
            ) from error
        if state.provider_failures and not state.normalized_offers:
            raise HTTPException(status_code=502, detail=state.provider_failures[0].message)
        return state

    return app

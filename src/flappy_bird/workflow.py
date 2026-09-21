"""LangGraph orchestration for the first deterministic decision slice."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from flappy_bird.decision import analyze_offers, assign_choice_roles
from flappy_bird.models import (
    ChoiceAssignment,
    FlightOffer,
    OfferAnalysis,
    TripRequest,
)
from flappy_bird.providers.fixture import IgnavObservationProvider
from flappy_bird.providers.ignav import (
    IgnavObservation,
    IgnavProviderError,
    normalize_ignav_response,
)

NodeName = Literal[
    "retrieve_offers",
    "normalize_offers",
    "analyze_offers",
    "construct_choice_set",
]


class WorkflowTraceEvent(BaseModel):
    """Inspectable record of one deterministic state transition."""

    node: NodeName
    writes: tuple[str, ...]


class ProviderFailure(BaseModel):
    """Sanitized provider failure retained in graph state."""

    provider: Literal["ignav"]
    message: str
    status_code: int | None = None


class FlightDecisionState(BaseModel):
    """Typed state shared by every node in the P0 flight-decision graph."""

    request: TripRequest
    provider_observation: IgnavObservation | None = None
    provider_failures: list[ProviderFailure] = Field(default_factory=list)
    normalized_offers: list[FlightOffer] = Field(default_factory=list)
    offer_analyses: dict[str, OfferAnalysis] = Field(default_factory=dict)
    choice_assignments: list[ChoiceAssignment] = Field(default_factory=list)
    trace: list[WorkflowTraceEvent] = Field(default_factory=list)


def build_flight_decision_graph(provider: IgnavObservationProvider):
    """Compile the deterministic P0 graph around an injected provider boundary."""

    def retrieve_offers(state: FlightDecisionState) -> dict[str, object]:
        try:
            observation = provider.retrieve(state.request)
        except IgnavProviderError as error:
            return {
                "provider_failures": [
                    ProviderFailure(
                        provider="ignav",
                        message=str(error),
                        status_code=error.status_code,
                    )
                ],
                "trace": _append_trace(state, "retrieve_offers", "provider_failures"),
            }
        return {
            "provider_observation": observation,
            "trace": _append_trace(state, "retrieve_offers", "provider_observation"),
        }

    def route_after_retrieval(
        state: FlightDecisionState,
    ) -> Literal["normalize_offers", "stop"]:
        if state.provider_observation is not None:
            return "normalize_offers"
        if state.provider_failures:
            return "stop"
        raise ValueError("retrieval produced neither an observation nor a provider failure")

    def normalize_offers(state: FlightDecisionState) -> dict[str, object]:
        observation = _require_observation(state)
        offers = normalize_ignav_response(
            observation.response,
            state.request,
            observation.retrieved_at,
        )
        return {
            "normalized_offers": offers,
            "trace": _append_trace(state, "normalize_offers", "normalized_offers"),
        }

    def analyze_normalized_offers(state: FlightDecisionState) -> dict[str, object]:
        if state.request.checked_bags is None:
            raise ValueError("P0 decision analysis requires an explicit checked-bag count")
        analyses = analyze_offers(
            state.normalized_offers,
            requested_checked_bags=state.request.checked_bags,
        )
        return {
            "offer_analyses": analyses,
            "trace": _append_trace(state, "analyze_offers", "offer_analyses"),
        }

    def construct_choice_set(state: FlightDecisionState) -> dict[str, object]:
        assignments = assign_choice_roles(
            state.normalized_offers,
            state.offer_analyses,
        )
        if len(assignments) != 3:
            raise ValueError(
                "P0 evaluation scenario requires exactly three distinct flight choices"
            )
        return {
            "choice_assignments": assignments,
            "trace": _append_trace(
                state,
                "construct_choice_set",
                "choice_assignments",
            ),
        }

    graph = StateGraph(FlightDecisionState)
    graph.add_node("retrieve_offers", retrieve_offers)
    graph.add_node("normalize_offers", normalize_offers)
    graph.add_node("analyze_offers", analyze_normalized_offers)
    graph.add_node("construct_choice_set", construct_choice_set)
    graph.add_edge(START, "retrieve_offers")
    graph.add_conditional_edges(
        "retrieve_offers",
        route_after_retrieval,
        {"normalize_offers": "normalize_offers", "stop": END},
    )
    graph.add_edge("normalize_offers", "analyze_offers")
    graph.add_edge("analyze_offers", "construct_choice_set")
    graph.add_edge("construct_choice_set", END)
    return graph.compile()


def _require_observation(state: FlightDecisionState) -> IgnavObservation:
    if state.provider_observation is None:
        raise ValueError("provider observation is missing")
    return state.provider_observation


def _append_trace(
    state: FlightDecisionState,
    node: NodeName,
    *writes: str,
) -> list[WorkflowTraceEvent]:
    return [*state.trace, WorkflowTraceEvent(node=node, writes=writes)]

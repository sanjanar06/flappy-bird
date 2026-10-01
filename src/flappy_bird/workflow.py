"""LangGraph orchestration for the first deterministic decision slice."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from flappy_bird.decision import analyze_offers, assign_choice_roles, select_choice_set
from flappy_bird.explanations import (
    ChoiceExplainer,
    ChoiceExplanation,
    ExplanationError,
    prepare_choice_evidence,
)
from flappy_bird.models import (
    ChoiceAssignment,
    FlightOffer,
    OfferAnalysis,
    TripRequest,
)
from flappy_bird.providers.base import ProviderSearchBatch
from flappy_bird.providers.fixture import IgnavObservationProvider
from flappy_bird.providers.ignav import (
    IgnavObservation,
    IgnavProviderError,
    normalize_ignav_response,
)
from flappy_bird.providers.serpapi import SerpApiProviderError
from flappy_bird.reconciliation import ReconciledItinerary, reconcile_offers

NodeName = Literal[
    "retrieve_offers",
    "normalize_offers",
    "reconcile_offers",
    "analyze_offers",
    "construct_choice_set",
    "explain_choices",
]


class WorkflowTraceEvent(BaseModel):
    """Inspectable record of one deterministic state transition."""

    node: NodeName
    writes: tuple[str, ...]


class ProviderFailure(BaseModel):
    """Sanitized provider failure retained in graph state."""

    provider: str
    message: str
    status_code: int | None = None


class FlightDecisionState(BaseModel):
    """Typed state shared by every node in the P0 flight-decision graph."""

    request: TripRequest
    provider_observation: IgnavObservation | None = None
    provider_failures: list[ProviderFailure] = Field(default_factory=list)
    provider_results_received: bool = False
    normalized_offers: list[FlightOffer] = Field(default_factory=list)
    reconciled_itineraries: list[ReconciledItinerary] = Field(default_factory=list)
    offer_analyses: dict[str, OfferAnalysis] = Field(default_factory=dict)
    choice_assignments: list[ChoiceAssignment] = Field(default_factory=list)
    choice_explanations: list[ChoiceExplanation] = Field(default_factory=list)
    explanation_failure: str | None = None
    trace: list[WorkflowTraceEvent] = Field(default_factory=list)


def build_flight_decision_graph(
    provider: IgnavObservationProvider,
    explainer: ChoiceExplainer | None = None,
):
    """Compile the deterministic P0 graph around an injected provider boundary."""

    def retrieve_offers(state: FlightDecisionState) -> dict[str, object]:
        try:
            observation = provider.retrieve(state.request)
        except (IgnavProviderError, SerpApiProviderError) as error:
            return {
                "provider_failures": [
                    ProviderFailure(
                        provider=getattr(provider, "name", "ignav"),
                        message=str(error),
                        status_code=error.status_code,
                    )
                ],
                "trace": _append_trace(state, "retrieve_offers", "provider_failures"),
            }
        if isinstance(observation, ProviderSearchBatch):
            failures = [
                ProviderFailure(
                    provider=failure.provider,
                    message=failure.message,
                    status_code=failure.status_code,
                )
                for failure in observation.failures
            ]
            return {
                "normalized_offers": observation.offers,
                "provider_failures": failures,
                "provider_results_received": True,
                "trace": _append_trace(
                    state,
                    "retrieve_offers",
                    "normalized_offers",
                    "provider_failures",
                ),
            }
        if isinstance(observation, list):
            return {
                "normalized_offers": observation,
                "provider_results_received": True,
                "trace": _append_trace(state, "retrieve_offers", "normalized_offers"),
            }
        return {
            "provider_observation": observation,
            "trace": _append_trace(state, "retrieve_offers", "provider_observation"),
        }

    def route_after_retrieval(
        state: FlightDecisionState,
    ) -> Literal["normalize_offers", "reconcile_offers", "stop"]:
        if state.provider_observation is not None:
            return "normalize_offers"
        if state.normalized_offers:
            return "reconcile_offers"
        if state.provider_failures:
            return "stop"
        if state.provider_results_received:
            return "reconcile_offers"
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

    def reconcile_normalized_offers(state: FlightDecisionState) -> dict[str, object]:
        itineraries = reconcile_offers(state.normalized_offers)
        representatives = [item.representative for item in itineraries]
        return {
            "normalized_offers": representatives,
            "reconciled_itineraries": itineraries,
            "trace": _append_trace(
                state,
                "reconcile_offers",
                "normalized_offers",
                "reconciled_itineraries",
            ),
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
        assignments = select_choice_set(assignments)
        return {
            "choice_assignments": assignments,
            "trace": _append_trace(
                state,
                "construct_choice_set",
                "choice_assignments",
            ),
        }

    def explain_choices(state: FlightDecisionState) -> dict[str, object]:
        if explainer is None:
            raise ValueError("explanation node requires an explainer")
        evidence = prepare_choice_evidence(
            state.normalized_offers,
            state.offer_analyses,
            state.choice_assignments,
        )
        try:
            explanations = explainer.explain(evidence)
        except ExplanationError as error:
            return {
                "explanation_failure": str(error),
                "trace": _append_trace(
                    state,
                    "explain_choices",
                    "explanation_failure",
                ),
            }
        return {
            "choice_explanations": explanations,
            "trace": _append_trace(
                state,
                "explain_choices",
                "choice_explanations",
            ),
        }

    graph = StateGraph(FlightDecisionState)
    graph.add_node("retrieve_offers", retrieve_offers)
    graph.add_node("normalize_offers", normalize_offers)
    graph.add_node("reconcile_offers", reconcile_normalized_offers)
    graph.add_node("analyze_offers", analyze_normalized_offers)
    graph.add_node("construct_choice_set", construct_choice_set)
    if explainer is not None:
        graph.add_node("explain_choices", explain_choices)
    graph.add_edge(START, "retrieve_offers")
    graph.add_conditional_edges(
        "retrieve_offers",
        route_after_retrieval,
        {
            "normalize_offers": "normalize_offers",
            "reconcile_offers": "reconcile_offers",
            "stop": END,
        },
    )
    graph.add_edge("normalize_offers", "reconcile_offers")
    graph.add_edge("reconcile_offers", "analyze_offers")
    graph.add_edge("analyze_offers", "construct_choice_set")
    if explainer is None:
        graph.add_edge("construct_choice_set", END)
    else:
        graph.add_edge("construct_choice_set", "explain_choices")
        graph.add_edge("explain_choices", END)
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

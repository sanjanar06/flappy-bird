import json
from datetime import date
from pathlib import Path

from flappy_bird.explanations import (
    ChoiceEvidence,
    ChoiceExplanation,
    ExplanationError,
)
from flappy_bird.models import InputSource, TripRequest
from flappy_bird.providers.fixture import FixtureIgnavProvider
from flappy_bird.providers.ignav import IgnavObservation
from flappy_bird.workflow import FlightDecisionState, build_flight_decision_graph

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"


class RecordingExplainer:
    def __init__(self) -> None:
        self.evidence: list[ChoiceEvidence] = []

    def explain(self, evidence: list[ChoiceEvidence]) -> list[ChoiceExplanation]:
        self.evidence = evidence
        return [
            ChoiceExplanation(
                offer_id=item.offer_id,
                summary="Grounded summary from prepared evidence.",
                tradeoff="The traveler should compare the deterministic facts.",
            )
            for item in evidence
        ]


class FailingExplainer:
    def explain(self, _evidence: list[ChoiceEvidence]) -> list[ChoiceExplanation]:
        raise ExplanationError("Groq explanation failed with HTTP 429")


def _request() -> TripRequest:
    return TripRequest(
        origin="ORD",
        destination="COK",
        departure_date=date(2026, 11, 10),
        checked_bags=2,
        field_sources={
            "origin": InputSource.FORM,
            "destination": InputSource.FORM,
            "departure_date": InputSource.FORM,
            "checked_bags": InputSource.FORM,
        },
    )


def _provider() -> FixtureIgnavProvider:
    observation = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
    return FixtureIgnavProvider(observation)


def test_explanation_node_receives_prepared_evidence_after_choice_construction() -> None:
    explainer = RecordingExplainer()
    graph = build_flight_decision_graph(_provider(), explainer)

    state = FlightDecisionState.model_validate(graph.invoke({"request": _request()}))

    assert len(explainer.evidence) == 3
    assert len(state.choice_explanations) == 3
    assert state.explanation_failure is None
    assert [event.node for event in state.trace] == [
        "retrieve_offers",
        "normalize_offers",
        "reconcile_offers",
        "analyze_offers",
        "construct_choice_set",
        "explain_choices",
    ]
    assert state.trace[-1].writes == ("choice_explanations",)


def test_explanation_failure_preserves_deterministic_choices() -> None:
    graph = build_flight_decision_graph(_provider(), FailingExplainer())

    state = FlightDecisionState.model_validate(graph.invoke({"request": _request()}))

    assert len(state.choice_assignments) == 3
    assert state.choice_explanations == []
    assert state.explanation_failure == "Groq explanation failed with HTTP 429"
    assert state.trace[-1].node == "explain_choices"
    assert state.trace[-1].writes == ("explanation_failure",)

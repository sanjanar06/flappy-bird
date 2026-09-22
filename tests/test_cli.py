import json
from io import StringIO
from pathlib import Path

from flappy_bird.cli import main
from flappy_bird.explanations import ChoiceEvidence, ChoiceExplanation
from flappy_bird.models import InputSource, TripRequest
from flappy_bird.providers.ignav import (
    IgnavObservation,
    IgnavProviderError,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ignav_ord_cok_2026-11-10.json"
SEARCH_ARGS = [
    "search",
    "--origin",
    "ORD",
    "--destination",
    "COK",
    "--date",
    "2026-11-10",
    "--checked-bags",
    "2",
]


class CapturingProvider:
    def __init__(self, observation: IgnavObservation) -> None:
        self.observation = observation
        self.request: TripRequest | None = None

    def retrieve(self, request: TripRequest) -> IgnavObservation:
        self.request = request
        return self.observation


class FailingProvider:
    def retrieve(self, _request: TripRequest) -> IgnavObservation:
        raise IgnavProviderError("Ignav request failed with HTTP 401", status_code=401)


class FixtureExplainer:
    def explain(self, evidence: list[ChoiceEvidence]) -> list[ChoiceExplanation]:
        return [
            ChoiceExplanation(
                offer_id=item.offer_id,
                summary="A grounded explanation.",
                tradeoff="A grounded tradeoff.",
            )
            for item in evidence
        ]


def test_cli_runs_fixture_graph_and_renders_normalized_choices() -> None:
    observation = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
    provider = CapturingProvider(observation)
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(SEARCH_ARGS, provider=provider, stdout=stdout, stderr=stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert provider.request is not None
    assert provider.request.field_sources["origin"] is InputSource.COMMAND_LINE
    assert provider.request.field_sources["max_stops"] is InputSource.DEFAULT
    rendered = stdout.getvalue()
    assert "3 flight choices" in rendered
    assert "Lowest apparent price" in rendered
    assert "Lowest connection risk" in rendered
    assert "Shortest total travel time" in rendered
    assert "$1,048.00 USD" in rendered
    assert "$1,487.00 USD" in rendered
    assert "Protection unknown" in rendered
    assert "Self-transfer" in rendered


def test_cli_returns_safe_error_for_provider_failure() -> None:
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(SEARCH_ARGS, provider=FailingProvider(), stdout=stdout, stderr=stderr)

    assert exit_code == 2
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "Flight search failed: Ignav request failed with HTTP 401\n"


def test_cli_renders_optional_validated_explanations() -> None:
    observation = IgnavObservation.model_validate(json.loads(FIXTURE_PATH.read_text()))
    stdout = StringIO()
    stderr = StringIO()

    exit_code = main(
        SEARCH_ARGS,
        provider=CapturingProvider(observation),
        explainer=FixtureExplainer(),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert "Explanation: A grounded explanation." in stdout.getvalue()
    assert "Tradeoff: A grounded tradeoff." in stdout.getvalue()

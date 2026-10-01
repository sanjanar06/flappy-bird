# Flappy Bird

Flappy Bird is a personal learning project for building an observable, agentic flight-decision workflow. It compares flight offers, makes uncertainty visible, and presents several explainable choices. It never books or pays for a flight; the traveler makes the final choice.

## Current implementation

The first implementation is deliberately small:

- one-way trips only;
- one adult in economy;
- a fixed departure date;
- airport-to-airport search;
- at most one stop;
- live Ignav search and an optional SerpApi Google Flights adapter;
- a natural-language CLI path with deterministic field validation;
- several choices, never an automatic winner.

Both CLI commands run the LangGraph decision path. Results show segment-level carriers,
airports, local times, connections, and unknown protection or baggage information.
The provider-neutral graph can combine normalized providers in parallel, reconcile
identical itineraries, preserve source observations, and report partial provider failures.

The SerpApi adapter requires an airport-to-IANA-timezone mapping because its documented
segment timestamps are local wall-clock values without UTC offsets. It does not infer
baggage fees or connection protection.

## Local setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
uv run pytest
uv run flappy-bird search --origin ORD --destination COK --date 2026-11-10 --checked-bags 2
uv run flappy-bird ask "Find ORD to COK on November 10, 2026 with two checked bags"
```

The `ask` command requires `GROQ_API_KEY`; live Ignav search requires `IGNAV_API_KEY`.
The FastAPI app is created with injected provider adapters through
`flappy_bird.api.create_app`, which keeps credentials and provider selection outside
the route handler.

## Scope boundaries

- No booking redirects, payment, ticket issuance, or order management.
- A form and natural-language request are equal entry points; both create the same typed request.
- Provider data is normalized and reconciled before it is compared.
- Unknown safety or protection details remain unknown; they are never assumed favorable.

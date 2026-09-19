# Flappy Bird

Flappy Bird is a personal learning project for building an observable, agentic flight-decision workflow. It compares flight offers, makes uncertainty visible, and presents several explainable choices. It never books or pays for a flight; the traveler makes the final choice.

## First vertical slice

The first implementation is deliberately small:

- one-way trips only;
- one adult in economy;
- a fixed departure date;
- airport-to-airport search;
- at most one stop;
- fixture data before live providers;
- several choices, never an automatic winner.

The initial code defines the stable language shared by later steps: a typed `TripRequest`, source provenance for every input field, and normalized flight-offer contracts. The next slice will add a fixture provider and the first LangGraph path.

## Local setup

This project uses Python 3.12 and `uv`.

```bash
uv sync
uv run pytest
```

## Scope boundaries

- No booking redirects, payment, ticket issuance, or order management.
- A form and natural-language request are equal entry points; both create the same typed request.
- Provider data is normalized and reconciled before it is compared.
- Unknown safety or protection details remain unknown; they are never assumed favorable.


# Flappy Bird project instructions

## Purpose and current boundary

Flappy Bird is a personal learning project for an observable flight decision-support workflow. It compares several evidence-backed choices; it does not book, pay for, issue, or manage travel.

The active first vertical slice is deliberately narrow: one-way, one adult, economy, a fixed departure date, airport-to-airport, and at most one stop. Use fixture data until the provider adapter and evaluation boundary are ready.

Treat the architecture as a learning artifact. Explain non-obvious choices in plain language when the user asks, and preserve the distinction between confirmed product behavior, an experiment, and a later idea.

## Core invariants

- Form and natural-language input are equal entry paths into the same `TripRequest`.
- Preserve field-level input provenance. An inferred value must never masquerade as an explicit user choice.
- The traveler owns the final choice. Return a small, meaningfully different choice set; do not silently select a winner.
- Normalize each provider response before comparing offers. Preserve provider, retrieval time, and disagreements.
- Treat safety-critical missing information as `UNKNOWN`, never as favorable. This includes ticket protection and self-transfer status.
- A provider-priced itinerary is atomic. When a leg is pinned or revised, search compatible whole itineraries; never combine independent legs and retain an old bundled price.
- Keep external credentials in environment variables or `.env`, never in code, fixtures, prompts, traces, or commits.

## Code and verification

- Use Python 3.12 and `uv`.
- Put application code in `src/flappy_bird/` and tests in `tests/`.
- Use Pydantic models at boundaries: user input, provider payloads, normalized offers, graph state, and output.
- Prefer deterministic calculations for durations, stops, layovers, hard filters, risk labels, matching, and provenance. An LLM may interpret language, ask a decision-changing clarification, or write a grounded explanation.
- After a behavior change, run `uv run ruff check .` and the smallest relevant `uv run pytest ...` suite. Run the full test suite before committing.
- Keep fixtures versioned, small, sanitized, and intentionally varied. Do not depend on a live provider in unit tests.
- Add dependencies only when the current implementation requires them; explain why in the change summary.

## LangGraph and observability

- Keep state typed and explicit. A node must have a clear input, output, and transition reason.
- Record node name, state transition, provider/tool call, retry, latency, error, and degraded-mode status in the trace design.
- Ask a clarification only when plausible answers can change the displayed choice set or ordering. Never turn the product into a survey.
- A provider failure should yield a labeled partial/degraded result when another provider can still answer.

## Documentation and scope

- Keep project documentation about the product and code, not a transcript of conversation history.
- Do not update Notion, push to GitHub, call live providers, or add credentials unless the user explicitly asks.
- Keep P1 UI, P3 historical reliability, hosting, and an MCP server out of a P0 change unless the task explicitly includes them.

## Code review rules

Flag or reject changes that:

- present an `UNKNOWN` protection or reliability value as safe, protected, or on time;
- drop provider provenance, observed prices, retrieval timestamps, or provider conflicts during normalization/reconciliation;
- allow a hard constraint to be outweighed by price or a model score;
- splice legs from separate priced itineraries;
- make a live provider call part of deterministic tests;
- add a model-generated factual claim without a field, calculation, user statement, or labeled assumption supporting it;
- introduce booking, payment, or a system-owned “best flight” decision.

## Repository-local Codex skills

- Use `$flight-evals` when adding or changing fixtures, provider normalization, choice-set behavior, or evaluation coverage.
- Use `$langgraph-changes` when adding or changing graph state, nodes, conditional edges, checkpoints, or trace behavior.


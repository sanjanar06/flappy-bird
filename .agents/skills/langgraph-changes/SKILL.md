---
name: langgraph-changes
description: "Safely evolve Flappy Bird's LangGraph state, nodes, conditional edges, checkpoints, and traces without weakening user control or deterministic safeguards."
---

# Langgraph Changes

Use this skill when a change affects the LangGraph workflow or its state. Do not use it for isolated model or fixture changes that leave graph behavior untouched.

## Outcome

The graph remains inspectable: a reader can tell what entered a node, what changed, why the next edge was selected, and what evidence supports the output.

## Before changing the graph

1. Name the user-visible behavior that is changing.
2. Identify the typed state fields read and written by each affected node.
3. State whether the change can trigger a provider search, reuse a snapshot, ask a question, or return a degraded result.
4. Add or update a focused graph test before relying on a live provider or model.

## Graph rules

- Keep request parsing, provider retrieval, normalization, reconciliation, hard filtering, choice construction, clarification, and presentation as distinguishable responsibilities.
- Conditional edges must be based on explicit state and have a traceable reason.
- A human checkpoint is appropriate only for a missing required fact or a clarification that can materially change choices.
- Preserve field provenance and explicit user overrides across every revision loop.
- Rerank from a compatible valid snapshot when possible; a pinned leg must create a whole-itinerary constraint, not an offer splice.
- Capture provider errors in state and allow a labeled partial result when possible.
- Generated explanations must consume prepared evidence; they must not decide protection, fare facts, or provider agreement.

## Verification

Run `uv run ruff check .` and focused graph tests after each behavior change. The full test suite must pass before committing. If a change adds a node or edge, include a test that reaches it and a trace assertion or equivalent observable transition check.

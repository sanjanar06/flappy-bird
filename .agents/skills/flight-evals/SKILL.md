---
name: flight-evals
description: "Create or change sanitized flight fixtures and evaluations when work affects provider normalization, risk rules, reconciliation, or the displayed choice set."
---

# Flight Evals

Use this skill for fixture providers, normalized-offer scenarios, deterministic tests, and evaluation cases. Do not use it for generic Python refactors that do not affect decision behavior.

## Outcome

Every scenario should make one decision-support behavior observable and testable. A good fixture contains enough evidence to explain why an offer is shown, filtered, labeled uncertain, or grouped with another provider observation.

## Workflow

1. State the scenario in plain language before creating data: user request, provider observations, intended risk/provenance facts, and expected choice-set result.
2. Use sanitized, deterministic fixture data. Include provider name, retrieval time, and explicit unknowns.
3. Assert objective behavior in tests: schema validation, normalized stops/durations, matching, preserved provenance, hard-constraint enforcement, and degraded-provider behavior.
4. For choice sets, assert meaningful differences and expected inclusion/exclusion rather than exact presentation wording.
5. Run `uv run ruff check .` and the relevant `uv run pytest ...` tests.

## Domain guardrails

- Do not manufacture facts that a provider would not have supplied. Missing protection, baggage, or reliability remains `UNKNOWN`.
- Do not write fixture data that implies independently priced legs can be combined.
- Prefer at least one tradeoff scenario over several nearly identical cheap offers.
- A provider timeout/failure fixture must prove the result is labeled partial or degraded, not silently complete.
- Keep model-quality tests separate from deterministic assertions. Do not make unit tests depend on live model output or exact generated prose.

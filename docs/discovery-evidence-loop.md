# Discovery Evidence Loop

## Goal

Add a small, testable discovery layer to Synapse-X: organize competing hypotheses, summarize supplied evidence, expose same-claim conflicts, and suggest a next test.

## Components

- discovery_engine/core.py: dependency-free evidence analysis primitives.
- tests/test_discovery_engine.py: unit tests for ranking, missing evidence, contradiction detection, next-test triage, and invalid inputs.
- skills/discovery-research/SKILL.md: OpenClaw-compatible workflow instructions. This is a skill, not a runtime plugin.
- Future adapter: translate agent output into structured input and return a structured report.

## Score semantics

descriptive_score = (support_count - refute_count) / (support_count + refute_count)

Neutral records are excluded. A score of 1 means all counted records support the hypothesis; -1 means all counted records refute it; 0 means equal counts. This is not a probability. The current version does not weight source quality, source dependence, or evidence strength.

## Local test

Run from the repository root:

    python -m unittest discover -s tests -v

## Acceptance criteria

- Invalid stances and unknown hypothesis IDs fail loudly.
- No-evidence hypotheses are labeled insufficient evidence, not false.
- Opposite stances are surfaced only when they share an explicit normalized claim key.
- A next-test suggestion includes its reason and caveat.
- No claim of model improvement is made until a baseline experiment is run.

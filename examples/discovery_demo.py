"""Run a small, transparent evidence-triage example from the repository root."""
import json
from discovery_engine import analyze_hypotheses, find_contradictions, suggest_next_test

hypotheses = [
    {"id": "h1", "claim": "The cache reduces repeated computation."},
    {"id": "h2", "claim": "The observed speedup is caused by measurement noise."},
]
evidence = [
    {"hypothesis_id": "h1", "stance": "support", "source": "benchmark-A", "note": "Repeated run shows lower latency."},
    {"hypothesis_id": "h1", "stance": "refute", "source": "benchmark-B", "note": "Cold-start run shows no improvement."},
    {"hypothesis_id": "h2", "stance": "support", "source": "measurement-review", "note": "Sample size is small."},
]
observations = [
    {"claim_key": "cache-improves-latency", "stance": "support", "source": "benchmark-A"},
    {"claim_key": "cache-improves-latency", "stance": "refute", "source": "benchmark-B"},
]
analysis = analyze_hypotheses(hypotheses, evidence)
report = {
    "hypothesis_analysis": analysis,
    "contradictions": find_contradictions(observations),
    "suggested_next_test": suggest_next_test(analysis),
    "interpretation_note": "Scores summarize supplied evidence counts; they are not probabilities or proof.",
}
print(json.dumps(report, indent=2))

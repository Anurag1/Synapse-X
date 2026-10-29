import unittest
from discovery_engine import analyze_hypotheses, find_contradictions, suggest_next_test

class DiscoveryEngineTests(unittest.TestCase):
    def test_rank_and_contested_status(self):
        hypotheses = [{"id": "h1", "claim": "A"}, {"id": "h2", "claim": "B"}]
        evidence = [
            {"hypothesis_id": "h1", "stance": "support", "source": "s1"},
            {"hypothesis_id": "h1", "stance": "refute", "source": "s2"},
            {"hypothesis_id": "h2", "stance": "support", "source": "s3"},
        ]
        results = analyze_hypotheses(hypotheses, evidence)
        self.assertEqual(results[0]["id"], "h2")
        self.assertEqual(results[0]["descriptive_score"], 1.0)
        self.assertEqual(results[1]["status"], "contested")

    def test_no_evidence_is_not_false(self):
        result = analyze_hypotheses([{"id": "h1", "claim": "A"}], [])[0]
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["descriptive_score"])

    def test_unknown_hypothesis_rejected(self):
        with self.assertRaises(ValueError):
            analyze_hypotheses([{"id": "h1", "claim": "A"}],
                [{"hypothesis_id": "missing", "stance": "support"}])

    def test_explicit_contradiction(self):
        conflicts = find_contradictions([
            {"claim_key": "claim-a", "stance": "support", "source": "a"},
            {"claim_key": "claim-a", "stance": "refute", "source": "b"},
            {"claim_key": "claim-b", "stance": "support", "source": "c"},
        ])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["status"], "unresolved_contradiction")

    def test_next_test_prefers_no_evidence(self):
        analysis = analyze_hypotheses(
            [{"id": "known", "claim": "Known"}, {"id": "unknown", "claim": "Unknown"}],
            [{"hypothesis_id": "known", "stance": "support"}])
        self.assertEqual(suggest_next_test(analysis)["hypothesis_id"], "unknown")

    def test_invalid_stance_rejected(self):
        with self.assertRaises(ValueError):
            analyze_hypotheses([{"id": "h1", "claim": "A"}],
                [{"hypothesis_id": "h1", "stance": "maybe"}])

if __name__ == "__main__":
    unittest.main()

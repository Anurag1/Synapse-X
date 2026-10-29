import json
import unittest
from pathlib import Path

from discovery_engine import analyze_hypotheses


class DiscoveryFixtureTests(unittest.TestCase):
    def test_all_declared_fixture_expectations(self):
        fixture_path = Path(__file__).resolve().parents[1] / "benchmarks" / "discovery_cases.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["version"], 1)
        for case in fixture["cases"]:
            with self.subTest(case=case["id"]):
                results = analyze_hypotheses(case["hypotheses"], case["evidence"])
                actual = {row["id"]: row["status"] for row in results}
                self.assertEqual(actual, case["expected_status_by_id"])


if __name__ == "__main__":
    unittest.main()

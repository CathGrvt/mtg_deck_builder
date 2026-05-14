import unittest

from eval.vertex_release_gate import GateThresholds, evaluate_gate


class VertexReleaseGateTests(unittest.TestCase):
    def test_gate_passes_when_metrics_meet_thresholds(self):
        rows = [
            {
                "metrics": {
                    "groundedness": 0.8,
                    "faithfulness": 0.4,
                    "citation_precision": 0.9,
                }
            },
            {
                "metrics": {
                    "groundedness": 0.7,
                    "faithfulness": 0.3,
                    "citation_precision": 0.8,
                }
            },
        ]
        report = evaluate_gate(rows, GateThresholds())
        self.assertTrue(report["gate_pass"])

    def test_gate_fails_when_groundedness_is_low(self):
        rows = [
            {
                "metrics": {
                    "groundedness": 0.2,
                    "faithfulness": 0.8,
                    "citation_precision": 0.9,
                }
            }
        ]
        report = evaluate_gate(rows, GateThresholds())
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["pass_groundedness"])


if __name__ == "__main__":
    unittest.main()

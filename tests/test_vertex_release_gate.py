import unittest

from eval.vertex_release_gate import GateThresholds, evaluate_gate


class VertexReleaseGateTests(unittest.TestCase):
    def test_gate_passes_when_metrics_meet_thresholds(self):
        rows = [
            {
                "metrics": {
                    "groundedness": 0.8,
                    "faithfulness": 0.4,
                    "topic_relevance": 0.2,
                    "citation_precision": 0.9,
                }
            },
            {
                "metrics": {
                    "groundedness": 0.7,
                    "faithfulness": 0.3,
                    "topic_relevance": 0.15,
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
                    "topic_relevance": 0.4,
                    "citation_precision": 0.9,
                }
            }
        ]
        report = evaluate_gate(rows, GateThresholds())
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["pass_groundedness"])

    def test_gate_fails_when_topic_relevance_is_low(self):
        rows = [
            {
                "metrics": {
                    "groundedness": 0.9,
                    "faithfulness": 0.5,
                    "topic_relevance": 0.01,
                    "citation_precision": 0.95,
                }
            }
        ]
        report = evaluate_gate(rows, GateThresholds())
        self.assertFalse(report["gate_pass"])
        self.assertFalse(report["pass_topic_relevance"])


if __name__ == "__main__":
    unittest.main()

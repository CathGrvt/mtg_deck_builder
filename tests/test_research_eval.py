import tempfile
import unittest
from pathlib import Path

from research_pipeline.eval.dataset import load_eval_cases
from research_pipeline.eval.metrics import classify_failure, evaluate_report


class EvalMetricsTests(unittest.TestCase):
    def test_metrics_detect_valid_citations(self):
        report = {
            "topic": "How efficient is Lightning Bolt as interaction in red decks?",
            "summary": "Summary",
            "claims": [
                {
                    "claim": "Lightning Bolt deals 3 damage and is efficient interaction.",
                    "citations": [
                        {
                            "doc_id": "card::lightning-bolt",
                            "chunk_id": "card::lightning-bolt::chunk-000",
                        }
                    ],
                    "confidence": 0.8,
                }
            ],
            "open_questions": [],
        }
        retrieved_chunks = [
            {
                "doc_id": "card::lightning-bolt",
                "chunk_id": "card::lightning-bolt::chunk-000",
                "source": "card_db",
                "title": "Lightning Bolt",
                "text": "Lightning Bolt is efficient interaction and deals 3 damage to any target.",
                "score": 0.9,
                "metadata": {},
            }
        ]

        metrics = evaluate_report(report, retrieved_chunks)
        self.assertEqual(metrics["claim_count"], 1)
        self.assertEqual(metrics["valid_citations"], 1)
        self.assertGreater(metrics["citation_precision"], 0.0)
        self.assertGreater(metrics["topic_relevance"], 0.0)

        failure_type, _ = classify_failure(metrics)
        self.assertEqual(failure_type, "ok")

    def test_failure_classifies_off_topic_claims(self):
        report = {
            "topic": "How much graveyard hate should commander decks run?",
            "summary": "Summary",
            "claims": [
                {
                    "claim": "Oracle text: target creature gets +3/+3 until end of turn.",
                    "citations": [
                        {
                            "doc_id": "card::giant-growth",
                            "chunk_id": "card::giant-growth::chunk-000",
                        }
                    ],
                    "confidence": 0.6,
                }
            ],
            "open_questions": [],
        }
        retrieved_chunks = [
            {
                "doc_id": "card::giant-growth",
                "chunk_id": "card::giant-growth::chunk-000",
                "source": "card_db",
                "title": "Giant Growth",
                "text": "Oracle text: target creature gets +3/+3 until end of turn.",
                "score": 0.9,
                "metadata": {},
            }
        ]

        metrics = evaluate_report(report, retrieved_chunks)
        failure_type, _ = classify_failure(metrics)
        self.assertEqual(failure_type, "off_topic_claim")

    def test_dataset_loader_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.jsonl"
            path.write_text(
                '{"id":"case-a","topic":"Topic A"}\n'
                '{"id":"case-b","topic":"Topic B","priority":"high"}\n',
                encoding="utf-8",
            )

            cases = load_eval_cases(str(path))
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].id, "case-a")
            self.assertEqual(cases[1].metadata["priority"], "high")


if __name__ == "__main__":
    unittest.main()

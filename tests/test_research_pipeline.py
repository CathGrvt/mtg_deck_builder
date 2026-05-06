import json
import tempfile
import unittest
from pathlib import Path

from research_pipeline.graph import ResearchPipeline
from research_pipeline.models import DocumentChunk
from research_pipeline.retrieval.index import HybridRetrievalIndex


class ResearchPipelineTests(unittest.TestCase):
    def test_pipeline_generates_structured_report_with_validation(self):
        chunks = [
            DocumentChunk(
                doc_id="deck::boros-midrange",
                chunk_id="deck::boros-midrange::chunk-000",
                source="decklist",
                title="Boros Midrange",
                text=(
                    "Boros Midrange often uses low-cost removal and steady card advantage engines. "
                    "These lists include Lightning Bolt and card draw effects."
                ),
            ),
            DocumentChunk(
                doc_id="card::lightning-bolt",
                chunk_id="card::lightning-bolt::chunk-000",
                source="card_db",
                title="Lightning Bolt",
                text=(
                    "Lightning Bolt is an instant that deals 3 damage to any target and is often used "
                    "as efficient single-target interaction."
                ),
            ),
            DocumentChunk(
                doc_id="card::seasoned-pyromancer",
                chunk_id="card::seasoned-pyromancer::chunk-000",
                source="card_db",
                title="Seasoned Pyromancer",
                text=(
                    "Seasoned Pyromancer helps Boros-style decks convert cards in hand into value and board presence."
                ),
            ),
        ]

        index = HybridRetrievalIndex(chunks=chunks, enable_semantic=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = str(Path(tmp_dir) / "trace.jsonl")
            pipeline = ResearchPipeline(
                index=index,
                max_iterations=2,
                max_questions=3,
                top_k_per_query=2,
                use_langgraph=False,
                trace_path=trace_path,
            )

            output = pipeline.run("What does Boros data suggest about removal and card advantage?")

            report = output["report"]
            self.assertIn("summary", report)
            self.assertGreater(len(report.get("claims", [])), 0)
            self.assertIn("validation", report)
            self.assertGreaterEqual(report["validation"].get("total_claims", 0), 1)

            first_claim = report["claims"][0]
            self.assertGreaterEqual(len(first_claim.get("citations", [])), 1)

            with open(trace_path, "r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            events = {(row["event"], row["node"]) for row in rows}
            self.assertIn(("node_start", "planner"), events)
            self.assertIn(("node_start", "retriever"), events)
            self.assertIn(("node_start", "critic"), events)
            self.assertIn(("node_start", "writer"), events)
            self.assertIn(("node_start", "validator"), events)


if __name__ == "__main__":
    unittest.main()

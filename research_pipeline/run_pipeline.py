from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List

from research_pipeline.graph import build_pipeline_from_local_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MTG research pipeline for a single topic.")
    parser.add_argument("topic", help="Research topic.")
    parser.add_argument("--cards", default="data/commander_cards.csv", help="Path to card CSV.")
    parser.add_argument("--decks", default="current_commander_decks", help="Path to decklist directory.")
    parser.add_argument(
        "--meta-json",
        nargs="*",
        default=None,
        help="Optional meta JSON paths; defaults to json_outputs/*.json.",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum critic loops.")
    parser.add_argument("--max-questions", type=int, default=5, help="Max planner subquestions.")
    parser.add_argument("--top-k-per-query", type=int, default=5, help="Retriever top-k per query.")
    parser.add_argument("--disable-semantic", action="store_true", help="Disable semantic retrieval.")
    parser.add_argument("--disable-langgraph", action="store_true", help="Force manual loop execution.")
    parser.add_argument(
        "--out-dir",
        default="runs",
        help="Output directory for report artifacts.",
    )
    return parser.parse_args()


def _report_to_markdown(report: dict) -> str:
    lines: List[str] = [
        f"# Research Report: {report.get('topic', '')}",
        "",
        "## Summary",
        report.get("summary", ""),
        "",
        "## Claims",
    ]

    claims = report.get("claims", [])
    if not claims:
        lines.append("- No claims produced.")
    else:
        for claim in claims:
            claim_text = claim.get("claim", "")
            confidence = claim.get("confidence", 0.0)
            citations = claim.get("citations", [])
            citation_text = ", ".join(
                [f"{item.get('doc_id')}::{item.get('chunk_id')}" for item in citations]
            )
            lines.append(
                f"- {claim_text} (confidence={confidence}, citations=[{citation_text}])"
            )

    lines.extend(["", "## Open Questions"])
    for question in report.get("open_questions", []):
        lines.append(f"- {question}")

    lines.extend(["", "## Validation", "```json", json.dumps(report.get("validation", {}), indent=2), "```", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    trace_path = os.path.join(run_dir, "trace.jsonl")

    pipeline = build_pipeline_from_local_data(
        cards_csv=args.cards,
        decks_dir=args.decks,
        meta_json_paths=args.meta_json,
        max_iterations=args.max_iterations,
        max_questions=args.max_questions,
        top_k_per_query=args.top_k_per_query,
        enable_semantic=not args.disable_semantic,
        use_langgraph=not args.disable_langgraph,
        trace_path=trace_path,
    )

    output = pipeline.run(args.topic)
    report = output["report"]

    report_json = os.path.join(run_dir, "report.json")
    state_json = os.path.join(run_dir, "state.json")
    report_md = os.path.join(run_dir, "report.md")

    with open(report_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    with open(state_json, "w", encoding="utf-8") as handle:
        json.dump(output["state"], handle, indent=2)

    with open(report_md, "w", encoding="utf-8") as handle:
        handle.write(_report_to_markdown(report))

    print("Research pipeline run completed.")
    print(f"run_dir: {run_dir}")
    print(f"report_json: {report_json}")
    print(f"report_md: {report_md}")
    print(f"state_json: {state_json}")
    print(f"trace_jsonl: {trace_path}")


if __name__ == "__main__":
    main()

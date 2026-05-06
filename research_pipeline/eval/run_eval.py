from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List

from research_pipeline.eval.dataset import EvalCase, load_eval_cases
from research_pipeline.eval.metrics import classify_failure, evaluate_report
from research_pipeline.graph import build_pipeline_from_local_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval harness for the research pipeline.")
    parser.add_argument("--dataset", default="eval/topics.jsonl", help="Path to eval dataset (.jsonl/.json).")
    parser.add_argument("--cards", default="data/commander_cards.csv", help="Path to card CSV.")
    parser.add_argument("--decks", default="current_commander_decks", help="Path to decklist directory.")
    parser.add_argument(
        "--meta-json",
        nargs="*",
        default=None,
        help="Optional meta JSON paths; defaults to json_outputs/*.json.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Optional cap on number of eval cases.")
    parser.add_argument("--max-iterations", type=int, default=3, help="Maximum critic loops.")
    parser.add_argument("--max-questions", type=int, default=5, help="Max planner subquestions.")
    parser.add_argument("--top-k-per-query", type=int, default=5, help="Retriever top-k per query.")
    parser.add_argument("--disable-semantic", action="store_true", help="Disable semantic retrieval.")
    parser.add_argument("--disable-langgraph", action="store_true", help="Force manual loop execution.")
    parser.add_argument(
        "--out-dir",
        default="eval_runs",
        help="Directory where eval artifacts are written.",
    )
    return parser.parse_args()


def _ensure_output_dir(base_dir: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_summary(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# Eval Summary\n\nNo rows were produced.\n"

    groundedness = [float(row["metrics"]["groundedness"]) for row in rows]
    faithfulness = [float(row["metrics"]["faithfulness"]) for row in rows]
    citation_precision = [float(row["metrics"]["citation_precision"]) for row in rows]

    failure_counts: Dict[str, int] = {}
    for row in rows:
        failure_type = row["failure"]["type"]
        failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

    lines = [
        "# Eval Summary",
        "",
        f"Cases: {len(rows)}",
        f"Mean groundedness: {mean(groundedness):.4f}",
        f"Mean faithfulness: {mean(faithfulness):.4f}",
        f"Mean citation precision: {mean(citation_precision):.4f}",
        "",
        "## Failure Breakdown",
    ]

    for failure_type, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {failure_type}: {count}")

    lines.extend(["", "## Per Case", ""])

    for row in rows:
        lines.append(
            "- "
            f"{row['case_id']}: groundedness={row['metrics']['groundedness']:.4f}, "
            f"faithfulness={row['metrics']['faithfulness']:.4f}, "
            f"citation_precision={row['metrics']['citation_precision']:.4f}, "
            f"failure={row['failure']['type']}"
        )

    return "\n".join(lines) + "\n"


def _render_failure_analysis(rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# Failure Analysis",
        "",
        "This report is generated from the eval harness outputs.",
        "",
    ]

    failing = [row for row in rows if row["failure"]["type"] != "ok"]
    if not failing:
        lines.append("No failing cases were detected.")
        return "\n".join(lines) + "\n"

    for row in failing:
        lines.extend(
            [
                f"## {row['case_id']} - {row['topic']}",
                f"- Failure type: {row['failure']['type']}",
                f"- Why: {row['failure']['reason']}",
                f"- Groundedness: {row['metrics']['groundedness']:.4f}",
                f"- Faithfulness: {row['metrics']['faithfulness']:.4f}",
                f"- Citation precision: {row['metrics']['citation_precision']:.4f}",
                "- Suggested fix:",
            ]
        )

        failure_type = row["failure"]["type"]
        if failure_type == "retrieval_miss":
            lines.append("  Increase query diversity or improve corpus coverage for this topic.")
        elif failure_type == "bad_citation":
            lines.append("  Add stricter citation validation in writer and retry before final output.")
        elif failure_type == "hallucinated_claim":
            lines.append("  Lower writer abstraction and force shorter evidence-linked claims.")
        else:
            lines.append("  Inspect trace logs to identify node-level regressions.")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_eval(args: argparse.Namespace) -> Dict[str, str]:
    cases: List[EvalCase] = load_eval_cases(args.dataset)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        raise ValueError("No eval cases loaded from dataset.")

    run_dir = _ensure_output_dir(args.out_dir)
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

    rows: List[Dict[str, Any]] = []
    for case in cases:
        output = pipeline.run(case.topic)
        state = output["state"]
        report = output["report"]

        metrics = evaluate_report(
            report_payload=report,
            retrieved_chunks_payload=state.get("retrieved_chunks", []),
        )
        failure_type, failure_reason = classify_failure(metrics)

        rows.append(
            {
                "case_id": case.id,
                "topic": case.topic,
                "metadata": case.metadata,
                "metrics": metrics,
                "failure": {
                    "type": failure_type,
                    "reason": failure_reason,
                },
                "report": report,
                "iterations": state.get("iteration", 0),
            }
        )

    results_path = os.path.join(run_dir, "results.jsonl")
    summary_path = os.path.join(run_dir, "summary.md")
    failure_path = os.path.join(run_dir, "failure_analysis.md")

    _write_jsonl(results_path, rows)

    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(_render_summary(rows))

    with open(failure_path, "w", encoding="utf-8") as handle:
        handle.write(_render_failure_analysis(rows))

    return {
        "run_dir": run_dir,
        "results": results_path,
        "summary": summary_path,
        "failure_analysis": failure_path,
        "trace": trace_path,
    }


def main() -> None:
    args = parse_args()
    outputs = run_eval(args)
    print("Eval completed.")
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

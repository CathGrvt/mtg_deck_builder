from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List


@dataclass
class GateThresholds:
    groundedness_min: float = 0.65
    faithfulness_min: float = 0.20
    citation_precision_min: float = 0.70


def load_results(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def evaluate_gate(rows: List[Dict], thresholds: GateThresholds) -> Dict[str, float | bool]:
    if not rows:
        raise ValueError("No eval rows loaded for release gate.")

    groundedness = [float(row["metrics"]["groundedness"]) for row in rows]
    faithfulness = [float(row["metrics"]["faithfulness"]) for row in rows]
    citation_precision = [float(row["metrics"]["citation_precision"]) for row in rows]

    agg = {
        "mean_groundedness": round(mean(groundedness), 4),
        "mean_faithfulness": round(mean(faithfulness), 4),
        "mean_citation_precision": round(mean(citation_precision), 4),
    }
    agg["pass_groundedness"] = agg["mean_groundedness"] >= thresholds.groundedness_min
    agg["pass_faithfulness"] = agg["mean_faithfulness"] >= thresholds.faithfulness_min
    agg["pass_citation_precision"] = agg["mean_citation_precision"] >= thresholds.citation_precision_min
    agg["gate_pass"] = bool(
        agg["pass_groundedness"]
        and agg["pass_faithfulness"]
        and agg["pass_citation_precision"]
    )
    return agg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vertex-style release gate for offline eval artifacts.")
    parser.add_argument("--results", required=True, help="Path to eval results.jsonl")
    parser.add_argument("--groundedness-min", type=float, default=0.65)
    parser.add_argument("--faithfulness-min", type=float, default=0.20)
    parser.add_argument("--citation-precision-min", type=float, default=0.70)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_results(args.results)
    report = evaluate_gate(
        rows,
        GateThresholds(
            groundedness_min=float(args.groundedness_min),
            faithfulness_min=float(args.faithfulness_min),
            citation_precision_min=float(args.citation_precision_min),
        ),
    )
    print(json.dumps(report, indent=2))
    if not bool(report["gate_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

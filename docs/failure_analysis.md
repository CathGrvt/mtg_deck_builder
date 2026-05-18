# Failure Analysis Workflow

This project includes a generated failure-analysis artifact in every eval run (`eval_runs/<timestamp>/failure_analysis.md`).

## What to inspect

- `results.jsonl`: per-case metrics and the produced report.
- `summary.md`: aggregate groundedness, faithfulness, topic relevance, citation precision.
- `failure_analysis.md`: categorized failure types and suggested remediations.
- `trace.jsonl`: node-level execution trace (`planner`, `retriever`, `critic`, `writer`, `validator`).

## Failure taxonomy

- `retrieval_miss`: retrieval context does not cover produced claims.
- `bad_citation`: missing or invalid citation bindings.
- `hallucinated_claim`: weak textual support from cited evidence.
- `off_topic_claim`: claims are evidence-supported but not aligned to the question topic.

## Suggested analysis loop

1. Sort failures by count and severity.
2. Pull 3-5 representative traces per failure type.
3. Identify the primary breakage node.
4. Propose a concrete fix and rerun eval.
5. Record before/after metrics in `summary.md`.

# Eval Dataset and Failure Analysis

## Dataset Scope

- Source file: `eval/topics.jsonl`
- Current size: **24 cases**
- Case schema:
  - `id`: stable identifier
  - `topic`: natural-language research prompt
  - `category`: coarse problem family for stratification
  - `difficulty`: `easy` | `medium` | `hard`

### Category Coverage (Current Starter Set)

- `interaction`: 6
- `archetype`: 4
- `mana_base`: 3
- `resilience`: 3
- `card_advantage`: 2
- `composition`: 2
- `ramp`: 2
- `consistency`: 1
- `synergy`: 1

This distribution is intentionally mixed so aggregate metrics are less sensitive to one narrow prompt style.

## Running the Eval Harness

```bash
python run_research_eval.py \
  --dataset eval/topics.jsonl \
  --cards data/commander_cards.csv \
  --decks current_commander_decks
```

Artifacts are written to `eval_runs/<timestamp>/`:

- `results.jsonl`
- `summary.md`
- `failure_analysis.md`
- `trace.jsonl`

## Latest Published Baseline

A committed benchmark snapshot is available at:

- `eval/benchmarks/2026-05-15_rule_lexical_topic_guarded.json`

Metrics from that run:

- Mean groundedness: `0.9167`
- Mean faithfulness: `0.2148`
- Mean topic relevance: `0.2148`
- Mean citation precision: `0.9167`

Note: faithfulness scoring uses evidence support (best span overlap) multiplied by topic alignment, which prevents perfect scores from pure span-copy behavior. The prior `eval/benchmarks/2026-05-15_rule_lexical_span_scoring.json` snapshot is kept for history but is superseded.

## Failure Analysis Writeup Template

Use this template after each eval run to keep iteration history interview-ready:

```markdown
# Eval Iteration YYYY-MM-DD

## Run Metadata
- Dataset: eval/topics.jsonl (24 cases)
- Model/runtime mode: <rule-based | OpenAI model>
- Retrieval config: lexical=<x>, semantic=<y>, top_k=<k>

## Aggregate Metrics
- Mean groundedness: <value>
- Mean faithfulness: <value>
- Mean topic relevance: <value>
- Mean citation precision: <value>

## Failures by Type
- retrieval_miss: <count>
- bad_citation: <count>
- hallucinated_claim: <count>
- off_topic_claim: <count>

## Top Failure Clusters
1. <cluster name> - <count> cases
2. <cluster name> - <count> cases
3. <cluster name> - <count> cases

## Fixes Applied
1. <change> (files: <paths>)
2. <change> (files: <paths>)

## Before/After (if rerun)
- groundedness: <before> -> <after>
- faithfulness: <before> -> <after>
- topic relevance: <before> -> <after>
- citation precision: <before> -> <after>
```

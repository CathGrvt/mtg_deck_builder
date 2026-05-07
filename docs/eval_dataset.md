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
- Mean citation precision: <value>

## Failures by Type
- retrieval_miss: <count>
- bad_citation: <count>
- hallucinated_claim: <count>

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
- citation precision: <before> -> <after>
```

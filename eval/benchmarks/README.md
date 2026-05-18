# Published Eval Benchmarks

This directory stores committed benchmark snapshots so repository readers can see real measured evaluation outcomes (not only thresholds or templates).

## Latest baseline (May 15, 2026 UTC)

- Snapshot: `2026-05-15_rule_lexical_topic_guarded.json`
- Dataset: `eval/topics.jsonl` (24 cases)
- Runtime mode: `MTG_LLM_PROVIDER=rule`
- Retrieval mode: lexical-only (`--disable-semantic`)
- Validation mode: evidence-support overlap with topic-alignment guard

### Aggregate metrics

- Mean groundedness: **0.9167**
- Mean faithfulness: **0.2148**
- Mean topic relevance: **0.2148**
- Mean citation precision: **0.9167**

### Gate status

Using `eval/vertex_release_gate.py` defaults:

- groundedness gate: pass
- faithfulness gate: pass
- topic relevance gate: pass
- citation precision gate: pass
- overall gate: pass

## Previous baselines

- Snapshot: `2026-05-15_rule_lexical.json`
- Validation mode: legacy full-chunk Jaccard overlap
- Mean groundedness: **0.8250**
- Mean faithfulness: **0.1375**
- Mean citation precision: **1.0000**

- Snapshot: `2026-05-15_rule_lexical_span_scoring.json`
- Validation mode: span overlap only (superseded)
- Mean groundedness: **1.0000**
- Mean faithfulness: **1.0000**
- Mean citation precision: **1.0000**

The span-only snapshot is retained for history, but it is superseded because copied evidence spans could pass without topic relevance checks.

## Reproduce

```bash
PYTHONPATH=. MTG_LLM_PROVIDER=rule ./.venv/bin/python run_research_eval.py \
  --dataset eval/topics.jsonl \
  --cards data/commander_cards.csv \
  --decks current_commander_decks \
  --disable-semantic

./.venv/bin/python eval/vertex_release_gate.py \
  --results eval_runs/<timestamp>/results.jsonl
```

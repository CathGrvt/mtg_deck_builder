# Published Eval Benchmarks

This directory stores committed benchmark snapshots so repository readers can see real measured evaluation outcomes (not only thresholds or templates).

## Latest baseline (May 15, 2026 UTC)

- Snapshot: `2026-05-15_rule_lexical.json`
- Dataset: `eval/topics.jsonl` (24 cases)
- Runtime mode: `MTG_LLM_PROVIDER=rule`
- Retrieval mode: lexical-only (`--disable-semantic`)

### Aggregate metrics

- Mean groundedness: **0.8250**
- Mean faithfulness: **0.1375**
- Mean citation precision: **1.0000**

### Gate status

Using `eval/vertex_release_gate.py` defaults:

- groundedness gate: pass
- faithfulness gate: fail
- citation precision gate: pass
- overall gate: fail

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

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvalCase:
    id: str
    topic: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_eval_cases(path: str) -> List[EvalCase]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()

    if not raw:
        return []

    items: List[Dict[str, Any]] = []
    if path.endswith(".jsonl"):
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                items.append(payload)
    else:
        payload = json.loads(raw)
        if isinstance(payload, list):
            items = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            maybe_cases = payload.get("cases") or payload.get("items") or []
            items = [item for item in maybe_cases if isinstance(item, dict)]

    cases: List[EvalCase] = []
    for idx, item in enumerate(items):
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        case_id = str(item.get("id", f"case-{idx + 1:03d}"))
        metadata = {
            key: value
            for key, value in item.items()
            if key not in {"id", "topic"}
        }
        cases.append(EvalCase(id=case_id, topic=topic, metadata=metadata))

    return cases

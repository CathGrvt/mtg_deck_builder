from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from research_pipeline.io_resolver import resolve_uri_to_local_path


@dataclass
class EvalCase:
    id: str
    topic: str
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_eval_cases(
    path: str = "",
    cache_dir: str | None = None,
    storage_client=None,
) -> List[EvalCase]:
    requested_path = str(path or "").strip()
    if not requested_path:
        requested_path = os.getenv("MTG_EVAL_DATASET_URI", "eval/topics.jsonl")

    local_path = resolve_uri_to_local_path(
        requested_path,
        cache_dir=cache_dir,
        storage_client=storage_client,
    )
    if not os.path.isfile(local_path):
        raise FileNotFoundError(f"Eval dataset not found: {requested_path}")

    with open(local_path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()

    if not raw:
        return []

    items: List[Dict[str, Any]] = []
    if local_path.endswith(".jsonl"):
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

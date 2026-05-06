from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class TraceLogger:
    """
    JSONL trace logger with lightweight redaction/truncation.
    """

    def __init__(self, path: str):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def log(self, event: str, node: str, payload: Optional[Dict[str, Any]] = None) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "node": node,
            "payload": self._to_jsonable(payload or {}),
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_node_start(self, node: str, payload: Optional[Dict[str, Any]] = None) -> float:
        started = time.perf_counter()
        self.log("node_start", node, payload)
        return started

    def log_node_end(self, node: str, started: float, payload: Optional[Dict[str, Any]] = None) -> None:
        duration_ms = (time.perf_counter() - started) * 1000.0
        merged_payload = dict(payload or {})
        merged_payload["duration_ms"] = round(duration_ms, 3)
        self.log("node_end", node, merged_payload)

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, str):
            text = value.strip()
            if len(text) > 1200:
                return text[:1200] + "..."
            return text
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        return str(value)

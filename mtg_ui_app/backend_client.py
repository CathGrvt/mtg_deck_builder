from __future__ import annotations

from typing import Any, Dict, List, Sequence

import requests


def call_backend_json(
    backend_url: str,
    payload: Dict[str, Any],
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    url = str(backend_url or "").strip()
    if not url:
        raise ValueError("backend_url is required.")

    response = requests.post(
        url,
        json=payload,
        timeout=max(5, int(timeout_sec)),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Backend response was not a JSON object.")
    return data


def build_research_backend_payload(
    session_id: str,
    topic: str,
    max_iterations: int,
    max_questions: int,
    top_k_per_query: int,
    enable_semantic: bool,
    use_langgraph: bool,
) -> Dict[str, Any]:
    return {
        "session_id": str(session_id).strip(),
        "topic": str(topic).strip(),
        "max_iterations": int(max_iterations),
        "max_questions": int(max_questions),
        "top_k_per_query": int(top_k_per_query),
        "enable_semantic": bool(enable_semantic),
        "use_langgraph": bool(use_langgraph),
    }


def build_chat_backend_payload(
    session_id: str,
    question: str,
    history: Sequence[Dict[str, Any]],
    top_k: int,
) -> Dict[str, Any]:
    normalized_history: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = "assistant" if str(item.get("role")) == "assistant" else "user"
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized_history.append({"role": role, "content": content})

    return {
        "session_id": str(session_id).strip(),
        "question": str(question).strip(),
        "history": normalized_history,
        "top_k": int(top_k),
    }

from __future__ import annotations

from typing import Any, Dict, Sequence

import requests

from mtg_shared.text import extract_json_object


def _chat_completions_url(base_url: str) -> str:
    return str(base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"


def post_openai_chat_completions(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    base_url: str = "https://api.openai.com/v1",
    timeout_sec: int = 45,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    if not str(api_key or "").strip():
        raise ValueError("OpenAI API key is required.")

    response = requests.post(
        _chat_completions_url(base_url),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": str(model),
            "temperature": float(temperature),
            "messages": list(messages),
        },
        timeout=max(5, int(timeout_sec)),
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload
    return {}


def extract_first_message_content(payload: Dict[str, Any], empty_message: str = "") -> str:
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not isinstance(choices, list) or not choices:
        return str(empty_message)

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content or "").strip() or str(empty_message)


def chat_completion_content(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    base_url: str = "https://api.openai.com/v1",
    timeout_sec: int = 45,
    temperature: float = 0.0,
    empty_message: str = "",
) -> str:
    payload = post_openai_chat_completions(
        api_key=api_key,
        model=model,
        messages=messages,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=temperature,
    )
    return extract_first_message_content(payload, empty_message=empty_message)


def chat_completion_json_object(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    base_url: str = "https://api.openai.com/v1",
    timeout_sec: int = 45,
    ) -> Dict[str, Any]:
    content = chat_completion_content(
        api_key=api_key,
        model=model,
        messages=messages,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=0.0,
        empty_message="",
    )
    return extract_json_object(content)


def safe_chat_completion_content(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    base_url: str = "https://api.openai.com/v1",
    timeout_sec: int = 45,
    temperature: float = 0.0,
    empty_message: str = "",
    fallback_message: str = "",
) -> str:
    try:
        return chat_completion_content(
            api_key=api_key,
            model=model,
            messages=messages,
            base_url=base_url,
            timeout_sec=timeout_sec,
            temperature=temperature,
            empty_message=empty_message,
        )
    except Exception:
        return str(fallback_message)

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import requests

from research_pipeline.llm import AgentLLM, RuleBasedLLM, build_default_llm
from research_pipeline.models import RetrievedChunk


@dataclass
class LLMProviderConfig:
    provider: str = "openai"
    openai_model: str = "gpt-4o-mini"
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url: str = "https://api.openai.com/v1"
    timeout_sec: int = 45
    vertex_model: str = "gemini-2.5-flash"
    vertex_project: str = ""
    vertex_location: str = "us-central1"

    @classmethod
    def from_env(cls) -> "LLMProviderConfig":
        provider = os.getenv("MTG_LLM_PROVIDER", "openai").strip().lower() or "openai"
        return cls(
            provider=provider,
            openai_model=os.getenv("MTG_OPENAI_MODEL", cls.openai_model),
            openai_api_key_env=os.getenv("MTG_OPENAI_API_KEY_ENV", cls.openai_api_key_env),
            openai_base_url=os.getenv("MTG_OPENAI_BASE_URL", cls.openai_base_url),
            timeout_sec=max(5, int(os.getenv("MTG_LLM_TIMEOUT_SEC", str(cls.timeout_sec)))),
            vertex_model=os.getenv("MTG_VERTEX_MODEL", cls.vertex_model),
            vertex_project=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
            vertex_location=os.getenv("GOOGLE_CLOUD_LOCATION", cls.vertex_location).strip() or cls.vertex_location,
        )


class LLMProviderRuntime:
    def __init__(self, config: Optional[LLMProviderConfig] = None):
        self.config = config or LLMProviderConfig.from_env()

    def model_for_reporting(self) -> str:
        provider = self.config.provider
        if provider == "rule":
            return "rule-based"
        if provider == "vertex":
            return self.config.vertex_model
        return self.config.openai_model

    def build_research_llm(self) -> AgentLLM:
        return build_default_llm(
            model=self.config.openai_model,
            api_key_env=self.config.openai_api_key_env,
            base_url=self.config.openai_base_url,
            timeout_sec=self.config.timeout_sec,
            provider=self.config.provider,
            vertex_model=self.config.vertex_model,
            vertex_project=self.config.vertex_project,
            vertex_location=self.config.vertex_location,
        )

    def try_chat_answer(
        self,
        question: str,
        history: Sequence[Dict[str, str]],
        evidence: Sequence[RetrievedChunk],
        temperature: float = 0.2,
    ) -> Optional[str]:
        provider = self.config.provider
        if provider == "rule":
            return None

        context_chunks = [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "score": round(float(item.score), 4),
                "text": item.text[:420],
            }
            for item in evidence[:8]
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an MTG research assistant. Answer using only the provided context. "
                    "If uncertain, say so. End with a short 'Sources' line using provided chunk ids."
                ),
            }
        ]
        for turn in history[-6:]:
            role = "assistant" if str(turn.get("role")) == "assistant" else "user"
            content = str(turn.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content[:1200]})

        user_payload = {
            "question": question,
            "context_chunks": context_chunks,
            "instructions": [
                "Ground claims in context chunks.",
                "Do not invent citations.",
                "Keep answer concise and actionable.",
            ],
        }
        messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})

        try:
            if provider == "vertex":
                return self._chat_vertex(messages=messages, temperature=temperature)
            if provider == "openai":
                return self._chat_openai(messages=messages, temperature=temperature)
        except Exception:
            return None

        return None

    def _chat_openai(self, messages: Sequence[Dict[str, str]], temperature: float) -> str:
        api_key = os.getenv(self.config.openai_api_key_env, "")
        if not api_key:
            raise ValueError(f"Environment variable '{self.config.openai_api_key_env}' is not set.")

        response = requests.post(
            self.config.openai_base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.openai_model,
                "temperature": float(temperature),
                "messages": list(messages),
            },
            timeout=self.config.timeout_sec,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()

    def _chat_vertex(self, messages: Sequence[Dict[str, str]], temperature: float) -> str:
        try:
            import vertexai
            from vertexai.generative_models import GenerationConfig, GenerativeModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Vertex chat requested but Vertex SDK is unavailable. Install requirements-gcp.txt."
            ) from exc

        init_kwargs = {}
        if self.config.vertex_project:
            init_kwargs["project"] = self.config.vertex_project
        if self.config.vertex_location:
            init_kwargs["location"] = self.config.vertex_location
        if init_kwargs:
            vertexai.init(**init_kwargs)

        prompt_lines = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            if content:
                prompt_lines.append(f"{role}: {content}")
        prompt = "\n\n".join(prompt_lines)

        model = GenerativeModel(self.config.vertex_model)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=float(temperature)),
        )
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text

        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content is not None else []
            for part in parts:
                part_text = str(getattr(part, "text", "") or "").strip()
                if part_text:
                    return part_text
        return ""


def build_rule_based_chat_answer(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    rule_llm = RuleBasedLLM()
    report = rule_llm.write_report(topic=question, retrieved_chunks=retrieved, gaps=[])

    lines = [report.summary, "", "Evidence-backed points:"]
    for claim in report.claims[:4]:
        citations = ", ".join(f"{item.doc_id}::{item.chunk_id}" for item in claim.citations)
        lines.append(f"- {claim.claim} [{citations}]")

    return "\n".join(lines).strip()

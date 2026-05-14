from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import requests

from research_pipeline.models import Citation, Claim, RetrievedChunk, StructuredReport


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
        return {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}

    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        return {}
    return {}


def _keyword_tokens(text: str) -> List[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "their",
        "about",
        "what",
        "which",
        "when",
        "where",
        "how",
        "why",
        "mtg",
        "magic",
        "gathering",
        "deck",
    }
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9']+", text.lower())
        if len(token) >= 4 and token not in stopwords
    ]
    return tokens


def _has_vertex_sdk() -> bool:
    try:
        import vertexai  # noqa: F401
        from vertexai.generative_models import GenerativeModel  # noqa: F401
    except Exception:
        return False
    return True


class AgentLLM(ABC):
    @abstractmethod
    def plan_subquestions(
        self,
        topic: str,
        previous_gaps: Sequence[str],
        max_questions: int,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def critique(
        self,
        topic: str,
        subquestions: Sequence[str],
        retrieved_chunks: Sequence[RetrievedChunk],
        iteration: int,
        max_iterations: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def write_report(
        self,
        topic: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        gaps: Sequence[str],
    ) -> StructuredReport:
        raise NotImplementedError


class RuleBasedLLM(AgentLLM):
    """
    Deterministic fallback for local/test usage.
    """

    def plan_subquestions(
        self,
        topic: str,
        previous_gaps: Sequence[str],
        max_questions: int,
    ) -> List[str]:
        questions: List[str] = []

        if previous_gaps:
            for gap in previous_gaps:
                gap_text = gap.strip()
                if gap_text:
                    questions.append(gap_text)

        questions.extend(
            [
                f"What are the core facts and current signals about {topic}?",
                f"Which deck data or card evidence supports claims about {topic}?",
                f"What tradeoffs, caveats, or uncertainty exist for {topic}?",
            ]
        )

        keywords = _keyword_tokens(topic)
        for token in keywords[: max(0, max_questions - len(questions))]:
            questions.append(f"How does {token} influence {topic}?")

        deduped: List[str] = []
        seen = set()
        for question in questions:
            key = question.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(question)

        if not deduped:
            deduped.append(f"What should be researched about {topic}?")

        return deduped[:max(1, max_questions)]

    def critique(
        self,
        topic: str,
        subquestions: Sequence[str],
        retrieved_chunks: Sequence[RetrievedChunk],
        iteration: int,
        max_iterations: int,
    ) -> Dict[str, Any]:
        del topic
        unique_docs = len({chunk.doc_id for chunk in retrieved_chunks})
        unique_chunks = len({chunk.chunk_id for chunk in retrieved_chunks})

        unresolved: List[str] = []
        for question in subquestions:
            q_tokens = set(_keyword_tokens(question))
            if not q_tokens:
                continue
            covered = False
            for chunk in retrieved_chunks:
                chunk_tokens = set(_keyword_tokens(chunk.text))
                if q_tokens & chunk_tokens:
                    covered = True
                    break
            if not covered:
                unresolved.append(f"Need evidence for: {question}")

        enough_context = unique_chunks >= 5 and unique_docs >= 3
        needs_more_research = bool(unresolved) or not enough_context
        if iteration >= max_iterations:
            needs_more_research = False

        reason = "sufficient coverage"
        if needs_more_research:
            reason = (
                f"coverage still thin (docs={unique_docs}, chunks={unique_chunks}, unresolved={len(unresolved)})"
            )

        return {
            "needs_more_research": needs_more_research,
            "gaps": unresolved[:5],
            "reason": reason,
        }

    def write_report(
        self,
        topic: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        gaps: Sequence[str],
    ) -> StructuredReport:
        if not retrieved_chunks:
            return StructuredReport(
                topic=topic,
                summary="No evidence was retrieved for this topic.",
                claims=[],
                open_questions=list(gaps) or ["Gather more domain documents and rerun retrieval."],
            )

        sorted_chunks = sorted(retrieved_chunks, key=lambda item: item.score, reverse=True)
        claims: List[Claim] = []

        for chunk in sorted_chunks[:6]:
            sentence = chunk.text.strip().split(". ")[0].strip()
            if not sentence:
                continue
            if len(sentence) > 220:
                sentence = sentence[:220].rstrip() + "..."
            confidence = max(0.2, min(1.0, 0.3 + chunk.score))
            claims.append(
                Claim(
                    claim=sentence,
                    citations=[Citation(doc_id=chunk.doc_id, chunk_id=chunk.chunk_id)],
                    confidence=round(confidence, 3),
                )
            )

        if not claims:
            claims.append(
                Claim(
                    claim="Retrieved evidence exists but could not be converted to claims.",
                    citations=[Citation(doc_id=sorted_chunks[0].doc_id, chunk_id=sorted_chunks[0].chunk_id)],
                    confidence=0.3,
                )
            )

        top_titles = []
        seen_titles = set()
        for chunk in sorted_chunks:
            if chunk.title and chunk.title not in seen_titles:
                seen_titles.add(chunk.title)
                top_titles.append(chunk.title)
            if len(top_titles) >= 3:
                break

        if top_titles:
            summary = (
                f"Research on '{topic}' synthesized {len(claims)} grounded claims from "
                f"{len({chunk.doc_id for chunk in sorted_chunks})} sources, including {', '.join(top_titles)}."
            )
        else:
            summary = (
                f"Research on '{topic}' synthesized {len(claims)} grounded claims from "
                f"{len({chunk.doc_id for chunk in sorted_chunks})} sources."
            )

        open_questions = list(gaps)
        if not open_questions:
            open_questions = [
                "Validate claim robustness against additional corpus slices.",
                "Check whether more recent metagame updates change conclusions.",
            ]

        return StructuredReport(
            topic=topic,
            summary=summary,
            claims=claims,
            open_questions=open_questions,
        )


class OpenAIChatLLM(AgentLLM):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_sec: int = 45,
        fallback: AgentLLM | None = None,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.fallback = fallback or RuleBasedLLM()

    def _chat_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable '{self.api_key_env}' is not set."
            )

        response = requests.post(
            self.base_url + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "temperature": 0.0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(user_payload, ensure_ascii=False),
                    },
                ],
            },
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return {}

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = str(message.get("content", "")).strip()
        return _extract_json_object(content)

    def plan_subquestions(
        self,
        topic: str,
        previous_gaps: Sequence[str],
        max_questions: int,
    ) -> List[str]:
        payload = {
            "task": "plan_research_subquestions",
            "topic": topic,
            "previous_gaps": list(previous_gaps),
            "max_questions": max_questions,
            "output_schema": {
                "subquestions": ["string"],
            },
        }
        system_prompt = (
            "You are a senior research planner. Produce concise investigative subquestions "
            "that maximize evidence coverage. Return JSON only."
        )

        try:
            parsed = self._chat_json(system_prompt=system_prompt, user_payload=payload)
            raw_questions = parsed.get("subquestions", [])
            questions = [str(item).strip() for item in raw_questions if str(item).strip()]
            if questions:
                return questions[: max(1, max_questions)]
        except Exception:
            pass

        return self.fallback.plan_subquestions(topic, previous_gaps, max_questions)

    def critique(
        self,
        topic: str,
        subquestions: Sequence[str],
        retrieved_chunks: Sequence[RetrievedChunk],
        iteration: int,
        max_iterations: int,
    ) -> Dict[str, Any]:
        chunk_briefs = [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "score": round(float(item.score), 4),
                "text": item.text[:400],
            }
            for item in retrieved_chunks[:15]
        ]

        payload = {
            "task": "critique_retrieval_coverage",
            "topic": topic,
            "subquestions": list(subquestions),
            "retrieved_chunks": chunk_briefs,
            "iteration": iteration,
            "max_iterations": max_iterations,
            "output_schema": {
                "needs_more_research": "boolean",
                "gaps": ["string"],
                "reason": "string",
            },
        }

        system_prompt = (
            "You are a retrieval critic. Judge if evidence is sufficient for synthesis. "
            "Prefer conservative grounding. Return JSON only."
        )

        try:
            parsed = self._chat_json(system_prompt=system_prompt, user_payload=payload)
            needs_more = bool(parsed.get("needs_more_research", False))
            gaps = [str(item).strip() for item in parsed.get("gaps", []) if str(item).strip()]
            reason = str(parsed.get("reason", ""))
            if iteration >= max_iterations:
                needs_more = False
            return {
                "needs_more_research": needs_more,
                "gaps": gaps[:5],
                "reason": reason or "critic response",
            }
        except Exception:
            return self.fallback.critique(topic, subquestions, retrieved_chunks, iteration, max_iterations)

    def write_report(
        self,
        topic: str,
        retrieved_chunks: Sequence[RetrievedChunk],
        gaps: Sequence[str],
    ) -> StructuredReport:
        chunk_briefs = [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "score": round(float(item.score), 4),
                "text": item.text[:450],
            }
            for item in retrieved_chunks[:20]
        ]

        payload = {
            "task": "write_grounded_structured_report",
            "topic": topic,
            "context_chunks": chunk_briefs,
            "open_gaps": list(gaps),
            "output_schema": {
                "topic": "string",
                "summary": "string",
                "claims": [
                    {
                        "claim": "string",
                        "citations": [{"doc_id": "string", "chunk_id": "string"}],
                        "confidence": "number 0..1",
                    }
                ],
                "open_questions": ["string"],
            },
            "instructions": [
                "Every claim must include at least one citation from the provided chunk ids.",
                "Do not invent citations or sources.",
                "Return strict JSON only.",
            ],
        }

        system_prompt = (
            "You are a technical writer for retrieval-augmented reports. "
            "Only use supplied context. Return strict JSON."
        )

        try:
            parsed = self._chat_json(system_prompt=system_prompt, user_payload=payload)
            report = StructuredReport.from_dict(parsed)
            if report.topic and report.claims:
                if not report.open_questions:
                    report.open_questions = list(gaps)
                return report
        except Exception:
            pass

        return self.fallback.write_report(topic, retrieved_chunks, gaps)


class VertexChatLLM(OpenAIChatLLM):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        project: str = "",
        location: str = "us-central1",
        timeout_sec: int = 45,
        fallback: AgentLLM | None = None,
    ):
        super().__init__(
            model=model,
            api_key_env="__vertex_not_used__",
            base_url="https://vertex.invalid",
            timeout_sec=timeout_sec,
            fallback=fallback,
        )
        self.project = project
        self.location = location

    def _chat_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            import vertexai
            from vertexai.generative_models import GenerationConfig, GenerativeModel
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Vertex research LLM requested but Vertex SDK is unavailable."
            ) from exc

        init_kwargs = {}
        if self.project:
            init_kwargs["project"] = self.project
        if self.location:
            init_kwargs["location"] = self.location
        if init_kwargs:
            vertexai.init(**init_kwargs)

        prompt = "\n\n".join(
            [
                f"SYSTEM: {system_prompt}",
                "USER_PAYLOAD_JSON:",
                json.dumps(user_payload, ensure_ascii=False),
                "Return strict JSON only.",
            ]
        )
        model = GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(temperature=0.0),
        )

        content = str(getattr(response, "text", "") or "").strip()
        if not content:
            candidates = getattr(response, "candidates", []) or []
            for candidate in candidates:
                body = getattr(candidate, "content", None)
                parts = getattr(body, "parts", []) if body is not None else []
                for part in parts:
                    text = str(getattr(part, "text", "") or "").strip()
                    if text:
                        content = text
                        break
                if content:
                    break
        return _extract_json_object(content)


def build_default_llm(
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1",
    timeout_sec: int = 45,
    provider: str | None = None,
    vertex_model: str = "gemini-2.5-flash",
    vertex_project: str = "",
    vertex_location: str = "us-central1",
) -> AgentLLM:
    selected_provider = str(provider or os.getenv("MTG_LLM_PROVIDER", "openai")).strip().lower() or "openai"
    if selected_provider == "rule":
        return RuleBasedLLM()
    if selected_provider == "vertex":
        if _has_vertex_sdk():
            return VertexChatLLM(
                model=vertex_model,
                project=vertex_project,
                location=vertex_location,
                timeout_sec=timeout_sec,
            )
        return RuleBasedLLM()

    if os.getenv(api_key_env, ""):
        return OpenAIChatLLM(
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            timeout_sec=timeout_sec,
        )
    return RuleBasedLLM()

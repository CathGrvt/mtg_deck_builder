from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from gcp_agent_runtime.contracts import SafetyVerdict


@dataclass
class SafetyPolicyConfig:
    blocked_patterns: List[str] = field(
        default_factory=lambda: [
            r"ignore\s+all\s+previous\s+instructions",
            r"disable\s+safety",
            r"bypass\s+guardrails",
            r"jailbreak",
            r"prompt\s+injection",
            r"exfiltrat(e|ion)",
            r"malware",
            r"phishing",
            r"drop\s+table",
            r"sudo\s+rm\s+-rf",
        ]
    )
    warning_patterns: List[str] = field(
        default_factory=lambda: [
            r"api[_\s-]?key",
            r"token\s+leak",
            r"credential",
            r"private\s+data",
            r"secret",
        ]
    )
    allowed_modes: List[str] = field(default_factory=lambda: ["deck_recommendation", "research_copilot"])


class SafetyGateAgent:
    def __init__(self, config: SafetyPolicyConfig | None = None):
        self.config = config or SafetyPolicyConfig()
        self._blocked_re = [re.compile(pattern, flags=re.IGNORECASE) for pattern in self.config.blocked_patterns]
        self._warning_re = [re.compile(pattern, flags=re.IGNORECASE) for pattern in self.config.warning_patterns]

    def evaluate_request(self, text: str, mode: str) -> SafetyVerdict:
        reasons: List[str] = []
        risk_score = 0.0
        blocked = False

        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in self.config.allowed_modes:
            blocked = True
            risk_score = 1.0
            reasons.append(f"unsupported_mode:{normalized_mode or 'empty'}")

        query = str(text or "")
        for pattern in self._blocked_re:
            if pattern.search(query):
                blocked = True
                risk_score = 1.0
                reasons.append(f"blocked_pattern:{pattern.pattern}")

        if not blocked:
            warning_hits = [pattern.pattern for pattern in self._warning_re if pattern.search(query)]
            if warning_hits:
                risk_score = min(0.8, 0.2 + 0.2 * len(warning_hits))
                reasons.extend(f"warning_pattern:{item}" for item in warning_hits[:4])

        status = "blocked" if blocked else ("review" if risk_score > 0 else "allow")
        return SafetyVerdict(
            status=status,
            reasons=reasons,
            risk_score=risk_score,
            blocked=blocked,
        )

    def evaluate_output(self, text: str) -> SafetyVerdict:
        query = str(text or "")
        reasons: List[str] = []
        risk_score = 0.0
        blocked = False

        for pattern in self._blocked_re:
            if pattern.search(query):
                blocked = True
                risk_score = 1.0
                reasons.append(f"blocked_output_pattern:{pattern.pattern}")

        status = "blocked" if blocked else ("review" if risk_score > 0 else "allow")
        return SafetyVerdict(
            status=status,
            reasons=reasons,
            risk_score=risk_score,
            blocked=blocked,
        )

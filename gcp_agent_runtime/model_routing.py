from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, Optional


@dataclass
class ModelRoutingConfig:
    default_model: str = "gemini-2.5-flash"
    escalation_model: str = "gemini-2.5-pro"
    fallback_model: str = "gemini-2.5-flash"
    pro_escalation_complexity_threshold: float = 0.65
    pro_escalation_low_confidence_threshold: float = 0.45
    # Lifecycle guard defaults can be overridden with env var:
    # MODEL_EOL_GEMINI_2_5_PRO=YYYY-MM-DD
    model_end_of_life: Dict[str, str] = field(
        default_factory=lambda: {
            "gemini-2.5-pro": "2026-06-17",
        }
    )


@dataclass
class ModelSelection:
    model_id: str
    reason: str
    escalated: bool
    lifecycle_fallback: bool


class ModelLifecycleGuard:
    def __init__(
        self,
        config: Optional[ModelRoutingConfig] = None,
        now_fn=None,
    ):
        self.config = config or ModelRoutingConfig()
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc).date())

    def _env_override_eol(self, model_id: str) -> Optional[str]:
        key = "MODEL_EOL_" + model_id.upper().replace("-", "_").replace(".", "_")
        return os.getenv(key, "").strip() or None

    def _model_eol_date(self, model_id: str) -> Optional[date]:
        raw = self._env_override_eol(model_id) or self.config.model_end_of_life.get(model_id)
        if not raw:
            return None
        try:
            return date.fromisoformat(raw)
        except ValueError:
            return None

    def is_model_retired(self, model_id: str) -> bool:
        eol = self._model_eol_date(model_id)
        if eol is None:
            return False
        return self.now_fn() > eol

    def choose_model(
        self,
        complexity_score: float,
        predicted_confidence: float,
    ) -> ModelSelection:
        cfg = self.config
        wants_escalation = (
            complexity_score >= cfg.pro_escalation_complexity_threshold
            or predicted_confidence <= cfg.pro_escalation_low_confidence_threshold
        )
        candidate = cfg.escalation_model if wants_escalation else cfg.default_model
        reason = (
            "escalated_to_pro_for_complexity_or_low_confidence"
            if wants_escalation
            else "default_flash_path"
        )

        if self.is_model_retired(candidate):
            return ModelSelection(
                model_id=cfg.fallback_model,
                reason=f"{reason}; lifecycle_guard_fallback",
                escalated=wants_escalation,
                lifecycle_fallback=True,
            )

        return ModelSelection(
            model_id=candidate,
            reason=reason,
            escalated=wants_escalation,
            lifecycle_fallback=False,
        )

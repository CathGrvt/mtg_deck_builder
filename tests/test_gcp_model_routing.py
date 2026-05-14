import unittest
from datetime import date

from gcp_agent_runtime.model_routing import ModelLifecycleGuard, ModelRoutingConfig


class ModelRoutingTests(unittest.TestCase):
    def test_defaults_to_flash_for_simple_case(self):
        guard = ModelLifecycleGuard()
        selection = guard.choose_model(complexity_score=0.2, predicted_confidence=0.9)
        self.assertEqual(selection.model_id, "gemini-2.5-flash")
        self.assertFalse(selection.escalated)

    def test_escalates_to_pro_for_complexity(self):
        guard = ModelLifecycleGuard()
        selection = guard.choose_model(complexity_score=0.9, predicted_confidence=0.8)
        self.assertEqual(selection.model_id, "gemini-2.5-pro")
        self.assertTrue(selection.escalated)

    def test_lifecycle_guard_falls_back_when_pro_retired(self):
        cfg = ModelRoutingConfig(
            default_model="gemini-2.5-flash",
            escalation_model="gemini-2.5-pro",
            fallback_model="gemini-2.5-flash",
            model_end_of_life={"gemini-2.5-pro": "2026-06-17"},
        )
        guard = ModelLifecycleGuard(config=cfg, now_fn=lambda: date(2026, 6, 18))
        selection = guard.choose_model(complexity_score=0.9, predicted_confidence=0.2)
        self.assertEqual(selection.model_id, "gemini-2.5-flash")
        self.assertTrue(selection.lifecycle_fallback)


if __name__ == "__main__":
    unittest.main()

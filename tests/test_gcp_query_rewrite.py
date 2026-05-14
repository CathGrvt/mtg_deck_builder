import unittest

from gcp_agent_runtime.contracts import DeckRecommendationRequest
from gcp_agent_runtime.query_rewrite import QueryRewriteAgent


class QueryRewriteAgentTests(unittest.TestCase):
    def test_rewrite_is_deterministic_for_same_input(self):
        request = DeckRecommendationRequest(
            session_id="s1",
            user_query="Recommend a resilient Boros deck with card draw and interaction.",
            format="commander",
            colors=["W", "R"],
            archetype_hint="midrange",
            must_include=["Lightning Bolt", "Smothering Tithe"],
            must_exclude=[],
            mode="deck_recommendation",
        )
        agent = QueryRewriteAgent()

        plan_a = agent.build_retrieval_plan(request)
        plan_b = agent.build_retrieval_plan(request)

        self.assertEqual(plan_a.rewritten_queries, plan_b.rewritten_queries)
        self.assertEqual(plan_a.corpus_targets, plan_b.corpus_targets)

    def test_rewrites_include_diversity_for_constraints(self):
        request = DeckRecommendationRequest(
            session_id="s2",
            user_query="Build me a Boros commander list that can survive wipes.",
            format="commander",
            colors=["W", "R"],
            archetype_hint="control",
            must_include=["Teferi's Protection"],
            must_exclude=["Mana Crypt"],
            mode="deck_recommendation",
        )
        agent = QueryRewriteAgent()
        plan = agent.build_retrieval_plan(request)

        self.assertGreaterEqual(len(plan.rewritten_queries), 3)
        self.assertTrue(any("color identity" in q.lower() for q in plan.rewritten_queries))
        self.assertTrue(any("archetype" in q.lower() for q in plan.rewritten_queries))
        self.assertTrue(any("include cards" in q.lower() for q in plan.rewritten_queries))


if __name__ == "__main__":
    unittest.main()

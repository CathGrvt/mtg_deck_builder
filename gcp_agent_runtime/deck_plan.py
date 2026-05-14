from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from ai_deck_generator import DeckSpec, build_name_index, generate_deck_from_meta
from gcp_agent_runtime.contracts import DeckCitation, DeckRecommendationRequest, RetrievalBundle
from gcp_agent_runtime.model_routing import ModelLifecycleGuard, ModelSelection
from mtg_io import load_card_database, load_decklists_from_directory


@dataclass
class DeckPlanResult:
    summary: str
    recommended_decklist: List[str]
    key_claims: List[str]
    citations: List[DeckCitation]
    confidence: float
    model_selection: ModelSelection


@dataclass
class DeckPlanSettings:
    cards_csv: str = "data/commander_cards.csv"
    decks_dir: str = "current_commander_decks"
    max_claims: int = 4
    seed: int = 42


class DeckPlanAgent:
    def __init__(
        self,
        routing_guard: ModelLifecycleGuard,
        settings: DeckPlanSettings | None = None,
    ):
        self.routing_guard = routing_guard
        self.settings = settings or DeckPlanSettings()
        self._card_db = None
        self._decklists = None

    def _ensure_data(self):
        if self._card_db is None:
            if not os.path.isfile(self.settings.cards_csv):
                raise FileNotFoundError(f"Card CSV not found: {self.settings.cards_csv}")
            self._card_db = load_card_database(self.settings.cards_csv)

        if self._decklists is None:
            if not os.path.isdir(self.settings.decks_dir):
                self._decklists = {}
            else:
                self._decklists = load_decklists_from_directory(
                    self.settings.decks_dir,
                    include_command_zone=True,
                )

    @staticmethod
    def _complexity_score(request: DeckRecommendationRequest, confidence_hint: float) -> float:
        score = 0.0
        score += min(0.25, 0.04 * len(request.must_include))
        score += min(0.15, 0.03 * len(request.must_exclude))
        score += 0.15 if bool(request.archetype_hint) else 0.0
        score += 0.15 if len(request.colors) >= 3 else 0.05 * len(request.colors)
        score += 0.2 if len(request.user_query.split()) >= 16 else 0.08
        score += max(0.0, 0.2 - confidence_hint * 0.2)
        return max(0.0, min(1.0, score))

    def _build_claims_and_citations(
        self,
        request: DeckRecommendationRequest,
        bundle: RetrievalBundle,
    ) -> tuple[List[str], List[DeckCitation]]:
        claims: List[str] = []
        citations: List[DeckCitation] = []
        for item in bundle.chunks[: max(1, self.settings.max_claims)]:
            sentence = item.text.strip().split(". ")[0].strip()
            if not sentence:
                continue
            if len(sentence) > 220:
                sentence = sentence[:220].rstrip() + "..."
            claims.append(sentence)
            citations.append(DeckCitation.from_evidence(item))

        if not claims:
            claims = [
                "Current evidence is limited; the generated deck emphasizes baseline color and archetype fit."
            ]

        if not citations and bundle.chunks:
            citations.append(DeckCitation.from_evidence(bundle.chunks[0]))

        return claims, citations

    def plan_deck(
        self,
        request: DeckRecommendationRequest,
        bundle: RetrievalBundle,
        predicted_confidence: float,
    ) -> DeckPlanResult:
        self._ensure_data()

        complexity = self._complexity_score(request=request, confidence_hint=predicted_confidence)
        selection = self.routing_guard.choose_model(
            complexity_score=complexity,
            predicted_confidence=predicted_confidence,
        )

        spec = DeckSpec(
            format=request.format,
            colors=list(request.colors),
            archetype=request.archetype_hint,
            include_cards=list(request.must_include),
            exclude_cards=list(request.must_exclude),
        )
        deck = generate_deck_from_meta(
            card_db=self._card_db,
            decklists=self._decklists or {},
            spec=spec,
            seed=self.settings.seed,
        )

        # Ensure must-include semantics stay strong even if input names are noisy.
        if request.must_include:
            canonical_index = build_name_index(self._card_db)
            canonical_names = {name.lower(): name for name in canonical_index.keys()}
            required = [canonical_names.get(item.lower(), item) for item in request.must_include]
            missing = [item for item in required if item not in deck]
            for card_name in missing[:3]:
                deck.insert(0, card_name)
            max_size = spec.effective_size()
            deck = deck[:max_size]

        key_claims, citations = self._build_claims_and_citations(request=request, bundle=bundle)
        confidence = max(0.35, min(0.98, predicted_confidence + (0.05 if selection.escalated else 0.0)))

        summary = (
            f"Generated a {request.format} deck recommendation with {len(deck)} cards using "
            f"{selection.model_id}. "
            f"Evidence was synthesized from {len({item.doc_id for item in bundle.chunks})} sources."
        )

        return DeckPlanResult(
            summary=summary,
            recommended_decklist=deck,
            key_claims=key_claims,
            citations=citations,
            confidence=round(confidence, 4),
            model_selection=selection,
        )

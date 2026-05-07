import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from ai_deck_generator import (
    DeckSpec,
    build_name_index,
    generate_deck_from_meta,
    resolve_requested_card_names,
)


def _sample_card_db() -> pd.DataFrame:
    rows = [
        {
            "name": "Lathril, Blade of the Elves",
            "full_name": "Lathril, Blade of the Elves",
            "type_line": "Legendary Creature — Elf Noble",
            "cmc": 4,
            "oracle_text": "",
            "color_identity": ["B", "G"],
            "is_land": False,
        },
        {
            "name": "Elvish Mystic",
            "full_name": "Elvish Mystic",
            "type_line": "Creature — Elf Druid",
            "cmc": 1,
            "oracle_text": "",
            "color_identity": ["G"],
            "is_land": False,
        },
        {
            "name": "Forest",
            "full_name": "Forest",
            "type_line": "Basic Land — Forest",
            "cmc": 0,
            "oracle_text": "",
            "color_identity": ["G"],
            "is_land": True,
        },
    ]
    return pd.DataFrame(rows)


class CardNameResolutionTests(unittest.TestCase):
    def test_resolve_name_without_comma(self):
        name_index = build_name_index(_sample_card_db())
        resolved, missing = resolve_requested_card_names(
            ["Lathril Blade of the Elves"],
            name_index,
        )
        self.assertEqual(resolved, ["Lathril, Blade of the Elves"])
        self.assertEqual(missing, [])

    def test_merge_adjacent_tokens_from_comma_split(self):
        name_index = build_name_index(_sample_card_db())
        resolved, missing = resolve_requested_card_names(
            ["Lathril", "Blade of the Elves", "Elvish Mystic"],
            name_index,
        )
        self.assertEqual(
            resolved,
            ["Lathril, Blade of the Elves", "Elvish Mystic"],
        )
        self.assertEqual(missing, [])

    def test_generator_uses_resolved_include_cards(self):
        card_db = _sample_card_db()
        spec = DeckSpec(
            format="commander",
            target_size=2,
            land_ratio=0.5,
            include_cards=["Lathril", "Blade of the Elves"],
        )

        log_capture = io.StringIO()
        with redirect_stdout(log_capture):
            deck = generate_deck_from_meta(
                card_db=card_db,
                decklists={},
                spec=spec,
                seed=1,
            )

        logs = log_capture.getvalue()
        self.assertIn("Lathril, Blade of the Elves", deck)
        self.assertNotIn(
            "requested include card 'Lathril' not found in card database.",
            logs,
        )
        self.assertNotIn(
            "requested include card 'Blade of the Elves' not found in card database.",
            logs,
        )


if __name__ == "__main__":
    unittest.main()

import tempfile
import textwrap
import unittest
from pathlib import Path

import pandas as pd

from deck_analysis import AdvancedDeckAnalyzer
from mtg_io import (
    load_decklists_from_directory,
    normalize_card_name,
    parse_decklist_lines,
    safe_parse_list,
)


class DecklistParserTests(unittest.TestCase):
    def test_normalize_card_name_single_slash(self):
        self.assertEqual(
            normalize_card_name("Fire // Ice"),
            "Fire // Ice",
        )
        self.assertEqual(
            normalize_card_name("Fire / Ice"),
            "Fire // Ice",
        )

    def test_safe_parse_list(self):
        self.assertEqual(safe_parse_list("['W', 'U']"), ["W", "U"])
        self.assertEqual(safe_parse_list("[]"), [])
        self.assertEqual(safe_parse_list("W"), ["W"])
        self.assertEqual(safe_parse_list(None), [])
        self.assertEqual(safe_parse_list("[W, U]"), ["W", "U"])

    def test_parse_commander_sections(self):
        lines = textwrap.dedent(
            """\
            Commander
            1 Atraxa, Praetors' Voice

            Deck
            1 Sol Ring
            1 Arcane Signet
            """
        ).splitlines()

        sections = parse_decklist_lines(lines)
        self.assertEqual(len(sections["commanders"]), 1)
        self.assertEqual(sections["commanders"][0], "Atraxa, Praetors' Voice")
        self.assertEqual(len(sections["mainboard"]), 2)
        self.assertEqual(len(sections["sideboard"]), 0)

    def test_parse_standard_blank_line_sideboard(self):
        lines = textwrap.dedent(
            """\
            4 Card A
            2 Card B

            1 Side C
            2 Side D
            """
        ).splitlines()

        sections = parse_decklist_lines(lines)
        self.assertEqual(len(sections["mainboard"]), 6)
        self.assertEqual(len(sections["sideboard"]), 3)

    def test_directory_loader_uses_shared_parser(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / "Deck - UW Control.txt").write_text(
                "4 Card A\n4 Card B\n\n2 Side Card\n",
                encoding="utf-8",
            )
            (tmp_path / "Deck - Atraxa.txt").write_text(
                "Commander\n1 Atraxa, Praetors' Voice\n\nDeck\n2 Card C\n",
                encoding="utf-8",
            )

            decklists = load_decklists_from_directory(tmp_dir, include_command_zone=True)
            self.assertEqual(len(decklists["Deck - UW Control"]), 8)
            self.assertEqual(len(decklists["Deck - Atraxa"]), 3)
            self.assertIn("Atraxa, Praetors' Voice", decklists["Deck - Atraxa"])


class CommanderLegalityTests(unittest.TestCase):
    def _card_db(self) -> pd.DataFrame:
        rows = [
            {
                "name": "Commander One",
                "full_name": "Commander One",
                "layout": "normal",
                "mana_cost": "{1}{W}{W}",
                "cmc": 3,
                "type_line": "Legendary Creature — Human",
                "oracle_text": "Vigilance",
                "colors": ["W"],
                "color_identity": ["W"],
                "power": "3",
                "toughness": "3",
                "rarity": "mythic",
                "set": "abc",
                "collector_number": "1",
                "keywords": [],
                "produced_mana": [],
                "legalities": {},
                "is_creature": True,
                "is_land": False,
                "is_instant_sorcery": False,
                "is_multicolored": False,
                "color_count": 1,
                "has_etb_effect": False,
                "is_legendary": True,
            },
            {
                "name": "Duplicate Spell",
                "full_name": "Duplicate Spell",
                "layout": "normal",
                "mana_cost": "{1}{W}",
                "cmc": 2,
                "type_line": "Instant",
                "oracle_text": "Draw a card.",
                "colors": ["W"],
                "color_identity": ["W"],
                "power": "",
                "toughness": "",
                "rarity": "common",
                "set": "abc",
                "collector_number": "2",
                "keywords": [],
                "produced_mana": [],
                "legalities": {},
                "is_creature": False,
                "is_land": False,
                "is_instant_sorcery": True,
                "is_multicolored": False,
                "color_count": 1,
                "has_etb_effect": False,
                "is_legendary": False,
            },
            {
                "name": "Plains",
                "full_name": "Plains",
                "layout": "normal",
                "mana_cost": "",
                "cmc": 0,
                "type_line": "Basic Land — Plains",
                "oracle_text": "{T}: Add {W}.",
                "colors": [],
                "color_identity": ["W"],
                "power": "",
                "toughness": "",
                "rarity": "common",
                "set": "abc",
                "collector_number": "3",
                "keywords": [],
                "produced_mana": ["W"],
                "legalities": {},
                "is_creature": False,
                "is_land": True,
                "is_instant_sorcery": False,
                "is_multicolored": False,
                "color_count": 0,
                "has_etb_effect": False,
                "is_legendary": False,
            },
        ]
        return pd.DataFrame(rows)

    def test_singleton_violations_detected(self):
        analyzer = AdvancedDeckAnalyzer(self._card_db())
        mainboard = ["Duplicate Spell", "Duplicate Spell"] + ["Plains"] * 97
        analysis = analyzer.analyze_deck(mainboard, commanders=["Commander One"])
        violations = analysis["commander_profile"]["singleton_violations"]

        self.assertTrue(
            any(item["card"] == "Duplicate Spell" and item["copies"] == 2 for item in violations)
        )
        self.assertFalse(any(item["card"] == "Plains" for item in violations))


if __name__ == "__main__":
    unittest.main()

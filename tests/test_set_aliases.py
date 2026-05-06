import unittest
from tempfile import TemporaryDirectory

import pandas as pd

from research_pipeline.retrieval.corpus import build_card_chunks
from research_pipeline.set_aliases import extract_set_codes_from_text, set_name_for_code


class SetAliasTests(unittest.TestCase):
    def test_extract_set_codes_from_natural_language(self):
        query = (
            "If I build Lathril, what cards from Lorwyn or Strixhaven are good?"
        )
        codes = extract_set_codes_from_text(query, valid_codes={"lrw", "stx", "khm"})
        self.assertIn("lrw", codes)
        self.assertIn("stx", codes)
        self.assertNotIn("if", codes)
        self.assertNotIn("what", codes)

    def test_set_name_for_code(self):
        self.assertEqual(set_name_for_code("lrw"), "Lorwyn")
        self.assertEqual(set_name_for_code("stx"), "Strixhaven: School of Mages")

    def test_card_chunks_include_set_context(self):
        with TemporaryDirectory() as tmp_dir:
            csv_path = f"{tmp_dir}/cards.csv"
            df = pd.DataFrame(
                [
                    {
                        "name": "Test Mage",
                        "type_line": "Creature — Elf Druid",
                        "oracle_text": "Add one mana of any color.",
                        "mana_cost": "{1}{G}",
                        "set": "stx",
                        "color_identity": "['G']",
                        "keywords": "[]",
                    }
                ]
            )
            df.to_csv(csv_path, index=False)

            chunks = build_card_chunks(cards_csv=csv_path, max_cards=10)
            self.assertTrue(chunks)
            text = chunks[0].text
            self.assertIn("Set code: stx", text)
            self.assertIn("Set name: Strixhaven: School of Mages", text)


if __name__ == "__main__":
    unittest.main()

import unittest

from ui_helpers import parse_card_list


class UIHelperTests(unittest.TestCase):
    def test_parse_card_list_handles_commas_and_newlines(self):
        raw = "Sol Ring, Arcane Signet\nSwords to Plowshares"
        parsed = parse_card_list(raw)
        self.assertEqual(parsed, ["Sol Ring", "Arcane Signet", "Swords to Plowshares"])

    def test_parse_card_list_dedupes_case_insensitive(self):
        raw = "Sol Ring, sol ring, SOL RING"
        parsed = parse_card_list(raw)
        self.assertEqual(parsed, ["Sol Ring"])


if __name__ == "__main__":
    unittest.main()

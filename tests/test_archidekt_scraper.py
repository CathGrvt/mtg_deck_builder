import unittest
from tempfile import TemporaryDirectory

from archidekt_deck_list_scraper import ArchidektScraper


class ArchidektScraperTests(unittest.TestCase):
    def test_extract_sections_commander(self):
        scraper = ArchidektScraper(format_name="commander", request_delay=0)
        deck_json = {
            "cards": [
                {
                    "quantity": 1,
                    "categories": ["Commander"],
                    "card": {"oracleCard": {"name": "Atraxa, Praetors' Voice", "types": ["Creature"]}},
                },
                {
                    "quantity": 1,
                    "companion": True,
                    "categories": ["Companion"],
                    "card": {"oracleCard": {"name": "Lurrus of the Dream-Den", "types": ["Creature"]}},
                },
                {
                    "quantity": 2,
                    "categories": ["Sideboard"],
                    "card": {"oracleCard": {"name": "Negate", "types": ["Instant"]}},
                },
                {
                    "quantity": 3,
                    "categories": ["Ramp"],
                    "card": {"oracleCard": {"name": "Cultivate", "types": ["Sorcery"]}},
                },
                {
                    "quantity": 1,
                    "categories": ["Maybeboard"],
                    "card": {"oracleCard": {"name": "Ponder", "types": ["Sorcery"]}},
                },
                {
                    "quantity": 1,
                    "categories": ["Deck"],
                    "card": {"oracleCard": {"name": "Soldier", "types": ["Token"]}},
                },
            ]
        }

        sections = scraper._extract_sections(deck_json)
        self.assertEqual(sections["commanders"], ["Atraxa, Praetors' Voice"])
        self.assertEqual(sections["companions"], ["Lurrus of the Dream-Den"])
        self.assertEqual(sections["mainboard"], ["Cultivate", "Cultivate", "Cultivate"])
        self.assertEqual(sections["sideboard"], ["Negate", "Negate"])

    def test_format_deck_content_commander(self):
        scraper = ArchidektScraper(format_name="commander", request_delay=0)
        text = scraper._format_deck_content(
            {
                "commanders": ["Krenko, Mob Boss"],
                "companions": [],
                "mainboard": ["Sol Ring", "Sol Ring", "Mountain"],
                "sideboard": ["Red Elemental Blast"],
            }
        )
        expected = (
            "Commander\n"
            "1 Krenko, Mob Boss\n\n"
            "Deck\n"
            "2 Sol Ring\n"
            "1 Mountain\n\n"
            "Sideboard\n"
            "1 Red Elemental Blast"
        )
        self.assertEqual(text, expected)

    def test_compute_meta_percentages(self):
        scraper = ArchidektScraper(format_name="standard", request_delay=0)
        ranked = scraper._compute_meta_percentages(
            [
                {
                    "deck_id": 1,
                    "archetype_name": "Deck A",
                    "url": "https://archidekt.com/decks/1",
                    "view_count": 1000,
                    "updated_at": "2026-05-04T10:00:00Z",
                },
                {
                    "deck_id": 2,
                    "archetype_name": "Deck B",
                    "url": "https://archidekt.com/decks/2",
                    "view_count": 10,
                    "updated_at": "2026-05-04T10:00:00Z",
                },
            ]
        )
        self.assertEqual(len(ranked), 2)
        self.assertGreater(ranked[0]["meta_percentage"], ranked[1]["meta_percentage"])
        total = ranked[0]["meta_percentage"] + ranked[1]["meta_percentage"]
        self.assertAlmostEqual(total, 100.0, places=3)

    def test_commander_size_and_commander_filters(self):
        with TemporaryDirectory() as tmp_dir:
            scraper = ArchidektScraper(
                format_name="commander",
                output_dir=tmp_dir,
                request_delay=0,
                min_total_cards=95,
                max_total_cards=120,
            )
            scraper.clear_output_directory()

            # No commander category -> reject.
            scraper._load_deck_details = lambda _: {
                "cards": [
                    {
                        "quantity": 100,
                        "categories": ["Deck"],
                        "card": {"oracleCard": {"name": "Mountain", "types": ["Land"]}},
                    }
                ]
            }
            bad_no_commander = scraper.download_and_save_deck(
                {
                    "deck_id": 1,
                    "archetype_name": "Bad Deck",
                    "url": "https://archidekt.com/decks/1",
                    "meta_percentage": 1.0,
                    "deck_count": 1,
                }
            )
            self.assertFalse(bad_no_commander)

            # Commander present but too small -> reject.
            scraper._load_deck_details = lambda _: {
                "cards": [
                    {
                        "quantity": 1,
                        "categories": ["Commander"],
                        "card": {"oracleCard": {"name": "Atraxa, Praetors' Voice", "types": ["Creature"]}},
                    },
                    {
                        "quantity": 10,
                        "categories": ["Deck"],
                        "card": {"oracleCard": {"name": "Cultivate", "types": ["Sorcery"]}},
                    },
                ]
            }
            bad_small = scraper.download_and_save_deck(
                {
                    "deck_id": 2,
                    "archetype_name": "Tiny Deck",
                    "url": "https://archidekt.com/decks/2",
                    "meta_percentage": 1.0,
                    "deck_count": 1,
                }
            )
            self.assertFalse(bad_small)

            # Size within range and commander present -> accept.
            scraper._load_deck_details = lambda _: {
                "cards": [
                    {
                        "quantity": 1,
                        "categories": ["Commander"],
                        "card": {"oracleCard": {"name": "Atraxa, Praetors' Voice", "types": ["Creature"]}},
                    },
                    {
                        "quantity": 99,
                        "categories": ["Deck"],
                        "card": {"oracleCard": {"name": "Cultivate", "types": ["Sorcery"]}},
                    },
                ]
            }
            good = scraper.download_and_save_deck(
                {
                    "deck_id": 3,
                    "archetype_name": "Good Deck",
                    "url": "https://archidekt.com/decks/3",
                    "meta_percentage": 1.0,
                    "deck_count": 1,
                }
            )
            self.assertTrue(good)


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests


FORMAT_CONFIG = {
    "standard": {
        "output_dir": "current_standard_decks",
        "meta_json": "deck_meta_representation.json",
        "api_format": "Standard",
        "min_total_cards": 55,
        "max_total_cards": 80,
    },
    "commander": {
        "output_dir": "current_commander_decks",
        "meta_json": "commander_meta_representation.json",
        "api_format": "Commander",
        "min_total_cards": 95,
        "max_total_cards": 120,
    },
}


ARCHIDEKT_FORMAT_ALIASES = {
    "edh": "Commander",
    "commander": "Commander",
    "standard": "Standard",
    "pioneer": "Pioneer",
    "modern": "Modern",
    "legacy": "Legacy",
    "vintage": "Vintage",
    "pauper": "Pauper",
    "historic": "Historic",
    "brawl": "Brawl",
    "timeless": "Timeless",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_utc_timestamp(timestamp: Optional[str]) -> Optional[datetime]:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ArchidektScraper:
    def __init__(
        self,
        format_name: str = "commander",
        output_dir: Optional[str] = None,
        min_meta_percentage: float = 0.2,
        max_decks: int = 300,
        max_pages: int = 6,
        request_delay: float = 0.25,
        top_k: int = 0,
        min_total_cards: int = 0,
        max_total_cards: int = 0,
    ):
        self.base_api_url = "https://archidekt.com/api"
        self.base_web_url = "https://archidekt.com"
        self.format_name = format_name.lower()
        defaults = FORMAT_CONFIG.get(self.format_name, {})

        self.api_format = defaults.get("api_format") or ARCHIDEKT_FORMAT_ALIASES.get(
            self.format_name, self.format_name.title()
        )
        self.output_dir = output_dir or defaults.get("output_dir", f"current_{self.format_name}_decks")
        self.meta_filename = defaults.get("meta_json", f"{self.format_name}_deck_meta.json")
        self.min_meta_percentage = max(0.0, float(min_meta_percentage))
        self.max_decks = max(1, int(max_decks))
        self.max_pages = max(1, int(max_pages))
        self.request_delay = max(0.0, float(request_delay))
        self.top_k = max(0, int(top_k))
        self.min_total_cards = max(0, int(min_total_cards or defaults.get("min_total_cards", 0)))
        self.max_total_cards = max(0, int(max_total_cards or defaults.get("max_total_cards", 0)))

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.8",
                "Connection": "keep-alive",
            }
        )

        self.meta_data: List[Dict[str, Any]] = []
        self._seen_filenames: set[str] = set()

    def clear_output_directory(self) -> None:
        if os.path.exists(self.output_dir):
            logger.info(f"Clearing existing directory: {self.output_dir}")
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Created fresh directory: {self.output_dir}")

    def _request_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = path if path.startswith("http") else f"{self.base_api_url}{path}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _list_recent_deck_summaries(self) -> List[Dict[str, Any]]:
        summaries: List[Dict[str, Any]] = []

        for page in range(1, self.max_pages + 1):
            if len(summaries) >= self.max_decks:
                break

            data = self._request_json(
                "/decks/v3/",
                params={"formats": self.api_format, "page": page},
            )
            results = data.get("results") or []
            if not results:
                break

            for item in results:
                if len(summaries) >= self.max_decks:
                    break

                deck_id = item.get("id")
                name = str(item.get("name") or "").strip()
                if not deck_id or not name:
                    continue

                summaries.append(
                    {
                        "deck_id": int(deck_id),
                        "archetype_name": name,
                        "url": f"{self.base_web_url}/decks/{deck_id}",
                        "view_count": _safe_int(item.get("viewCount"), 0),
                        "updated_at": item.get("updatedAt"),
                    }
                )

            logger.info(
                "Collected %s deck summaries (page %s, format=%s)",
                len(summaries),
                page,
                self.api_format,
            )

            if not data.get("next"):
                break

            if self.request_delay > 0:
                time.sleep(self.request_delay)

        return summaries[: self.max_decks]

    def _compute_meta_percentages(
        self,
        summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        scored: List[Tuple[Dict[str, Any], float]] = []

        for deck in summaries:
            views = max(0, _safe_int(deck.get("view_count"), 0))
            updated_at = _parse_utc_timestamp(deck.get("updated_at"))
            age_days = (
                max(0.0, (now - updated_at).total_seconds() / 86400.0)
                if updated_at is not None
                else 365.0
            )
            recency_bonus = math.exp(-age_days / 45.0)
            score = math.log1p(views) + recency_bonus
            scored.append((deck, max(score, 1e-6)))

        if not scored:
            return []

        total_score = sum(score for _, score in scored)
        if total_score <= 0:
            total_score = float(len(scored))

        ranked: List[Dict[str, Any]] = []
        for deck, score in scored:
            meta_percentage = (100.0 * score / total_score) if total_score > 0 else (100.0 / len(scored))
            ranked.append(
                {
                    **deck,
                    "meta_percentage": round(meta_percentage, 4),
                    "deck_count": 1,
                }
            )

        ranked.sort(key=lambda row: row["meta_percentage"], reverse=True)
        return ranked

    def collect_meta_information(self) -> List[Dict[str, Any]]:
        logger.info(
            "Fetching Archidekt deck summaries (format=%s, max_pages=%s, max_decks=%s)...",
            self.api_format,
            self.max_pages,
            self.max_decks,
        )
        summaries = self._list_recent_deck_summaries()
        ranked = self._compute_meta_percentages(summaries)
        filtered = [row for row in ranked if row["meta_percentage"] >= self.min_meta_percentage]
        if self.top_k > 0:
            filtered = filtered[: self.top_k]

        logger.info(
            "Ranked %s decks, %s selected after min-meta %.3f%% and top-k=%s.",
            len(ranked),
            len(filtered),
            self.min_meta_percentage,
            self.top_k if self.top_k > 0 else "all",
        )
        return filtered

    def _load_deck_details(self, deck_id: int) -> Dict[str, Any]:
        return self._request_json(f"/decks/{deck_id}/")

    def _extract_card_name(self, card_entry: Dict[str, Any]) -> Optional[str]:
        card_info = card_entry.get("card")
        if not isinstance(card_info, dict):
            return None

        oracle = card_info.get("oracleCard")
        if not isinstance(oracle, dict):
            return None

        name = str(oracle.get("name") or "").strip()
        if not name:
            return None

        oracle_types = oracle.get("types") or []
        if isinstance(oracle_types, list):
            lowered = {str(t).strip().lower() for t in oracle_types}
            if "token" in lowered or "emblem" in lowered:
                return None

        return name

    def _extract_sections(self, deck_json: Dict[str, Any]) -> Dict[str, List[str]]:
        sections: Dict[str, List[str]] = {
            "commanders": [],
            "companions": [],
            "mainboard": [],
            "sideboard": [],
        }

        for entry in deck_json.get("cards") or []:
            quantity = _safe_int(entry.get("quantity"), 0)
            if quantity <= 0:
                continue

            name = self._extract_card_name(entry)
            if not name:
                continue

            categories = {str(c).strip().lower() for c in (entry.get("categories") or [])}
            is_maybeboard = "maybeboard" in categories
            is_sideboard = "sideboard" in categories
            is_commander = self.format_name == "commander" and "commander" in categories
            is_companion = bool(entry.get("companion")) or "companion" in categories

            if is_maybeboard:
                continue

            target_section = "mainboard"
            if is_commander:
                target_section = "commanders"
            elif is_companion:
                target_section = "companions"
            elif is_sideboard:
                target_section = "sideboard"

            sections[target_section].extend([name] * quantity)

        return sections

    def _collapse_with_order(self, cards: List[str]) -> List[Tuple[str, int]]:
        counts: Dict[str, int] = {}
        order: List[str] = []
        for card in cards:
            if card not in counts:
                counts[card] = 0
                order.append(card)
            counts[card] += 1
        return [(card, counts[card]) for card in order]

    def _format_deck_content(self, sections: Dict[str, List[str]]) -> str:
        lines: List[str] = []

        if self.format_name == "commander":
            if sections["commanders"]:
                lines.append("Commander")
                for card, qty in self._collapse_with_order(sections["commanders"]):
                    lines.append(f"{qty} {card}")
                lines.append("")

            if sections["companions"]:
                lines.append("Companion")
                for card, qty in self._collapse_with_order(sections["companions"]):
                    lines.append(f"{qty} {card}")
                lines.append("")

            lines.append("Deck")
            for card, qty in self._collapse_with_order(sections["mainboard"]):
                lines.append(f"{qty} {card}")

            if sections["sideboard"]:
                lines.append("")
                lines.append("Sideboard")
                for card, qty in self._collapse_with_order(sections["sideboard"]):
                    lines.append(f"{qty} {card}")
        else:
            for card, qty in self._collapse_with_order(sections["mainboard"]):
                lines.append(f"{qty} {card}")

            if sections["sideboard"]:
                lines.append("")
                for card, qty in self._collapse_with_order(sections["sideboard"]):
                    lines.append(f"{qty} {card}")

        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _safe_filename(self, archetype_name: str) -> str:
        clean_name = re.sub(r"[\\/:*?\"<>|]+", " ", archetype_name)
        clean_name = re.sub(r"\s+", " ", clean_name).strip() or "Untitled Deck"
        base = f"Deck - {clean_name}"
        candidate = base
        suffix = 2

        while f"{candidate}.txt" in self._seen_filenames:
            candidate = f"{base} ({suffix})"
            suffix += 1

        filename = f"{candidate}.txt"
        self._seen_filenames.add(filename)
        return filename

    def download_and_save_deck(self, deck_info: Dict[str, Any]) -> bool:
        deck_id = deck_info["deck_id"]
        archetype_name = deck_info["archetype_name"]
        url = deck_info["url"]

        try:
            details = self._load_deck_details(deck_id)
            sections = self._extract_sections(details)

            if not sections["mainboard"] and not sections["commanders"]:
                logger.warning("Skipping deck with no parsable cards: %s (%s)", archetype_name, deck_id)
                return False

            if self.format_name == "commander" and not sections["commanders"]:
                logger.warning("Skipping commander deck without Commander section: %s (%s)", archetype_name, deck_id)
                return False

            total_cards = len(sections["mainboard"]) + len(sections["commanders"]) + len(sections["companions"])
            if self.min_total_cards > 0 and total_cards < self.min_total_cards:
                logger.warning(
                    "Skipping deck below minimum size (%s < %s): %s (%s)",
                    total_cards,
                    self.min_total_cards,
                    archetype_name,
                    deck_id,
                )
                return False
            if self.max_total_cards > 0 and total_cards > self.max_total_cards:
                logger.warning(
                    "Skipping deck above maximum size (%s > %s): %s (%s)",
                    total_cards,
                    self.max_total_cards,
                    archetype_name,
                    deck_id,
                )
                return False

            content = self._format_deck_content(sections)
            if not content.strip():
                logger.warning("Skipping empty deck after formatting: %s (%s)", archetype_name, deck_id)
                return False

            filename = self._safe_filename(archetype_name)
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            self.meta_data.append(
                {
                    "archetype": archetype_name,
                    "meta_percentage": float(deck_info["meta_percentage"]),
                    "deck_count": int(deck_info.get("deck_count", 1)),
                    "url": url,
                    "source": "archidekt",
                    "view_count": int(deck_info.get("view_count", 0)),
                    "updated_at": deck_info.get("updated_at"),
                }
            )

            logger.info("Saved deck list: %s", path)
            return True
        except Exception as e:
            logger.error("Failed to fetch/save deck %s (%s): %s", archetype_name, deck_id, e)
            return False

    def export_meta_json(self) -> str:
        json_dir = "json_outputs"
        os.makedirs(json_dir, exist_ok=True)
        path = os.path.join(json_dir, self.meta_filename)
        sorted_data = sorted(self.meta_data, key=lambda row: row["meta_percentage"], reverse=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sorted_data, f, indent=2)
        logger.info("Exported meta data to: %s", path)
        return path

    def run(self) -> None:
        try:
            self.clear_output_directory()

            meta_decks = self.collect_meta_information()
            if not meta_decks:
                logger.warning(
                    "No decks passed min-meta %.3f%%. Try lowering --min-meta or increasing --max-decks.",
                    self.min_meta_percentage,
                )
                return

            successful = 0
            for deck_info in meta_decks:
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                if self.download_and_save_deck(deck_info):
                    successful += 1

            if successful > 0:
                self.export_meta_json()
            logger.info("Successfully saved %s out of %s decks", successful, len(meta_decks))
        except Exception as e:
            logger.error("An error occurred: %s", e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape recent Archidekt decklists for a format and export deck/meta outputs."
    )
    parser.add_argument(
        "--format",
        default="commander",
        help="Format to scrape (default: commander).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory for downloaded decklists.",
    )
    parser.add_argument(
        "--min-meta",
        type=float,
        default=0.2,
        help="Minimum computed meta percentage required to include a deck.",
    )
    parser.add_argument(
        "--max-decks",
        type=int,
        default=300,
        help="Maximum number of deck summaries to consider (default: 300).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=6,
        help="Maximum number of Archidekt pages to fetch (default: 6).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.25,
        help="Delay in seconds between API calls (default: 0.25).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Optional hard cap on selected decks after filtering (default: 0 = keep all).",
    )
    parser.add_argument(
        "--min-total-cards",
        type=int,
        default=0,
        help=(
            "Minimum cards required after parsing (default: format preset; "
            "set 0 to use preset, negative values are treated as 0)."
        ),
    )
    parser.add_argument(
        "--max-total-cards",
        type=int,
        default=0,
        help=(
            "Maximum cards allowed after parsing (default: format preset; "
            "set 0 to use preset, negative values are treated as 0)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scraper = ArchidektScraper(
        format_name=args.format,
        output_dir=args.output_dir,
        min_meta_percentage=args.min_meta,
        max_decks=args.max_decks,
        max_pages=args.max_pages,
        request_delay=args.request_delay,
        top_k=args.top_k,
        min_total_cards=args.min_total_cards,
        max_total_cards=args.max_total_cards,
    )
    scraper.run()

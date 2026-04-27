import argparse
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from mtg_io import (
    normalize_card_name as shared_normalize_card_name,
    parse_decklist_file,
    safe_parse_list,
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMMANDER_DUPLICATE_EXCEPTIONS = {
    "Relentless Rats",
    "Rat Colony",
    "Shadowborn Apostle",
    "Persistent Petitioners",
    "Dragon's Approach"
}

class DeckArchetype(Enum):
    AGGRO = "aggro"
    MIDRANGE = "midrange"
    CONTROL = "control"
    TEMPO = "tempo"
    COMBO = "combo"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"

    @classmethod
    def get_characteristics(cls) -> Dict['DeckArchetype', Dict[str, Tuple[float, float]]]:
        """
        Provides dynamic characteristics for each archetype
        These are more flexible than hardcoded values
        """
        return {
            cls.AGGRO: {
                'creature_ratio': (0.45, 0.75),
                'removal_ratio': (0.1, 0.3),
                'curve_peak': (1, 2),
                'avg_cmc': (1.5, 2.5),
                'interaction_ratio': (0.0, 0.2),
                'card_advantage_ratio': (0.0, 0.15)
            },
            cls.MIDRANGE: {
                'creature_ratio': (0.35, 0.55),
                'removal_ratio': (0.15, 0.35),
                'curve_peak': (2, 4),
                'avg_cmc': (2.5, 3.5),
                'interaction_ratio': (0.1, 0.3),
                'card_advantage_ratio': (0.1, 0.3)
            },
            cls.CONTROL: {
                'creature_ratio': (0.1, 0.3),
                'removal_ratio': (0.25, 0.45),
                'curve_peak': (2, 5),
                'avg_cmc': (3.0, 4.5),
                'interaction_ratio': (0.25, 0.5),
                'card_advantage_ratio': (0.2, 0.4)
            },
            cls.TEMPO: {
                'creature_ratio': (0.3, 0.5),
                'removal_ratio': (0.15, 0.35),
                'curve_peak': (1, 3),
                'avg_cmc': (2.0, 3.0),
                'interaction_ratio': (0.2, 0.4),
                'card_advantage_ratio': (0.1, 0.25)
            }
        }

class AdvancedDeckAnalyzer:
    """
    A comprehensive deck analysis system with dynamic archetype detection
    """
    def __init__(self, card_database: pd.DataFrame):
        self.card_db = card_database
        self.archetype_characteristics = DeckArchetype.get_characteristics()
        self.decklist = []
        self.commanders: List[str] = []
        self.companions: List[str] = []
        self._count_cache: Dict[str, int] = {}
        
    def _normalize_card_name(self, card_name: str) -> str:
        """
        Normalize card name by standardizing single slash to double slash format
        """
        return shared_normalize_card_name(card_name)
        
    def _extract_card_mechanics(self, card: pd.Series) -> Dict[str, int]:
        """
        Extract detailed mechanics from a card, handling both sides of dual-faced cards
        """
        mechanics = defaultdict(int)
        
        # Process oracle text for the card (combining both sides for dual-faced cards)
        oracle_texts = []
        
        # Handle the primary oracle text
        if pd.notna(card['oracle_text']):
            oracle_texts.append(str(card['oracle_text']).lower())
        
        # Check if this is a split/dual-faced card by looking at the full_name
        if pd.notna(card['full_name']) and ' // ' in card['full_name']:
            # This is a split card - we need to find the back face oracle text
            card_name = card['name']
            full_name = card['full_name']
            
            # Extract the back face name
            if ' // ' in full_name:
                front_face, back_face = full_name.split(' // ', 1)
                
                # Look for the back face in the card database
                back_face_data = self.card_db[self.card_db['name'] == back_face]
                if not back_face_data.empty:
                    back_oracle = back_face_data['oracle_text'].iloc[0]
                    if pd.notna(back_oracle):
                        oracle_texts.append(str(back_oracle).lower())
        
        # Combine all oracle texts for analysis
        combined_oracle_text = ' '.join(oracle_texts)
        
        # Comprehensive mechanic patterns
        mechanic_patterns = {
            'card_draw': [r'draw \d+ cards?', r'look at the top \d+ cards?'],
            'removal': [r'destroy target', r'exile target', r'deals? \d+ damage to', r'-\d+/-\d+'],
            'counter_spell': [r'counter target spell', r'counter target'],
            'ramp': [r'add \d+ mana', r'search your library for a land'],
            'token_generation': [r'create.*token', r'create a.*token'],
            'life_gain': [r'gain \d+ life'],
            'scry': [r'scry \d+'],
            'surveil': [r'surveil \d+'],
            'card_selection': [r'look at', r'reveal', r'choose'],
            'recursion': [r'return.*from your graveyard'],
        }
        
        # Check card type mechanics
        if card['is_creature'] == True:
            mechanics['creature'] += 1
            
            # Power-based mechanics
            try:
                power = float(card['power']) if pd.notna(card['power']) else 0
                toughness = float(card['toughness']) if pd.notna(card['toughness']) else 0
                
                if power >= 3:
                    mechanics['threat'] += 1
                if power + toughness >= 5:
                    mechanics['big_creature'] += 1
            except (ValueError, TypeError):
                pass
        
        # Special handling for Room cards
        if 'Room' in str(card['type_line']):
            mechanics['room'] += 1
            
            if 'unlock' in combined_oracle_text:
                mechanics['unlock_mechanic'] += 1
                
            if 'door' in combined_oracle_text:
                mechanics['door_mechanic'] += 1
        
        # Check keyword mechanics
        if isinstance(card['keywords'], list):
            for keyword in card['keywords']:
                mechanics[keyword.lower()] += 1
        
        # Check text-based mechanics
        for mechanic, patterns in mechanic_patterns.items():
            if any(re.search(pattern, combined_oracle_text) for pattern in patterns):
                mechanics[mechanic] += 1
        
        return dict(mechanics)
        
    def analyze_deck(
        self,
        decklist: List[str],
        commanders: Optional[List[str]] = None,
        companions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive deck analysis with dynamic archetype detection
        """
        try:
            # Normalize card names in the decklist (convert single slash to double slash)
            normalized_decklist = [self._normalize_card_name(card) for card in decklist]
            normalized_commanders = [self._normalize_card_name(card) for card in (commanders or [])]
            normalized_companions = [self._normalize_card_name(card) for card in (companions or [])]
            
            # Store the original decklist for later use in mana curve calculation
            self.decklist = normalized_decklist
            self._count_cache = {}
            self.commanders = normalized_commanders
            self.companions = normalized_companions
            
            # Verify cards in database - enhanced to handle split cards
            deck_cards = self._match_cards_in_database(normalized_decklist)
            verified_cards = deck_cards['name'].tolist()
            commander_cards = self._match_cards_in_database(normalized_commanders) if normalized_commanders else pd.DataFrame()
            companion_cards = self._match_cards_in_database(normalized_companions) if normalized_companions else pd.DataFrame()
            if not commander_cards.empty:
                verified_cards.extend(commander_cards['name'].tolist())
            if not companion_cards.empty:
                verified_cards.extend(companion_cards['name'].tolist())
            
            # Basic deck statistics
            stats = self._calculate_deck_statistics(deck_cards, normalized_decklist)
            
            # Card mechanics analysis
            mechanics_breakdown = self._analyze_deck_mechanics(deck_cards)
            
            # Archetype detection
            archetype_scores = self._detect_archetype(stats)
            
            # Commander-specific analysis
            commander_profile = self._analyze_commander_profile(
                deck_cards=deck_cards,
                commander_cards=commander_cards,
                mainboard_list=normalized_decklist
            )
            
            # Final analysis
            analysis = {
                'deck_size': len(decklist),
                'total_with_command_zone': len(decklist) + len(normalized_commanders) + len(normalized_companions),
                'statistics': stats,
                'mechanics': mechanics_breakdown,
                'archetype_scores': archetype_scores,
                'primary_archetype': max(archetype_scores.items(), key=lambda x: x[1])[0],
                'verified_cards': verified_cards,
                'commanders': normalized_commanders,
                'companions': normalized_companions,
                'commander_profile': commander_profile
            }
            
            return analysis
        
        except Exception as e:
            logger.error("Error in deck analysis: {}".format(str(e)))
            raise
            
    def _match_cards_in_database(self, decklist: List[str]) -> pd.DataFrame:
        """
        Enhanced card matching to handle split cards and variations
        """
        # Create a set of unique cards for faster lookup
        unique_cards = set(decklist)
        matched_cards = pd.DataFrame()
        remaining_cards = unique_cards.copy()
        
        # First try exact name match
        exact_matches = self.card_db[self.card_db['name'].isin(unique_cards)]
        matched_cards = pd.concat([matched_cards, exact_matches])
        
        # Remove matched cards from the remaining set
        remaining_cards -= set(exact_matches['name'])
        
        # For remaining cards, try matching against full_name column
        if remaining_cards:
            for card_name in list(remaining_cards):
                # Check if this card name appears in any full_name field
                full_name_matches = self.card_db[self.card_db['full_name'] == card_name]
                
                if not full_name_matches.empty:
                    matched_cards = pd.concat([matched_cards, full_name_matches])
                    remaining_cards.remove(card_name)
                    continue
                
                # Check if this is a normalized name (with //) that matches a full_name
                if '//' in card_name:
                    full_name_matches = self.card_db[self.card_db['full_name'] == card_name]
                    if not full_name_matches.empty:
                        matched_cards = pd.concat([matched_cards, full_name_matches])
                        remaining_cards.remove(card_name)
                        continue
                    
                    # Also check if it's one face of a split card in the full_name
                    front_face, back_face = card_name.split(' // ', 1)
                    
                    # Try to match just the front face
                    front_matches = self.card_db[self.card_db['name'] == front_face]
                    if not front_matches.empty:
                        matched_cards = pd.concat([matched_cards, front_matches])
                        remaining_cards.remove(card_name)
                        continue
                        
                    # Try to match using the front face in a full_name
                    front_full_matches = self.card_db[self.card_db['full_name'].str.startswith(front_face + ' //', na=False)]
                    if not front_full_matches.empty:
                        matched_cards = pd.concat([matched_cards, front_full_matches])
                        remaining_cards.remove(card_name)
                        continue
                
        return matched_cards
    
    def _calculate_deck_statistics(self, deck_cards: pd.DataFrame, decklist: List[str]) -> Dict[str, float]:
        """
        Calculate comprehensive deck statistics
        """
        # Improved boolean filtering
        nonland_cards = deck_cards[deck_cards['is_land'] != True]
        land_cards = deck_cards[deck_cards['is_land'] == True]
        total_cards = len(decklist)
        
        land_total = sum(self._get_card_count(name) for name in land_cards['name'])
        nonland_total = max(total_cards - land_total, 0)
        creature_total = sum(self._get_card_count(name) for name in nonland_cards[nonland_cards['is_creature'] == True]['name'])
        
        cmc_samples = []
        for _, card in nonland_cards.iterrows():
            if pd.isna(card['cmc']):
                continue
            count = self._get_card_count(card['name'])
            if count <= 0:
                continue
            cmc_samples.extend([card['cmc']] * count)
        
        # Calculate various ratios and statistics
        deck_stats = {
            'land_ratio': land_total / total_cards if total_cards > 0 else 0,
            'creature_ratio': creature_total / nonland_total if nonland_total > 0 else 0,
            'avg_cmc': float(np.mean(cmc_samples)) if cmc_samples else 0,
            'median_cmc': float(np.median(cmc_samples)) if cmc_samples else 0,
            'color_diversity': self._calculate_color_diversity(deck_cards),
            'curve': self._calculate_mana_curve(nonland_cards)
        }
        
        # Calculate advanced ratios
        deck_stats.update(self._calculate_advanced_ratios(nonland_cards, nonland_total))
        
        return deck_stats
    
    def _get_card_count(self, card_name: str) -> int:
        """
        Return the number of times a card appears in the current decklist,
        accounting for split/dual-faced name variations.
        """
        cache_key = card_name.lower()
        if cache_key in self._count_cache:
            return self._count_cache[cache_key]
        
        count = sum(1 for name in self.decklist if name == card_name)
        
        if count == 0:
            if ' // ' in card_name:
                faces = [face.strip() for face in card_name.split(' // ')]
                for face in faces:
                    count += sum(1 for name in self.decklist if name == face)
            else:
                for deck_card in self.decklist:
                    if ' // ' in deck_card:
                        faces = [face.strip() for face in deck_card.split(' // ')]
                        if card_name in faces:
                            count += 1
        
        self._count_cache[cache_key] = count
        return count
    
    def _calculate_mana_curve(self, nonland_cards: pd.DataFrame) -> Dict[int, int]:
        """
        Calculate detailed mana curve considering card copies
        """
        # Create a dictionary to track card counts
        card_counts = {}
        for card_name in nonland_cards['name']:
            count = self._get_card_count(card_name)
            if count == 0:
                continue
            
            # Get the CMC for this card
            card_cmc = nonland_cards[nonland_cards['name'] == card_name]['cmc'].iloc[0]
            
            # Round CMC to nearest integer
            cmc_int = int(round(card_cmc))
            
            # Add to curve, respecting the number of copies
            card_counts[cmc_int] = card_counts.get(cmc_int, 0) + count
        
        return card_counts
        
    def _calculate_advanced_ratios(self, nonland_cards: pd.DataFrame, nonland_total: int) -> Dict[str, float]:
        """
        Calculate advanced deck ratios
        """
        advanced_ratios = {}
        
        # Prevent division by zero
        if nonland_total == 0:
            return {
                'interaction_ratio': 0,
                'removal_ratio': 0,
                'card_advantage_ratio': 0
            }
        
        # Interaction ratio
        interaction_patterns = [
            r'counter target', r'destroy target', r'exile target', 
            r'tap target', r'can\'t attack', r'can\'t block'
        ]
        interaction_total = 0
        for _, card in nonland_cards.iterrows():
            text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, text) for pattern in interaction_patterns):
                interaction_total += self._get_card_count(card['name'])
        advanced_ratios['interaction_ratio'] = interaction_total / nonland_total
        
        # Removal ratio
        removal_patterns = [
            r'destroy target', r'exile target', r'deals? \d+ damage to', 
            r'target creature gets -\d+/-\d+'
        ]
        removal_total = 0
        for _, card in nonland_cards.iterrows():
            text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, text) for pattern in removal_patterns):
                removal_total += self._get_card_count(card['name'])
        advanced_ratios['removal_ratio'] = removal_total / nonland_total
        
        # Card advantage ratio
        card_advantage_patterns = [
            r'draw \d+ cards?', r'return.*from your graveyard', 
            r'search your library', r'look at the top \d+ cards?'
        ]
        card_advantage_total = 0
        for _, card in nonland_cards.iterrows():
            text = str(card['oracle_text']).lower() if pd.notna(card['oracle_text']) else ''
            if any(re.search(pattern, text) for pattern in card_advantage_patterns):
                card_advantage_total += self._get_card_count(card['name'])
        advanced_ratios['card_advantage_ratio'] = card_advantage_total / nonland_total
        
        return advanced_ratios
    
    def _calculate_color_diversity(self, deck_cards: pd.DataFrame) -> float:
        """
        Calculate color diversity of the deck
        """
        all_colors = []
        for _, card in deck_cards.iterrows():
            if isinstance(card['colors'], list):
                count = self._get_card_count(card['name'])
                if count <= 0:
                    continue
                for _ in range(count):
                    all_colors.extend(card['colors'])
        
        # Use entropy to measure color diversity
        if not all_colors:
            return 0
        
        color_counts = Counter(all_colors)
        total_colors = len(all_colors)
        
        # Prevent log(0) issues
        entropy = -sum((count/total_colors) * np.log(count/total_colors) 
                      for count in color_counts.values() if count > 0)
        
        return entropy
    
    def _analyze_deck_mechanics(self, deck_cards: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze mechanics across the entire deck
        """
        mechanics_breakdown = defaultdict(int)
        
        for _, card in deck_cards.iterrows():
            card_mechanics = self._extract_card_mechanics(card)
            for mechanic, count in card_mechanics.items():
                mechanics_breakdown[mechanic] += count
        
        return dict(mechanics_breakdown)
    
    def _detect_archetype(self, stats: Dict[str, float]) -> Dict[DeckArchetype, float]:
        """
        Detect deck archetype with confidence scores
        """
        archetype_scores = {}
        
        for archetype, characteristics in self.archetype_characteristics.items():
            score = 0
            total_checks = 0
            
            # Check each characteristic
            for characteristic, (min_val, max_val) in characteristics.items():
                if characteristic in stats:
                    value = stats.get(characteristic, 0)
                    if min_val <= value <= max_val:
                        score += 1
                    total_checks += 1
            
            # Calculate archetype confidence
            archetype_scores[archetype] = score / total_checks if total_checks > 0 else 0
        
        # Check for potential hybrid archetype
        sorted_scores = sorted(archetype_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.7 * sorted_scores[0][1]:
            archetype_scores[DeckArchetype.HYBRID] = (sorted_scores[0][1] + sorted_scores[1][1]) / 2
        
        return archetype_scores

    def _analyze_commander_profile(
        self,
        deck_cards: pd.DataFrame,
        commander_cards: pd.DataFrame,
        mainboard_list: List[str]
    ) -> Dict[str, Any]:
        """
        Apply Commander-specific validations and heuristics.
        """
        commanders_present = bool(self.commanders)
        profile = {
            'has_commander': commanders_present,
            'commanders': self.commanders,
            'companions': self.companions,
            'mainboard_size': len(mainboard_list),
            'expected_mainboard_size': (100 - len(self.commanders)) if commanders_present else None,
            'size_is_legal': True,
            'singleton_violations': [],
            'color_identity_violations': [],
            'color_identity': [],
            'missing_commanders': [],
            'ramp_sources': 0,
            'ramp_ratio': 0.0
        }
        
        if not commanders_present:
            return profile
        
        # Deck size check (Commander decks = 100 cards including command zone)
        expected = max(0, 100 - len(self.commanders))
        profile['size_is_legal'] = len(mainboard_list) == expected
        profile['expected_mainboard_size'] = expected
        
        # Commander presence check
        commander_names_found = commander_cards['name'].tolist() if not commander_cards.empty else []
        profile['missing_commanders'] = [cmd for cmd in self.commanders if cmd not in commander_names_found]
        
        # Prepare lookups
        deck_unique = deck_cards.drop_duplicates(subset='name') if not deck_cards.empty else deck_cards
        deck_unique_full = deck_cards.drop_duplicates(subset='full_name').dropna(subset=['full_name']) if not deck_cards.empty else deck_cards
        deck_lookup = deck_unique.set_index('name') if not deck_unique.empty else pd.DataFrame()
        deck_lookup_full = deck_unique_full.set_index('full_name') if not deck_unique_full.empty else pd.DataFrame()
        
        def lookup_card(card_name: str) -> Optional[pd.Series]:
            if not deck_cards.empty:
                if card_name in deck_lookup.index:
                    return deck_lookup.loc[card_name]
                if card_name in deck_lookup_full.index:
                    return deck_lookup_full.loc[card_name]
                if ' // ' in card_name:
                    front_face = card_name.split(' // ')[0].strip()
                    if front_face in deck_lookup.index:
                        return deck_lookup.loc[front_face]
            return None
        
        # Determine commander color identity
        allowed_colors: Set[str] = set()
        if not commander_cards.empty:
            for _, commander in commander_cards.iterrows():
                color_identity = commander.get('color_identity') or commander.get('colors') or []
                if isinstance(color_identity, str):
                    color_identity = safe_parse_list(color_identity)
                allowed_colors.update(color_identity)
        profile['color_identity'] = sorted(allowed_colors)
        
        # Check color identity violations
        if allowed_colors and not deck_cards.empty:
            for _, card in deck_unique.iterrows():
                card_identity = card.get('color_identity') or []
                if isinstance(card_identity, str):
                    card_identity = safe_parse_list(card_identity)
                if card_identity and not set(card_identity).issubset(allowed_colors):
                    profile['color_identity_violations'].append(card['name'])
        
        # Singleton violations (allow basic lands + special exceptions)
        card_counts = Counter(mainboard_list)
        violations = []
        for card_name, count in card_counts.items():
            if count <= 1:
                continue
            card_row = lookup_card(card_name)
            if card_row is None:
                continue
            type_line = card_row.get('type_line') or ''
            is_basic_land = 'Basic' in type_line and 'Land' in type_line
            if is_basic_land or card_name in COMMANDER_DUPLICATE_EXCEPTIONS:
                continue
            violations.append({'card': card_name, 'copies': count})
        profile['singleton_violations'] = violations
        
        # Ramp density heuristics
        def is_ramp_card(card_row: Optional[pd.Series]) -> bool:
            if card_row is None:
                return False
            oracle_text = (card_row.get('oracle_text') or '').lower()
            type_line = card_row.get('type_line') or ''
            produced_mana = card_row.get('produced_mana') or []
            if isinstance(produced_mana, str):
                produced_mana = safe_parse_list(produced_mana)
            # Mana rocks / dorks
            if produced_mana and any(keyword in type_line for keyword in ['Artifact', 'Creature', 'Enchantment']):
                return True
            ramp_patterns = [
                'add {',
                'search your library for a land',
                'create a treasure token',
                'create two treasure tokens',
                'treasure token',
                'untap target land',
                'landfall — add'
            ]
            return any(pattern in oracle_text for pattern in ramp_patterns)
        
        ramp_sources = 0
        nonland_total = 0
        for card_name, count in card_counts.items():
            card_row = lookup_card(card_name)
            if card_row is None:
                continue
            is_land = bool(card_row.get('is_land'))
            if not is_land:
                nonland_total += count
            if is_ramp_card(card_row):
                ramp_sources += count
        profile['ramp_sources'] = ramp_sources
        profile['ramp_ratio'] = (ramp_sources / nonland_total) if nonland_total else 0.0
        
        return profile

def infer_commander_hints_from_filename(filepath: str) -> List[str]:
    """
    Infer commander names from deck filename patterns such as 'Deck - Commander Name.txt'.
    """
    filename = os.path.basename(filepath)
    base, _ = os.path.splitext(filename)
    prefix = "deck - "
    if base.lower().startswith(prefix):
        base = base[len(prefix):]
    base = base.strip()
    if not base:
        return []
    
    # Split on common partner separators without touching commas in names
    parts = re.split(r'\s*(?:&|/|\+)\s*', base)
    hints = [part.strip() for part in parts if part.strip()]
    return hints


def load_decklist(filepath: str, commander_hints: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    Load decklist from a file, capturing Commander sections when present.
    """
    sections = parse_decklist_file(filepath)
    
    # Auto-infer commanders from filename if no explicit section
    if not sections['commanders'] and commander_hints:
        normalized_hints = [normalize_card_name(name) for name in commander_hints]
        
        def remove_card(card_name: str) -> Optional[str]:
            for idx, existing in enumerate(sections['mainboard']):
                if existing.lower() == card_name.lower():
                    return sections['mainboard'].pop(idx)
            return None
        
        for hint in normalized_hints:
            removed = remove_card(hint)
            if removed:
                sections['commanders'].append(removed)
            else:
                sections['commanders'].append(hint)
    
    return sections

def normalize_card_name(card_name: str) -> str:
    """
    Normalize card name by standardizing single slash to double slash format
    """
    return shared_normalize_card_name(card_name)

def analyze_deck(cards_df: pd.DataFrame, decklist_path: str):
    """
    Advanced deck analysis function
    """
    # Validate input
    if not os.path.exists(decklist_path):
        raise FileNotFoundError("Decklist file not found: {}".format(decklist_path))
    
    # Load decklist
    try:
        commander_hints = infer_commander_hints_from_filename(decklist_path)
        sections = load_decklist(decklist_path, commander_hints=commander_hints)
    except Exception as e:
        raise ValueError("Error loading decklist: {}".format(e))
    
    mainboard = sections.get('mainboard', [])
    sideboard = sections.get('sideboard', [])
    commanders = sections.get('commanders', [])
    companions = sections.get('companions', [])
    
    # Validate mainboard
    if not mainboard:
        raise ValueError("Decklist is empty")
    
    # Initialize analyzer
    analyzer = AdvancedDeckAnalyzer(cards_df)
    
    # Analyze mainboard
    try:
        analysis = analyzer.analyze_deck(mainboard, commanders=commanders, companions=companions)
    except Exception as e:
        raise RuntimeError("Deck analysis failed: {}".format(e))
    
    # Display results with enhanced readability
    print("\n=== Deck Analysis: {} ===\n".format(os.path.basename(decklist_path)))
    
    # Deck Overview
    total_cards = analysis['total_with_command_zone']
    print(f"- Mainboard Cards: {analysis['deck_size']}")
    if analysis['commanders']:
        print(f"- Commanders: {', '.join(analysis['commanders'])}")
    if analysis['companions']:
        print(f"- Companion(s): {', '.join(analysis['companions'])}")
    print(f"- Total including command zone: {total_cards}")
    print("- Lands: {:.0f}%".format(analysis['statistics']['land_ratio'] * 100))
    print("- Creatures: {:.0f}%".format(analysis['statistics']['creature_ratio'] * 100))
    
    # Room cards detection
    if 'room' in analysis['mechanics']:
        room_count = analysis['mechanics']['room']
        print("- Room Cards: {}".format(room_count))
    
    # Archetype Analysis
    print("\nArchetype Analysis:")
    archetype_descriptions = {
        DeckArchetype.AGGRO: "Aggressive strategy focused on quick, low-cost threats",
        DeckArchetype.MIDRANGE: "Balanced strategy with mid-cost creatures and flexible answers",
        DeckArchetype.CONTROL: "Defensive strategy focused on controlling the game and answering threats",
        DeckArchetype.TEMPO: "Strategy that balances threats with interaction to maintain board advantage",
        DeckArchetype.COMBO: "Strategy relying on specific card combinations to win",
        DeckArchetype.HYBRID: "Mixed strategy combining elements of multiple archetypes"
    }
    
    sorted_archetypes = sorted(analysis['archetype_scores'].items(), key=lambda x: x[1], reverse=True)
    top_archetypes = [arch for arch, score in sorted_archetypes if score > 0]
    
    print("Primary Strategy: {}".format(analysis['primary_archetype'].value.title()))
    if len(top_archetypes) > 1:
        print("Secondary Strategies: {}".format(
            ", ".join(arch.value.title() for arch in top_archetypes[1:] if arch != analysis['primary_archetype'])
        ))
    
    # Detailed Strategy Description
    primary_archetype = analysis['primary_archetype']
    print("- {}".format(archetype_descriptions.get(primary_archetype, "Unique strategy")))
    
    # Mana Curve Analysis
    print("\nMana Curve Insights:")
    curve = analysis['statistics']['curve']
    print("Average Converted Mana Cost (CMC): {:.2f}".format(analysis['statistics']['avg_cmc']))
    print("Median Converted Mana Cost (CMC): {:.2f}".format(analysis['statistics']['median_cmc']))
    
    print("Spell Distribution by Mana Cost:")
    for cmc, count in sorted(curve.items()):
        bar = '█' * min(count, 30)  # Limit bar length for display purposes
        print("{} CMC: {} {} cards".format(cmc, bar, count))
    
    # Deck Composition Analysis
    print("\nDeck Composition:")
    print("Interaction Density: {:.0f}%".format(analysis['statistics']['interaction_ratio'] * 100))
    print("Removal Density: {:.0f}%".format(analysis['statistics']['removal_ratio'] * 100))
    print("Card Advantage Potential: {:.0f}%".format(analysis['statistics']['card_advantage_ratio'] * 100))
    
    # Commander-specific reporting
    profile = analysis.get('commander_profile', {})
    if profile.get('has_commander'):
        print("\nCommander Compliance:")
        expected = profile.get('expected_mainboard_size')
        actual = profile.get('mainboard_size')
        size_label = "OK" if profile.get('size_is_legal') else "Needs Adjustment"
        print(f"- Mainboard Size: {actual} (expected {expected}) [{size_label}]")
        
        color_identity = profile.get('color_identity', [])
        if color_identity:
            print(f"- Commander Color Identity: {''.join(color_identity) or 'Colorless'}")
        if profile.get('missing_commanders'):
            print("- Commander cards missing from database: {}".format(
                ", ".join(profile['missing_commanders'])
            ))
        if profile.get('color_identity_violations'):
            print("- Color Identity Violations: {}".format(
                ", ".join(profile['color_identity_violations'])
            ))
        if profile.get('singleton_violations'):
            violation_strings = [
                f"{item['card']} (x{item['copies']})" for item in profile['singleton_violations']
            ]
            print("- Singleton Violations: {}".format(", ".join(violation_strings)))
        ramp_sources = profile.get('ramp_sources', 0)
        ramp_ratio = profile.get('ramp_ratio', 0.0) * 100
        print(f"- Ramp Sources: {ramp_sources} (~{ramp_ratio:.0f}% of nonlands)")
    
    # Mechanics Breakdown
    print("\nKey Deck Mechanics:")
    sorted_mechanics = sorted(analysis['mechanics'].items(), key=lambda x: x[1], reverse=True)
    top_mechanics = sorted_mechanics[:8]  # Top 8 mechanics
    
    mechanic_descriptions = {
        'card_draw': "Ability to draw additional cards",
        'card_selection': "Ability to look at and potentially rearrange top cards",
        'removal': "Spells that eliminate opponent's creatures or threats",
        'surveil': "Look at top cards and put them in graveyard or back on top",
        'token_generation': "Create additional creature tokens",
        'big_creature': "Powerful, high-impact creatures",
        'threat': "Creatures that demand an immediate answer",
        'life_gain': "Ability to restore life points",
        'room': "Room enchantments that provide utility",
        'unlock_mechanic': "Ability to unlock room doors for additional effects",
        'door_mechanic': "Mechanics involving door unlocking and room transitions"
    }
    
    for mechanic, count in top_mechanics:
        desc = mechanic_descriptions.get(mechanic, "Unique mechanical interaction")
        print("- {}: {} ({})".format(mechanic.replace('_', ' ').title(), count, desc))
    
    # Find cards in decklist that weren't matched
    normalized_mainboard = [normalize_card_name(card) for card in mainboard]
    verified_names = set(analysis['verified_cards'])
    
    # Also check full_names in card_db for potential matches
    all_names_in_db = set(cards_df['name'])
    all_full_names_in_db = set(cards_df['full_name'].dropna())
    
    missing_cards = []
    cards_to_verify = normalized_mainboard + [normalize_card_name(c) for c in commanders] + [normalize_card_name(c) for c in companions]
    for card in cards_to_verify:
        # Skip if already found in verified cards
        if card in verified_names:
            continue
            
        # Check if found in database names or full names
        if card not in all_names_in_db and card not in all_full_names_in_db:
            # For split cards, also check if front face exists
            if '//' in card:
                front_face = card.split(' // ')[0].strip()
                if front_face not in all_names_in_db:
                    missing_cards.append(card)
            else:
                missing_cards.append(card)
    
    if missing_cards:
        print("\nWarning - Cards not found in database:")
        for card in missing_cards:
            print("- {}".format(card))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze an MTG deck with Commander awareness.")
    parser.add_argument('deck_path', help="Path to the decklist file to analyze.")
    parser.add_argument(
        '--cards',
        default='data/commander_cards.csv',
        help="Path to the card database CSV (default: data/commander_cards.csv)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load card database
    try:
        cards_df = pd.read_csv(args.cards)
        
        # Convert boolean columns to actual booleans if they are strings
        bool_columns = ['is_creature', 'is_land', 'is_instant_sorcery', 
                        'is_multicolored', 'has_etb_effect', 'is_legendary']
        for col in bool_columns:
            if col in cards_df.columns:
                cards_df[col] = cards_df[col].map({'True': True, 'False': False, True: True, False: False})
        
        # Convert lists stored as strings to actual lists
        list_columns = ['colors', 'color_identity', 'keywords', 'produced_mana']
        for col in list_columns:
            if col in cards_df.columns:
                cards_df[col] = cards_df[col].apply(safe_parse_list)
        
    except FileNotFoundError:
        print(f"Error: Card database file not found. Please ensure '{args.cards}' exists.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading card database: {e}")
        sys.exit(1)
    
    # Analyze the specified deck
    decklist_path = args.deck_path
    
    try:
        analyze_deck(cards_df, decklist_path)
    except Exception as e:
        print("An error occurred during deck analysis: {}".format(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

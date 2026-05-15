from typing import List

from mtg_shared.deck_utils import parse_card_list as _parse_card_list


def parse_card_list(raw: str) -> List[str]:
    return _parse_card_list(raw)

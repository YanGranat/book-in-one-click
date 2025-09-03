#!/usr/bin/env python3
"""
Language utilities.
"""


def detect_lang_from_text(text: str) -> str:
    """
    Very simple language heuristic: return 'ru' if Cyrillic letters present, else 'en'.

    This lightweight approach mirrors the smoke test and is sufficient for choosing
    output language based on a short topic string like "Transposons" or "Транспозоны".
    """
    has_cyrillic = False
    for ch in text:
        lower = ch.lower()
        if 'а' <= lower <= 'я' or lower == 'ё':
            has_cyrillic = True
            break
    return 'ru' if has_cyrillic else 'en'



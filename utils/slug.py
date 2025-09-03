#!/usr/bin/env python3
"""
Slug helper for safe filenames on Windows.
"""


def safe_slug(text: str) -> str:
    forbidden = '<>:"/\\|?*\n\r\t'
    cleaned = ''.join('_' if ch in forbidden else ch for ch in text)
    cleaned = cleaned.replace(' ', '_')
    return cleaned.lower()


def safe_filename_base(text: str) -> str:
    """
    Produce a filesystem-safe filename base while preserving non-Latin characters and case.
    Replaces forbidden characters with underscores and spaces with underscores.
    """
    forbidden = '<>:"/\\|?*\n\r\t'
    base = ''.join('_' if ch in forbidden else ch for ch in text)
    base = base.replace(' ', '_').strip().strip('. ')
    # Collapse consecutive underscores
    while '__' in base:
        base = base.replace('__', '_')
    return base or 'untitled'



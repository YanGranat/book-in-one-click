#!/usr/bin/env python3
"""
Slug helper for safe filenames on Windows.
"""


def safe_slug(text: str) -> str:
    forbidden = '<>:"/\\|?*\n\r\t'
    cleaned = ''.join('_' if ch in forbidden else ch for ch in text)
    cleaned = cleaned.replace(' ', '_')
    return cleaned.lower()



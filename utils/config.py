#!/usr/bin/env python3
"""
Configuration loader.
Reads JSON config from config/defaults.json with safe fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_CACHE: Dict[str, Any] | None = None


def load_config(current_file: str) -> Dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    project_root = Path(current_file).resolve().parents[1]
    config_path = project_root / "config" / "defaults.json"

    default: Dict[str, Any] = {
        "generator_label": "OpenAI Agents SDK",
        "format": "md",
        "metadata": {
            "include_generator": True,
            "include_pipeline": True,
            "include_timestamp": True,
        },
        "pipelines": {
            "post": {
                "model": "gpt-4o",
                "word_count_min": 400,
                "word_count_max": 800,
                "output_subdir": "popular_science_post",
                "style_pack": "pop_sci",
            },
            "article": {
                "model": "gpt-4o",
                "word_count_min": 1200,
                "word_count_max": 2000,
                "output_subdir": "deep_popular_science_article",
                "style_pack": "pop_sci",
            },
            "book": {
                "model": "gpt-4o",
                "output_subdir": "deep_popular_science_book",
                "style_pack": "pop_sci",
            },
        },
    }

    if config_path.exists():
        try:
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
            # Shallow merge: loaded keys override defaults
            default.update(loaded)
        except Exception:
            # Keep defaults if parsing fails
            pass

    _CACHE = default
    return _CACHE



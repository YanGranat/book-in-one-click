#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_MODELS_CACHE: Optional[Dict[str, Any]] = None


def _load_models_config() -> Dict[str, Any]:
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE
    # Resolve repo root relative to this file
    base = Path(__file__).resolve().parents[1] / "config" / "models.json"
    try:
        data = json.loads(base.read_text(encoding="utf-8"))
    except Exception:
        # Fallback minimal defaults
        data = {
            "openai": {"tiers": {"fast": {"model": "gpt-5-mini"}, "heavy": {"model": "gpt-5"}}},
            "gemini": {"tiers": {"fast": {"model": "gemini-2.5-flash"}, "heavy": {"model": "gemini-2.5-pro"}}},
            "claude": {"tiers": {"fast": {"model": "claude-haiku-4-0"}, "heavy": {"model": "claude-sonnet-4-0"}}},
        }
    _MODELS_CACHE = data
    return data


def get_model(provider: str, tier: str = "heavy") -> str:
    """Return the model name for a provider and tier from config/models.json.
    ENV overrides are honored if present but not required by the user.
    """
    p = (provider or "openai").strip().lower()
    t = (tier or "heavy").strip().lower()
    # normalize aliases
    if t == "superfast":
        t = "fast"
    if t == "medium":
        t = "heavy"
    cfg = _load_models_config()
    # Optional ENV overrides for compatibility
    if p == "openai":
        if t == "fast" and os.getenv("OPENAI_FAST_MODEL"):
            return os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini")
        if t == "heavy" and os.getenv("OPENAI_MODEL"):
            return os.getenv("OPENAI_MODEL", "gpt-5")
    if p in {"gemini", "google"}:
        if t == "fast" and os.getenv("GEMINI_FAST_MODEL"):
            return os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash")
        if t == "heavy" and os.getenv("GEMINI_MODEL"):
            return os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    if p == "claude":
        if os.getenv("ANTHROPIC_MODEL") and t in {"fast", "heavy"}:
            # Single model override covers both tiers unless FAST specified separately
            if t == "fast" and os.getenv("ANTHROPIC_FAST_MODEL"):
                return os.getenv("ANTHROPIC_FAST_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-0"))
            return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-0")
        if t == "fast" and os.getenv("ANTHROPIC_FAST_MODEL"):
            return os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-4-0")
        # Hard fallback updated to 4.5 when no overrides and config missing
        # (config normally has 4.5 already)
    # Config file fallback
    prov = cfg.get(p) or cfg.get("gemini" if p == "google" else p) or {}
    tiers = (prov.get("tiers") or {})
    model = ((tiers.get(t) or {}).get("model")) or ""
    if not model:
        # Final hard fallback
        if p == "openai":
            return "gpt-5" if t != "fast" else "gpt-5-mini"
        if p in {"gemini", "google"}:
            return "gemini-2.5-pro" if t != "fast" else "gemini-2.5-flash"
        if p == "claude":
            return "claude-sonnet-4-0" if t != "fast" else "claude-haiku-4-0"
    return model


def get_json_mode(provider: str) -> Dict[str, Any]:
    """Return provider-specific JSON mode hints from config, if any."""
    p = (provider or "openai").strip().lower()
    cfg = _load_models_config()
    prov = cfg.get(p) or cfg.get("gemini" if p == "google" else p) or {}
    return (prov.get("json_mode") or {})


def is_json_supported(provider: str) -> bool:
    jm = get_json_mode(provider)
    return bool(jm.get("supported"))


def get_thinking_config(provider: str) -> Dict[str, Any]:
    """Return provider-specific thinking hints from config, if any.
    Currently used for Anthropic Claude to pass optional 'thinking' object with budget_tokens.
    """
    p = (provider or "openai").strip().lower()
    cfg = _load_models_config()
    prov = cfg.get(p) or cfg.get("gemini" if p == "google" else p) or {}
    return (prov.get("thinking") or {})



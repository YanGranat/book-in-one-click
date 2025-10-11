#!/usr/bin/env python3
from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from utils.models import get_model, get_json_mode, is_json_supported


def _ensure_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception:
            pass


class ProviderRunner:
    def __init__(self, provider: str):
        self.provider = (provider or "openai").strip().lower()

    # --- OpenAI ---
    def _openai_text(self, system: str, user_message: str, *, tier: str = "heavy") -> str:
        _ensure_loop()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        from agents import Agent, Runner  # type: ignore
        model = get_model("openai", tier)
        tools = []
        try:
            from agents import WebSearchTool  # type: ignore
            tools = [WebSearchTool()]
        except Exception:
            tools = []
        agent = Agent(name="Agent", instructions=system, model=model, tools=tools)
        res = Runner.run_sync(agent, user_message)
        return getattr(res, "final_output", "")

    # --- Gemini ---
    def _gemini_text(self, system: str, user_message: str, *, tier: str = "heavy", json_mode: bool = False) -> str:
        _ensure_loop()
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
        genai.configure(api_key=api_key)
        mname = get_model("gemini", tier)
        # Try to enable Google Search grounding when SDK supports it; otherwise fall back gracefully
        tools = None
        try:
            # Prefer typed Tool/GoogleSearch if available (newer SDKs)
            from google.generativeai.types import Tool as _Tool, GoogleSearch as _GoogleSearch  # type: ignore
            tools = [_Tool(google_search=_GoogleSearch())]
        except Exception:
            try:
                # Older SDKs sometimes accept dict shape
                tools = [{"google_search": {}}]
            except Exception:
                tools = None
        gen_cfg = {"max_output_tokens": 8192}
        if json_mode and is_json_supported("gemini"):
            jm = get_json_mode("gemini")
            mime = jm.get("response_mime_type") or "application/json"
            gen_cfg["response_mime_type"] = mime
        # Create model; if tools are not recognized by SDK or backend, retry without tools
        try:
            model = genai.GenerativeModel(
                model_name=mname,
                system_instruction=system,
                tools=tools if tools else None,
                generation_config=gen_cfg or None,
            )
            resp = model.generate_content(user_message)
        except Exception:
            # Retry without tools to avoid "Unknown field for FunctionDeclaration: google_search"
            model = genai.GenerativeModel(
                model_name=mname,
                system_instruction=system,
                generation_config=gen_cfg or None,
            )
            resp = model.generate_content(user_message)
        txt = (getattr(resp, "text", None) or "").strip()
        if not txt:
            parts: List[str] = []
            try:
                for c in getattr(resp, "candidates", []) or []:
                    for part in getattr(getattr(c, "content", None), "parts", []) or []:
                        t = getattr(part, "text", None)
                        if t:
                            parts.append(t)
            except Exception:
                pass
            txt = ("\n".join(parts)).strip()
        return txt

    # --- Claude ---
    def _claude_text(self, system: str, user_message: str, *, tier: str = "heavy", json_mode: bool = False) -> str:
        _ensure_loop()
        import anthropic  # type: ignore
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Claude API key not found. Set ANTHROPIC_API_KEY environment variable.")
        client = anthropic.Anthropic(api_key=api_key)
        mname = get_model("claude", tier)
        kwargs: Dict[str, Any] = {
            "model": mname,
            "max_tokens": 8192,
            "system": system,
            "messages": [{"role": "user", "content": user_message}]
        }
        if json_mode and is_json_supported("claude"):
            kwargs["response_format"] = get_json_mode("claude").get("response_format", {"type": "json_object"})
        msg = client.messages.create(**kwargs)
        parts: List[str] = []
        for blk in getattr(msg, "content", []) or []:
            txt = getattr(blk, "text", None)
            if txt:
                parts.append(txt)
        return ("\n\n".join(parts)).strip()

    # --- Public API ---
    def run_text(self, system: str, user_message: str, *, speed: str = "heavy") -> str:
        p = self.provider
        tier = "fast" if (speed or "").strip().lower() == "fast" else "heavy"
        if p == "openai":
            return self._openai_text(system, user_message, tier=tier)
        if p in {"gemini", "google"}:
            return self._gemini_text(system, user_message, tier=tier, json_mode=False)
        return self._claude_text(system, user_message, tier=tier, json_mode=False)

    def run_json(self, system: str, user_message: str, *, speed: str = "fast") -> str:
        p = self.provider
        tier = "fast" if (speed or "").strip().lower() == "fast" else "heavy"
        if p == "openai":
            # OpenAI Agents SDK returns text; let caller parse JSON if needed
            return self._openai_text(system, user_message, tier=tier)
        if p in {"gemini", "google"}:
            return self._gemini_text(system, user_message, tier=tier, json_mode=True)
        return self._claude_text(system, user_message, tier=tier, json_mode=True)



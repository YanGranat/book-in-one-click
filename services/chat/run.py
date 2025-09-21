#!/usr/bin/env python3
from __future__ import annotations

import os
import asyncio
import re
from typing import Optional


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed."
        ) from e


def _run_openai_with(system: str, user_message: str, model_name: Optional[str] = None) -> str:
    _ensure_loop()
    Agent, Runner = _try_import_sdk()
    # Try add WebSearchTool for browsing capability
    tools = []
    try:
        from agents import WebSearchTool  # type: ignore
        tools = [WebSearchTool()]
    except Exception:
        tools = []
    agent = Agent(
        name="Чат-бот ассистент",
        instructions=system,
        model=(model_name or os.getenv("OPENAI_MODEL", "gpt-5")),
        tools=tools,
    )
    res = Runner.run_sync(agent, user_message)
    return getattr(res, "final_output", "")


def _run_gemini_with(system: str, user_message: str, model_name: Optional[str] = None) -> str:
    _ensure_loop()
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    preferred = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    fallbacks = [
        preferred,
        os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash"),
        "gemini-1.5-pro",
    ]
    last_err = None
    for mname in fallbacks:
        try:
            # Enable Google Search grounding if available via 'tools'
            tools = []
            try:
                from google.generativeai import types as gen_types  # type: ignore
                # Some SDKs expose grounding with a special tool; if not, leave empty
                # Placeholder for future: tools=[gen_types.Tool(google_search=gen_types.GoogleSearch())]
                tools = []
            except Exception:
                tools = []
            model = genai.GenerativeModel(model_name=mname, system_instruction=system, tools=tools)
            resp = model.generate_content(user_message)
            txt = (getattr(resp, "text", None) or "").strip()
            if not txt:
                parts = []
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
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini request failed; last error: {last_err}")


def _run_claude_with(system: str, user_message: str, model_name: Optional[str] = None) -> str:
    _ensure_loop()
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    preferred = model_name or os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet-latest")
    fallbacks = [
        preferred,
        "claude-4-haiku-latest",
    ]
    last_err = None
    for mname in fallbacks:
        try:
            # Prepare a minimal tool schema for web_search; the tool execution handled by pre-built context above
            tools = [
                {
                    "name": "web_search",
                    "description": "Search the web and return brief results for grounding.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ]
            msg = client.messages.create(
                model=mname,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=tools,
            )
            parts = []
            for blk in getattr(msg, "content", []) or []:
                txt = getattr(blk, "text", None)
                if txt:
                    parts.append(txt)
            return ("\n\n".join(parts)).strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Claude request failed; last error: {last_err}")


def run_chat_message(provider: str, system: str, user_message: str) -> str:
    _ensure_loop()
    prov = (provider or "openai").strip().lower()
    # Augment with web context for non-OpenAI providers too (and as fallback for all)
    try:
        if _should_ground(user_message):
            web_ctx = _build_web_context(user_message)
            if web_ctx:
                system = f"{system}\n\nДополнительный контекст из веб-поиска (используй для проверки актуальности):\n{web_ctx}"
    except Exception:
        pass
    if prov == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        return _run_openai_with(system, user_message, model)
    if prov in {"gemini", "google"}:
        # Support both Pro and Flash via ENV
        mname = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        return _run_gemini_with(system, user_message, mname)
    # default to Claude (Claude 4 by default if configured)
    cname = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet-latest")
    return _run_claude_with(system, user_message, cname)


def build_system_prompt(*, chat_lang: str, kind: str, full_content: str) -> str:
    # Load Russian Telegram prompt from prompts/chat_telegram/system.md and append full context
    try:
        from pathlib import Path as _P
        p = _P(__file__).resolve().parents[2] / "prompts" / "chat_telegram" / "assistant.md"
        base = p.read_text(encoding="utf-8")
    except Exception:
        base = (
            "Ты — ассистент в Telegram-боте 'Book in One Click'.\n\n"
            "Поведение: отвечай на языке пользователя, будь лаконичен и практичен."
        )
    base = base.replace("{chat_lang}", chat_lang or "ru")
    kind_norm = (kind or "result").strip().lower()
    ctx = [
        f"\nПолный контекст последней генерации (тип: {kind_norm}):",
        "=== LAST_GENERATION START ===",
        full_content or "",
        "=== LAST_GENERATION END ===",
    ]
    return base + "\n" + "\n".join(ctx)


def _ensure_loop() -> None:
    try:
        # If there's already a running loop in this thread, do nothing
        asyncio.get_running_loop()
    except RuntimeError:
        # Create and set a new event loop for this background thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception:
            pass


def run_chat_summary(provider: str, text: str) -> str:
    """Summarize long chat history into a short, lossless recap.
    Uses fast model variants where possible.
    """
    _ensure_loop()
    prov = (provider or "openai").strip().lower()
    sys_prompt = (
        "Summarize the following chat context into a short recap (5-10 bullet points).\n"
        "Keep key facts, decisions, and user intent. Preserve language.\n"
    )
    if prov == "openai":
        model = os.getenv("OPENAI_FAST_MODEL", os.getenv("OPENAI_MODEL", "gpt-5"))
        return _run_openai_with(sys_prompt, text, model)
    if prov in {"gemini", "google"}:
        mname = os.getenv("GEMINI_FAST_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        return _run_gemini_with(sys_prompt, text, mname)
    # Claude fast fallback
    cname = os.getenv("ANTHROPIC_FAST_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-4-haiku-latest"))
    return _run_claude_with(sys_prompt, text, cname)


def _should_ground(user_message: str) -> bool:
    """Heuristic: decide when to attach web search context."""
    text = (user_message or "").lower()
    patterns = [
        r"актуал",
        r"обнов",
        r"правда",
        r"верно",
        r"2025",
        r"now|current|latest|updat",
    ]
    return any(re.search(p, text) for p in patterns)


def _build_web_context(user_message: str) -> str:
    """Build compact web context via DuckDuckGo and HTML fetch.
    Uses existing utils.web tools; safe no-op on failures.
    """
    try:
        from utils.web import build_search_context
    except Exception:
        return ""
    q = (user_message or "").strip()
    # Add lightweight expansions
    queries = [q, f"{q} 2025", f"{q} latest update"]
    try:
        ctx = build_search_context(queries=queries, per_query=2, max_chars=3000)
    except Exception:
        ctx = ""
    return ctx



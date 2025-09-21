#!/usr/bin/env python3
from __future__ import annotations

import os
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
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    preferred = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    fallbacks = [
        preferred,
        os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05"),
        "gemini-2.0-pro",
        os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash"),
        "gemini-1.5-pro-latest",
    ]
    last_err = None
    for mname in fallbacks:
        try:
            model = genai.GenerativeModel(model_name=mname, system_instruction=system)
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
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    preferred = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
    fallbacks = [
        preferred,
        "claude-3-5-sonnet-latest",
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
    ]
    last_err = None
    for mname in fallbacks:
        try:
            msg = client.messages.create(
                model=mname,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_message}],
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
    prov = (provider or "openai").strip().lower()
    if prov == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-5")
        return _run_openai_with(system, user_message, model)
    if prov in {"gemini", "google"}:
        mname = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        return _run_gemini_with(system, user_message, mname)
    # default to Claude
    cname = os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-latest")
    return _run_claude_with(system, user_message, cname)


def build_system_prompt(*, chat_lang: str, kind: str, full_content: str) -> str:
    # Load Russian Telegram prompt from prompts/chat_telegram/system.md and append full context
    try:
        from pathlib import Path as _P
        p = _P(__file__).resolve().parents[2] / "prompts" / "chat_telegram" / "system.md"
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



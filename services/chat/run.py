#!/usr/bin/env python3
from __future__ import annotations

import os
import asyncio
import re
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed."
        ) from e


def _ensure_sessions_db() -> Path:
    """Ensure sessions DB path exists and return it."""
    base = Path(__file__).resolve().parents[2] / "output"
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return base / "sessions.db"


def _parse_session_id(session_id: Optional[str]) -> Tuple[int, int, str]:
    """Parse "<chat_id>:<user_id>:<provider>" → tuple; fallback safe defaults."""
    try:
        if not session_id:
            return (0, 0, "openai")
        parts = str(session_id).split(":", 2)
        chat_id = int(parts[0]) if len(parts) >= 1 and parts[0].isdigit() else 0
        user_id = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
        provider = (parts[2] if len(parts) >= 3 else "openai").strip().lower()
        return (chat_id, user_id, provider or "openai")
    except Exception:
        return (0, 0, "openai")


def _kv_get_history(telegram_id: int, chat_id: int, provider: str, limit: int = 200) -> List[Dict[str, str]]:
    try:
        from server.kv import chat_get  # type: ignore
    except Exception:
        return []
    try:
        import asyncio as _aio
        loop = None
        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # We may be on a worker thread without running loop; create one temporarily
            return _aio.run(chat_get(telegram_id, chat_id, provider, limit=limit))  # type: ignore
        else:
            return _aio.run(chat_get(telegram_id, chat_id, provider, limit=limit))  # type: ignore
    except Exception:
        return []


def _kv_append(telegram_id: int, chat_id: int, provider: str, role: str, content: str) -> None:
    try:
        from server.kv import chat_append  # type: ignore
    except Exception:
        return
    try:
        import asyncio as _aio
        loop = None
        try:
            loop = _aio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            # Fire-and-forget on background loop
            _aio.get_event_loop().create_task(chat_append(telegram_id, chat_id, provider, role, content))  # type: ignore
        else:
            _aio.run(chat_append(telegram_id, chat_id, provider, role, content))  # type: ignore
    except Exception:
        return


def _run_openai_with(system: str, user_message: str, *, model_name: Optional[str] = None, session_id: Optional[str] = None) -> str:
    _ensure_loop()
    Agent, Runner = _try_import_sdk()
    # Optional web search tool (native)
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
    # Prefer native session memory for OpenAI
    session = None
    if session_id:
        try:
            from agents import SQLiteSession  # type: ignore
            sessions_db = _ensure_sessions_db()
            session = SQLiteSession(str(session_id), str(sessions_db))
        except Exception:
            session = None
    res = Runner.run_sync(agent, user_message, session=session)
    return getattr(res, "final_output", "")


def _run_gemini_with(system: str, user_message: str, *, model_name: Optional[str] = None, session_id: Optional[str] = None) -> str:
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
            # Try enabling Google Search grounding if available via safety_settings/tools kwargs
            tools = []
            try:
                # Recent SDKs expose tool config via dicts
                tools = [{"google_search": {}}]
            except Exception:
                tools = []
            model = genai.GenerativeModel(model_name=mname, system_instruction=system, tools=tools)
            chat = None
            history_msgs: List[Dict[str, str]] = []
            c_id, u_id, prov = _parse_session_id(session_id)
            if c_id and u_id:
                history_msgs = _kv_get_history(u_id, c_id, prov, limit=200)
            # Convert KV history to Gemini parts
            ghistory = []
            for it in history_msgs[-60:]:
                role = (it.get("role") or "user").lower()
                text = it.get("content") or ""
                ghistory.append({"role": ("user" if role == "user" else "model"), "parts": [text]})
            if ghistory:
                chat = model.start_chat(history=ghistory)
                resp = chat.send_message(user_message)
            else:
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
            # Append to KV fallback history
            if session_id:
                c_id, u_id, prov = _parse_session_id(session_id)
                if c_id and u_id:
                    _kv_append(u_id, c_id, prov, "user", user_message)
                    _kv_append(u_id, c_id, prov, "assistant", txt)
            return txt
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini request failed; last error: {last_err}")


def _run_claude_with(system: str, user_message: str, *, model_name: Optional[str] = None, session_id: Optional[str] = None) -> str:
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
            # Prefer simple message flow; if tool_use appears, handle minimal cycle for web_search stub
            # Build message list using KV fallback history as conversation
            history: List[Dict[str, Any]] = []
            c_id, u_id, prov = _parse_session_id(session_id)
            if c_id and u_id:
                for it in _kv_get_history(u_id, c_id, prov, limit=200)[-60:]:
                    role = (it.get("role") or "user").lower()
                    text = it.get("content") or ""
                    history.append({"role": ("user" if role == "user" else "assistant"), "content": text})
            history.append({"role": "user", "content": user_message})
            msg = client.messages.create(
                model=mname,
                max_tokens=4096,
                system=system,
                messages=history,
                tools=[{
                    "name": "web_search",
                    "description": "Search the web and return brief results for grounding.",
                    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                }],
            )
            # If the model returned a tool_use, respond once with our prebuilt context
            def _extract_text(m: Any) -> str:
                parts = []
                for blk in getattr(m, "content", []) or []:
                    t = getattr(blk, "text", None)
                    if t:
                        parts.append(t)
                return ("\n\n".join(parts)).strip()

            used_tool = False
            tool_query = None
            for blk in getattr(msg, "content", []) or []:
                if getattr(blk, "type", "") == "tool_use" and getattr(blk, "name", "") == "web_search":
                    used_tool = True
                    # Capture query if available
                    try:
                        tool_query = getattr(blk, "input", {}).get("query")  # type: ignore
                    except Exception:
                        tool_query = None
            if used_tool:
                # Build web context once and send as tool_result
                web_ctx = ""
                try:
                    q = tool_query or user_message
                    web_ctx = _build_web_context(q)
                except Exception:
                    web_ctx = ""
                tool_result = [{
                    "role": "tool",
                    "name": "web_search",
                    "content": web_ctx or "",
                }]
                msg2 = client.messages.create(
                    model=mname,
                    max_tokens=4096,
                    system=system,
                    messages=history + tool_result,  # continue conversation providing tool output
                )
                out = _extract_text(msg2)
            else:
                out = _extract_text(msg)
            if session_id:
                c_id, u_id, prov = _parse_session_id(session_id)
                if c_id and u_id:
                    _kv_append(u_id, c_id, prov, "user", user_message)
                    _kv_append(u_id, c_id, prov, "assistant", out)
            return out
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Claude request failed; last error: {last_err}")


def run_chat_message(provider: str, system: str, user_message: str, *, session_id: str | None = None) -> str:
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
        # Also push to KV for cross-provider continuity
        try:
            c_id, u_id, p = _parse_session_id(session_id)
            if c_id and u_id:
                # Append user before call; assistant after
                _kv_append(u_id, c_id, p, "user", user_message)
        except Exception:
            pass
        reply = _run_openai_with(system, user_message, model_name=model, session_id=session_id)
        try:
            c_id, u_id, p = _parse_session_id(session_id)
            if c_id and u_id:
                _kv_append(u_id, c_id, p, "assistant", reply)
        except Exception:
            pass
        return reply
    if prov in {"gemini", "google"}:
        # Support both Pro and Flash via ENV
        mname = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        return _run_gemini_with(system, user_message, model_name=mname, session_id=session_id)
    # default to Claude (Claude 4 by default if configured)
    cname = os.getenv("ANTHROPIC_MODEL", "claude-4-sonnet-latest")
    return _run_claude_with(system, user_message, model_name=cname, session_id=session_id)


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
        return _run_openai_with(sys_prompt, text, model_name=model)
    if prov in {"gemini", "google"}:
        mname = os.getenv("GEMINI_FAST_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        return _run_gemini_with(sys_prompt, text, model_name=mname)
    # Claude fast fallback
    cname = os.getenv("ANTHROPIC_FAST_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-4-haiku-latest"))
    return _run_claude_with(sys_prompt, text, model_name=cname)


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



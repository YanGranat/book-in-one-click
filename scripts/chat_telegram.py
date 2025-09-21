#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from utils.env import ensure_project_root_on_syspath, load_env_from_root


def _setup_env():
    ensure_project_root_on_syspath(__file__)
    load_env_from_root(__file__)


def _read_result_content_by_id(result_id: int) -> tuple[str, str]:
    from server.db import SessionLocal
    if SessionLocal is None:
        raise RuntimeError("DB is not configured (SessionLocal is None)")
    from sqlalchemy import select
    from server.db import ResultDoc
    async def _fetch():
        async with SessionLocal() as s:
            r = await s.execute(select(ResultDoc).where(ResultDoc.id == int(result_id)))
            row = r.scalars().first()
            if not row:
                raise RuntimeError("Result not found")
            return (getattr(row, "kind", "result") or "result", getattr(row, "content", "") or "")
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_fetch())


def _detect_lang(s: str) -> str:
    try:
        from utils.lang import detect_lang_from_text
        det = (detect_lang_from_text(s or "") or "").lower()
        return "en" if det.startswith("en") else "ru"
    except Exception:
        return "ru"


def _run_chat(chat_lang: str, user_input: str, *, context_kind: Optional[str], context_content: Optional[str]) -> str:
    if context_content:
        from services.chat.run import build_system_prompt, run_chat_message
        system = build_system_prompt(chat_lang=chat_lang, kind=(context_kind or "result"), full_content=context_content)
        provider = (os.getenv("CHAT_PROVIDER") or os.getenv("OPENAI_PROVIDER") or os.getenv("DEFAULT_PROVIDER") or "openai").strip().lower()
        return run_chat_message(provider, system, user_input)
    else:
        from llm_agents.chat_telegram.assistant import build_chat_telegram_assistant
        from agents import Runner  # type: ignore
        agent = build_chat_telegram_assistant(chat_lang=chat_lang)
        res = Runner.run_sync(agent, user_input)
        return getattr(res, "final_output", "")


def _maybe_save_md(text: str, force_save: bool) -> bool:
    import re
    m = re.search(r"<md_output([^>]*)>([\s\S]*?)</md_output>", text)
    if not m and not force_save:
        return False
    body = m.group(2) if m else text
    title = None
    if m:
        import re as _re
        mt = _re.search(r"title=\"([^\"]+)\"", m.group(1) or "")
        if mt:
            title = mt.group(1)
    from utils.slug import safe_filename_base
    from utils.io import ensure_output_dir, next_available_filepath
    out_dir = ensure_output_dir("chat")
    base = safe_filename_base(title or "chat_result")
    path = next_available_filepath(out_dir, base, ".md")
    path.write_text(body if body.endswith("\n") else (body + "\n"), encoding="utf-8")
    print(f"Saved: {path}")
    return True


def main() -> None:
    _setup_env()
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="auto", help="ru|en|auto")
    p.add_argument("--context-id", type=int, default=None, help="ResultDoc id to use as context")
    p.add_argument("--save-md", action="store_true", help="Force save response to .md if no <md_output>")
    args = p.parse_args()

    chat_lang = args.lang
    context_kind = None
    context_content = None
    if args.context_id is not None:
        try:
            context_kind, context_content = _read_result_content_by_id(int(args.context_id))
        except Exception as e:
            print(f"Context error: {e}")

    print("Chat started. Type your message. Ctrl+C to exit.")
    while True:
        try:
            user_in = input("> ")
        except KeyboardInterrupt:
            print("\nBye")
            break
        if not user_in.strip():
            continue
        eff_lang = chat_lang
        if (chat_lang or "auto").lower() == "auto":
            eff_lang = _detect_lang(user_in)
        try:
            reply = _run_chat(eff_lang, user_in, context_kind=context_kind, context_content=context_content)
        except Exception as e:
            print(f"Error: {e}")
            continue
        if not _maybe_save_md(reply, args.save_md):
            print(reply)


if __name__ == "__main__":
    main()



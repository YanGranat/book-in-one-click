from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_prompt(chat_lang: str = "ru") -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "chat_telegram" / "assistant.md"
    txt = prompt_path.read_text(encoding="utf-8")
    return txt.replace("{chat_lang}", chat_lang or "ru")


def try_import_sdk():
    from agents import Agent  # type: ignore
    try:
        from agents import WebSearchTool  # type: ignore
    except Exception:
        WebSearchTool = None  # type: ignore
    return Agent, WebSearchTool


def build_chat_telegram_assistant(*, model: str | None = None, chat_lang: str = "ru", instructions_override: str | None = None) -> Any:
    Agent, WebSearchTool = try_import_sdk()
    tools = []
    if WebSearchTool is not None:
        tools.append(WebSearchTool())
    return Agent(
        name="Чат-бот ассистент",
        instructions=instructions_override or _load_prompt(chat_lang),
        model=model or "gpt-5",
        tools=tools,
    )



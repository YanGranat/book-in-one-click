from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import ArticleOutline


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "deep_popular_science_article" / "module_02_research" / "outline_editor.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_outline_editor_agent(model: str | None = None) -> Any:
    Agent = try_import_sdk()
    return Agent(
        name="Deep Article · Outline Editor",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=ArticleOutline,
    )



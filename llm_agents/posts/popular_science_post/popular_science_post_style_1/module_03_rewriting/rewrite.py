from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_post_prompt() -> str:
    base = Path(__file__).resolve().parents[5] / "prompts" / "posts" / "popular_science_post" / "popular_science_post_style_1"
    p1 = base / "module_01_writing" / "writer.md"
    return p1.read_text(encoding="utf-8") if p1.exists() else ""


def _load_rewrite_prompt() -> str:
    base = Path(__file__).resolve().parents[5] / "prompts" / "posts" / "popular_science_post" / "popular_science_post_style_1"
    p1 = base / "module_03_rewriting" / "rewrite.md"
    return p1.read_text(encoding="utf-8") if p1.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_rewrite_agent() -> Any:
    Agent = try_import_sdk()
    combined_instructions = (
        _load_rewrite_prompt() + "\n\n" +
        "<style_contract_original_writer>\n" + _load_post_prompt() + "\n</style_contract_original_writer>\n"
    )
    return Agent(
        name="Post Rewriter (Style 1)",
        instructions=combined_instructions,
        model="gpt-5",
    )



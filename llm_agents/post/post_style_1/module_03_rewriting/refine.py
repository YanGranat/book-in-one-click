from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_post_prompt() -> str:
    base = Path(__file__).resolve().parents[4] / "prompts" / "post"
    p1 = base / "post_style_1" / "module_01_writing" / "writer.md"
    return p1.read_text(encoding="utf-8") if p1.exists() else ""


def _load_refine_prompt() -> str:
    base = Path(__file__).resolve().parents[4] / "prompts" / "post"
    p1 = base / "post_style_1" / "module_03_rewriting" / "refine.md"
    return p1.read_text(encoding="utf-8") if p1.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_refine_agent() -> Any:
    Agent = try_import_sdk()
    combined_instructions = (
        _load_refine_prompt() + "\n\n" +
        "<style_contract_original_writer>\n" + _load_post_prompt() + "\n</style_contract_original_writer>\n"
    )
    return Agent(
        name="Post Style Refiner (Style 1)",
        instructions=combined_instructions,
        model="gpt-5",
    )



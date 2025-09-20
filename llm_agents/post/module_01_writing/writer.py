from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "post"
        / "module_01_writing"
        / "post.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent  # type: ignore

    return Agent


def build_post_writer_agent(
    model: str | None = None,
    *,
    instructions_override: str | None = None,
) -> Any:
    Agent = try_import_sdk()
    return Agent(
        name="Popular Science Post Writer",
        instructions=instructions_override or _load_prompt(),
        model=model or "gpt-5",
    )




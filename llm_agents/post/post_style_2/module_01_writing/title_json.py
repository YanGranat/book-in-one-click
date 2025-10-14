from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[4]
        / "prompts"
        / "post"
        / "post_style_2"
        / "module_01_writing"
        / "title_json.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, ModelSettings  # type: ignore
    return Agent, ModelSettings


def build_title_json_agent(model: str | None = None) -> Any:
    Agent, ModelSettings = try_import_sdk()
    a = Agent(
        name="Style2 Title+JSON Formatter",
        instructions=_load_prompt(),
        model=model or "gpt-5",
    )
    try:
        a.model_settings = ModelSettings(reasoning={"effort": "low"})
    except Exception:
        pass
    return a



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
    from agents import Agent  # type: ignore
    from schemas import JsonSchema as AgentOutputSchema  # type: ignore
    return Agent, AgentOutputSchema


def build_title_json_agent(model: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    # JSON-only output; schema is unconstrained here (validated downstream)
    return Agent(
        name="Style2 Title+JSON Formatter",
        instructions=_load_prompt(),
        model=model or "gpt-5",
    )



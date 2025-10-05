from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import TitleProposal


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "deep_popular_science_article" / "module_03_writing" / "title_namer.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore
    return Agent, AgentOutputSchema


def build_title_namer_agent(model: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    return Agent(
        name="Deep Article · Title Namer",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=AgentOutputSchema(TitleProposal, strict_json_schema=False),
    )



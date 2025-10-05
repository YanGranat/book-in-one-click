from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import LeadChunk


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "deep_popular_science_article" / "module_03_writing" / "section_lead_writer.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore
    return Agent, AgentOutputSchema


def build_section_lead_writer_agent(model: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    return Agent(
        name="Deep Article · Section Lead Writer",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=AgentOutputSchema(LeadChunk, strict_json_schema=False),
    )



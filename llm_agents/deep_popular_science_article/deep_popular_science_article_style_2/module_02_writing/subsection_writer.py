from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import SectionDraftChunk
from utils.models import get_model


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[5]
        / "prompts"
        / "deep_popular_science_article"
        / "deep_popular_science_article_style_2"
        / "module_02_writing"
        / "subsection_writer.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_section_writer_agent(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "heavy")
    return Agent(
        name="Deep Article · Section Writer (Style 2)",
        instructions=_load_prompt(),
        model=eff_model,
        output_type=AgentOutputSchema(SectionDraftChunk, strict_json_schema=False),
    )



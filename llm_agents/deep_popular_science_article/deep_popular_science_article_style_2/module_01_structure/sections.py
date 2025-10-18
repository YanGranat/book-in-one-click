from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import ArticleOutline
from utils.models import get_model


def _load_prompt() -> str:
    base_dir = (
        Path(__file__).resolve().parents[4]
        / "prompts"
        / "deep_popular_science_article"
        / "deep_popular_science_article_style_2"
        / "module_01_structure"
    )
    prompt_path = base_dir / "sections.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_sections_agent(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "medium")
    return Agent(
        name="Deep Article · Outline Sections Builder (Style 2)",
        instructions=_load_prompt(),
        model=eff_model,
        output_type=AgentOutputSchema(ArticleOutline, strict_json_schema=False),
    )



from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import ExtendedTopic
from utils.models import get_model


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[4]
        / "prompts"
        / "article"
        / "deep_popular_science_article"
        / "deep_popular_science_article_style_3"
        / "module_01_concept"
        / "agent_1_extended_topic.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_agent_1_extended_topic(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "medium")
    try:
        return Agent(
            name="Style 3 · Agent 1 · Extended Topic",
            instructions=_load_prompt(),
            model=eff_model,
            output_type=AgentOutputSchema(ExtendedTopic, strict_json_schema=False),
        )
    except Exception:
        return Agent(
            name="Style 3 · Agent 1 · Extended Topic (plain)",
            instructions=_load_prompt(),
            model=eff_model,
        )



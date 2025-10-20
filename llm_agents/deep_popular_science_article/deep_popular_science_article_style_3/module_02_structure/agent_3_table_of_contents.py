from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import Style3ToC
from utils.models import get_model


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[4]
        / "prompts"
        / "article"
        / "deep_popular_science_article"
        / "deep_popular_science_article_style_3"
        / "module_02_structure"
        / "agent_3_table_of_contents.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_agent_3_toc(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "medium")
    try:
        return Agent(
            name="Style 3 · Agent 3 · Table of Contents",
            instructions=_load_prompt(),
            model=eff_model,
            output_type=AgentOutputSchema(Style3ToC, strict_json_schema=False),
        )
    except Exception:
        return Agent(
            name="Style 3 · Agent 3 · Table of Contents (plain)",
            instructions=_load_prompt(),
            model=eff_model,
        )



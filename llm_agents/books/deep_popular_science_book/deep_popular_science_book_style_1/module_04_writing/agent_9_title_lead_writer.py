from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import ArticleTitleLead
from utils.models import get_model


def _load_prompt() -> str:
    base_dir = (
        Path(__file__).resolve().parents[5]
        / "prompts"
        / "books"
        / "deep_popular_science_book"
        / "deep_popular_science_book_style_1"
        / "module_04_writing"
    )
    return (base_dir / "agent_9_title_lead_writer.md").read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_agent_9_title_lead_writer(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "heavy")
    try:
        return Agent(
            name="Deep Book · Agent 9 · Title & Book Lead",
            instructions=_load_prompt(),
            model=eff_model,
            output_type=AgentOutputSchema(ArticleTitleLead, strict_json_schema=False),
        )
    except Exception:
        return Agent(
            name="Deep Book · Agent 9 · Title & Book Lead (plain)",
            instructions=_load_prompt(),
            model=eff_model,
        )



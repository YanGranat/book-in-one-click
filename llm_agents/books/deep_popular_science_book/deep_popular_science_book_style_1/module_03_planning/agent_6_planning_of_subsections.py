from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.book import SubsectionPlan
from utils.models import get_model


def _load_prompt() -> str:
    base_dir = (
        Path(__file__).resolve().parents[5]
        / "prompts"
        / "books"
        / "deep_popular_science_book"
        / "deep_popular_science_book_style_1"
        / "module_03_planning"
    )
    return (base_dir / "agent_6_planning_of_subsections.md").read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_agent_6_subsection_plan(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "heavy")
    try:
        return Agent(
            name="Deep Book · Agent 6 · Subsection Plan",
            instructions=_load_prompt(),
            model=eff_model,
            output_type=AgentOutputSchema(SubsectionPlan, strict_json_schema=False),
        )
    except Exception:
        return Agent(
            name="Deep Book · Agent 6 · Subsection Plan (plain)",
            instructions=_load_prompt(),
            model=eff_model,
        )



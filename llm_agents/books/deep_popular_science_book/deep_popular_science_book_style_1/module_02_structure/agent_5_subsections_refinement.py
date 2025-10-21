from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.book import BookOutline
from utils.models import get_model


def _load_prompt() -> str:
    base_dir = (
        Path(__file__).resolve().parents[5]
        / "prompts"
        / "books"
        / "deep_popular_science_book"
        / "deep_popular_science_book_style_1"
        / "module_02_structure"
    )
    return (base_dir / "agent_5_subsections_refinement.md").read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore

    return Agent, AgentOutputSchema


def build_agent_5_subsections_refine(model: str | None = None, provider: str | None = None) -> Any:
    Agent, AgentOutputSchema = try_import_sdk()
    eff_provider = (provider or "openai").strip().lower()
    eff_model = model or get_model(eff_provider, "heavy")
    try:
        return Agent(
            name="Deep Book · Agent 5 · Subsections Refinement",
            instructions=_load_prompt(),
            model=eff_model,
            output_type=AgentOutputSchema(BookOutline, strict_json_schema=False),
        )
    except Exception:
        return Agent(
            name="Deep Book · Agent 5 · Subsections Refinement (plain)",
            instructions=_load_prompt(),
            model=eff_model,
        )



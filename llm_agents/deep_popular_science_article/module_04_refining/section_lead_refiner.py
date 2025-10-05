from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import LeadChunk


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "deep_popular_science_article" / "module_04_refining" / "section_lead_refiner.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_section_lead_refiner_agent(model: str | None = None) -> Any:
    Agent = try_import_sdk()
    return Agent(
        name="Deep Article · Section Lead Refiner",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=LeadChunk,
    )



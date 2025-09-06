from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.research import ResearchIterationNote


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "post" / "module_02_review" / "iterative_research.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""


def try_import_sdk():
    from agents import Agent  # type: ignore
    return Agent


def build_iterative_research_agent() -> Any:
    # Add WebSearchTool to enable actual lookups
    Agent = try_import_sdk()
    from agents import WebSearchTool  # type: ignore
    # Use fast model if available
    from utils.config import load_config
    cfg = load_config(__file__)
    fast_model = cfg.get("fast_model", "gpt-5-mini")
    return Agent(
        name="Iterative Researcher",
        instructions=_load_prompt(),
        model=fast_model,
        tools=[WebSearchTool()],
        output_type=ResearchIterationNote,
    )



from __future__ import annotations

from pathlib import Path
from typing import Any

from schemas.article import OutlineChangeList


def _load_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[3] / "prompts" / "deep_popular_science_article" / "module_02_research" / "topic_research.md"
    return prompt_path.read_text(encoding="utf-8")


def try_import_sdk():
    from agents import Agent, AgentOutputSchema  # type: ignore
    try:
        from agents import WebSearchTool  # type: ignore
    except Exception:
        WebSearchTool = None  # type: ignore
    return Agent, AgentOutputSchema, WebSearchTool


def build_topic_research_agent(model: str | None = None) -> Any:
    Agent, AgentOutputSchema, WebSearchTool = try_import_sdk()
    tools = []
    try:
        if WebSearchTool is not None:
            tools = [WebSearchTool()]
    except Exception:
        tools = []
    return Agent(
        name="Deep Article · Topic Research Planner",
        instructions=_load_prompt(),
        model=model or "gpt-5",
        output_type=AgentOutputSchema(OutlineChangeList, strict_json_schema=False),
        tools=tools or None,
    )



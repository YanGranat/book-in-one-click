from __future__ import annotations

from agents import Agent
from pathlib import Path


def build_meme_extractor_agent(model: str) -> Agent:
    prompt_path = Path("prompts") / "memes" / "meme_extractor" / "meme_extractor_style_1" / "meme_extractor.md"
    instructions = prompt_path.read_text(encoding="utf-8")
    # Hardening appended at caller layer for flexibility
    return Agent(name="Meme Extractor", instructions=instructions, model=model)

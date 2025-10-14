from pathlib import Path


def build_instructions(topic: str, lang: str, style: str = "post_style_1") -> str:
    # Return only the canonical system prompt text from file (single source of truth)
    style_key = (style or "post_style_1").strip().lower()
    if style_key not in {"post_style_1", "post_style_2"}:
        style_key = "post_style_1"
    prompt_path = (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "post"
        / style_key
        / "module_01_writing"
        / "writer.md"
    )
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
# Second pass formatter removed — system prompt is the single source of truth



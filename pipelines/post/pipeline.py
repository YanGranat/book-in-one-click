from pathlib import Path


def _post_style_dir(style_key: str) -> Path:
    base = Path(__file__).resolve().parents[2] / "prompts" / "posts"
    if style_key == "post_style_2":
        return base / "john_oliver_explains_post" / "john_oliver_explains_post_style_1"
    # default: popular science post style 1
    return base / "popular_science_post" / "popular_science_post_style_1"


def build_instructions(topic: str, lang: str, style: str = "post_style_1") -> str:
    # Return only the canonical system prompt text from file (single source of truth)
    style_key = (style or "post_style_1").strip().lower()
    if style_key not in {"post_style_1", "post_style_2"}:
        style_key = "post_style_1"
    prompt_path = _post_style_dir(style_key) / "module_01_writing" / "writer.md"
    return prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
# Second pass formatter removed — system prompt is the single source of truth



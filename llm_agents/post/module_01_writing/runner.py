from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import os


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except Exception as e:  # pragma: no cover
        raise


def _load_prompt() -> str:
    prompt_path = (
        Path(__file__).resolve().parents[3]
        / "prompts"
        / "post"
        / "module_01_writing"
        / "post.md"
    )
    return prompt_path.read_text(encoding="utf-8")


def _run_openai_with(system: str, user_message: str, model: Optional[str] = None) -> str:
    Agent, Runner = _try_import_sdk()
    agent = Agent(name="Popular Science Post Writer", instructions=system, model=(model or os.getenv("OPENAI_MODEL", "gpt-5")))
    res = Runner.run_sync(agent, user_message)
    return getattr(res, "final_output", "")


def _run_gemini_with(system: str, user_message: str, model_name: Optional[str] = None) -> str:
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    preferred = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    fallbacks = [
        preferred,
        os.getenv("GEMINI_MODEL", "gemini-2.0-pro-exp-02-05"),
        "gemini-2.0-pro",
        os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash"),
        "gemini-1.5-pro-latest",
    ]
    last_err = None
    for mname in fallbacks:
        try:
            model = genai.GenerativeModel(model_name=mname, system_instruction=system)
            resp = model.generate_content(user_message)
            return (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gemini request failed; last error: {last_err}")


def _run_claude_with(system: str, user_message: str, model_name: Optional[str] = None) -> str:
    import anthropic  # type: ignore
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    preferred = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    fallbacks = [preferred, "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
    last_err = None
    for mname in fallbacks:
        try:
            msg = client.messages.create(
                model=mname,
                max_tokens=4096,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            parts = []
            for blk in getattr(msg, "content", []) or []:
                txt = getattr(blk, "text", None)
                if txt:
                    parts.append(txt)
            return ("\n\n".join(parts)).strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Claude request failed; last error: {last_err}")


def run_post_writer(provider: str, *, instructions_override: Optional[str], user_message: str, speed: str = "heavy") -> str:
    system = instructions_override or _load_prompt()
    prov = (provider or "openai").strip().lower()
    if prov == "openai":
        model = os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini") if speed == "fast" else os.getenv("OPENAI_MODEL", "gpt-5")
        return _run_openai_with(system, user_message, model)
    if prov in {"gemini", "google"}:
        mname = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash") if speed == "fast" else os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        return _run_gemini_with(system, user_message, mname)
    # Claude
    cname = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    return _run_claude_with(system, user_message, cname)




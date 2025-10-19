#!/usr/bin/env python3
from __future__ import annotations

import os
import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from utils.models import get_model, get_json_mode, is_json_supported, get_thinking_config
from utils.logging import PipelineLogger


def _ensure_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception:
            pass


class ProviderRunner:
    def __init__(self, provider: str, *, logger: Optional[PipelineLogger] = None):
        # Normalize provider aliases and auto
        p = (provider or "openai").strip().lower()
        if p in {"", "auto"}:
            p = "openai"
        if p == "google":
            p = "gemini"
        if p == "anthropic":
            p = "claude"
        self.provider = p
        self.logger = logger

    # --- OpenAI ---
    def _openai_text(self, system: str, user_message: str, *, tier: str = "heavy") -> str:
        _ensure_loop()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        try:
            from agents import Agent, Runner  # type: ignore
            model = get_model("openai", tier)
            t0 = time.monotonic()
            try:
                if self.logger:
                    self.logger.info("LLM_START", extra={
                        "provider": "openai",
                        "tier": tier,
                        "model": model,
                        "system_len": len(system or ""),
                        "user_len": len(user_message or ""),
                    })
            except Exception:
                pass
            tools = []
            try:
                from agents import WebSearchTool  # type: ignore
                tools = [WebSearchTool()]
            except Exception:
                tools = []
            agent = Agent(name="Agent", instructions=system, model=model, tools=tools)
            res = Runner.run_sync(agent, user_message)
            out = getattr(res, "final_output", "")
            try:
                if self.logger:
                    self.logger.success("LLM_OK", extra={
                        "provider": "openai",
                        "tier": tier,
                        "model": model,
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "out_len": len(out or ""),
                    })
            except Exception:
                pass
            return out
        except Exception as e:
            print(f"❌ OpenAI API error (tier={tier}, model={get_model('openai', tier)}): {type(e).__name__}: {str(e)[:300]}")
            try:
                if self.logger:
                    self.logger.error("LLM_FAIL", exception=e, extra={
                        "provider": "openai",
                        "tier": tier,
                        "model": get_model("openai", tier),
                    })
            except Exception:
                pass
            raise

    # --- Gemini ---
    def _gemini_text(self, system: str, user_message: str, *, tier: str = "heavy", json_mode: bool = False) -> str:
        _ensure_loop()
        try:
            import google.generativeai as genai  # type: ignore
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
            genai.configure(api_key=api_key)
            mname = get_model("gemini", tier)
            t0 = time.monotonic()
            try:
                if self.logger:
                    self.logger.info("LLM_START", extra={
                        "provider": "gemini",
                        "tier": tier,
                        "model": mname,
                        "json_mode": bool(json_mode),
                        "system_len": len(system or ""),
                        "user_len": len(user_message or ""),
                    })
            except Exception:
                pass
            # Try to enable Google Search grounding when SDK supports it; otherwise fall back gracefully
            tools = None
            try:
                # Prefer typed Tool/GoogleSearch if available (newer SDKs)
                from google.generativeai.types import Tool as _Tool, GoogleSearch as _GoogleSearch  # type: ignore
                tools = [_Tool(google_search=_GoogleSearch())]
            except Exception:
                try:
                    # Older SDKs sometimes accept dict shape
                    tools = [{"google_search": {}}]
                except Exception:
                    tools = None
            gen_cfg = {"max_output_tokens": 8192}
            if json_mode and is_json_supported("gemini"):
                jm = get_json_mode("gemini")
                mime = jm.get("response_mime_type") or "application/json"
                gen_cfg["response_mime_type"] = mime
            # Create model; if tools are not recognized by SDK or backend, retry without tools
            try:
                model = genai.GenerativeModel(
                    model_name=mname,
                    system_instruction=system,
                    tools=tools if tools else None,
                    generation_config=gen_cfg or None,
                )
                resp = model.generate_content(user_message)
            except Exception:
                # Retry without tools to avoid "Unknown field for FunctionDeclaration: google_search"
                model = genai.GenerativeModel(
                    model_name=mname,
                    system_instruction=system,
                    generation_config=gen_cfg or None,
                )
                resp = model.generate_content(user_message)
            txt = (getattr(resp, "text", None) or "").strip()
            if not txt:
                parts: List[str] = []
                try:
                    for c in getattr(resp, "candidates", []) or []:
                        for part in getattr(getattr(c, "content", None), "parts", []) or []:
                            t = getattr(part, "text", None)
                            if t:
                                parts.append(t)
                except Exception:
                    pass
                txt = ("\n".join(parts)).strip()
            try:
                if self.logger:
                    self.logger.success("LLM_OK", extra={
                        "provider": "gemini",
                        "tier": tier,
                        "model": mname,
                        "json_mode": bool(json_mode),
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "out_len": len(txt or ""),
                    })
            except Exception:
                pass
            return txt
        except Exception as e:
            print(f"❌ Gemini API error (tier={tier}, model={get_model('gemini', tier)}, json_mode={json_mode}): {type(e).__name__}: {str(e)[:300]}")
            try:
                if self.logger:
                    self.logger.error("LLM_FAIL", exception=e, extra={
                        "provider": "gemini",
                        "tier": tier,
                        "model": get_model("gemini", tier),
                        "json_mode": bool(json_mode),
                    })
            except Exception:
                pass
            raise

    # --- Claude ---
    def _claude_text(self, system: str, user_message: str, *, tier: str = "heavy", json_mode: bool = False) -> str:
        _ensure_loop()
        try:
            import anthropic  # type: ignore
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Claude API key not found. Set ANTHROPIC_API_KEY environment variable.")
            client = anthropic.Anthropic(api_key=api_key)
            mname = get_model("claude", tier)
            t0 = time.monotonic()
            try:
                if self.logger:
                    self.logger.info("LLM_START", extra={
                        "provider": "claude",
                        "tier": tier,
                        "model": mname,
                        "json_mode": bool(json_mode),
                        "system_len": len(system or ""),
                        "user_len": len(user_message or ""),
                    })
            except Exception:
                pass
            # Claude relies on prompt instructions for JSON output, not response_format parameter
            # If JSON mode is requested, augment system prompt
            system_augmented = system
            if json_mode:
                system_augmented = f"{system}\n\nIMPORTANT: You MUST respond with valid JSON only. No explanations, no markdown fences, just pure JSON."
            kwargs: Dict[str, Any] = {
                "model": mname,
                "max_tokens": 8192,
                "system": system_augmented,
                "messages": [{"role": "user", "content": user_message}]
            }
            # Optional thinking support (if configured in models.json)
            try:
                tcfg = get_thinking_config("claude")
                if bool(tcfg.get("supported")):
                    cfg_budget = int(tcfg.get("default_budget_tokens", 0) or 0)
                    if cfg_budget > 0:
                        # Ensure Claude API invariant and leave headroom for output tokens
                        cur_max = int(kwargs.get("max_tokens", 8192))
                        # Hard-coded cap and simple policy: heavy Claude runs with thinking enabled
                        # Max tokens hard default 60k; budget from config but not exceeding cap-1
                        cap = 60000
                        final_budget = int(min(cfg_budget, cap - 1))
                        final_max = int(max(cap, final_budget + 1))  # ensure API invariant
                        kwargs["max_tokens"] = final_max
                        kwargs["thinking"] = {"type": "enabled", "budget_tokens": final_budget}
                        # Diagnostics for resolved limits
                        try:
                            import sys as _sys
                            print(
                                f"[CLAUDE][LIMITS] budget_cfg={cfg_budget} cur_max={cur_max} cap={cap} -> max_tokens={kwargs.get('max_tokens')} thinking={kwargs.get('thinking')}",
                                file=_sys.stderr,
                                flush=True,
                            )
                        except Exception:
                            pass
            except Exception:
                pass
            # Diagnostics: log request params
            try:
                import sys as _sys
                _think = kwargs.get("thinking")
                print(f"[CLAUDE][REQ] model={mname} max_tokens={kwargs.get('max_tokens')} thinking={_think}", file=_sys.stderr, flush=True)
            except Exception:
                pass
            # Prefer streaming for long‑running requests (extended thinking)
            import time as _time
            _t0 = _time.monotonic()
            try:
                out: List[str] = []
                _chunks = 0
                try:
                    # Anthropic SDK streaming helper (high‑level)
                    with client.messages.stream(**kwargs) as stream:  # type: ignore
                        try:
                            for chunk in getattr(stream, "text_stream", []):
                                out.append(str(chunk))
                                _chunks += 1
                        except Exception:
                            # Fallback: iterate raw events
                            try:
                                for ev in stream:  # type: ignore
                                    try:
                                        if getattr(ev, "type", "") == "content_block_delta":
                                            delta = getattr(ev, "delta", None)
                                            if delta and getattr(delta, "type", "") == "text_delta":
                                                t = getattr(delta, "text", "")
                                                if t:
                                                    out.append(str(t))
                                                    _chunks += 1
                                    except Exception:
                                        continue
                            except Exception:
                                pass
                        # Ensure completion and log usage
                        try:
                            _final = stream.get_final_message()  # type: ignore
                            try:
                                import sys as _sys
                                _usage = getattr(_final, "usage", None)
                                _stop = getattr(_final, "stop_reason", None)
                                _elapsed = _time.monotonic() - _t0
                                if _usage:
                                    print(f"[CLAUDE][USAGE][stream] usage={_usage} stop={_stop} chunks={_chunks} elapsed={_elapsed:.1f}s out_len={sum(len(s) for s in out)}", file=_sys.stderr, flush=True)
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    out = out or []
                txt_all = ("".join(out)).strip()
                if txt_all:
                    try:
                        if self.logger:
                            self.logger.success("LLM_OK", extra={
                                "provider": "claude",
                                "tier": tier,
                                "model": mname,
                                "json_mode": bool(json_mode),
                                "duration_ms": int((time.monotonic() - t0) * 1000),
                                "out_len": len(txt_all or ""),
                            })
                    except Exception:
                        pass
                    return txt_all
            except Exception:
                pass
            # Fallback to non‑streaming
            msg = client.messages.create(**kwargs)
            try:
                import sys as _sys
                _usage2 = getattr(msg, "usage", None)
                _stop2 = getattr(msg, "stop_reason", None)
                _elapsed2 = _time.monotonic() - _t0
                if _usage2:
                    print(f"[CLAUDE][USAGE][nonstream] usage={_usage2} stop={_stop2} elapsed={_elapsed2:.1f}s", file=_sys.stderr, flush=True)
            except Exception:
                pass
            parts: List[str] = []
            for blk in getattr(msg, "content", []) or []:
                txt = getattr(blk, "text", None)
                if txt:
                    parts.append(txt)
            txt_final = ("\n\n".join(parts)).strip()
            try:
                if self.logger:
                    self.logger.success("LLM_OK", extra={
                        "provider": "claude",
                        "tier": tier,
                        "model": mname,
                        "json_mode": bool(json_mode),
                        "duration_ms": int((time.monotonic() - t0) * 1000),
                        "out_len": len(txt_final or ""),
                    })
            except Exception:
                pass
            return txt_final
        except Exception as e:
            print(f"❌ Claude API error (tier={tier}, model={get_model('claude', tier)}, json_mode={json_mode}): {type(e).__name__}: {str(e)[:300]}")
            try:
                if self.logger:
                    self.logger.error("LLM_FAIL", exception=e, extra={
                        "provider": "claude",
                        "tier": tier,
                        "model": get_model("claude", tier),
                        "json_mode": bool(json_mode),
                    })
            except Exception:
                pass
            raise

    # --- Public API ---
    def run_text(self, system: str, user_message: str, *, speed: str = "heavy") -> str:
        p = self.provider
        tier = "fast" if (speed or "").strip().lower() == "fast" else "heavy"
        if p == "openai":
            return self._openai_text(system, user_message, tier=tier)
        if p in {"gemini", "google"}:
            return self._gemini_text(system, user_message, tier=tier, json_mode=False)
        return self._claude_text(system, user_message, tier=tier, json_mode=False)

    def run_json(self, system: str, user_message: str, *, speed: str = "fast") -> str:
        p = self.provider
        tier = "fast" if (speed or "").strip().lower() == "fast" else "heavy"
        if p == "openai":
            # OpenAI Agents SDK returns text; let caller parse JSON if needed
            return self._openai_text(system, user_message, tier=tier)
        if p in {"gemini", "google"}:
            return self._gemini_text(system, user_message, tier=tier, json_mode=True)
        return self._claude_text(system, user_message, tier=tier, json_mode=True)



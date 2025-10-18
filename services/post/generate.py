#!/usr/bin/env python3
"""
Reusable generation function for popular science posts.
Extracted from scripts/Popular_science_post.py and adapted for server/bot usage.

Key guarantees:
- No interactive input; accepts all parameters explicitly
- Can be safely called from a background thread
- Returns a Path to the saved Markdown file
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Type, Any
import os
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.models import get_model
from utils.logging import create_logger
from services.providers.runner import ProviderRunner
from utils.slug import safe_filename_base
from utils.web import build_search_context
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from pipelines.post.pipeline import build_instructions as build_post_instructions
from utils.json_parse import parse_json_best_effort
from schemas.research import ResearchPlan, QueryPack, ResearchIterationNote, SufficiencyDecision, Recommendation


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


def generate_post(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",  # openai|gemini|claude
    style: str = "post_style_1",  # post_style_1 | post_style_2
    factcheck: bool = True,
    factcheck_max_items: int = 0,
    research_iterations: int = 2,
    research_concurrency: int = 6,  # Parallel workers for fact-checking
    output_subdir: str = "post",
    on_progress: Optional[Callable[[str], None]] = None,
    job_meta: Optional[dict] = None,
    return_log_path: bool = False,
    use_refine: bool = True,
    # Series / overrides
    instructions_override: Optional[str] = None,
    series_topics: Optional[list[dict[str, Any]]] = None,
    series_current_id: Optional[str] = None,
    series_done_ids: Optional[list[str]] = None,
    # Side effects control (for series orchestrator)
    disable_db_record: bool = False,
    disable_file_save: bool = False,
    disable_sidecar_log: bool = False,
    return_content: bool = False,
) -> Path:
    """
    Generate a popular science post and save it to output/<output_subdir>/.

    Returns the Path to the saved .md file.
    """
    # Ensure project modules are importable and env is loaded
    _ensure_root(__file__)
    load_env_from_root(__file__)

    if not topic or not topic.strip():
        raise ValueError("Topic must be a non-empty string")

    _prov = (provider or "openai").strip().lower()
    if _prov == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found in environment")
    elif _prov == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment")
    elif _prov in {"gemini", "google"}:
        if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not found in environment")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Ensure this thread has an event loop for libs that expect it
    try:
        asyncio.get_event_loop()
    except Exception:
        asyncio.set_event_loop(asyncio.new_event_loop())

    Agent, Runner = _try_import_sdk()

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass
    def _fallback_title_from_text(text: str) -> str:
        try:
            import re as _re
            s = (text or "").strip()
            if not s:
                return ""
            # Use first non-empty line/sentence
            first_line = s.splitlines()[0].strip()
            # Strip markdown bullets/prefixes
            first_line = _re.sub(r"^[#>\-\*\s]+", "", first_line)
            # Truncate
            if len(first_line) > 80:
                first_line = first_line[:77].rstrip() + "…"
            return first_line
        except Exception:
            return ""


    _emit("start:post")
    from datetime import datetime
    started_at = datetime.utcnow()
    started_perf = time.perf_counter()
    
    # Initialize structured logger
    logger = create_logger("post", show_debug=bool(os.getenv("DEBUG_LOGS")))
    logger.info(f"Starting post generation: '{topic[:100]}'")
    logger.info(f"Configuration: provider={_prov}, lang={lang}, style={style_key}")
    
    log_lines = []
    def log(section: str, body: str):
        """Log to markdown file - keep it readable and high-level."""
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")
    
    def log_summary(emoji: str, title: str, items: list[str]):
        """Log a clean summary without technical details."""
        content = "\n".join(f"- {item}" for item in items if item)
        log_lines.append(f"---\n\n## {emoji} {title}\n\n{content}\n")
    # Normalize style
    style_key = (style or "post_style_1").strip().lower()
    if style_key not in {"post_style_1", "post_style_2"}:
        style_key = "post_style_1"

    # Log generation configuration
    log_summary("⚙️", "Конфигурация генерации", [
        f"Провайдер: {_prov}",
        f"Язык: {lang}",
        f"Стиль: {style_key}",
        f"Тема: {topic[:100]}{'...' if len(topic) > 100 else ''}"
    ])

    instructions = instructions_override or build_post_instructions(topic, lang, style_key)
    # Defaults for variables referenced by nested functions in non-OpenAI factcheck path
    pref: list[str] = []
    p_iter: Optional[str] = None
    p_suff: Optional[str] = None
    p_rec: Optional[str] = None
    p_qs: Optional[str] = None

    def _run_openai_with(system: str, user_message_local: str, model: Optional[str] = None) -> str:
        agent = Agent(
            name="Generic Agent",
            instructions=system,
            model=(model or get_model("openai", "heavy")),
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    def _run_openai() -> str:
        agent = Agent(
            name="Popular Science Post Writer",
            instructions=instructions,
            model=get_model("openai", "heavy"),
        )
        user_message_local = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
            f"</input>"
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    # Provider-agnostic runners

    def run_with_provider(system: str, user_inp: str, speed: str = "heavy") -> str:
        pr = ProviderRunner(_prov)
        if _prov == "openai":
            # Prefer Agents SDK path for OpenAI to keep tools/session parity
            model = get_model("openai", "fast" if speed == "fast" else "heavy")
            return _run_openai_with(system, user_inp, model)
        return pr.run_text(system, user_inp, speed=("fast" if speed == "fast" else "heavy"))

    def run_json_with_provider(system: str, user_inp: str, cls: Type, speed: str = "fast"):
        import json
        pr = ProviderRunner(_prov)
        if _prov in {"gemini", "google", "claude"}:
            txt = pr.run_json(system, user_inp, speed=("fast" if speed == "fast" else "heavy"))
        else:
            txt = pr.run_text(system, user_inp, speed=("fast" if speed == "fast" else "heavy"))
        # strip code fences if any
        if txt.strip().startswith("```") and txt.strip().endswith("```"):
            txt = "\n".join([line for line in txt.strip().splitlines()[1:-1]])

        def _norm_key(s: str) -> str:
            return "".join(ch for ch in (s or "").lower() if ch.isalnum())

        def _unwrap(obj):
            try:
                # unwrap named root {"ClassName": {...}}
                if isinstance(obj, dict):
                    keys = list(obj.keys())
                    cname = _norm_key(cls.__name__)
                    if len(keys) == 1 and _norm_key(keys[0]) == cname and isinstance(obj[keys[0]], (dict, list)):
                        obj = obj[keys[0]]
                    # unwrap common wrappers when they are the only key
                    for k in ("data", "output", "result", "response"):
                        if isinstance(obj, dict) and list(obj.keys()) == [k] and isinstance(obj[k], (dict, list)):
                            obj = obj[k]
                return obj
            except Exception:
                return obj

        def _snake_case(name: str) -> str:
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", "_", name)
            s = s.replace("-", "_")
            return s.lower()

        def _to_confidence(value):
            if isinstance(value, (int, float)):
                v = float(value)
                return max(0.0, min(1.0, v if v <= 1.0 else v / 100.0))
            if isinstance(value, str):
                s = value.strip().lower()
                scale = {
                    "low": 0.25,
                    "medium": 0.5,
                    "med": 0.5,
                    "mid": 0.5,
                    "high": 0.75,
                    "very high": 0.9,
                    "vh": 0.9,
                    "confident": 0.8,
                }
                if s in scale:
                    return scale[s]
                try:
                    num = float(s)
                    return max(0.0, min(1.0, num if num <= 1.0 else num / 100.0))
                except Exception:
                    return 0.0
            return 0.0

        def _normalize(obj):
            # Recursively convert keys to snake_case and coerce common fields
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    nk = _snake_case(k)
                    out[nk] = _normalize(v)
                # Heuristics for expected schemas
                if _norm_key(cls.__name__) == "querypack":
                    if "point_id" not in out:
                        if isinstance(out.get("point"), dict) and "id" in out["point"]:
                            out["point_id"] = out["point"].get("id")
                        elif "id" in out:
                            out["point_id"] = out.get("id")
                    if isinstance(out.get("queries"), str):
                        raw = out["queries"].replace("\r", "\n")
                        items = [q.strip() for q in raw.replace(";", "\n").split("\n") if q.strip()]
                        out["queries"] = items
                if _norm_key(cls.__name__) == "sufficiencydecision":
                    if isinstance(out.get("done"), str):
                        out["done"] = out["done"].strip().lower() in {"true", "yes", "1"}
                    if "confidence" in out:
                        out["confidence"] = _to_confidence(out.get("confidence"))
                if _norm_key(cls.__name__) == "recommendation":
                    act = out.get("action")
                    if isinstance(act, str):
                        a = act.strip().lower()
                        syn = {
                            "retain": "keep",
                            "keep": "keep",
                            "ok": "keep",
                            "clarify": "clarify",
                            "edit": "rewrite",
                            "rewrite": "rewrite",
                            "reword": "rewrite",
                            "remove": "remove",
                            "delete": "remove",
                        }
                        out["action"] = syn.get(a, a)
                if _norm_key(cls.__name__) == "researchiterationnote":
                    if isinstance(out.get("queries"), str):
                        raw = out["queries"].replace("\r", "\n")
                        items = [q.strip() for q in raw.replace(";", "\n").split("\n") if q.strip()]
                        out["queries"] = items
                    # findings can come as list of strings
                    if isinstance(out.get("findings"), list):
                        out["findings"] = "\n".join(str(x) for x in out["findings"]) or ""
                    # evidence may be list of strings or dicts with url/title
                    ev = out.get("evidence")
                    if isinstance(ev, list):
                        norm_ev = []
                        for item in ev:
                            if isinstance(item, str):
                                norm_ev.append({"url": item})
                            elif isinstance(item, dict):
                                url = item.get("url") or item.get("link") or item.get("href") or item.get("source")
                                title = item.get("title") or item.get("name")
                                snippet = item.get("snippet") or item.get("summary") or item.get("text")
                                if url:
                                    norm_ev.append({"url": url, "title": title, "snippet": snippet})
                        out["evidence"] = norm_ev
                    # confidence may come as string labels
                    if "confidence" in out:
                        out["confidence"] = _to_confidence(out.get("confidence"))
                return out
            if isinstance(obj, list):
                return [_normalize(x) for x in obj]
            return obj

        try:
            data_obj = parse_json_best_effort(txt)
            data_obj = _unwrap(data_obj)
            data_obj = _normalize(data_obj)
            return cls.model_validate(data_obj)
        except Exception as e:
            if _prov in {"gemini", "google", "claude"} and speed != "heavy":
                return run_json_with_provider(system, user_inp, cls, speed="heavy")
            raise RuntimeError(f"Failed to parse JSON for {cls.__name__}: {str(e)[:200]}")

    import json as _json
    series_block = ""
    if series_topics is not None:
        try:
            series_block = (
                "\n<series>\n"
                f"<topics_json>{_json.dumps(series_topics, ensure_ascii=False)}</topics_json>\n"
                f"<current_id>{series_current_id or ''}</current_id>\n"
                f"<done_ids>{','.join(series_done_ids or [])}</done_ids>\n"
                "</series>\n"
            )
        except Exception:
            # Fallback: omit series context on failure
            series_block = ""
    user_message_local_writer = (
        f"<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{(lang or 'auto').strip()}</lang>\n"
        f"{series_block}"
        f"</input>"
    )
    # Log writer input for transparency (full prompt for tracking evolution)
    log("⬇️ Writer · Input", user_message_local_writer)
    # Use explicit Agent for OpenAI via writer module; others use provider runner
    if _prov == "openai":
        try:
            if style_key == "post_style_2":
                from llm_agents.post.post_style_2.module_01_writing.writer import build_post_writer_agent  # type: ignore
            else:
                from llm_agents.post.post_style_1.module_01_writing.writer import build_post_writer_agent  # type: ignore
            if style_key == "post_style_2":
                # For style 2: send only user message (no system instructions)
                from agents import Agent as _Agent, ModelSettings as _MS  # type: ignore
                # Use chat-latest; do NOT set reasoning (unsupported for this model)
                agent = _Agent(name="Style2 Writer (User-only)", instructions="", model="gpt-5-chat-latest")
                # Build user message strictly from writer.md with only <topic> substituted
                from pathlib import Path as _P
                _writer_prompt = (
                    _P(__file__).resolve().parents[2]
                    / "prompts"
                    / "post"
                    / "post_style_2"
                    / "module_01_writing"
                    / "writer.md"
                ).read_text(encoding="utf-8")
                tmpl = _writer_prompt.replace("<topic>", topic)
                res_local = Runner.run_sync(agent, tmpl)
            else:
                agent = build_post_writer_agent(
                    model=os.getenv("OPENAI_MODEL", "gpt-5"),
                    instructions_override=instructions,
                )
                res_local = Runner.run_sync(agent, user_message_local_writer)
            content_raw = getattr(res_local, "final_output", "")
            # Log raw writer output for tracking
            try:
                log("✍️ Writer · Raw", f"len={len(str(content_raw or ''))}")
            except Exception:
                pass
            if style_key == "post_style_2":
                # Second step: run dedicated agent built on title_json.md prompt (SDK path),
                # which removes disclaimers and returns strict JSON {title, text}.
                from llm_agents.post.post_style_2.module_01_writing.title_json import build_title_json_agent  # type: ignore
                # Force gpt-5 for title agent to support reasoning
                title_agent = build_title_json_agent(model="gpt-5")
                
                # Pass target language alongside text for potential translation to <lang>
                tj_payload = (
                    "<input>\n"
                    f"<lang>{(lang or 'auto').strip()}</lang>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<text>\n{str(content_raw or '')}\n</text>\n"
                    "</input>"
                )
                tj_res = Runner.run_sync(title_agent, tj_payload)
                tj_raw = getattr(tj_res, "final_output", "")
                from utils.json_parse import parse_json_best_effort as _pjson
                try:
                    obj = _pjson(tj_raw)
                    title = str((obj or {}).get("title") or "").strip() or _fallback_title_from_text(str(content_raw or ""))
                    body = str((obj or {}).get("text") or (obj or {}).get("post") or "").strip()
                    header = f"**{title}**\n\n" if title else ""
                    content = (header + body).strip()
                    # Log title extraction for tracking
                    try:
                        log("🧩 Title JSON · Agent", f"title={title!r}, body_len={len(body)}")
                    except Exception:
                        pass
                except Exception as e:
                    # Provider JSON fallback if SDK agent output isn't clean JSON
                    try:
                        from pathlib import Path as _P
                        tprompt = (
                            _P(__file__).resolve().parents[2]
                            / "prompts"
                            / "post"
                            / "post_style_2"
                            / "module_01_writing"
                            / "title_json.md"
                        ).read_text(encoding="utf-8")
                        pr_local = ProviderRunner(_prov)
                        tj = pr_local.run_json(tprompt, tj_payload, speed="heavy")
                        obj = _pjson(tj)
                        title = str((obj or {}).get("title") or "").strip() or _fallback_title_from_text(str(content_raw or ""))
                        body = str((obj or {}).get("text") or (obj or {}).get("post") or "").strip()
                        header = f"**{title}**\n\n" if title else ""
                        content = (header + body).strip()
                        # Log fallback extraction
                        try:
                            log("🧩 Title JSON · Fallback(JSON)", f"title={title!r}, body_len={len(body)}")
                        except Exception:
                            pass
                    except Exception as e2:
                        # Final fallback: derive a title and use raw body
                        title = _fallback_title_from_text(str(content_raw or ""))
                        header = f"**{title}**\n\n" if title else ""
                        content = (header + str(content_raw or "")).strip()
                        # Log final fallback reason
                        try:
                            log("⚠️ Title JSON · Fallback(Text)", f"reason={type(e).__name__}/{type(e2).__name__}")
                        except Exception:
                            pass
            else:
                content = str(content_raw).strip()
        except Exception:
            # For style 2, do not fallback; surface the error to the caller
            if style_key == "post_style_2":
                raise
            content = run_with_provider(instructions, user_message_local_writer, speed="heavy")
    else:
        # For non-OpenAI Style 2: send user-only prompt and then wrap with title_json via provider runner
        if style_key == "post_style_2":
            # step 1: writer
            tmpl = (instructions or "").replace("<topic>", topic).replace("<lang>", (lang or "auto").strip())
            writer_text = run_with_provider("", tmpl, speed="heavy")
            # Log raw writer output for tracking
            try:
                log("✍️ Writer · Raw", f"len={len(str(writer_text or ''))}")
            except Exception:
                pass
            # step 2: title json (prefer provider JSON mode for non-OpenAI)
            from pathlib import Path as _P
            tprompt = (_P(__file__).resolve().parents[2] / "prompts" / "post" / "post_style_2" / "module_01_writing" / "title_json.md").read_text(encoding="utf-8")
            tj_payload = (
                "<input>\n"
                f"<lang>{(lang or 'auto').strip()}</lang>\n"
                f"<topic>{topic}</topic>\n"
                f"<text>\n{str(writer_text or '')}\n</text>\n"
                "</input>"
            )
            try:
                pr_local = ProviderRunner(_prov)
                tj = pr_local.run_json(tprompt, tj_payload, speed="heavy")
            except Exception:
                tj = run_with_provider(tprompt, tj_payload, speed="heavy")
            try:
                from utils.json_parse import parse_json_best_effort as _pjson
                obj = _pjson(tj)
                title = str((obj or {}).get("title") or "").strip() or _fallback_title_from_text(writer_text)
                body = str((obj or {}).get("text") or (obj or {}).get("post") or "").strip()
                header = f"**{title}**\n\n" if title else ""
                content = (header + body).strip()
            except Exception:
                title = _fallback_title_from_text(writer_text or "")
                header = f"**{title}**\n\n" if title else ""
                content = (header + (writer_text or "")).strip()
        else:
            content = run_with_provider(instructions, user_message_local_writer, speed="heavy")
    # Log final writer output for tracking evolution
    log("📦 Final · Output", content)
    
    if not content:
        raise RuntimeError("Empty result from writer agent")
    
    # Log writing summary
    content_lines = content.count('\n') + 1
    content_chars = len(content)
    log_summary("✍️", "Первый черновик готов", [
        f"Длина: {content_chars} символов",
        f"Строк: {content_lines}",
        f"Первые 150 символов:",
        f"_{content[:150]}..._"
    ])

    # Define helper classes at function scope to ensure instances persist across blocks
    class _SimpleItem:
        def __init__(self, claim_text: str, verdict: str, reason: str):
            self.claim_text = claim_text
            self.verdict = verdict
            self.reason = reason

    class _SimpleReport:
        def __init__(self, items):
            self.items = items

        def model_dump_json(self):
            import json
            return json.dumps(
                {
                    "summary": "Issues only for rewrite (exclude confirmed)",
                    "items": [
                        {
                            "claim_text": i.claim_text,
                            "verdict": i.verdict,
                            "reason": i.reason,
                        }
                        for i in self.items
                    ],
                },
                ensure_ascii=False,
            )

    class _TmpReport:
        def __init__(self, point_id, notes):
            self.point_id = point_id
            self.notes = notes
            self.synthesis = ""

        def model_dump_json(self):
            import json
            return json.dumps(
                {
                    "point_id": self.point_id,
                    "notes": [n.model_dump() for n in self.notes],
                    "synthesis": self.synthesis,
                },
                ensure_ascii=False,
            )

    report = None
    # Style 2: skip fact-check entirely
    if factcheck and style_key == "post_style_2":
        factcheck = False

    if factcheck:
        if _prov == "openai":
            # Preserve original OpenAI Agents SDK flow (with WebSearchTool)
            if style_key == "post_style_2":
                # Safety: should not happen due to guard above
                report = None
            else:
                from llm_agents.post.post_style_1.module_02_review.identify_points import build_identify_points_agent  # type: ignore
                from llm_agents.post.post_style_1.module_02_review.iterative_research import build_iterative_research_agent  # type: ignore
                from llm_agents.post.post_style_1.module_02_review.recommendation import build_recommendation_agent  # type: ignore
                from llm_agents.post.post_style_1.module_02_review.sufficiency import build_sufficiency_agent  # type: ignore
            from utils.config import load_config

            _emit("factcheck:init")
            logger.stage("Fact-Checking", total_stages=4, current_stage=2)
            logger.info(f"Fact-checking enabled: max {factcheck_max_items} points, {research_iterations} iterations")
            identify_agent = build_identify_points_agent()
            identify_result = Runner.run_sync(identify_agent, f"<post>\n{content}\n</post>\n<lang>{lang}</lang>")
            plan = identify_result.final_output  # type: ignore
            points = plan.points or []
            if factcheck_max_items and factcheck_max_items > 0:
                points = points[: factcheck_max_items]
            
            # Log fact-check plan for OpenAI
            try:
                log("🔎 Fact-check · Plan (OpenAI)", f"points={len(points)}")
            except Exception:
                pass
            
            if not points:
                log_summary("ℹ️", "Факт-чек пропущен", [
                    "Рискованных утверждений не обнаружено"
                ])
                report = None
            else:
                # Log agent initialization
                log("🔧 Building agents", f"Starting fact-check for {len(points)} points")
                logger.step(f"Verifying {len(points)} points with web search")
                research_agent = build_iterative_research_agent()
                suff_agent = build_sufficiency_agent()
                rec_agent = build_recommendation_agent()
                log("✓ Agents ready", "All agents initialized")

                def _run_sync_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
                    # In ThreadPoolExecutor threads, asyncio.run() will create and manage its own event loop
                    for i in range(attempts):
                        try:
                            result = asyncio.run(Runner.run(agent, inp))
                            return result
                        except Exception as e:
                            if i == attempts - 1:
                                raise
                            time.sleep(base_delay * (2 ** i))

                def process_point_sync(p):
                    # Skip query synthesis for OpenAI - WebSearchTool handles this internally
                    # The agent will autonomously decide what to search for

                    notes = []
                    for step in range(1, max(1, int(research_iterations)) + 1):
                        rr_input = (
                            "<input>\n"
                            f"<point>{p.model_dump_json()}</point>\n"
                            f"<step>{step}</step>\n"
                            "</input>"
                        )
                        note_res = _run_sync_with_retries(research_agent, rr_input)
                        note = note_res.final_output  # type: ignore
                        notes.append(note)

                        suff_input = (
                            "<input>\n"
                            f"<point>{p.model_dump_json()}</point>\n"
                            f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                            "</input>"
                        )
                        decision_res = _run_sync_with_retries(suff_agent, suff_input)
                        decision = decision_res.final_output  # type: ignore
                        if decision.done:
                            break

                    rr = _TmpReport(p.id, notes)
                    rec_res = _run_sync_with_retries(
                        rec_agent,
                        f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n</input>",
                    )
                    rec = rec_res.final_output  # type: ignore
                    return p, rec, notes

                # Parallelize across points using threads (Runner.run_sync is synchronous)
                # Bound concurrency by research_concurrency to avoid provider rate limits
                max_workers = max(1, int(research_concurrency))
                log("🔍 Processing points", f"total={len(points)}, workers={max_workers}")
                logger.parallel_start(f"Checking {len(points)} points", total_jobs=len(points), max_workers=max_workers)
                results = []
                fc_t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    future_map = {pool.submit(process_point_sync, p): p for p in points}
                    for fut in as_completed(list(future_map.keys())):
                        try:
                            result = fut.result()
                            results.append(result)
                            # Log each processed point for tracking
                            log("✓ Point processed", f"{result[0].id}: {result[0].text[:60]}...")
                        except Exception as e:
                            # Skip failed point; continue others
                            p_failed = future_map[fut]
                            log("✗ Point failed", f"{p_failed.id}: {str(e)[:200]}")
                            pass
                fc_duration = time.perf_counter() - fc_t0
                log("🔍 Processing complete", f"successful={len(results)}/{len(points)}")
                logger.parallel_complete(succeeded=len(results), failed=len(points)-len(results), duration=fc_duration)

                simple_items = []
                kept_count = 0
                issues_by_action = {"clarify": 0, "rewrite": 0, "remove": 0}
                for (p, r, notes) in results:
                    action = getattr(r, "action", "keep")
                    if action == "keep":
                        kept_count += 1
                        continue
                    if r.action == "clarify":
                        verdict = "uncertain"
                        issues_by_action["clarify"] += 1
                    elif r.action == "rewrite" or r.action == "remove":
                        verdict = "fail"
                        issues_by_action[r.action] += 1
                    else:
                        verdict = "fail"
                    reason = getattr(r, "explanation", "") or ""
                    simple_items.append(_SimpleItem(p.text, verdict, reason))
                    # Log each issue for tracking
                    log("⚠️ Issue found", f"action={action}, point={p.text[:60]}...")

                if kept_count > 0:
                    log("✅ Points confirmed", f"{kept_count} point(s) passed fact-check")

                report = _SimpleReport(simple_items) if simple_items else None
                
                # Log fact-check summary JSON for tracking
                if report is not None:
                    log("factcheck_summary", report.model_dump_json())
                else:
                    log("✅ Fact-check complete", "No issues found, skipping rewrite")
                
                # Log readable summary
                if report is not None:
                    issues_str = ", ".join([f"{k}: {v}" for k, v in issues_by_action.items() if v > 0])
                    log_summary("🔍", "Результат факт-чека", [
                        f"Проверено утверждений: {len(results)}",
                        f"Подтверждено: {kept_count}",
                        f"Требуют внимания: {len(simple_items)}",
                        f"  ({issues_str})" if issues_str else ""
                    ])
                else:
                    log_summary("✅", "Факт-чек пройден", [
                        f"Проверено утверждений: {len(results)}",
                        "Все утверждения подтверждены"
                    ])
        else:
            from utils.config import load_config
            from pathlib import Path

            _emit("factcheck:init")
            if style_key == "post_style_2":
                # Non-OpenAI path also skips FC for style 2
                try:
                    log("ℹ️ Fact-check skipped", "Style 2 does not support fact-checking")
                except Exception:
                    pass
                points = []
            else:
                base = Path(__file__).resolve().parents[2] / "prompts" / "post" / style_key / "module_02_review"
                p_ident = (base / "identify_risky_points.md").read_text(encoding="utf-8")
                plan = run_json_with_provider(
                p_ident
                + "\n\n<format>\nВерни строго JSON-объект ResearchPlan без пояснений.\n</format>\n",
                f"<post>\n{content}\n</post>\n<lang>{lang}</lang>",
                ResearchPlan,
                speed="fast",
            )
                points = plan.points or []
            # Fallback: if Gemini/Claude produced empty plan, retry on heavy model with strict JSON requirement and min points
            if not points:
                try:
                    strict_ident = (
                        p_ident
                        + "\n\n<requirements>\n"
                        + "- Верни строго JSON-объект вида {\"points\":[...]}.\n"
                        + "- Минимум 3 пункта (p01, p02, p03), даже если текст общий.\n"
                        + "- Пункты атомарные, конкретные (факт/цифра/датировка/причинно-следственная связь).\n"
                        + "</requirements>\n"
                    )
                    plan_heavy = run_json_with_provider(strict_ident, f"<post>\n{content}\n</post>\n<lang>{lang}</lang>", ResearchPlan, speed="heavy")
                    points = plan_heavy.points or []
                    if points:
                        plan = plan_heavy  # Update plan to heavy version if it succeeded
                    # Log fallback plan
                    try:
                        log("🔎 Fact-check · Plan (fallback)", f"points={len(points)}")
                    except Exception:
                        pass
                except Exception:
                    # keep empty; downstream will no-op
                    points = []
            if factcheck_max_items and factcheck_max_items > 0:
                points = points[: factcheck_max_items]
            
            # Log fact-check plan JSON for tracking (use latest plan)
            try:
                if hasattr(plan, 'model_dump_json'):
                    log("🔎 Fact-check · Plan", f"```json\n{plan.model_dump_json()}\n```")
                else:
                    log("🔎 Fact-check · Plan", f"points={len(points)}")
            except Exception:
                log("🔎 Fact-check · Plan", f"points={len(points)}")
            
            if not points:
                log_summary("ℹ️", "Факт-чек пропущен", [
                    "Рискованных утверждений не обнаружено"
                ])

            cfg = load_config(__file__)
            pref = (cfg.get("research", {}) or {}).get("preferred_domains", [])
            if style_key == "post_style_1":
                p_iter = (base / "iterative_research.md").read_text(encoding="utf-8")
                p_suff = (base / "sufficiency.md").read_text(encoding="utf-8")
                p_rec = (base / "recommendation.md").read_text(encoding="utf-8")
                p_qs = (base / "query_synthesizer.md").read_text(encoding="utf-8")
            else:
                p_iter = p_suff = p_rec = p_qs = None

            async def _run_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
                for i in range(attempts):
                    try:
                        return await Runner.run(agent, inp)
                    except Exception:
                        if i == attempts - 1:
                            raise
                        await asyncio.sleep(base_delay * (2 ** i))

            async def process_point_async(p):
                # Query synthesis
                cfg_pref = ",".join(pref)
                qp = run_json_with_provider(
                    (p_qs or ""),
                    f"<input>\n<point>{p.model_dump_json()}</point>\n<preferred_domains>{cfg_pref}</preferred_domains>\n</input>",
                    QueryPack,
                    speed="fast",
                )
                # Web context build (provider-agnostic)
                queries = getattr(qp, "queries", []) or []
                # Reduce external fetch workload to improve latency under rate limits
                try:
                    _per_q = int(os.getenv("FC_WEB_PER_QUERY", "1"))
                except Exception:
                    _per_q = 1
                try:
                    _max_chars = int(os.getenv("FC_WEB_MAX_CHARS", "1600"))
                except Exception:
                    _max_chars = 1600
                web_ctx = build_search_context(queries, per_query=max(1, _per_q), max_chars=max(200, _max_chars))
                # Log web queries and sources for tracking
                if queries:
                    log("🌐 Web · Queries", "\n".join([f"- {q}" for q in queries]))
                # Extract sources (best-effort)
                try:
                    import re as _re
                    urls = _re.findall(r"url=\"([^\"]+)\"", web_ctx)
                    if urls:
                        log("🌐 Web · Sources", "\n".join([f"- {u}" for u in urls[:20]]))
                except Exception:
                    pass

                notes = []
                for step in range(1, max(1, int(research_iterations)) + 1):
                    rr_input = (
                        "<input>\n"
                        f"<point>{p.model_dump_json()}</point>\n"
                        f"<step>{step}</step>\n"
                        f"<web_context>\n{web_ctx}\n</web_context>\n"
                        "</input>"
                    )
                    note = run_json_with_provider(p_iter or "", rr_input, ResearchIterationNote, speed="fast")
                    notes.append(note)

                    suff_input = (
                        "<input>\n"
                        f"<point>{p.model_dump_json()}</point>\n"
                        f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                        f"<lang>{lang}</lang>\n"
                        "</input>"
                    )
                    decision = run_json_with_provider(p_suff or "", suff_input, SufficiencyDecision, speed="fast")
                    if decision.done:
                        break

                rr = _TmpReport(p.id, notes)
                rec = run_json_with_provider(
                    (p_rec or ""),
                    f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n<lang>{lang}</lang>\n</input>",
                    Recommendation,
                    speed="fast",
                )
                return p, rec, notes

            # Process all points sequentially for non-OpenAI providers (with DuckDuckGo)
            results = []
            # Ensure we have an event loop for async operations
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if points:
                logger.step(f"Verifying {len(points)} points with web search")
            
            for p in points or []:
                try:
                    result = loop.run_until_complete(process_point_async(p))
                    results.append(result)
                    # Log each processed point for tracking
                    log("✓ Point processed", f"{p.id}: {p.text[:60]}...")
                except Exception as e:
                    # Log failed points
                    log("✗ Point failed", f"{p.id}: {str(e)[:200]}")
                    pass
            
            # Log processing complete
            log("🔍 Processing complete", f"successful={len(results)}/{len(points or [])}")

            simple_items = []
            kept_count = 0
            issues_by_action = {"clarify": 0, "rewrite": 0, "remove": 0}
            for (p, r, notes) in results:
                action = getattr(r, "action", "keep")
                if action == "keep":
                    kept_count += 1
                    continue  # confirmed parts не передаём в переписывание
                if r.action == "clarify":
                    verdict = "uncertain"
                    issues_by_action["clarify"] += 1
                elif r.action == "rewrite" or r.action == "remove":
                    verdict = "fail"
                    issues_by_action[r.action] += 1
                else:
                    verdict = "fail"
                reason = getattr(r, "explanation", "") or ""
                simple_items.append(_SimpleItem(p.text, verdict, reason))
        
            # Log points confirmed
            if kept_count > 0:
                log("✅ Points confirmed", f"{kept_count} point(s) passed fact-check")
        
            report = _SimpleReport(simple_items) if simple_items else None
        
            # Log fact-check summary JSON for tracking
            if report is not None:
                log("factcheck_summary", report.model_dump_json())
        
            # Log readable summary for non-OpenAI
            if points:
                if report is not None and simple_items:
                    issues_str = ", ".join([f"{k}: {v}" for k, v in issues_by_action.items() if v > 0])
                    log_summary("🔍", "Результат факт-чека", [
                        f"Проверено утверждений: {len(results)}",
                        f"Подтверждено: {kept_count}",
                        f"Требуют внимания: {len(simple_items)}",
                        f"  ({issues_str})" if issues_str else ""
                    ])
                else:
                    log_summary("✅", "Факт-чек пройден", [
                        f"Проверено утверждений: {len(results)}",
                        "Все утверждения подтверждены"
                    ])

    # Rewrite and refine
    final_content = content
    if report is not None:
        needs_rewrite = any(i.verdict != "pass" for i in report.items)
        if needs_rewrite:
            _emit("rewrite:init")
            logger.stage("Rewriting Content", total_stages=4, current_stage=3)
            logger.info(f"Rewriting needed for {sum(1 for i in report.items if i.verdict != 'pass')} points")
            from pathlib import Path
            if style_key == "post_style_1":
                p_rewrite = (Path(__file__).resolve().parents[2] / "prompts" / "post" / style_key / "module_03_rewriting" / "rewrite.md").read_text(encoding="utf-8")
            else:
                p_rewrite = ""
            rw_input = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<post>\n\n{content}\n\n</post>\n"
                f"<critique_json>\n\n{report.model_dump_json()}\n\n</critique_json>\n"
                "</input>"
            )
            # Log rewrite input for tracking evolution
            log("⬇️ Rewrite · Input", rw_input)
            logger.step(f"Rewriting content based on {len(report.items)} critique points")
            final_content = run_with_provider(p_rewrite, rw_input, speed="heavy") or content
            # Log rewrite output
            log("🛠️ Rewrite · Output", final_content)
            log_summary("🛠️", "Контент переписан", [
                f"Исправлено проблем: {sum(1 for i in report.items if i.verdict != 'pass')}",
                f"Новая длина: {len(final_content)} символов"
            ])

    from pathlib import Path
    if use_refine and style_key == "post_style_1":
        _emit("refine:init")
        logger.stage("Final Refinement", total_stages=4, current_stage=4)
        logger.step("Polishing content for publication")
        p_refine = (Path(__file__).resolve().parents[2] / "prompts" / "post" / style_key / "module_03_rewriting" / "refine.md").read_text(encoding="utf-8")
        refine_input = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<post>\n\n{final_content}\n\n</post>\n"
            "</input>"
        )
        # Log refine input for tracking evolution
        log("⬇️ Refine · Input", refine_input)
        final_content = run_with_provider(p_refine, refine_input, speed="heavy") or final_content
        # Log refine output
        log("✨ Refine · Output", final_content)
        log_summary("✨", "Контент отполирован", [
            f"Финальная длина: {len(final_content)} символов",
            "Пост готов к публикации"
        ])
    else:
        if style_key != "post_style_2":
            log("✨ Refine · Skipped", "Refine disabled by configuration or unsupported by style")

    # Save final (optional)
    _emit("save:init")
    logger.step("Saving post to file")
    filepath = None
    if not disable_file_save:
        output_dir = ensure_output_dir(output_subdir)
        base = f"{safe_filename_base(topic)}_post"
        filepath = next_available_filepath(output_dir, base, ".md")
        save_markdown(
            filepath,
            title=topic,
            generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
            pipeline="PopularSciencePost",
            content=final_content,
        )
    # Save log sidecar .md (optional) and register in DB if available
    log_dir = ensure_output_dir(output_subdir)
    log_path = log_dir / f"{safe_filename_base(topic)}_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    header = (
        f"# 🧾 Generation Log\n\n"
        f"- provider: {_prov}\n"
        f"- lang: {lang}\n"
        f"- style: {style_key}\n"
        f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- duration: {duration_s:.1f}s\n"
        f"- topic: {topic}\n"
        f"- factcheck: {bool(factcheck)}\n"
        f"- refine: {bool(use_refine)}\n"
    )
    full_log_content = header + "\n".join(log_lines)
    if not disable_sidecar_log:
        save_markdown(log_path, title=f"Log: {topic}", generator="bio1c", pipeline="Log", content=full_log_content)
    # Record log and result in DB if available
    try:
        from server.db import JobLog, ResultDoc
        # Gate by DB_URL presence, not by async SessionLocal state
        if os.getenv("DB_URL", "").strip() and not disable_db_record:
            # Use sync approach to avoid event loop conflicts in thread executor
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            
            # Create sync connection from same DB_URL
            db_url = os.getenv("DB_URL", "")
            if db_url:
                # Convert async URL to sync with psycopg2 driver (strip query params)
                sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
                parts = urlsplit(sync_url)
                qs = parse_qs(parts.query or "")
                # psycopg2 не принимает нестандартные query-параметры (например, ssl=true)
                base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
                cargs = {}
                # Ensure SSL for psycopg2 if not explicitly set
                if "sslmode" not in {k.lower() for k in qs.keys()}:
                    cargs["sslmode"] = "require"
                # Increased pool for concurrent users: pool_size=15, max_overflow=10 = 25 total
                sync_engine = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True, pool_size=15, max_overflow=10, pool_timeout=30)
                SyncSession = sessionmaker(sync_engine)
                
                with SyncSession() as s:
                    # Import sync models
                    from server.db import JobLog, ResultDoc
                    # Store relative path for portability
                    try:
                        rel_path = str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path)
                    except ValueError:
                        # If relative_to fails, use absolute path
                        rel_path = str(log_path)
                    # Safely convert job_id to int
                    try:
                        job_id = int((job_meta or {}).get("job_id", 0))
                    except (ValueError, TypeError):
                        job_id = 0
                    # Store log content directly (avoid FS race/ephemeral issues)
                    # 1) Persist JobLog first (transactionally)
                    jl = JobLog(job_id=job_id, kind="md", path=rel_path, content=full_log_content)
                    result_job_id = job_id
                    try:
                        s.add(jl)
                        s.flush()
                        s.commit()
                    except Exception as _e:
                        s.rollback()
                        print(f"[ERROR] JobLog commit failed: {_e}")
                    # Ensure we have a valid Job.id for history join; if missing, create one based on user_id
                    try:
                        if not result_job_id or int(result_job_id) <= 0:
                            # Prefer db_user_id (User.id) from job_meta if available; else resolve by telegram_id
                            from server.db import User, Job as _Job
                            tg_uid_meta = int((job_meta or {}).get("user_id", 0) or 0)
                            db_user_id = (job_meta or {}).get("db_user_id")  # Try User.id first
                            print(f"[DEBUG] Fallback triggered: telegram_id={tg_uid_meta}, db_user_id from meta={db_user_id}")
                            if db_user_id is None or int(db_user_id or 0) <= 0:
                                # Fallback: resolve User by telegram_id
                                tg_uid = int((job_meta or {}).get("user_id", 0) or 0)
                                if tg_uid > 0:
                                    try:
                                        urow = s.query(User).filter(User.telegram_id == tg_uid).first()
                                    except Exception:
                                        urow = None
                                    if urow is None:
                                        from sqlalchemy.exc import IntegrityError as _IntegrityError
                                        try:
                                            urow = User(telegram_id=tg_uid, credits=0)
                                            s.add(urow)
                                            s.flush()
                                            s.commit()  # Commit User creation immediately
                                        except _IntegrityError:
                                            # Race condition: another request created this user
                                            s.rollback()
                                            try:
                                                urow = s.query(User).filter(User.telegram_id == tg_uid).first()
                                            except Exception:
                                                urow = None
                                        except Exception as other_err:
                                            # Other DB errors
                                            print(f"[ERROR] Fallback User create failed (non-IntegrityError) for telegram_id={tg_uid}: {type(other_err).__name__}: {str(other_err)[:200]}")
                                            s.rollback()
                                            urow = None
                                    if urow is not None:
                                        db_user_id = int(getattr(urow, "id", 0) or 0)
                            else:
                                db_user_id = int(db_user_id)
                            # Create a Job row linked to this user (ALWAYS use User.id, never telegram_id directly)
                            if db_user_id:
                                try:
                                    import json as __json
                                    params = {
                                        "topic": topic,
                                        "lang": lang,
                                        "provider": _prov,
                                        "factcheck": bool(factcheck),
                                        "refine": bool(use_refine),
                                    }
                                    jrow = _Job(user_id=db_user_id, type="post", status="done", params_json=__json.dumps(params, ensure_ascii=False), cost=1, file_path=str(filepath) if filepath else None)
                                    s.add(jrow)
                                    s.flush()
                                    result_job_id = int(getattr(jrow, "id", 0) or 0)
                                    s.commit()
                                except Exception as _e:
                                    s.rollback()
                                    print(f"[ERROR] Fallback Job create failed: {_e}")
                            else:
                                print(f"[WARN] Cannot create fallback Job: User not found/created for telegram_id {tg_uid}")
                    except Exception as outer_e:
                        # Non-fatal: keep result_job_id as-is
                        print(f"[ERROR] Fallback Job creation outer exception: {outer_e}")
                    # 2) Persist final ResultDoc independently (even if JobLog failed)
                    if filepath is not None:
                        try:
                            rel_doc = str(filepath.relative_to(Path.cwd())) if filepath.is_absolute() else str(filepath)
                        except ValueError:
                            rel_doc = str(filepath)
                        final_job_id = int(result_job_id or 0)
                        rd = ResultDoc(
                            job_id=final_job_id,
                            kind=output_subdir,
                            path=rel_doc,
                            topic=topic,
                            provider=_prov,
                            lang=lang,
                            content=final_content,
                            hidden=1 if ((job_meta or {}).get("incognito") is True) else 0,
                        )
                        try:
                            s.add(rd)
                            s.flush()
                            s.commit()
                        except Exception as _e:
                            s.rollback()
                            print(f"[ERROR] ResultDoc commit failed: {_e}")
                            # Attempt to create table on-the-fly if missing, then retry once
                            try:
                                from server.db import ResultDoc as _RD
                                try:
                                    _RD.__table__.create(bind=sync_engine, checkfirst=True)
                                except Exception:
                                    pass
                                s.add(rd)
                                s.flush()
                                s.commit()
                                print("[INFO] ResultDoc table created on-the-fly and record inserted")
                            except Exception as _e2:
                                s.rollback()
                                print(f"[ERROR] ResultDoc create+retry failed: {_e2}")
                    try:
                        print(f"[INFO] Log recorded in DB: id={getattr(jl,'id',None)}, path={log_path}, content_size={len(full_log_content)}; result_job_id={result_job_id}")
                    except Exception:
                        pass
                try:
                    sync_engine.dispose()
                except Exception:
                    pass
            else:
                print(f"[INFO] No DB_URL configured, log saved to filesystem only: {log_path}")
        else:
            print(f"[INFO] Log saved to filesystem only: {log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to record log in DB: {e}")
        print(f"[INFO] Log available on filesystem: {log_path}")
        # Continue execution - log file is still created even if DB fails
    _emit("done")
    logger.total_duration()
    logger.success(f"Post generation complete{f': {filepath.name}' if filepath else ''}", show_duration=False)
    if return_content:
        # Return the content string instead of path
        return final_content  # type: ignore[return-value]
    if return_log_path:
        return log_path
    return filepath if filepath is not None else log_path



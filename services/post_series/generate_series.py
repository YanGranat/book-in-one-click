#!/usr/bin/env python3
"""
Series generator for popular science posts.

Features:
- Auto and Fixed modes
- Iterative list building (builder → sufficiency → extend)
- Switch to heavy model for sufficiency after a threshold
- Sequential writing of posts using services.post.generate (with series context)
- Output modes: single .md aggregate or folder of per-post files (no per-post DB records)
- Single aggregate ResultDoc(kind='post_series') and a single JobLog

Credits integration is left to server/bot layer. This module exposes knobs
for budget-like limits but does not mutate user credits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Type
import os
import asyncio
import time
import json

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.slug import safe_filename_base
from schemas.series import PostIdea, PostIdeaList, ListSufficiency, ExtendResponse, PrioritizedList


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


class _ProviderRunner:
    def __init__(self, provider: str):
        self.provider = (provider or "openai").strip().lower()

    def _run_openai_with(self, system: str, user_message: str, model: Optional[str] = None) -> str:
        Agent, Runner = _try_import_sdk()
        agent = Agent(name="Series Agent", instructions=system, model=(model or os.getenv("OPENAI_MODEL", "gpt-5")))
        res = Runner.run_sync(agent, user_message)
        return getattr(res, "final_output", "")

    def _run_gemini_with(self, system: str, user_message: str, model_name: Optional[str] = None) -> str:
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

    def _run_gemini_json_with(self, system: str, user_message: str, model_name: Optional[str] = None) -> str:
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
                model = genai.GenerativeModel(
                    model_name=mname,
                    system_instruction=system,
                    generation_config={"response_mime_type": "application/json"},
                )
                resp = model.generate_content(user_message)
                txt = (getattr(resp, "text", None) or "").strip()
                if not txt:
                    parts = []
                    try:
                        for c in getattr(resp, "candidates", []) or []:
                            for part in getattr(getattr(c, "content", None), "parts", []) or []:
                                t = getattr(part, "text", None)
                                if t:
                                    parts.append(t)
                    except Exception:
                        pass
                    txt = ("\n".join(parts)).strip()
                return txt
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Gemini JSON request failed; last error: {last_err}")

    def _run_claude_with(self, system: str, user_message: str, model_name: Optional[str] = None) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        preferred = model_name or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        fallbacks = [preferred, "claude-3-7-sonnet-latest", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
        last_err = None
        for mname in fallbacks:
            try:
                msg = client.messages.create(
                    model=mname, max_tokens=4096, system=system, messages=[{"role": "user", "content": user_message}]
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

    def run_text(self, system: str, user_message: str, speed: str = "heavy") -> str:
        if self.provider == "openai":
            model = os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini") if speed == "fast" else os.getenv("OPENAI_MODEL", "gpt-5")
            return self._run_openai_with(system, user_message, model)
        if self.provider in {"gemini", "google"}:
            mname = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash") if speed == "fast" else os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            return self._run_gemini_with(system, user_message, mname)
        # Claude: one heavy model
        cname = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        return self._run_claude_with(system, user_message, cname)

    def run_json(self, system: str, user_message: str, cls: Type, speed: str = "fast"):
        txt = None
        if self.provider in {"gemini", "google"}:
            mname = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash") if speed == "fast" else os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            txt = self._run_gemini_json_with(system, user_message, mname)
        else:
            txt = self.run_text(system, user_message, speed)
        if txt.strip().startswith("```") and txt.strip().endswith("```"):
            txt = "\n".join([line for line in txt.strip().splitlines()[1:-1]])

        def _norm_key(s: str) -> str:
            return "".join(ch for ch in (s or "").lower() if ch.isalnum())

        def _snake_case(name: str) -> str:
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", "_", name)
            s = s.replace("-", "_")
            return s.lower()

        def _normalize(obj):
            if isinstance(obj, dict):
                out = {}
                for k, v in obj.items():
                    out[_snake_case(k)] = _normalize(v)
                return out
            if isinstance(obj, list):
                return [_normalize(x) for x in obj]
            return obj

        try:
            data = json.loads(txt)
        except Exception:
            import re as _re
            m = _re.search(r"\{[\s\S]*\}\s*$", txt)
            if not m:
                if self.provider in {"gemini", "google"} and speed != "heavy":
                    return self.run_json(system, user_message, cls, speed="heavy")
                raise RuntimeError(f"Failed to parse JSON for {cls.__name__}")
            data = json.loads(m.group(0))
        data = _normalize(data)
        return cls.model_validate(data)


def _load_prompt(rel_path: str) -> str:
    base = Path(__file__).resolve().parents[2] / "prompts" / "post_series"
    return (base / rel_path).read_text(encoding="utf-8")


def _dedupe(ideas: list[PostIdea]) -> list[PostIdea]:
    seen = set()
    out: list[PostIdea] = []
    import re
    def norm(s: str) -> str:
        s2 = s.lower().strip()
        s2 = re.sub(r"\s+", " ", s2)
        s2 = s2.replace("’", "'")
        return s2
    for it in ideas:
        key = (norm(it.title), norm(it.angle or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _next_id(existing: list[PostIdea]) -> str:
    mx = 0
    for it in existing:
        try:
            if it.id and it.id[0].lower() == 't':
                mx = max(mx, int(it.id[1:]))
        except Exception:
            continue
    return f"t{mx+1:02d}"


def generate_series(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",
    mode: str = "auto",  # auto|fixed
    count: int = 0,
    max_iterations: int = 1,
    sufficiency_heavy_after: int = 3,
    output_mode: str = "single",  # single|folder
    output_subdir: str = "post_series",
    factcheck: bool = False,
    research_iterations: int = 2,
    refine: bool = False,
    on_progress: Optional[Callable[[str], None]] = None,
    job_meta: Optional[dict] = None,
) -> Path:
    """
    Generate a series of popular science posts for a topic.

    Returns path to the aggregate .md file (even for folder mode; aggregate contains full inline content).
    """
    _ensure_root(__file__)
    load_env_from_root(__file__)

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

    # Ensure loop
    try:
        asyncio.get_event_loop()
    except Exception:
        asyncio.set_event_loop(asyncio.new_event_loop())

    Agent, Runner = _try_import_sdk()
    pr = _ProviderRunner(_prov)

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass

    _emit("start:series")
    from datetime import datetime
    started_at = datetime.utcnow()
    started_perf = time.perf_counter()
    log_lines: list[str] = []
    def log(section: str, body: str):
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")

    log("🧭 Config", f"provider={_prov}\nlang={lang}\nmode={mode}\ncount={count}\nmax_iterations={max_iterations}\noutput_mode={output_mode}\nfactcheck={bool(factcheck)}\nrefine={bool(refine)}")

    # 1) Build initial ideas
    p_builder = _load_prompt("builder.md")
    user_builder = f"<input>\n<topic>{topic}</topic>\n<lang>{(lang or 'auto').strip()}</lang>\n</input>"
    plan = pr.run_json(p_builder, user_builder, PostIdeaList, speed="fast")
    ideas: list[PostIdea] = _dedupe(plan.items or [])
    log("🧱 Builder · Ideas", f"```json\n{PostIdeaList(items=ideas).model_dump_json()}\n```")

    # 2) Iterate sufficiency/extend
    iterations = max(0, int(max_iterations))
    for i in range(iterations):
        p_suff = _load_prompt("sufficiency.md")
        suff_user = f"<input>\n<topic>{topic}</topic>\n<ideas_json>{PostIdeaList(items=ideas).model_dump_json()}</ideas_json>\n</input>"
        speed = "heavy" if i + 1 > int(sufficiency_heavy_after) else "fast"
        suff = pr.run_json(p_suff, suff_user, ListSufficiency, speed=speed)
        log("🧪 Sufficiency", f"```json\n{suff.model_dump_json()}\n```")
        if suff.done:
            break
        missing = suff.missing_areas or []
        p_ext = _load_prompt("extend.md")
        needed = 0
        ext_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<ideas_json>{PostIdeaList(items=ideas).model_dump_json()}</ideas_json>\n"
            f"<missing_areas_json>{json.dumps(missing, ensure_ascii=False)}</missing_areas_json>\n"
            f"<needed>{needed}</needed>\n"
            "</input>"
        )
        ext = pr.run_json(p_ext, ext_user, ExtendResponse, speed="fast")
        new_items = ext.items or []
        # Re-sequence IDs to continue numbering if needed
        cur = ideas[:]
        for ni in new_items:
            if not ni.id:
                ni.id = _next_id(cur)
            cur.append(ni)
        ideas = _dedupe(cur)
        log("➕ Extend", f"added={len(new_items)} total={len(ideas)}")

    # 3) Selection for fixed/auto
    selected: list[PostIdea] = []
    if (mode or "auto").strip().lower() == "fixed":
        n = max(1, int(count or 0))
        p_pri = _load_prompt("prioritize.md")
        pri_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<ideas_json>{PostIdeaList(items=ideas).model_dump_json()}</ideas_json>\n"
            f"<N>{n}</N>\n"
            "</input>"
        )
        pri = pr.run_json(p_pri, pri_user, PrioritizedList, speed="fast")
        selected = pri.items or []
        if len(selected) != n:
            # Fallback: truncate or pad from ideas
            selected = (selected + [x for x in ideas if x.id not in {i.id for i in selected}])[:n]
    else:
        # Auto: take all planned ideas
        selected = ideas[:]

    # 4) Write posts sequentially
    from services.post.generate import generate_post as generate_single_post
    posts_contents: list[tuple[PostIdea, str, Optional[Path]]] = []
    done_ids: list[str] = []
    for idx, idea in enumerate(selected, start=1):
        _emit(f"write:{idx}")
        log("✍️ Writer · Topic", f"{idea.id}: {idea.title}")
        # series writer prompt override
        p_writer = _load_prompt("writer.md")
        series_topics_min = [
            {"id": it.id, "title": it.title, "angle": it.angle, "tags": it.tags}
            for it in selected
        ]
        # Choose behavior based on output mode
        if (output_mode or "single").strip().lower() == "single":
            content = generate_single_post(
                idea.title,
                lang=lang,
                provider=provider,
                factcheck=factcheck,
                research_iterations=research_iterations,
                output_subdir=output_subdir,
                job_meta=job_meta,
                use_refine=refine,
                instructions_override=p_writer,
                series_topics=series_topics_min,
                series_current_id=idea.id,
                series_done_ids=done_ids,
                disable_db_record=True,
                disable_file_save=True,
                disable_sidecar_log=True,
                return_content=True,
            )
            posts_contents.append((idea, content, None))
        else:
            path = generate_single_post(
                idea.title,
                lang=lang,
                provider=provider,
                factcheck=factcheck,
                research_iterations=research_iterations,
                output_subdir=output_subdir,
                job_meta=job_meta,
                use_refine=refine,
                instructions_override=p_writer,
                series_topics=series_topics_min,
                series_current_id=idea.id,
                series_done_ids=done_ids,
                disable_db_record=True,
                disable_sidecar_log=True,
            )
            # Read content back to include inline in aggregate
            content = Path(path).read_text(encoding="utf-8") if isinstance(path, Path) else Path(path).read_text(encoding="utf-8")
            posts_contents.append((idea, content, path if isinstance(path, Path) else Path(path)))
        done_ids.append(idea.id)

    # 5) Build aggregate content
    header_lines = [
        f"# Серия постов: {topic}",
        "",
        "## Список тем",
    ]
    for it in selected:
        tags_str = (", ".join(it.tags or [])) if it.tags else ""
        angle_str = (f" — {it.angle}" if it.angle else "")
        header_lines.append(f"- {it.id}. {it.title}{angle_str}{(' ['+tags_str+']') if tags_str else ''}")
    header_lines.append("")
    body_lines = []
    for (it, content, pth) in posts_contents:
        body_lines.append(f"\n\n---\n\n## {it.id}. {it.title}\n\n")
        # Extract only markdown content from saved files if they contain metadata wrappers
        body_lines.append(content)
    aggregate_content = "\n".join(header_lines + body_lines)

    # 6) Save aggregate file
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_series"
    aggregate_path = next_available_filepath(output_dir, base, ".md")
    save_markdown(
        aggregate_path,
        title=f"Series: {topic}",
        generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
        pipeline="PostSeries",
        content=aggregate_content,
    )

    # 7) Save series log and single DB aggregate record
    log_dir = ensure_output_dir(output_subdir)
    from datetime import datetime
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    log_header = (
        f"# 🧾 Series Generation Log\n\n"
        f"- provider: {_prov}\n"
        f"- lang: {lang}\n"
        f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- duration: {duration_s:.1f}s\n"
        f"- topic: {topic}\n"
        f"- mode: {mode}\n"
        f"- count: {count}\n"
        f"- max_iterations: {max_iterations}\n"
        f"- output_mode: {output_mode}\n"
        f"- factcheck: {bool(factcheck)}\n"
        f"- refine: {bool(refine)}\n"
    )
    full_log_content = log_header + "\n".join(log_lines)
    log_path = log_dir / f"{safe_filename_base(topic)}_series_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    save_markdown(log_path, title=f"Series Log: {topic}", generator="bio1c", pipeline="LogSeries", content=full_log_content)

    # DB aggregate only
    try:
        from server.db import JobLog, ResultDoc
        if os.getenv("DB_URL", "").strip():
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from urllib.parse import urlsplit, urlunsplit, parse_qs

            db_url = os.getenv("DB_URL", "")
            if db_url:
                sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
                parts = urlsplit(sync_url)
                qs = parse_qs(parts.query or "")
                base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
                cargs = {}
                if "sslmode" not in {k.lower() for k in qs.keys()}:
                    cargs["sslmode"] = "require"
                sync_engine = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True, pool_size=3, max_overflow=0)
                SyncSession = sessionmaker(sync_engine)
                with SyncSession() as s:
                    try:
                        rel_log = str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path)
                    except ValueError:
                        rel_log = str(log_path)
                    try:
                        job_id = int((job_meta or {}).get("job_id", 0))
                    except (ValueError, TypeError):
                        job_id = 0
                    jl = JobLog(job_id=job_id, kind="md", path=rel_log, content=full_log_content)
                    s.add(jl)
                    s.flush()
                    # Aggregate record
                    try:
                        rel_doc = str(aggregate_path.relative_to(Path.cwd())) if aggregate_path.is_absolute() else str(aggregate_path)
                    except ValueError:
                        rel_doc = str(aggregate_path)
                    rd = ResultDoc(
                        job_id=job_id,
                        kind="post_series",
                        path=rel_doc,
                        topic=topic,
                        provider=_prov,
                        lang=lang,
                        content=aggregate_content,
                        hidden=1 if ((job_meta or {}).get("incognito") is True) else 0,
                    )
                    s.add(rd)
                    s.flush()
                    s.commit()
                try:
                    sync_engine.dispose()
                except Exception:
                    pass
        else:
            print(f"[INFO] Series log saved to filesystem only: {log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to record series in DB: {e}")
        print(f"[INFO] Series log available on filesystem: {log_path}")

    return aggregate_path




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
from utils.logging import create_logger
from schemas.series import PostIdea, PostIdeaList, ListSufficiency, ExtendResponse, PrioritizedList
from services.providers.runner import ProviderRunner


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


class _ProviderAdapter:
    def __init__(self, provider: str):
        self._runner = ProviderRunner(provider)

    def run_text(self, system: str, user_message: str, speed: str = "heavy") -> str:
        return self._runner.run_text(system, user_message, speed=speed)

    def run_json(self, system: str, user_message: str, cls: Type, speed: str = "fast"):
        import json as _json
        txt = self._runner.run_json(system, user_message, speed=speed)
        if txt.strip().startswith("```") and txt.strip().endswith("```"):
            txt = "\n".join([line for line in txt.strip().splitlines()[1:-1]])

        def _snake_case(name: str) -> str:
            import re as _re
            s = _re.sub(r"(?<!^)(?=[A-Z])", "_", name)
            s = s.replace("-", "_")
            return s.lower()

        def _normalize(obj):
            if isinstance(obj, dict):
                return { _snake_case(k): _normalize(v) for k, v in obj.items() }
            if isinstance(obj, list):
                return [_normalize(x) for x in obj]
            return obj

        try:
            data = _json.loads(txt)
        except Exception as e:
            import re as _re
            m = _re.search(r"\{[\s\S]*\}\s*$", txt)
            if not m:
                if speed != "heavy":
                    return self.run_json(system, user_message, cls, speed="heavy")
                raise RuntimeError(f"Failed to parse JSON for {cls.__name__}: {str(e)[:200]}")
            data = _json.loads(m.group(0))
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
    pr = _ProviderAdapter(_prov)

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
    
    # Initialize structured logger
    logger = create_logger("series", show_debug=bool(os.getenv("DEBUG_LOGS")))
    logger.info(f"Starting series generation: '{topic[:100]}'")
    logger.info(f"Configuration: provider={_prov}, lang={lang}, mode={mode}, count={count}")
    
    log_lines: list[str] = []
    def log(section: str, body: str):
        """Log to markdown file - keep it readable and high-level."""
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")
    
    def log_summary(emoji: str, title: str, items: list[str]):
        """Log a clean summary without technical details."""
        content = "\n".join(f"- {item}" for item in items if item)
        log_lines.append(f"---\n\n## {emoji} {title}\n\n{content}\n")

    # Log generation configuration
    log_summary("⚙️", "Конфигурация генерации", [
        f"Провайдер: {_prov}",
        f"Язык: {lang}",
        f"Режим: {mode}",
        f"Постов в серии: {count}",
        f"Тема: {topic[:100]}{'...' if len(topic) > 100 else ''}"
    ])

    # 1) Build initial ideas
    logger.stage("Planning Series", total_stages=2, current_stage=1)
    logger.step("Generating initial post ideas")
    p_builder = _load_prompt("module_01_planning/builder.md")
    user_builder = f"<input>\n<topic>{topic}</topic>\n<lang>{(lang or 'auto').strip()}</lang>\n</input>"
    plan = pr.run_json(p_builder, user_builder, PostIdeaList, speed="fast")
    ideas: list[PostIdea] = _dedupe(plan.items or [])
    logger.success(f"Generated {len(ideas)} initial post ideas", show_duration=False)
    
    # Log full ideas JSON for tracking evolution
    log("🧱 Builder · Ideas", f"```json\n{PostIdeaList(items=ideas).model_dump_json()}\n```")
    
    # Log readable summary
    idea_titles = [f"**{idea.title}**" for idea in ideas[:10]]
    if len(ideas) > 10:
        idea_titles.append(f"...и ещё {len(ideas) - 10} идей")
    log_summary("💡", "Начальные идеи для постов", [
        f"Всего идей: {len(ideas)}",
        "",
        "Основные темы:",
        *idea_titles
    ])

    # 2) Iterate sufficiency/extend
    iterations = max(0, int(max_iterations))
    for i in range(iterations):
        p_suff = _load_prompt("module_01_planning/sufficiency.md")
        suff_user = f"<input>\n<topic>{topic}</topic>\n<ideas_json>{PostIdeaList(items=ideas).model_dump_json()}</ideas_json>\n<lang>{(lang or 'auto').strip()}</lang>\n</input>"
        speed = "heavy" if i + 1 > int(sufficiency_heavy_after) else "fast"
        suff = pr.run_json(p_suff, suff_user, ListSufficiency, speed=speed)
        # Log sufficiency check for tracking
        log("🧪 Sufficiency", f"```json\n{suff.model_dump_json()}\n```")
        if suff.done:
            logger.step(f"Coverage check (iteration {i+1}): sufficient")
            break
        else:
            logger.step(f"Coverage check (iteration {i+1}): extending")
        missing = suff.missing_areas or []
        p_ext = _load_prompt("module_01_planning/extend.md")
        needed = 0
        ext_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<ideas_json>{PostIdeaList(items=ideas).model_dump_json()}</ideas_json>\n"
            f"<missing_areas_json>{json.dumps(missing, ensure_ascii=False)}</missing_areas_json>\n"
            f"<needed>{needed}</needed>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
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
        # Log extension for tracking
        log("➕ Extend", f"added={len(new_items)} total={len(ideas)}")
        logger.success(f"Added {len(new_items)} ideas, total: {len(ideas)}", show_duration=False)

    # 3) Selection for fixed/auto
    selected: list[PostIdea] = []
    if (mode or "auto").strip().lower() == "fixed":
        n = max(1, int(count or 0))
        p_pri = _load_prompt("module_01_planning/prioritize.md")
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

    # Log full selected topics JSON for tracking
    try:
        log("🗂️ Selected · Topics", f"```json\n{PostIdeaList(items=selected).model_dump_json()}\n```")
    except Exception:
        pass
    
    # Log readable summary
    selected_titles = [f"**{idea.title}**" for idea in selected]
    log_summary("📋", "Выбранные темы для серии", [
        f"Постов в серии: {len(selected)}",
        "",
        "Посты:",
        *[f"{i+1}. {title}" for i, title in enumerate(selected_titles)]
    ])

    # 4) Write posts sequentially
    logger.stage("Writing Posts", total_stages=2, current_stage=2)
    logger.info(f"Writing {len(selected)} posts for series")
    from services.post.generate import generate_post as generate_single_post
    posts_contents: list[tuple[PostIdea, str, Optional[Path]]] = []
    done_ids: list[str] = []
    for idx, idea in enumerate(selected, start=1):
        _emit(f"write:{idx}")
        logger.step(f"Writing post: {idea.title}", current=idx, total=len(selected))
        # Log topic being written
        log("✍️ Writer · Topic", f"{idea.id}: {idea.title}")
        # series writer prompt override
        p_writer = _load_prompt("module_02_writing/writer.md")
        # Pass full list of topics as JSON (all fields of PostIdea)
        series_topics_full = [it.model_dump() for it in selected]
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
                series_topics=series_topics_full,
                series_current_id=idea.id,
                series_done_ids=done_ids,
                disable_db_record=True,
                disable_file_save=True,
                disable_sidecar_log=True,
                return_content=True,
            )
            # Log post output for tracking evolution
            try:
                log("✍️ Writer · Output", f"## {idea.title}\n\n{content}")
            except Exception:
                pass
            posts_contents.append((idea, content, None))
            logger.success(f"Post written: {idea.title[:50]}, {len(content)} chars", show_duration=False)
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
                series_topics=series_topics_full,
                series_current_id=idea.id,
                series_done_ids=done_ids,
                disable_db_record=True,
                disable_sidecar_log=True,
            )
            # Read content back to include inline in aggregate and in series log
            content = Path(path).read_text(encoding="utf-8") if isinstance(path, Path) else Path(path).read_text(encoding="utf-8")
            # Log post output for tracking evolution
            try:
                log("✍️ Writer · Output", f"## {idea.title}\n\n{content}")
            except Exception:
                pass
            logger.success(f"Post written: {idea.title[:50]}, {len(content)} chars", show_duration=False)
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
        header_lines.append(f"- {it.title}{angle_str}{(' ['+tags_str+']') if tags_str else ''}")
    header_lines.append("")
    body_lines = []
    for (it, content, pth) in posts_contents:
        body_lines.append(f"\n\n---\n\n## {it.title}\n\n")
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
                # Increased pool for concurrent users: pool_size=15, max_overflow=10 = 25 total
                sync_engine = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True, pool_size=15, max_overflow=10, pool_timeout=30)
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

    logger.total_duration()
    logger.success(f"Series generation complete: {len(selected)} posts written", show_duration=False)
    return aggregate_path




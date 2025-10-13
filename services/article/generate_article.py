#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import asyncio
import time
import json as _json
import sys as _sys
import traceback as _tb
from time import sleep as _sleep

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.slug import safe_filename_base
from services.providers.runner import ProviderRunner

from schemas.article import ArticleOutline, DraftChunk, ArticleTitleLead


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


def generate_article(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",  # openai|gemini|claude
    output_subdir: str = "deep_article",
    on_progress: Optional[Callable[[str], None]] = None,
    job_meta: Optional[dict] = None,
    return_log_path: bool = False,
    enable_research: bool = False,
    enable_refine: bool = False,
    max_parallel: Optional[int] = None,
) -> Path | str:
    """
    Generate a deep popular science article and save it to output/<output_subdir>/.

    Returns Path to saved .md, or log path when return_log_path=True.
    """
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

    _emit("start:article")
    from datetime import datetime
    started_at = datetime.utcnow()
    started_perf = time.perf_counter()
    log_lines: list[str] = []
    def log(section: str, body: str):
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")

    def srvlog(tag: str, text: str) -> None:
        try:
            print(f"[ARTICLE][{tag}] {text}", file=_sys.stderr)
        except Exception:
            pass

    def _run_with_retries_sync(agent: Any, user_input: str, *, attempts: int = 3, backoff_seq: tuple[int, ...] = (2, 5, 8)):
        """Run agent with simple retries on provider transient errors."""
        for i in range(attempts):
            try:
                return Runner.run_sync(agent, user_input)
            except Exception as e:
                err_s = str(e)[:300]
                srvlog("RETRY", f"attempt={i+1}/{attempts} err={type(e).__name__}: {err_s}")
                if i == attempts - 1:
                    raise
                _sleep(backoff_seq[min(i, len(backoff_seq)-1)])
    log("🧭 Config", f"provider={_prov}\nlang={lang}\nresearch={bool(enable_research)}\nrefine={bool(enable_refine)}")
    # Determine parallelism (bounded to avoid provider rate limits)
    par_env = (os.getenv("ARTICLE_MAX_PAR", "").strip() or None)
    try:
        _par = int(par_env) if par_env is not None else (int(max_parallel) if max_parallel is not None else 4)
    except Exception:
        _par = 4
    max_workers = max(1, min(16, _par))
    srvlog("CONF_PAR", f"max_workers={max_workers} par_env='{par_env or ''}' max_parallel={max_parallel}")

    # Agents import
    from llm_agents.deep_popular_science_article.module_01_structure.sections_and_subsections import build_sections_and_subsections_agent
    # content_of_subsections removed: structure module now only builds sections; writing fills content
    from llm_agents.deep_popular_science_article.module_02_writing.subsection_writer import (
        build_subsection_writer_agent,
    )
    from llm_agents.deep_popular_science_article.module_02_writing.article_title_lead_writer import (
        build_article_title_lead_writer_agent,
    )

    # Module 1
    srvlog("START", f"topic='{topic[:100]}' lang={lang} provider={_prov}")
    outline_agent = build_sections_and_subsections_agent(provider=_prov)
    user_outline = f"<input>\n<topic>{topic}</topic>\n<lang>{lang}</lang>\n</input>"
    try:
        t0 = time.perf_counter()
        srvlog("OUTLINE_START", "pass=1 (initial)")
        outline: ArticleOutline = Runner.run_sync(outline_agent, user_outline).final_output  # type: ignore
        log("📑 Outline · Sections", f"```json\n{outline.model_dump_json()}\n```")
        sec_count = len(outline.sections)
        sub_count = sum(len(s.subsections) for s in outline.sections)
        srvlog("OUTLINE_OK", f"pass=1 sections={sec_count} subsections={sub_count} dur_ms={int((time.perf_counter()-t0)*1000)}")
        if sec_count == 0:
            raise ValueError("Outline has no sections")
    except Exception as e:
        srvlog("OUTLINE_ERR", f"{type(e).__name__}: {e}")
        _tb.print_exc()
        raise

    # Skip legacy content-of-subsections step (removed in 2-module pipeline)

    # Research module removed in 2‑module pipeline

    # Second pass: refine/improve outline with the same agent if necessary
    try:
        improve_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            "</input>"
        )
        t1 = time.perf_counter()
        srvlog("OUTLINE_START", "pass=2 (improve)")
        improved_outline: ArticleOutline = Runner.run_sync(outline_agent, improve_user).final_output  # type: ignore
        # Accept improved outline when it still has sections and not smaller by an extreme margin
        if improved_outline and improved_outline.sections:
            outline = improved_outline
            log("📑 Outline · Improved", f"```json\n{outline.model_dump_json()}\n```")
            srvlog("OUTLINE_IMPROVED", f"pass=2 sections={len(outline.sections)} dur_ms={int((time.perf_counter()-t1)*1000)}")
    except Exception as e:
        srvlog("OUTLINE_IMPROVE_ERR", f"{type(e).__name__}: {e}")

    # Third pass: expand content items per subsection (outline_json + expand_content=true)
    try:
        expand_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<expand_content>true</expand_content>\n"
            "</input>"
        )
        t2 = time.perf_counter()
        srvlog("OUTLINE_START", "pass=3 (expand_content)")
        expanded_outline: ArticleOutline = Runner.run_sync(outline_agent, expand_user).final_output  # type: ignore
        if expanded_outline and expanded_outline.sections:
            outline = expanded_outline
            log("📑 Outline · Expanded Content", f"```json\n{outline.model_dump_json()}\n```")
            srvlog("OUTLINE_EXPANDED", f"pass=3 sections={len(outline.sections)} dur_ms={int((time.perf_counter()-t2)*1000)}")
    except Exception as e:
        srvlog("OUTLINE_EXPAND_ERR", f"{type(e).__name__}: {e}")

    # Module 2 (Writing): Subsections drafts in parallel across all subsections
    ssw_agent = build_subsection_writer_agent(provider=_prov)
    drafts_by_subsection: dict[tuple[str, str], DraftChunk] = {}
    all_subs_writing = [(sec, sub) for sec in outline.sections for sub in sec.subsections]
    srvlog("DRAFT_SETUP", f"total_jobs={len(all_subs_writing)} max_workers={max_workers}")

    def _run_subsection_draft(sec_obj, sub_obj):
        ssw_user_local = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec_obj.id}</section_id>\n"
            f"<subsection_id>{sub_obj.id}</subsection_id>\n"
            # If content_items exist for this subsection, pass them in
            f"<content_items_json>{_json.dumps([{'id': getattr(ci, 'id', ''), 'point': getattr(ci, 'point', '')} for ci in (getattr(sub_obj, 'content_items', []) or [])], ensure_ascii=False)}</content_items_json>\n"
            "</input>"
        )
        # Retry inside worker thread
        backoff_seq = (2, 4, 8)
        for i in range(3):
            try:
                tloc = time.perf_counter()
                srvlog("DRAFT_START", f"{sec_obj.id}/{sub_obj.id} attempt={i+1}/3")
                res = asyncio.run(Runner.run(ssw_agent, ssw_user_local))  # type: ignore
                srvlog("DRAFT_DONE", f"{sec_obj.id}/{sub_obj.id} dur_ms={int((time.perf_counter()-tloc)*1000)}")
                return res.final_output  # type: ignore
            except Exception as ex:
                srvlog(
                    "DRAFT_ERR",
                    f"{sec_obj.id}/{sub_obj.id}: attempt={i+1}/3 {type(ex).__name__}: {str(ex)[:200]}",
                )
                if i == 2:
                    break
                _sleep(backoff_seq[min(i, len(backoff_seq)-1)])
        raise RuntimeError(f"Draft generation failed for {sec_obj.id}/{sub_obj.id}")

    # Round-based generation until all subsections are produced, or hard fail
    max_rounds = 3
    try:
        max_rounds = int(os.getenv("ARTICLE_MAX_ROUNDS", "3"))
    except Exception:
        max_rounds = 3
    round_backoff = 6
    try:
        round_backoff = int(os.getenv("ARTICLE_ROUND_BACKOFF_S", "6"))
    except Exception:
        round_backoff = 6
    srvlog("DRAFT_CONF", f"max_rounds={max_rounds} round_backoff_s={round_backoff}")

    draft_result_by_key: dict[tuple[str, str], DraftChunk] = {}
    remaining = [(sec, sub) for (sec, sub) in all_subs_writing]
    writing_t0 = time.perf_counter()
    for r in range(1, max_rounds + 1):
        if not remaining:
            break
        srvlog("DRAFT_ROUND", f"round={r}/{max_rounds} jobs={len(remaining)} max_workers={max_workers}")
        failed_pairs: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_map = {ex.submit(_run_subsection_draft, sec, sub): (sec.id, sub.id) for (sec, sub) in remaining}
            for fut in as_completed(list(fut_map.keys())):
                key = fut_map[fut]
                try:
                    res = fut.result()
                    draft_result_by_key[key] = res
                    srvlog("DRAFT_OK", f"{key[0]}/{key[1]}")
                except Exception as ex:
                    failed_pairs.append(key)
                    srvlog("DRAFT_FAIL", f"{key[0]}/{key[1]}: {type(ex).__name__}: {str(ex)[:200]}")
        remaining = []
        # Next round will try failed only
        for (sid, ssid) in failed_pairs:
            # re-find sec/sub objects by ids to preserve structure
            for sec in outline.sections:
                if sec.id == sid:
                    for sub in sec.subsections:
                        if sub.id == ssid:
                            remaining.append((sec, sub))
                            break
        if remaining and r < max_rounds:
            _sleep(round_backoff)

    if remaining:
        # Final sequential attempts (strong retry via sync runner)
        srvlog("DRAFT_FINAL", f"count={len(remaining)}")
        still_missing: list[tuple[str, str]] = []
        for (sec, sub) in remaining:
            ssw_user_inline = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<subsection_id>{sub.id}</subsection_id>\n"
                "</input>"
            )
            try:
                out = _run_with_retries_sync(ssw_agent, ssw_user_inline)
                d: DraftChunk = out.final_output  # type: ignore
                draft_result_by_key[(sec.id, sub.id)] = d
                srvlog("FINAL_OK", f"{sec.id}/{sub.id}")
            except Exception as ex:
                still_missing.append((sec.id, sub.id))
                srvlog("FINAL_FAIL", f"{sec.id}/{sub.id}: {type(ex).__name__}: {str(ex)[:200]}")

        if still_missing:
            # Hard fail: do not finish pipeline with gaps
            missing_str = ", ".join([f"{sid}/{ssid}" for (sid, ssid) in still_missing])
            raise RuntimeError(f"Some subsections failed to generate: {missing_str}")

    for sec in outline.sections:
        for sub in sec.subsections:
            key = (sec.id, sub.id)
            d = draft_result_by_key.get(key)
            if not d:
                continue
            drafts_by_subsection[key] = d
            log("✍️ Draft · Subsection", f"{sec.id}/{sub.id} → ```json\n{d.model_dump_json()}\n```")
    srvlog("DRAFT_SUMMARY", f"ok={len(drafts_by_subsection)} total={len(all_subs_writing)} dur_ms={int((time.perf_counter()-writing_t0)*1000)}")

    # Refining module removed in 2‑module pipeline

    # Assemble final Markdown
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_article"
    article_path = next_available_filepath(output_dir, base, ".md")

    # Build body to feed Title&Lead agent
    def _section_label(idx: int) -> str | None:
        lang_l = (lang or "auto").strip().lower()
        if lang_l.startswith("ru"):
            return f"Раздел {idx}."
        if lang_l.startswith("en"):
            return f"Section {idx}."
        # lang=auto или другой язык → как раньше: без префикса и без нумерации
        return None

    def _toc_title() -> str:
        lang_l = (lang or "auto").strip().lower()
        if lang_l.startswith("ru"):
            return "Оглавление"
        if lang_l.startswith("en"):
            return "Table of Contents"
        # lang=auto или другой язык → как раньше
        return "Оглавление"

    toc_lines = [f"## {_toc_title()}"]
    for i, sec in enumerate(outline.sections, start=1):
        toc_lines.append(f"- {i}. {sec.title}")
        for j, sub in enumerate(sec.subsections, start=1):
            toc_lines.append(f"  - {i}.{j} {sub.title}")
    body_lines: list[str] = []
    # Initialize Title & Lead agent once and reuse for section leads and article lead
    atl_agent = build_article_title_lead_writer_agent(provider=_prov)
    for i, sec in enumerate(outline.sections, start=1):
        _lbl = _section_label(i)
        if _lbl:
            body_lines.append(f"\n\n## {_lbl} {sec.title}\n\n")
        else:
            body_lines.append(f"\n\n## {sec.title}\n\n")
        # Per-section lead
        try:
            sec_md_parts = []
            for sub in sec.subsections:
                d = drafts_by_subsection.get((sec.id, sub.id))
                sub_title = d.title if d and d.title else sub.title
                sub_md = d.markdown if d else ""
                sec_md_parts.append(f"\n### {sub_title}\n\n{sub_md}\n")
            sec_body_text = "".join(sec_md_parts)
            # Trim very long sections for lead agent
            try:
                sec_max_chars = int(os.getenv("SECTION_LEAD_MAX_CHARS", "2000000"))
            except Exception:
                sec_max_chars = 2000000
            used_sec = sec_body_text if len(sec_body_text) <= sec_max_chars else sec_body_text[:sec_max_chars]
            srvlog("SECTION_LEAD_INPUT", f"sec={sec.id} title_len={len(sec.title or '')} body_len={len(sec_body_text)} used_len={len(used_sec)} max_chars={sec_max_chars}")
            sec_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<article_markdown>{sec.title}\n\n{used_sec}</article_markdown>\n"
                f"<section_id>{sec.id}</section_id>\n"
                "</input>"
            )
            t_sec = time.perf_counter()
            sec_lead_obj: ArticleTitleLead = _run_with_retries_sync(atl_agent, sec_user).final_output  # type: ignore
            sec_lead = (sec_lead_obj.lead_markdown or "").strip()
            if sec_lead:
                body_lines.append(f"{sec_lead}\n\n")
                srvlog("SECTION_LEAD_OK", f"sec={sec.id} lead_len={len(sec_lead)} dur_ms={int((time.perf_counter()-t_sec)*1000)}")
            else:
                # Retry with only subsection titles to help agent summarize
                titles_bullets = "\n".join([f"- {s.title}" for s in sec.subsections])
                sec_user2 = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<article_markdown>{sec.title}\n\n{titles_bullets}</article_markdown>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    "</input>"
                )
                try:
                    t_sec2 = time.perf_counter()
                    sec_lead_obj2: ArticleTitleLead = _run_with_retries_sync(atl_agent, sec_user2).final_output  # type: ignore
                    sec_lead2 = (sec_lead_obj2.lead_markdown or "").strip()
                    if sec_lead2:
                        body_lines.append(f"{sec_lead2}\n\n")
                        srvlog("SECTION_LEAD_OK", f"sec={sec.id} retry=1 lead_len={len(sec_lead2)} dur_ms={int((time.perf_counter()-t_sec2)*1000)}")
                    else:
                        srvlog("SECTION_LEAD_EMPTY", f"{sec.id}: lead empty after retries")
                except Exception as e2:
                    srvlog("SECTION_LEAD_RETRY_ERR", f"{sec.id}: {type(e2).__name__}: {e2}")
        except Exception as e:
            srvlog("SECTION_LEAD_ERR", f"{sec.id}: {type(e).__name__}: {e}")
        for sub in sec.subsections:
            d = drafts_by_subsection.get((sec.id, sub.id))
            sub_title = d.title if d and d.title else sub.title
            sub_md = d.markdown if d else ""
            body_lines.append(f"\n### {sub_title}\n\n{sub_md}\n")
    toc_text = "\n".join(toc_lines)
    body_text = "\n".join(body_lines)

    # Title & Lead based on full article content
    max_chars = 2000000
    try:
        max_chars = int(os.getenv("TITLE_LEAD_MAX_CHARS", "2000000"))
    except Exception:
        max_chars = 2000000
    used_body = body_text if len(body_text) <= max_chars else body_text[:max_chars]
    srvlog("TITLE_LEAD_INPUT", f"toc_len={len(toc_text)} body_len={len(body_text)} used_len={len(used_body)} max_chars={max_chars}")
    atl_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<article_markdown>{toc_text}\n\n{used_body}</article_markdown>\n"
        "</input>"
    )
    try:
        t_atl = time.perf_counter()
        atl: ArticleTitleLead = _run_with_retries_sync(atl_agent, atl_user).final_output  # type: ignore
        log("🧾 Title & Lead", f"```json\n{atl.model_dump_json()}\n```")
        srvlog("TITLE_LEAD_OK", f"title_len={len(atl.title or '')} lead_len={len(atl.lead_markdown or '')} dur_ms={int((time.perf_counter()-t_atl)*1000)}")
    except Exception as e:
        srvlog("TITLE_LEAD_ERR", f"{type(e).__name__}: {e}")
        _tb.print_exc()
        # Graceful fallback: use outline title/topic, empty lead
        atl = ArticleTitleLead(title=(outline.title or topic), lead_markdown="")
        log("🧾 Title & Lead (fallback)", f"```json\n{atl.model_dump_json()}\n```")

    title_text = atl.title or (outline.title or topic)
    article_md = (
        f"# {title_text}\n\n"
        f"{atl.lead_markdown}\n\n"
        f"{toc_text}\n\n"
        f"{body_text}\n"
    )
    try:
        save_markdown(
            article_path,
            title=title_text,
            generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
            pipeline="DeepArticle",
            content=article_md,
        )
        srvlog("SAVE_OK", f"article_path={article_path} size={len(article_md)}")
    except Exception as e:
        srvlog("SAVE_ERR", f"{type(e).__name__}: {e}")
        _tb.print_exc()
        raise

    # Save log to filesystem and DB if available
    log_dir = ensure_output_dir(output_subdir)
    log_path = log_dir / f"{safe_filename_base(topic)}_article_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    header = (
        f"# 🧾 Article Generation Log\n\n"
        f"- provider: {_prov}\n"
        f"- lang: {lang}\n"
        f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- duration: {duration_s:.1f}s\n"
        f"- topic: {topic}\n"
    )
    full_log_content = header + "".join(log_lines)
    save_markdown(log_path, title=f"Log: {topic}", generator="bio1c", pipeline="LogDeepArticle", content=full_log_content)

    # DB record
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
                    try:
                        rel_doc = str(article_path.relative_to(Path.cwd())) if article_path.is_absolute() else str(article_path)
                    except ValueError:
                        rel_doc = str(article_path)
                    rd = ResultDoc(
                        job_id=job_id,
                        kind="article",
                        path=rel_doc,
                        topic=topic,
                        provider=_prov,
                        lang=lang,
                        content=article_md,
                        hidden=1 if ((job_meta or {}).get("incognito") is True) else 0,
                    )
                    s.add(rd)
                    s.flush()
                    s.commit()
                try:
                    sync_engine.dispose()
                except Exception:
                    pass
    except Exception as e:
        srvlog("DB_ERR", f"{type(e).__name__}: {e}")
        _tb.print_exc()

    srvlog("DONE", f"path={article_path}")
    if return_log_path:
        return log_path
    return article_path



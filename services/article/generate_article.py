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
from utils.logging import create_logger
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
    style: str = "article_style_1",  # article_style_1 | article_style_2
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
    provider_in = None
    try:
        provider_in = (str((job_meta or {}).get("provider_in", "")).strip().lower() or None) if isinstance(job_meta, dict) else None
    except Exception:
        provider_in = None
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
    
    # Normalize and validate style key early (used in logs below)
    style_key = (style or "article_style_1").strip().lower()
    if style_key not in {"article_style_1", "article_style_2"}:
        style_key = "article_style_1"
    
    # Initialize structured logger
    logger = create_logger("article", show_debug=bool(os.getenv("DEBUG_LOGS")))
    logger.info(f"Starting article generation: '{topic[:100]}'")
    logger.info(f"Configuration: provider_in={(provider_in or provider)}, resolved={_prov}, lang={lang}, style={style_key}")
    
    log_lines: list[str] = []
    def log(section: str, body: str):
        """Log to markdown file - keep it readable and high-level."""
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")
    
    def log_summary(emoji: str, title: str, items: list[str]):
        """Log a clean summary without technical details."""
        content = "\n".join(f"- {item}" for item in items if item)
        log_lines.append(f"---\n\n## {emoji} {title}\n\n{content}\n")

    def _run_with_retries_sync(agent: Any, user_input: str, *, attempts: int = 3, backoff_seq: tuple[int, ...] = (2, 5, 8)):
        """Run agent with simple retries on provider transient errors."""
        for i in range(attempts):
            try:
                return Runner.run_sync(agent, user_input)
            except Exception as e:
                if i < attempts - 1:
                    logger.retry(i + 1, attempts, reason=f"{type(e).__name__}: {str(e)[:100]}")
                    _sleep(backoff_seq[min(i, len(backoff_seq)-1)])
                else:
                    raise
    # Log generation configuration
    log_summary("⚙️", "Конфигурация генерации", [
        f"Провайдер: {provider_in or provider}",
        f"Язык: {lang}",
        f"Стиль: {style_key}",
        f"Тема: {topic[:100]}{'...' if len(topic) > 100 else ''}"
    ])
    # Determine parallelism (bounded to avoid provider rate limits)
    par_env = (os.getenv("ARTICLE_MAX_PAR", "").strip() or None)
    try:
        _par = int(par_env) if par_env is not None else (int(max_parallel) if max_parallel is not None else 4)
    except Exception:
        _par = 4
    max_workers = max(1, min(16, _par))
    logger.info(f"Parallelism configured: {max_workers} workers")

    # Agents import per style
    if style_key == "article_style_2":
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_2.module_01_structure.sections import (  # type: ignore
            build_sections_agent,
        )
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_2.module_02_writing.section_writer import (  # type: ignore
            build_section_writer_agent,
        )
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_2.module_02_writing.article_title_lead_writer import (  # type: ignore
            build_article_title_lead_writer_agent,
        )
    else:
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_1.module_01_structure.sections_and_subsections import (  # type: ignore
            build_sections_agent,
        )
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_1.module_02_writing.subsection_writer import (  # type: ignore
            build_subsection_writer_agent,
        )
        from llm_agents.deep_popular_science_article.deep_popular_science_article_style_1.module_02_writing.article_title_lead_writer import (  # type: ignore
            build_article_title_lead_writer_agent,
        )

    # Module 1: Structure Generation
    logger.stage("Structure Generation", total_stages=3, current_stage=1)
    
    outline_agent = build_sections_agent(provider=_prov)
    user_outline = f"<input>\n<topic>{topic}</topic>\n<lang>{lang}</lang>\n</input>"
    try:
        t0 = time.perf_counter()
        logger.step("Generating initial outline (pass 1 of 3)")
        outline: ArticleOutline = Runner.run_sync(outline_agent, user_outline).final_output  # type: ignore
        sec_count = len(outline.sections)
        sub_count = sum(len(s.subsections) for s in outline.sections)
        duration = time.perf_counter() - t0
        
        # Log full outline JSON for tracking evolution
        log("📑 Outline · Sections", f"```json\n{outline.model_dump_json()}\n```")
        
        # Log readable summary
        outline_items = [f"**{sec.title}**" for sec in outline.sections[:5]]
        if len(outline.sections) > 5:
            outline_items.append(f"...и ещё {len(outline.sections) - 5} разделов")
        log_summary("📋", "Структура статьи создана", [
            f"Разделов: {sec_count}",
            f"Подразделов: {sub_count}",
            "",
            "Основные разделы:",
            *outline_items
        ])
        
        logger.success(f"Initial outline generated: {sec_count} sections, {sub_count} subsections", show_duration=False)
        logger.debug(f"Outline generation took {int(duration*1000)}ms")
        if sec_count == 0:
            raise ValueError("Outline has no sections")
    except Exception as e:
        logger.error("Failed to generate initial outline", exception=e)
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
        logger.step("Refining outline (pass 2 of 3)")
        improved_outline: ArticleOutline = Runner.run_sync(outline_agent, improve_user).final_output  # type: ignore
        # Accept improved outline when it still has sections and not smaller by an extreme margin
        if improved_outline and improved_outline.sections:
            outline = improved_outline
            duration = time.perf_counter() - t1
            # Log full improved outline
            log("📑 Outline · Improved", f"```json\n{outline.model_dump_json()}\n```")
            log_summary("✨", "Структура улучшена", [
                f"Итого разделов: {len(outline.sections)}",
                f"Итого подразделов: {sum(len(s.subsections) for s in outline.sections)}"
            ])
            logger.success(f"Outline refined: {len(outline.sections)} sections", show_duration=False)
            logger.debug(f"Refine took {int(duration*1000)}ms")
    except Exception as e:
        logger.warning(f"Outline refinement failed: {type(e).__name__}: {str(e)[:100]}")

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
        logger.step("Expanding content points (pass 3 of 3)")
        expanded_outline: ArticleOutline = Runner.run_sync(outline_agent, expand_user).final_output  # type: ignore
        if expanded_outline and expanded_outline.sections:
            outline = expanded_outline
            duration = time.perf_counter() - t2
            total_content_items = sum(len(getattr(sub, 'content_items', []) or []) for sec in outline.sections for sub in sec.subsections)
            # Log full expanded outline with content items
            log("📑 Outline · Expanded Content", f"```json\n{outline.model_dump_json()}\n```")
            log_summary("📝", "Контент-пункты добавлены", [
                f"Детализировано подразделов: {sum(len(s.subsections) for s in outline.sections)}",
                f"Контент-пунктов для раскрытия: {total_content_items}"
            ])
            logger.success(f"Content points expanded: {len(outline.sections)} sections ready", show_duration=False)
            logger.debug(f"Expansion took {int(duration*1000)}ms")
    except Exception as e:
        logger.warning(f"Content expansion failed: {type(e).__name__}: {str(e)[:100]}")

    # Module 2: Writing Content
    logger.stage("Content Writing", total_stages=3, current_stage=2)
    
    drafts_by_subsection: dict[tuple[str, str], DraftChunk] = {}
    drafts_by_section: dict[str, Any] = {}
    if style_key == "article_style_2":
        ssw_agent = build_section_writer_agent(provider=_prov)  # type: ignore[name-defined]
        all_sections = [sec for sec in outline.sections]
        logger.info(f"Writing style: section-level (style 2, sequential)")
    else:
        ssw_agent = build_subsection_writer_agent(provider=_prov)  # type: ignore[name-defined]
        all_subs_writing = [(sec, sub) for sec in outline.sections for sub in sec.subsections]
        logger.info(f"Writing style: subsection-level (style 1)")
        logger.parallel_start("Writing subsections", total_jobs=len(all_subs_writing), max_workers=max_workers)

    def _run_subsection_draft(sec_obj, sub_obj):
        ssw_user_local = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec_obj.id}</section_id>\n"
            f"<subsection_id>{sub_obj.id}</subsection_id>\n"
            f"<content_items_json>{_json.dumps([{'id': getattr(ci, 'id', ''), 'point': getattr(ci, 'point', '')} for ci in (getattr(sub_obj, 'content_items', []) or [])], ensure_ascii=False)}</content_items_json>\n"
            "</input>"
        )
        backoff_seq = (2, 4, 8)
        for i in range(3):
            try:
                tloc = time.perf_counter()
                logger.debug(f"Writing subsection {sec_obj.id}/{sub_obj.id} (attempt {i+1}/3)")
                res = asyncio.run(Runner.run(ssw_agent, ssw_user_local))  # type: ignore
                duration = time.perf_counter() - tloc
                logger.debug(f"Subsection {sec_obj.id}/{sub_obj.id} completed in {int(duration*1000)}ms")
                return res.final_output  # type: ignore
            except Exception as ex:
                logger.debug(f"Subsection {sec_obj.id}/{sub_obj.id} failed (attempt {i+1}/3): {type(ex).__name__}")
                if i == 2:
                    break
                _sleep(backoff_seq[min(i, len(backoff_seq)-1)])
        raise RuntimeError(f"Draft generation failed for {sec_obj.id}/{sub_obj.id}")

    def _run_section_draft(sec_obj):
        ssw_user_local = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec_obj.id}</section_id>\n"
            f"<content_items_json>{_json.dumps([{'id': getattr(ci, 'id', ''), 'point': getattr(ci, 'point', '')} for ci in (getattr(sec_obj, 'content_items', []) or [])], ensure_ascii=False)}</content_items_json>\n"
            f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n"
            "</input>"
        )
        backoff_seq = (2, 4, 8)
        for i in range(3):
            try:
                tloc = time.perf_counter()
                logger.debug(f"Writing section {sec_obj.id} (attempt {i+1}/3)")
                res = asyncio.run(Runner.run(ssw_agent, ssw_user_local))  # type: ignore
                duration = time.perf_counter() - tloc
                logger.debug(f"Section {sec_obj.id} completed in {int(duration*1000)}ms")
                return res.final_output  # type: ignore
            except Exception as ex:
                logger.debug(f"Section {sec_obj.id} failed (attempt {i+1}/3): {type(ex).__name__}")
                if i == 2:
                    break
                _sleep(backoff_seq[min(i, len(backoff_seq)-1)])
        raise RuntimeError(f"Draft generation failed for {sec_obj.id}")

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
    logger.info(f"Round configuration: max {max_rounds} rounds, {round_backoff}s backoff between rounds")

    draft_result_by_key: dict[tuple[str, str], DraftChunk] = {}
    section_result_by_id: dict[str, Any] = {}
    if style_key == "article_style_2":
        remaining_secs = [sec for sec in outline.sections]
    else:
        remaining = [(sec, sub) for (sec, sub) in all_subs_writing]
    writing_t0 = time.perf_counter()
    for r in range(1, max_rounds + 1):
        if style_key == "article_style_2":
            if not remaining_secs:
                break
            logger.step(f"Sequential writing round {r}/{max_rounds}", current=r, total=max_rounds)
            logger.info(f"Processing {len(remaining_secs)} sections sequentially")
            failed_secs: list[str] = []
            for sec in remaining_secs:
                try:
                    res = _run_section_draft(sec)
                    section_result_by_id[sec.id] = res
                    logger.debug(f"Section {sec.id} written successfully")
                except Exception as ex:
                    failed_secs.append(sec.id)
                    logger.debug(f"Section {sec.id} failed: {type(ex).__name__}")
            succeeded = len(remaining_secs) - len(failed_secs)
            logger.info(f"Round {r} complete: {succeeded} succeeded, {len(failed_secs)} failed")
            remaining_secs = [sec for sec in outline.sections if sec.id in set(failed_secs)]
            if remaining_secs and r < max_rounds:
                logger.info(f"Retrying {len(remaining_secs)} failed sections after {round_backoff}s")
                _sleep(round_backoff)
        else:
            if not remaining:
                break
            logger.step(f"Parallel writing round {r}/{max_rounds}", current=r, total=max_rounds)
            logger.info(f"Processing {len(remaining)} subsections with {max_workers} workers")
            failed_pairs: list[tuple[str, str]] = []
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut_map = {ex.submit(_run_subsection_draft, sec, sub): (sec.id, sub.id) for (sec, sub) in remaining}
                for fut in as_completed(list(fut_map.keys())):
                    key = fut_map[fut]
                    try:
                        res = fut.result()
                        draft_result_by_key[key] = res
                        logger.debug(f"Subsection {key[0]}/{key[1]} written successfully")
                    except Exception as ex:
                        failed_pairs.append(key)
                        logger.debug(f"Subsection {key[0]}/{key[1]} failed: {type(ex).__name__}")
            succeeded = len(remaining) - len(failed_pairs)
            logger.info(f"Round {r} complete: {succeeded} succeeded, {len(failed_pairs)} failed")
            remaining = []
            for (sid, ssid) in failed_pairs:
                for sec in outline.sections:
                    if sec.id == sid:
                        for sub in sec.subsections:
                            if sub.id == ssid:
                                remaining.append((sec, sub))
                                break
            if remaining and r < max_rounds:
                logger.info(f"Retrying {len(remaining)} failed subsections after {round_backoff}s")
                _sleep(round_backoff)

    if style_key == "article_style_2":
        if remaining_secs:
            logger.step(f"Final sequential attempt for {len(remaining_secs)} failed sections")
            still_missing_s: list[str] = []
            for i, sec in enumerate(remaining_secs, 1):
                ssw_user_inline = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n"
                    "</input>"
                )
                try:
                    logger.debug(f"Final attempt for section {sec.id} ({i}/{len(remaining_secs)})")
                    out = _run_with_retries_sync(ssw_agent, ssw_user_inline)
                    d = out.final_output  # type: ignore
                    section_result_by_id[sec.id] = d
                    logger.debug(f"Section {sec.id} recovered successfully")
                except Exception as ex:
                    still_missing_s.append(sec.id)
                    logger.error(f"Section {sec.id} failed permanently", exception=ex)
            if still_missing_s:
                missing_str = ", ".join(still_missing_s)
                raise RuntimeError(f"Some sections failed to generate: {missing_str}")
    else:
        if 'remaining' in locals() and remaining:
            logger.step(f"Final sequential attempt for {len(remaining)} failed subsections")
            still_missing: list[tuple[str, str]] = []
            for i, (sec, sub) in enumerate(remaining, 1):
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
                    logger.debug(f"Final attempt for subsection {sec.id}/{sub.id} ({i}/{len(remaining)})")
                    out = _run_with_retries_sync(ssw_agent, ssw_user_inline)
                    d: DraftChunk = out.final_output  # type: ignore
                    draft_result_by_key[(sec.id, sub.id)] = d
                    logger.debug(f"Subsection {sec.id}/{sub.id} recovered successfully")
                except Exception as ex:
                    still_missing.append((sec.id, sub.id))
                    logger.error(f"Subsection {sec.id}/{sub.id} failed permanently", exception=ex)
            if still_missing:
                missing_str = ", ".join([f"{sid}/{ssid}" for (sid, ssid) in still_missing])
                raise RuntimeError(f"Some subsections failed to generate: {missing_str}")

    if style_key == "article_style_2":
        for sec in outline.sections:
            d = section_result_by_id.get(sec.id)
            if not d:
                continue
            drafts_by_section[sec.id] = d
            # Log each section draft for tracking evolution
            try:
                log("✍️ Draft · Section", f"{sec.id} → ```json\n{d.model_dump_json()}\n```")
            except Exception:
                pass
        writing_duration = time.perf_counter() - writing_t0
        
        # Log writing summary
        written_sections = [sec.title for sec in outline.sections if sec.id in drafts_by_section][:5]
        if len(drafts_by_section) > 5:
            written_sections.append(f"...и ещё {len(drafts_by_section) - 5}")
        log_summary("✍️", "Разделы написаны", [
            f"Успешно: {len(drafts_by_section)} из {len(outline.sections)}",
            f"Время написания: {int(writing_duration)}с",
            "",
            "Написаны разделы:",
            *[f"✓ {title}" for title in written_sections]
        ])
        logger.parallel_complete(succeeded=len(drafts_by_section), failed=len(outline.sections)-len(drafts_by_section), duration=writing_duration)
    else:
        for sec in outline.sections:
            for sub in sec.subsections:
                key = (sec.id, sub.id)
                d = draft_result_by_key.get(key)
                if not d:
                    continue
                drafts_by_subsection[key] = d
                # Log each subsection draft for tracking evolution
                log("✍️ Draft · Subsection", f"{sec.id}/{sub.id} → ```json\n{d.model_dump_json()}\n```")
        writing_duration = time.perf_counter() - writing_t0
        total_subs = sum(len(sec.subsections) for sec in outline.sections)
        
        # Log writing summary  
        log_summary("✍️", "Подразделы написаны", [
            f"Успешно: {len(drafts_by_subsection)} из {total_subs}",
            f"Время написания: {int(writing_duration)}с",
            f"Разделов обработано: {len(outline.sections)}"
        ])
        logger.parallel_complete(succeeded=len(drafts_by_subsection), failed=total_subs-len(drafts_by_subsection), duration=writing_duration)

    # Refining module removed in 2‑module pipeline

    # Module 3: Titles and Leads
    logger.stage("Titles and Leads Generation", total_stages=3, current_stage=3)
    
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
        if style_key == "article_style_2":
            _lbl = _section_label(i)
            if _lbl:
                toc_lines.append(f"- {_lbl} {sec.title}")
            else:
                toc_lines.append(f"- {i}. {sec.title}")
        else:
            toc_lines.append(f"{i}. {sec.title}")
            for j, sub in enumerate(sec.subsections, start=1):
                toc_lines.append(f"  {i}.{j} {sub.title}")
    body_lines: list[str] = []
    # Initialize Title & Lead agent once and reuse for section leads and article lead
    atl_agent = build_article_title_lead_writer_agent(provider=_prov)
    
    logger.step("Generating sections", current=1, total=len(outline.sections)+1)
    for i, sec in enumerate(outline.sections, start=1):
        _lbl = _section_label(i)
        if body_lines:
            body_lines.append("")
        if _lbl:
            body_lines.append(f"## {_lbl} {sec.title}")
        else:
            body_lines.append(f"## {sec.title}")
        try:
            if style_key == "article_style_2":
                # Style 2: no per-section leads, only append section body
                d = drafts_by_section.get(sec.id)
                sec_body_text = (getattr(d, "markdown", "") or "").strip()
                if sec_body_text:
                    body_lines.append("")
                    body_lines.append(sec_body_text)
            else:
                # Style 1: generate section lead and then append subsections
                sec_md_parts = []
                for sub in sec.subsections:
                    d = drafts_by_subsection.get((sec.id, sub.id))
                    sub_title = d.title if d and d.title else sub.title
                    sub_md = (d.markdown if d else "").strip()
                    sec_md_parts.append(f"### {sub_title}\n\n{sub_md}")
                sec_body_text = "\n\n".join(sec_md_parts)
                try:
                    sec_max_chars = int(os.getenv("SECTION_LEAD_MAX_CHARS", "2000000"))
                except Exception:
                    sec_max_chars = 2000000
                used_sec = sec_body_text if len(sec_body_text) <= sec_max_chars else sec_body_text[:sec_max_chars]
                logger.debug(f"Generating lead for section {sec.id}: body_len={len(sec_body_text)}, used_len={len(used_sec)}")
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
                    body_lines.append("")
                    body_lines.append(sec_lead)
                    duration = time.perf_counter() - t_sec
                    logger.debug(f"Section lead {sec.id} generated: {len(sec_lead)} chars in {int(duration*1000)}ms")
                else:
                    logger.debug(f"Section lead {sec.id} empty, retrying with bullet titles")
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
                            body_lines.append("")
                            body_lines.append(sec_lead2)
                            duration2 = time.perf_counter() - t_sec2
                            logger.debug(f"Section lead {sec.id} retry successful: {len(sec_lead2)} chars in {int(duration2*1000)}ms")
                        else:
                            logger.warning(f"Section lead {sec.id} empty after retry")
                    except Exception as e2:
                        logger.warning(f"Section lead {sec.id} retry failed: {type(e2).__name__}")
                # Append subsections content
                for sub in sec.subsections:
                    d = drafts_by_subsection.get((sec.id, sub.id))
                    sub_title = d.title if d and d.title else sub.title
                    sub_md = (d.markdown if d else "").strip()
                    body_lines.append("")
                    body_lines.append(f"### {sub_title}")
                    if sub_md:
                        body_lines.append("")
                        body_lines.append(sub_md)
        except Exception as e:
            logger.error(f"Failed to generate section lead for {sec.id}", exception=e)
        
    toc_text = "\n".join(toc_lines)
    body_text = "\n".join(body_lines)

    # Title & Lead based on full article content
    logger.step("Generating article title and lead")
    max_chars = 2000000
    try:
        max_chars = int(os.getenv("TITLE_LEAD_MAX_CHARS", "2000000"))
    except Exception:
        max_chars = 2000000
    # Truncate inputs conservatively to help providers that reject very long prompts
    # Keep ToC + up to max_chars of body
    used_body = body_text if len(body_text) <= max_chars else body_text[:max_chars]
    logger.debug(f"Article title/lead input: toc_len={len(toc_text)}, body_len={len(body_text)}, used_len={len(used_body)}")
    atl_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<article_markdown>{toc_text}\n\n{used_body}</article_markdown>\n"
        + (f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n" if style_key == "article_style_2" else "")
        + "</input>"
    )
    try:
        t_atl = time.perf_counter()
        _res = _run_with_retries_sync(atl_agent, atl_user)
        _out = getattr(_res, "final_output", _res)
        if isinstance(_out, ArticleTitleLead):
            atl = _out
        else:
            from utils.json_parse import parse_json_best_effort as _parse
            try:
                data = _parse(str(_out)) or {}
            except Exception:
                data = {}
            # Safe construct to avoid validation errors when fields are missing
            _title_val = (data.get("title") or "").strip() if isinstance(data, dict) else ""
            _lead_val = (data.get("lead_markdown") or "").strip() if isinstance(data, dict) else ""
            atl = ArticleTitleLead(title=_title_val, lead_markdown=_lead_val)
        duration = time.perf_counter() - t_atl
        # If fields are empty, try a plain-agent retry below
        need_plain_retry = (not (getattr(atl, "title", "") or "").strip()) or (not (getattr(atl, "lead_markdown", "") or "").strip())
        if not need_plain_retry:
            # Log full title & lead JSON for tracking
            log("🧾 Title & Lead", f"```json\n{atl.model_dump_json()}\n```")
            # Log readable summary
            log_summary("📰", "Заголовок и вступление", [
                f"**Заголовок:** {atl.title or '(не создан)'}",
                "",
                "**Вступление:**",
                f"{(atl.lead_markdown or '')[:200]}{'...' if len(atl.lead_markdown or '') > 200 else ''}"
            ])
            logger.success(f"Article title and lead generated: title_len={len(atl.title or '')}, lead_len={len(atl.lead_markdown or '')}", show_duration=False)
            logger.debug(f"Title/lead generation took {int(duration*1000)}ms")
        else:
            raise RuntimeError("empty_title_or_lead")
    except Exception as e:
        # If the failure may be due to unsupported output_type, retry once with plain agent (no output schema) and manual JSON parse
        try:
            msg = str(e)
        except Exception:
            msg = ""
        did_retry_plain = False
        if ("output_type" in msg) or ("json_schema" in msg) or ("not supported" in msg) or ("empty_title_or_lead" in msg):
            try:
                did_retry_plain = True
                from agents import Agent  # type: ignore
                from utils.models import get_model as _get_model
                # Load prompt directly
                style_dir = "deep_popular_science_article_style_2" if style_key == "article_style_2" else "deep_popular_science_article_style_1"
                prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "deep_popular_science_article" / style_dir / "module_02_writing" / "article_title_lead_writer.md"
                prompt_text = prompt_path.read_text(encoding="utf-8")
                # Ensure provider-in model is used; if it fails, fall back to a general heavy model
                try:
                    model_id = _get_model(provider_in or _prov, "heavy")
                except Exception:
                    model_id = _get_model(_prov, "heavy")
                plain_agent = Agent(name="Deep Article · Title & Lead (plain)", instructions=prompt_text, model=model_id)
                t_atl2 = time.perf_counter()
                _res2 = _run_with_retries_sync(plain_agent, atl_user)
                _out2 = getattr(_res2, "final_output", _res2)
                from utils.json_parse import parse_json_best_effort as _parse2
                data2 = {}
                try:
                    data2 = _parse2(str(_out2)) or {}
                except Exception:
                    data2 = {}
                atl = ArticleTitleLead(title=(data2.get("title") or ""), lead_markdown=(data2.get("lead_markdown") or ""))
                # If still empty, try minimal input (ToC only + optional main_idea)
                if not (atl.title or '').strip() or not (atl.lead_markdown or '').strip():
                    atl_user_min = (
                        "<input>\n"
                        f"<topic>{topic}</topic>\n"
                        f"<lang>{lang}</lang>\n"
                        f"<article_markdown>{toc_text}</article_markdown>\n"
                        + (f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n" if style_key == "article_style_2" else "")
                        + "</input>"
                    )
                    _res3 = _run_with_retries_sync(plain_agent, atl_user_min)
                    _out3 = getattr(_res3, "final_output", _res3)
                    try:
                        data3 = _parse2(str(_out3)) or {}
                    except Exception:
                        data3 = {}
                    # Preserve any non-empty fields from previous attempt
                    if data3:
                        title3 = (data3.get("title") or "").strip()
                        lead3 = (data3.get("lead_markdown") or "").strip()
                        atl = ArticleTitleLead(
                            title=title3 or (atl.title or ""),
                            lead_markdown=lead3 or (atl.lead_markdown or ""),
                        )
                duration2 = time.perf_counter() - t_atl2
                log("🧾 Title & Lead", f"```json\n{atl.model_dump_json()}\n```")
                log_summary("📰", "Заголовок и вступление", [
                    f"**Заголовок:** {atl.title or '(не создан)'}",
                    "",
                    "**Вступление:**",
                    f"{(atl.lead_markdown or '')[:200]}{'...' if len(atl.lead_markdown or '') > 200 else ''}"
                ])
                logger.success(f"Article title and lead generated (plain retry): title_len={len(atl.title or '')}, lead_len={len(atl.lead_markdown or '')}", show_duration=False)
                logger.debug(f"Title/lead plain retry took {int(duration2*1000)}ms")
            except Exception as e2:
                logger.error("Plain retry for title/lead failed", exception=e2)
        if not did_retry_plain:
            logger.error("Failed to generate article title and lead", exception=e)
        if 'atl' not in locals() or not isinstance(atl, ArticleTitleLead) or not (atl.title or '').strip():
            # Graceful fallback: use outline title/topic, empty lead
            atl = ArticleTitleLead(title=(outline.title or topic), lead_markdown="")
            logger.warning("Using fallback title from outline")
            log("🧾 Title & Lead (fallback)", f"```json\n{atl.model_dump_json()}\n```")
            log_summary("⚠️", "Заголовок (резервный)", [
                f"Использован заголовок из структуры: {atl.title}"
            ])

    title_text = atl.title or (outline.title or topic)
    lead_text = (atl.lead_markdown or "").strip()
    article_md = (
        f"# {title_text}\n\n"
        f"{lead_text}\n\n"
        f"{toc_text}\n\n"
        f"{body_text}\n"
    )
    logger.step("Saving article to file")
    try:
        save_markdown(
            article_path,
            title=title_text,
            generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
            pipeline="DeepArticle",
            content=article_md,
        )
        logger.success(f"Article saved: {article_path.name} ({len(article_md)} chars)", show_duration=False)
    except Exception as e:
        logger.error("Failed to save article", exception=e)
        _tb.print_exc()
        raise

    # Save log to filesystem and DB if available
    log_dir = ensure_output_dir(output_subdir)
    log_path = log_dir / f"{safe_filename_base(topic)}_article_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    provider_in = (job_meta or {}).get("provider_in") if isinstance(job_meta, dict) else None
    header = (
        f"# 🧾 Article Generation Log\n\n"
        f"- provider: {provider_in or _prov}\n"
        f"- style: {style_key}\n"
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
                        provider=(provider_in or provider),
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
        logger.warning(f"Failed to save to database: {type(e).__name__}")
        logger.debug(f"DB error details: {str(e)[:200]}")
        _tb.print_exc()

    logger.total_duration()
    logger.success(f"Article generation complete: {article_path.name}", show_duration=False)
    if return_log_path:
        return log_path
    return article_path



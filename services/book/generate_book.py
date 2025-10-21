#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any
import os
import asyncio
import time
import json as _json
import sys as _sys

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.lang import detect_lang_from_text
from utils.slug import safe_filename_base
from utils.logging import create_logger

from schemas.book import BookOutline
from schemas.article import DraftChunk, ArticleTitleLead


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed."
        ) from e


def generate_book(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",
    output_subdir: str = "deep_book",
    on_progress: Optional[Callable[[str], None]] = None,
    job_meta: Optional[dict] = None,
) -> Path:
    _ensure_root(__file__)
    load_env_from_root(__file__)
    if not topic or not topic.strip():
        raise ValueError("Topic must be a non-empty string")

    Agent, Runner = _try_import_sdk()

    # Resolve provider early (used in initial logs)
    _prov = (provider or "openai").strip().lower()

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass

    logger = create_logger("book", show_debug=bool(os.getenv("DEBUG_LOGS")))
    logger.stage("Initialization", total_stages=5, current_stage=1)
    from datetime import datetime
    started_at = datetime.utcnow()
    logger.info(f"Starting book generation: '{topic[:100]}'", extra={
        "provider": _prov,
        "lang": lang,
        "output_subdir": output_subdir,
    })

    # Markdown log aggregator (similar to articles)
    log_lines: list[str] = []
    def log(section: str, body: str):
        log_lines.append(f"---\n\n## {section}\n\n{body}\n")
    def log_summary(emoji: str, title: str, items: list[str]):
        content = "\n".join(f"- {it}" for it in items if it)
        log_lines.append(f"---\n\n## {emoji} {title}\n\n{content}\n")

    # Import agents
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_01_main_idea.agent_1_main_idea import (  # type: ignore
        build_agent_1_main_idea,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_02_structure.agent_2_table_of_contents import (  # type: ignore
        build_agent_2_toc,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_02_structure.agent_3_table_of_contents_refinement import (  # type: ignore
        build_agent_3_toc_refine,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_02_structure.agent_4_adding_subsections import (  # type: ignore
        build_agent_4_add_subsections,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_02_structure.agent_5_subsections_refinement import (  # type: ignore
        build_agent_5_subsections_refine,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_03_planning.agent_6_planning_of_subsections import (  # type: ignore
        build_agent_6_subsection_plan,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_04_writing.agent_7_subsection_writer import (  # type: ignore
        build_agent_7_subsection_writer,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_04_writing.agent_8_section_lead_writer import (  # type: ignore
        build_agent_8_section_lead_writer,
    )
    from llm_agents.books.deep_popular_science_book.deep_popular_science_book_style_1.module_04_writing.agent_9_title_lead_writer import (  # type: ignore
        build_agent_9_title_lead_writer,
    )

    # _prov already resolved above

    def _run_with_retries_sync(agent: Any, user_input: str, *, attempts: int = 3, backoff_seq: tuple[int, ...] = (2, 5, 8)):
        for i in range(attempts):
            try:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    return loop.run_until_complete(Runner.run(agent, user_input))
                finally:
                    try:
                        loop.close()
                    except Exception:
                        pass
                    try:
                        asyncio.set_event_loop(None)
                    except Exception:
                        pass
            except Exception as e:
                if i < attempts - 1:
                    logger.retry(i + 1, attempts, reason=f"{type(e).__name__}: {str(e)[:100]}")
                    time.sleep(backoff_seq[min(i, len(backoff_seq)-1)])
                else:
                    raise

    # Agent 1: Main idea
    logger.stage("Agent 1 · Main Idea", total_stages=5, current_stage=1)
    a1 = build_agent_1_main_idea(provider=_prov)
    a1_in = f"<input>\n- topic: {topic}\n- lang: {lang}\n</input>"
    t_a1 = time.perf_counter()
    try:
        logger.info("A1_RUN", extra={"input_len": len(a1_in)})
        a1_res = _run_with_retries_sync(a1, a1_in)
        main_idea_obj = getattr(a1_res, "final_output", None)
        main_idea = getattr(main_idea_obj, "main_idea", "") if main_idea_obj else ""
        logger.success("A1_OK", show_duration=True, extra={"main_idea_len": len(main_idea or "")})
        try:
            import json as __json
            log("🧩 Concept · Main Idea", f"```json\n{__json.dumps({'main_idea': main_idea}, ensure_ascii=False)}\n```")
        except Exception:
            pass
    except Exception as e:
        logger.error("A1_FAIL", exception=e)
        main_idea = ""

    # Agent 2: ToC (sections with purposes)
    logger.stage("Agent 2 · Build ToC", total_stages=5, current_stage=2)
    a2 = build_agent_2_toc(provider=_prov)
    a2_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" "</input>"
    )
    try:
        logger.info("A2_RUN", extra={"input_len": len(a2_in)})
        toc_outline: BookOutline = _run_with_retries_sync(a2, a2_in).final_output  # type: ignore
        sec_count = len(getattr(toc_outline, 'sections', []) or [])
        logger.success("A2_OK", show_duration=True, extra={"sections": sec_count})
        try:
            log("📑 ToC · Initial", f"```json\n{toc_outline.model_dump_json()}\n```")
            log_summary("📋", "Оглавление создано", [f"Разделов: {sec_count}"])
        except Exception:
            pass
    except Exception as e:
        logger.error("A2_FAIL", exception=e)
        raise

    # Agent 3: ToC refinement
    logger.stage("Agent 3 · Refine ToC", total_stages=5, current_stage=2)
    a3 = build_agent_3_toc_refine(provider=_prov)
    a3_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    try:
        logger.info("A3_RUN", extra={"input_len": len(a3_in)})
        toc_outline = _run_with_retries_sync(a3, a3_in).final_output  # type: ignore
        logger.success("A3_OK", show_duration=True, extra={"sections": len(toc_outline.sections)})
        try:
            log("📑 ToC · Refined", f"```json\n{toc_outline.model_dump_json()}\n```")
        except Exception:
            pass
    except Exception as e:
        logger.error("A3_FAIL", exception=e)
        raise

    # Agent 4: Add subsections
    logger.stage("Agent 4 · Add Subsections", total_stages=5, current_stage=2)
    a4 = build_agent_4_add_subsections(provider=_prov)
    a4_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    try:
        logger.info("A4_RUN", extra={"input_len": len(a4_in)})
        toc_outline = _run_with_retries_sync(a4, a4_in).final_output  # type: ignore
        subs = sum(len(s.subsections) for s in toc_outline.sections)
        logger.success("A4_OK", show_duration=True, extra={"sections": len(toc_outline.sections), "subsections": subs})
        try:
            log("📑 ToC · With Subsections", f"```json\n{toc_outline.model_dump_json()}\n```")
            log_summary("📝", "Подразделы добавлены", [f"Подразделов: {subs}"])
        except Exception:
            pass
    except Exception as e:
        logger.error("A4_FAIL", exception=e)
        raise

    # Agent 5: Subsections refinement
    logger.stage("Agent 5 · Refine Subsections", total_stages=5, current_stage=2)
    a5 = build_agent_5_subsections_refine(provider=_prov)
    a5_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    try:
        logger.info("A5_RUN", extra={"input_len": len(a5_in)})
        toc_outline = _run_with_retries_sync(a5, a5_in).final_output  # type: ignore
        subs = sum(len(s.subsections) for s in toc_outline.sections)
        logger.success("A5_OK", show_duration=True, extra={"sections": len(toc_outline.sections), "subsections": subs})
        try:
            log("📑 ToC · Subsections Refined", f"```json\n{toc_outline.model_dump_json()}\n```")
        except Exception:
            pass
    except Exception as e:
        logger.error("A5_FAIL", exception=e)
        raise

    # Agent 6: Plan per subsection → build mapping plans
    logger.stage("Agent 6 · Plan Subsections", total_stages=5, current_stage=3)
    plan_agent = build_agent_6_subsection_plan(provider=_prov)
    plans: dict[tuple[str, str], list[str]] = {}
    for sec in toc_outline.sections:
        for sub in sec.subsections:
            pi_in = (
                "<input>\n"
                f"- topic: {topic}\n"
                f"- lang: {lang}\n"
                f"- main_idea: {main_idea}\n"
                f"- section_id: {sec.id}\n"
                f"- subsection_id: {sub.id}\n"
                f"- section_purpose: {(sec.purpose or '').strip()}\n"
                f"- subsection_purpose: {(sub.purpose or '').strip()}\n"
                f"- toc_json: {toc_outline.model_dump_json()}\n"
                "</input>"
            )
            try:
                logger.step("A6_RUN", current=None, total=None)
                t_a6 = time.perf_counter()
                res = _run_with_retries_sync(plan_agent, pi_in)
                sp = getattr(res, "final_output", None)
                items = list(getattr(sp, "plan_items", []) or [])
                logger.info("A6_OK", extra={"section": sec.id, "subsection": sub.id, "items": len(items), "t_ms": int((time.perf_counter()-t_a6)*1000)})
            except Exception as e:
                logger.warning("A6_FAIL", extra={"section": sec.id, "subsection": sub.id, "error": type(e).__name__})
                items = []
            plans[(sec.id, sub.id)] = items
    try:
        planned = sum(len(v) for v in plans.values())
        log_summary("🧩", "Планы подразделов", [f"Итого пунктов плана: {planned}"])
    except Exception:
        pass

    # Agent 7: Write subsections (parallel rounds similar to article style 1)
    logger.stage("Agent 7 · Write Subsections", total_stages=5, current_stage=4)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ssw = build_agent_7_subsection_writer(provider=_prov)
    all_pairs = [(sec, sub) for sec in toc_outline.sections for sub in sec.subsections]
    drafts: dict[tuple[str, str], DraftChunk] = {}
    max_workers = max(1, min(8, int(os.getenv("ARTICLE_MAX_PAR", "4") or 4)))
    logger.parallel_start("Write subsections", total_jobs=len(all_pairs), max_workers=max_workers)
    t_w = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for sec, sub in all_pairs:
            content_items = [{"id": str(i + 1), "point": p} for i, p in enumerate(plans.get((sec.id, sub.id), []))]
            user = (
                "<input>\n"
                f"- topic: {topic}\n"
                f"- lang: {lang}\n"
                f"- main_idea: {main_idea}\n"
                f"- section_id: {sec.id}\n"
                f"- subsection_id: {sub.id}\n"
                f"- toc_json: {toc_outline.model_dump_json()}\n"
                f"- content_items_json: {_json.dumps(content_items, ensure_ascii=False)}\n"
                "</input>"
            )
            def _job(u: str, _agent: Any):
                try:
                    return _run_with_retries_sync(_agent, u)
                except Exception as _e:
                    raise _e
            futs[ex.submit(_job, user, ssw)] = (sec.id, sub.id)
        failed = 0
        for fut in as_completed(list(futs.keys())):
            key = futs[fut]
            try:
                out = fut.result()
                d = getattr(out, "final_output", None)
                if d:
                    drafts[key] = d
            except Exception as e:
                failed += 1
                logger.warning("A7_WRITE_FAIL", extra={"section": key[0], "subsection": key[1], "error": type(e).__name__})
                continue
    logger.parallel_complete(succeeded=len(drafts), failed=failed, duration=(time.perf_counter()-t_w))
    try:
        log_summary("✍️", "Подразделы написаны", [f"Успешно: {len(drafts)}", f"Ошибок: {failed}"])
    except Exception:
        pass

    # Agent 8: Section leads
    logger.stage("Agent 8 · Section Leads", total_stages=5, current_stage=4)
    sec_leads: dict[str, str] = {}
    a8 = build_agent_8_section_lead_writer(provider=_prov)
    for sec in toc_outline.sections:
        # Concatenate subsections for lead context
        parts = []
        for sub in sec.subsections:
            d = drafts.get((sec.id, sub.id))
            if not d:
                continue
            parts.append(f"### {d.title or sub.title}\n\n{(d.markdown or '').strip()}")
        sec_body = "\n\n".join(parts)
        user = (
            "<input>\n"
            f"- topic: {topic}\n"
            f"- lang: {lang}\n"
            f"- main_idea: {main_idea}\n"
            f"- section_id: {sec.id}\n"
            f"- section_title: {sec.title}\n"
            f"- section_purpose: {(sec.purpose or '').strip()}\n"
            f"- section_markdown: {sec_body}\n"
            "</input>"
        )
        try:
            t_a8 = time.perf_counter()
            res = _run_with_retries_sync(a8, user)
            lead_obj = getattr(res, "final_output", None)
            sec_lead = getattr(lead_obj, "lead_markdown", "") if lead_obj else ""
            logger.info("A8_OK", extra={"section": sec.id, "lead_len": len(sec_lead or ""), "t_ms": int((time.perf_counter()-t_a8)*1000)})
        except Exception as e:
            logger.warning("A8_FAIL", extra={"section": sec.id, "error": type(e).__name__})
            sec_lead = ""
        sec_leads[sec.id] = sec_lead
    try:
        have_leads = sum(1 for v in sec_leads.values() if (v or '').strip())
        log_summary("📰", "Лиды разделов", [f"Сгенерировано: {have_leads} из {len(toc_outline.sections)}"])
    except Exception:
        pass

    # Assemble body
    body_lines: list[str] = []
    # Resolve display language for headings (ru/en); if lang=auto, detect by topic
    _lang_l = (lang or "auto").strip().lower()
    _disp_lang = _lang_l if _lang_l in {"ru", "en"} else detect_lang_from_text(topic or "")
    toc_lines: list[str] = ["## Оглавление" if _disp_lang == "ru" else "## Table of Contents"]
    def _book_section_label(idx: int) -> str | None:
        if _disp_lang == "ru":
            return f"Глава {idx}."
        if _disp_lang == "en":
            return f"Chapter {idx}."
        return None
    for i, sec in enumerate(toc_outline.sections, start=1):
        # ToC section line with language-specific label for ru/en
        _lbl = _book_section_label(i)
        if _lbl:
            toc_lines.append(f"{_lbl} {sec.title}")
        else:
            toc_lines.append(f"{i}. {sec.title}")
        # Include subsections in ToC
        for j, sub in enumerate(getattr(sec, "subsections", []) or [], start=1):
            # Use NBSPs to increase visual indent without triggering Markdown code blocks
            _indent = "\u00A0\u00A0\u00A0\u00A0"  # 4×NBSP
            toc_lines.append(f"{_indent}{i}.{j}. {sub.title}")
        # Section title (H2) with language-specific chapter label
        body_lines.append("")
        _lbl2 = _book_section_label(i)
        if _lbl2:
            body_lines.append(f"## {_lbl2} {sec.title}")
        else:
            body_lines.append(f"## {sec.title}")
        # Section lead
        lead = (sec_leads.get(sec.id) or "").strip()
        if lead:
            body_lines.append("")
            body_lines.append(lead)
        # Subsections
        for sub in sec.subsections:
            d = drafts.get((sec.id, sub.id))
            sub_title = (getattr(d, "title", "") or sub.title)
            sub_md = (getattr(d, "markdown", "") or "").strip()
            body_lines.append("")
            body_lines.append(f"### {sub_title}")
            if sub_md:
                body_lines.append("")
                body_lines.append(sub_md)

    body_text = "\n".join(body_lines)
    toc_text = "\n".join(toc_lines)

    # Agent 9: Book title & lead (full body)
    logger.stage("Agent 9 · Title & Book Lead", total_stages=5, current_stage=5)
    a9 = build_agent_9_title_lead_writer(provider=_prov)
    a9_user = (
        "<input>\n"
        f"- topic: {topic}\n"
        f"- lang: {lang}\n"
        f"- main_idea: {main_idea}\n"
        f"- toc_json: {toc_outline.model_dump_json()}\n"
        f"- book_markdown: {toc_text}\n\n{body_text}\n"
        "</input>"
    )
    try:
        logger.info("A9_RUN", extra={"input_len": len(a9_user)})
        atl = _run_with_retries_sync(a9, a9_user).final_output  # type: ignore
        logger.success("A9_OK", show_duration=True, extra={"title_len": len(getattr(atl, 'title', '') or ''), "lead_len": len(getattr(atl, 'lead_markdown', '') or '')})
        try:
            import json as __json
            log("🧾 Title & Lead", f"```json\n{__json.dumps({'title': getattr(atl, 'title', ''), 'lead_markdown': getattr(atl, 'lead_markdown', '')[:400]}, ensure_ascii=False)}\n```")
        except Exception:
            pass
    except Exception as e:
        logger.error("A9_FAIL", exception=e)
        atl = ArticleTitleLead(title=(topic or ""), lead_markdown="")

    title_text = getattr(atl, "title", None) or topic
    lead_text = (getattr(atl, "lead_markdown", "") or "").strip()

    # Save markdown
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_book"
    book_path = next_available_filepath(output_dir, base, ".md")
    content = f"# {title_text}\n\n{lead_text}\n\n{toc_text}\n\n{body_text}\n"
    logger.stage("Save Result", total_stages=5, current_stage=5)
    save_markdown(
        book_path,
        title=title_text,
        generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
        pipeline="DeepBook",
        content=content,
    )
    logger.success("Book saved", show_duration=True, extra={"path": str(book_path), "chars": len(content)})
    # Save log (compact) and DB records (best-effort)
    try:
        log_dir = ensure_output_dir(output_subdir)
        log_path = log_dir / f"{safe_filename_base(topic)}_book_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
        finished_at = datetime.utcnow()
        header = (
            f"# 🧾 Book Generation Log\n\n"
            f"- provider: {_prov}\n"
            f"- lang: {lang}\n"
            f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
            f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
            f"- topic: {topic}\n"
            f"- sections: {len(toc_outline.sections)}\n"
            f"- subsections: {sum(len(s.subsections) for s in toc_outline.sections)}\n\n"
            f"## Outline JSON\n\n```json\n{toc_outline.model_dump_json()}\n```\n"
        )
        full_log_content = header + "".join(log_lines)
        save_markdown(log_path, title=f"Log: {topic}", generator="bio1c", pipeline="LogDeepBook", content=full_log_content)
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
                            rel_doc = str(book_path.relative_to(Path.cwd())) if book_path.is_absolute() else str(book_path)
                        except ValueError:
                            rel_doc = str(book_path)
                        rd = ResultDoc(
                            job_id=job_id,
                            kind="book",
                            path=rel_doc,
                            topic=topic,
                            provider=_prov,
                            lang=lang,
                            content=content,
                            hidden=1 if ((job_meta or {}).get("incognito") is True) else 0,
                        )
                        s.add(rd)
                        s.flush()
                        s.commit()
                    try:
                        sync_engine.dispose()
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    logger.total_duration()
    return book_path



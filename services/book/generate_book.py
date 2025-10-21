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

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass

    logger = create_logger("book", show_debug=bool(os.getenv("DEBUG_LOGS")))
    logger.info(f"Starting book generation: '{topic[:100]}'")

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

    _prov = (provider or "openai").strip().lower()

    # Agent 1: Main idea
    a1 = build_agent_1_main_idea(provider=_prov)
    a1_in = f"<input>\n- topic: {topic}\n- lang: {lang}\n</input>"
    a1_res = Runner.run_sync(a1, a1_in)
    main_idea_obj = getattr(a1_res, "final_output", None)
    main_idea = getattr(main_idea_obj, "main_idea", "") if main_idea_obj else ""
    logger.info(f"Main idea len={len(main_idea or '')}")

    # Agent 2: ToC (sections with purposes)
    a2 = build_agent_2_toc(provider=_prov)
    a2_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" "</input>"
    )
    toc_outline: BookOutline = Runner.run_sync(a2, a2_in).final_output  # type: ignore
    logger.info(f"ToC sections={len(getattr(toc_outline,'sections',[]) or [])}")

    # Agent 3: ToC refinement
    a3 = build_agent_3_toc_refine(provider=_prov)
    a3_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    toc_outline = Runner.run_sync(a3, a3_in).final_output  # type: ignore

    # Agent 4: Add subsections
    a4 = build_agent_4_add_subsections(provider=_prov)
    a4_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    toc_outline = Runner.run_sync(a4, a4_in).final_output  # type: ignore

    # Agent 5: Subsections refinement
    a5 = build_agent_5_subsections_refine(provider=_prov)
    a5_in = (
        "<input>\n" f"- topic: {topic}\n" f"- lang: {lang}\n" f"- main_idea: {main_idea}\n" f"- toc_json: {toc_outline.model_dump_json()}\n" "</input>"
    )
    toc_outline = Runner.run_sync(a5, a5_in).final_output  # type: ignore

    # Agent 6: Plan per subsection → build mapping plans
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
                res = Runner.run_sync(plan_agent, pi_in)
                sp = getattr(res, "final_output", None)
                items = list(getattr(sp, "plan_items", []) or [])
            except Exception:
                items = []
            plans[(sec.id, sub.id)] = items

    # Agent 7: Write subsections (parallel rounds similar to article style 1)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ssw = build_agent_7_subsection_writer(provider=_prov)
    all_pairs = [(sec, sub) for sec in toc_outline.sections for sub in sec.subsections]
    drafts: dict[tuple[str, str], DraftChunk] = {}
    max_workers = max(1, min(8, int(os.getenv("ARTICLE_MAX_PAR", "4") or 4)))
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
            futs[ex.submit(Runner.run, ssw, user)] = (sec.id, sub.id)
        for fut in as_completed(list(futs.keys())):
            key = futs[fut]
            try:
                out = asyncio.get_event_loop().run_until_complete(fut)  # use existing loop
                d = getattr(out, "final_output", None)
                if d:
                    drafts[key] = d
            except Exception:
                continue

    # Agent 8: Section leads
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
            res = Runner.run_sync(a8, user)
            lead_obj = getattr(res, "final_output", None)
            sec_lead = getattr(lead_obj, "lead_markdown", "") if lead_obj else ""
        except Exception:
            sec_lead = ""
        sec_leads[sec.id] = sec_lead

    # Assemble body
    body_lines: list[str] = []
    toc_lines: list[str] = ["## Оглавление" if (lang or "auto").lower().startswith("ru") else "## Table of Contents"]
    for i, sec in enumerate(toc_outline.sections, start=1):
        toc_lines.append(f"{i}. {sec.title}")
        # Section title
        body_lines.append("")
        body_lines.append(f"# {sec.title}")
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
            body_lines.append(f"## {sub_title}")
            if sub_md:
                body_lines.append("")
                body_lines.append(sub_md)

    body_text = "\n".join(body_lines)
    toc_text = "\n".join(toc_lines)

    # Agent 9: Book title & lead (full body)
    a9 = build_agent_9_title_lead_writer(provider=_prov)
    a9_user = (
        "<input>\n"
        f"- topic: {topic}\n"
        f"- lang: {lang}\n"
        f"- main_idea: {main_idea}\n"
        f"- book_markdown: {toc_text}\n\n{body_text}\n"
        "</input>"
    )
    try:
        atl = Runner.run_sync(a9, a9_user).final_output  # type: ignore
    except Exception:
        atl = ArticleTitleLead(title=(topic or ""), lead_markdown="")

    title_text = getattr(atl, "title", None) or topic
    lead_text = (getattr(atl, "lead_markdown", "") or "").strip()

    # Save markdown
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_book"
    book_path = next_available_filepath(output_dir, base, ".md")
    content = f"# {title_text}\n\n{lead_text}\n\n{toc_text}\n\n{body_text}\n"
    save_markdown(
        book_path,
        title=title_text,
        generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
        pipeline="DeepBook",
        content=content,
    )
    return book_path



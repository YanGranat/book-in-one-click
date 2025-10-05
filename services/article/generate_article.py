#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any
import os
import asyncio
import time
import json as _json

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.slug import safe_filename_base
from services.providers.runner import ProviderRunner

from schemas.article import ArticleOutline, OutlineChangeList, DraftChunk, LeadChunk, TitleProposal


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
    enable_research: bool = True,
    enable_refine: bool = True,
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
    log("🧭 Config", f"provider={_prov}\nlang={lang}\nresearch={bool(enable_research)}\nrefine={bool(enable_refine)}")

    # Agents import
    from llm_agents.deep_popular_science_article.module_01_structure.sections_and_subsections import build_sections_and_subsections_agent
    from llm_agents.deep_popular_science_article.module_01_structure.content_of_subsections import build_subsections_content_agent
    from llm_agents.deep_popular_science_article.module_02_research.topic_research import build_topic_research_agent
    from llm_agents.deep_popular_science_article.module_02_research.section_research import build_section_research_agent
    from llm_agents.deep_popular_science_article.module_02_research.subsection_research_evidence import build_subsection_research_agent
    from llm_agents.deep_popular_science_article.module_02_research.outline_editor import build_outline_editor_agent
    from llm_agents.deep_popular_science_article.module_03_writing.title_namer import build_title_namer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.article_lead_writer import build_article_lead_writer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.section_lead_writer import build_section_lead_writer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.subsection_writer import build_subsection_writer_agent
    from llm_agents.deep_popular_science_article.module_04_refining.section_lead_refiner import build_section_lead_refiner_agent
    from llm_agents.deep_popular_science_article.module_04_refining.subsection_refiner import build_subsection_refiner_agent

    # Module 1
    outline_agent = build_sections_and_subsections_agent()
    user_outline = f"<input>\n<topic>{topic}</topic>\n<lang>{lang}</lang>\n</input>"
    outline: ArticleOutline = Runner.run_sync(outline_agent, user_outline).final_output  # type: ignore
    log("📑 Outline · Sections", f"```json\n{outline.model_dump_json()}\n```")

    content_agent = build_subsections_content_agent()
    user_content = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    outline = Runner.run_sync(content_agent, user_content).final_output  # type: ignore
    log("📑 Outline · Subsections Content", f"```json\n{outline.model_dump_json()}\n```")

    # Module 2 (optional)
    if enable_research:
        tr_agent = build_topic_research_agent()
        tr_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            "</input>"
        )
        top_changes: OutlineChangeList = Runner.run_sync(tr_agent, tr_user).final_output  # type: ignore
        log("🔎 Research · Topic Changes", f"```json\n{top_changes.model_dump_json()}\n```")
        ed_agent = build_outline_editor_agent()
        ed_user = (
            "<input>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<changes_json>{top_changes.model_dump_json()}</changes_json>\n"
            "</input>"
        )
        outline = Runner.run_sync(ed_agent, ed_user).final_output  # type: ignore
        log("🛠️ Outline · Applied Topic Changes", f"```json\n{outline.model_dump_json()}\n```")

        sr_agent = build_section_research_agent()
        ssr_agent = build_subsection_research_agent()
        evidence_list: list[dict[str, Any]] = []
        for sec in outline.sections:
            sr_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                "</input>"
            )
            sec_changes: OutlineChangeList = Runner.run_sync(sr_agent, sr_user).final_output  # type: ignore
            log("🔎 Research · Section Changes", f"{sec.id} → ```json\n{sec_changes.model_dump_json()}\n```")
            ed_user2 = (
                "<input>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<changes_json>{sec_changes.model_dump_json()}</changes_json>\n"
                "</input>"
            )
            outline = Runner.run_sync(ed_agent, ed_user2).final_output  # type: ignore
            # Subsections
            for sub in sec.subsections:
                ss_user = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    f"<subsection_id>{sub.id}</subsection_id>\n"
                    "</input>"
                )
                ss_res = Runner.run_sync(ssr_agent, ss_user).final_output  # type: ignore
                try:
                    ed_user3 = (
                        "<input>\n"
                        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                        f"<changes_json>{ss_res.changes.model_dump_json()}</changes_json>\n"
                        "</input>"
                    )
                    outline = Runner.run_sync(ed_agent, ed_user3).final_output  # type: ignore
                except Exception:
                    pass
                try:
                    evidence_list.append(_json.loads(ss_res.evidence.model_dump_json()))
                except Exception:
                    pass
        try:
            log("📚 Evidence · Collected", f"```json\n{_json.dumps(evidence_list, ensure_ascii=False)}\n```")
        except Exception:
            pass
    else:
        log("🔎 Research · Skipped", "Research module disabled by configuration")

    # Module 3
    t_agent = build_title_namer_agent()
    t_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    title: TitleProposal = Runner.run_sync(t_agent, t_user).final_output  # type: ignore
    log("🧾 Title", f"```json\n{title.model_dump_json()}\n```")

    al_agent = build_article_lead_writer_agent()
    al_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    article_lead: LeadChunk = Runner.run_sync(al_agent, al_user).final_output  # type: ignore
    log("✍️ Article Lead", f"```json\n{article_lead.model_dump_json()}\n```")

    sl_agent = build_section_lead_writer_agent()
    ssw_agent = build_subsection_writer_agent()
    leads_by_section: dict[str, LeadChunk] = {}
    drafts_by_subsection: dict[tuple[str, str], DraftChunk] = {}
    for sec in outline.sections:
        sl_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec.id}</section_id>\n"
            "</input>"
        )
        lead: LeadChunk = Runner.run_sync(sl_agent, sl_user).final_output  # type: ignore
        leads_by_section[sec.id] = lead
        log("✍️ Section Lead", f"{sec.id} → ```json\n{lead.model_dump_json()}\n```")
        for sub in sec.subsections:
            ssw_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<subsection_id>{sub.id}</subsection_id>\n"
                "</input>"
            )
            d: DraftChunk = Runner.run_sync(ssw_agent, ssw_user).final_output  # type: ignore
            drafts_by_subsection[(sec.id, sub.id)] = d
            log("✍️ Draft · Subsection", f"{sec.id}/{sub.id} → ```json\n{d.model_dump_json()}\n``>")

    # Module 4 (optional)
    if enable_refine:
        slr_agent = build_section_lead_refiner_agent()
        ssr_agent2 = build_subsection_refiner_agent()
        for sec in outline.sections:
            lead = leads_by_section.get(sec.id)
            if lead:
                slr_user = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    f"<lead_chunk>{lead.model_dump_json()}</lead_chunk>\n"
                    "</input>"
                )
                rlead: LeadChunk = Runner.run_sync(slr_agent, slr_user).final_output  # type: ignore
                leads_by_section[sec.id] = rlead
                log("✨ Refine · Section Lead", f"{sec.id} → ```json\n{rlead.model_dump_json()}\n```")
            for sub in sec.subsections:
                d = drafts_by_subsection.get((sec.id, sub.id))
                if not d:
                    continue
                ssr_user2 = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    f"<subsection_id>{sub.id}</subsection_id>\n"
                    f"<draft_chunk>{d.model_dump_json()}</draft_chunk>\n"
                    "</input>"
                )
                rd: DraftChunk = Runner.run_sync(ssr_agent2, ssr_user2).final_output  # type: ignore
                drafts_by_subsection[(sec.id, sub.id)] = rd
                log("✨ Refine · Subsection", f"{sec.id}/{sub.id} → ```json\n{rd.model_dump_json()}\n```")
    else:
        log("✨ Refine · Skipped", "Refine module disabled by configuration")

    # Assemble final Markdown
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_article"
    article_path = next_available_filepath(output_dir, base, ".md")

    toc_lines = ["## Оглавление"]
    for i, sec in enumerate(outline.sections, start=1):
        toc_lines.append(f"- {i}. {sec.title}")
        for j, sub in enumerate(sec.subsections, start=1):
            toc_lines.append(f"  - {i}.{j} {sub.title}")
    body_lines: list[str] = []
    for sec in outline.sections:
        lead_md = (leads_by_section.get(sec.id).markdown if sec.id in leads_by_section else "")
        body_lines.append(f"\n\n## {sec.title}\n\n{lead_md}\n")
        for sub in sec.subsections:
            d = drafts_by_subsection.get((sec.id, sub.id))
            sub_title = d.title if d and d.title else sub.title
            sub_md = d.markdown if d else ""
            body_lines.append(f"\n### {sub_title}\n\n{sub_md}\n")

    title_text = (title.title if isinstance(title, TitleProposal) else (outline.title or topic))
    toc_text = "\n".join(toc_lines)
    body_text = "\n".join(body_lines)
    article_md = (
        f"# {title_text}\n\n"
        f"{(article_lead.markdown if isinstance(article_lead, LeadChunk) else '')}\n\n"
        f"{toc_text}\n\n"
        f"{body_text}\n"
    )
    save_markdown(article_path, title=title_text, generator=("OpenAI Agents SDK" if _prov == "openai" else _prov), pipeline="DeepArticle", content=article_md)

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
        print(f"[ERROR] Failed to record article in DB: {e}")

    if return_log_path:
        return log_path
    return article_path



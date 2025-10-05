#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
import argparse
from typing import Any, Optional

# Ensure project root on sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils.env import ensure_project_root_on_syspath as ensure_root, load_env_from_root
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.slug import safe_filename_base
from utils.json_parse import parse_json_best_effort
from services.providers.runner import ProviderRunner

from schemas.article import (
    ArticleOutline,
    OutlineChangeList,
    DraftChunk,
    LeadChunk,
    TitleProposal,
)


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed."
        ) from e


def _load_prompt(rel: str) -> str:
    return (Path(__file__).resolve().parents[1] / "prompts" / "deep_popular_science_article" / rel).read_text(encoding="utf-8")


def _log_append(lines: list[str], section: str, body: str) -> None:
    lines.append(f"---\n\n## {section}\n\n{body}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deep popular science article")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider: openai|gemini|claude")
    parser.add_argument("--out", type=str, default="deep_article", help="Output subdirectory")
    parser.add_argument("--include-logs", action="store_true", help="Save detailed process log")
    args = parser.parse_args()

    ensure_root(__file__)
    load_env_from_root(__file__)

    provider = (args.provider or "openai").strip().lower()
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден")
        return
    if provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY не найден")
        return
    if provider in {"gemini", "google"} and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        print("❌ GOOGLE_API_KEY/GEMINI_API_KEY не найден")
        return

    topic = (args.topic or "").strip()
    if not topic:
        print("Введите тему:")
        topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    Agent, Runner = _try_import_sdk()
    pr = ProviderRunner(provider)

    logs: list[str] = []
    _log_append(logs, "🧭 Config", f"provider={provider}\nlang={args.lang}\ntopic={topic}")

    # Module 1: Outline
    from llm_agents.deep_popular_science_article.module_01_structure.sections_and_subsections import (
        build_sections_and_subsections_agent,
    )
    from llm_agents.deep_popular_science_article.module_01_structure.content_of_subsections import (
        build_subsections_content_agent,
    )

    user_outline = f"<input>\n<topic>{topic}</topic>\n<lang>{args.lang}</lang>\n</input>"
    outline_agent = build_sections_and_subsections_agent()
    outline_res = Runner.run_sync(outline_agent, user_outline)
    outline: ArticleOutline = outline_res.final_output  # type: ignore
    _log_append(logs, "📑 Outline · Sections", f"```json\n{outline.model_dump_json()}\n```")

    content_agent = build_subsections_content_agent()
    user_content = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{args.lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    outline_res2 = Runner.run_sync(content_agent, user_content)
    outline = outline_res2.final_output  # type: ignore
    _log_append(logs, "📑 Outline · Subsections Content", f"```json\n{outline.model_dump_json()}\n```")

    # Module 2: Research (single‑pass, no multi‑level loops; one section/subsection at a time)
    from llm_agents.deep_popular_science_article.module_02_research.topic_research import build_topic_research_agent
    from llm_agents.deep_popular_science_article.module_02_research.section_research import build_section_research_agent
    from llm_agents.deep_popular_science_article.module_02_research.subsection_research_evidence import (
        build_subsection_research_agent,
    )
    from llm_agents.deep_popular_science_article.module_02_research.outline_editor import build_outline_editor_agent

    # Topic-level adjustments
    tr_agent = build_topic_research_agent()
    tr_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{args.lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    tr_res = Runner.run_sync(tr_agent, tr_user)
    top_changes: OutlineChangeList = tr_res.final_output  # type: ignore
    _log_append(logs, "🔎 Research · Topic Changes", f"```json\n{top_changes.model_dump_json()}\n```")
    ed_agent = build_outline_editor_agent()
    ed_user = (
        "<input>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        f"<changes_json>{top_changes.model_dump_json()}</changes_json>\n"
        "</input>"
    )
    outline = Runner.run_sync(ed_agent, ed_user).final_output  # type: ignore
    _log_append(logs, "🛠️ Outline · Applied Topic Changes", f"```json\n{outline.model_dump_json()}\n```")

    # Section-level research and edits, one section at a time
    sr_agent = build_section_research_agent()
    for sec in outline.sections:
        sr_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{args.lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec.id}</section_id>\n"
            "</input>"
        )
        sr_res = Runner.run_sync(sr_agent, sr_user)
        sec_changes: OutlineChangeList = sr_res.final_output  # type: ignore
        _log_append(logs, "🔎 Research · Section Changes", f"{sec.id} → ```json\n{sec_changes.model_dump_json()}\n```")
        ed_user = (
            "<input>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<changes_json>{sec_changes.model_dump_json()}</changes_json>\n"
            "</input>"
        )
        outline = Runner.run_sync(ed_agent, ed_user).final_output  # type: ignore

    # Subsection-level research and evidence; apply changes per subsection
    ssr_agent = build_subsection_research_agent()
    evidence_list: list[dict[str, Any]] = []
    # Iterate after possible structure updates
    for sec in outline.sections:
        for sub in sec.subsections:
            ss_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{args.lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<subsection_id>{sub.id}</subsection_id>\n"
                "</input>"
            )
            ss_res = Runner.run_sync(ssr_agent, ss_user)
            ss_obj = ss_res.final_output  # type: ignore
            try:
                # Apply changes
                ed_user = (
                    "<input>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<changes_json>{ss_obj.changes.model_dump_json()}</changes_json>\n"
                    "</input>"
                )
                outline = Runner.run_sync(ed_agent, ed_user).final_output  # type: ignore
            except Exception:
                pass
            try:
                evidence_list.append(parse_json_best_effort(ss_obj.evidence.model_dump_json()))
            except Exception:
                pass

    try:
        ev_json = json.dumps(evidence_list, ensure_ascii=False)
    except Exception:
        ev_json = "[]"
    _log_append(logs, "📚 Evidence · Collected", f"```json\n{ev_json}\n```")

    # Module 3: Writing
    from llm_agents.deep_popular_science_article.module_03_writing.title_namer import build_title_namer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.article_lead_writer import build_article_lead_writer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.section_lead_writer import build_section_lead_writer_agent
    from llm_agents.deep_popular_science_article.module_03_writing.subsection_writer import build_subsection_writer_agent

    t_agent = build_title_namer_agent()
    t_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{args.lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    title: TitleProposal = Runner.run_sync(t_agent, t_user).final_output  # type: ignore
    _log_append(logs, "🧾 Title", f"```json\n{title.model_dump_json()}\n```")

    al_agent = build_article_lead_writer_agent()
    al_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{args.lang}</lang>\n"
        f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
        "</input>"
    )
    article_lead: LeadChunk = Runner.run_sync(al_agent, al_user).final_output  # type: ignore
    _log_append(logs, "✍️ Article Lead", f"```json\n{article_lead.model_dump_json()}\n```")

    sl_agent = build_section_lead_writer_agent()
    leads_by_section: dict[str, LeadChunk] = {}
    for sec in outline.sections:
        sl_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{args.lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec.id}</section_id>\n"
            "</input>"
        )
        lead: LeadChunk = Runner.run_sync(sl_agent, sl_user).final_output  # type: ignore
        leads_by_section[sec.id] = lead
        _log_append(logs, "✍️ Section Lead", f"{sec.id} → ```json\n{lead.model_dump_json()}\n```")

    ssw_agent = build_subsection_writer_agent()
    drafts_by_subsection: dict[tuple[str, str], DraftChunk] = {}
    for sec in outline.sections:
        for sub in sec.subsections:
            ssw_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{args.lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<subsection_id>{sub.id}</subsection_id>\n"
                "</input>"
            )
            d: DraftChunk = Runner.run_sync(ssw_agent, ssw_user).final_output  # type: ignore
            drafts_by_subsection[(sec.id, sub.id)] = d
            _log_append(logs, "✍️ Draft · Subsection", f"{sec.id}/{sub.id} → ```json\n{d.model_dump_json()}\n```")

    # Module 4: Refining
    from llm_agents.deep_popular_science_article.module_04_refining.section_lead_refiner import build_section_lead_refiner_agent
    from llm_agents.deep_popular_science_article.module_04_refining.subsection_refiner import build_subsection_refiner_agent

    slr_agent = build_section_lead_refiner_agent()
    for sec in outline.sections:
        lead = leads_by_section.get(sec.id)
        if not lead:
            continue
        slr_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{args.lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<section_id>{sec.id}</section_id>\n"
            f"<lead_chunk>{lead.model_dump_json()}</lead_chunk>\n"
            "</input>"
        )
        rlead: LeadChunk = Runner.run_sync(slr_agent, slr_user).final_output  # type: ignore
        leads_by_section[sec.id] = rlead
        _log_append(logs, "✨ Refine · Section Lead", f"{sec.id} → ```json\n{rlead.model_dump_json()}\n```")

    ssr_agent2 = build_subsection_refiner_agent()
    for sec in outline.sections:
        for sub in sec.subsections:
            d = drafts_by_subsection.get((sec.id, sub.id))
            if not d:
                continue
            ssr_user2 = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{args.lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<subsection_id>{sub.id}</subsection_id>\n"
                f"<draft_chunk>{d.model_dump_json()}</draft_chunk>\n"
                "</input>"
            )
            rd: DraftChunk = Runner.run_sync(ssr_agent2, ssr_user2).final_output  # type: ignore
            drafts_by_subsection[(sec.id, sub.id)] = rd
            _log_append(logs, "✨ Refine · Subsection", f"{sec.id}/{sub.id} → ```json\n{rd.model_dump_json()}\n```")

    # Assemble final Markdown
    output_dir = ensure_output_dir(args.out)
    base = f"{safe_filename_base(topic)}_article"
    article_path = next_available_filepath(output_dir, base, ".md")

    # Build TOC and content
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
    article_md = (
        f"# {title_text}\n\n"
        f"{(article_lead.markdown if isinstance(article_lead, LeadChunk) else '')}\n\n"
        f"{os.linesep.join(toc_lines)}\n\n"
        f"{os.linesep.join(body_lines)}\n"
    )

    save_markdown(article_path, title=title_text, generator=("OpenAI Agents SDK" if provider == "openai" else provider), pipeline="DeepArticle", content=article_md)

    # Save log file if requested
    if args.include_logs:
        log_dir = ensure_output_dir(args.out)
        log_path = next_available_filepath(log_dir, f"{safe_filename_base(topic)}_article_log", ".md")
        from datetime import datetime
        header = (
            f"# 🧾 Article Generation Log\n\n"
            f"- provider: {provider}\n"
            f"- lang: {args.lang}\n"
            f"- started_at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}\n"
            f"- topic: {topic}\n"
        )
        full_log = header + "".join(logs)
        save_markdown(log_path, title=f"Log: {topic}", generator="bio1c", pipeline="LogDeepArticle", content=full_log)
        print(f"📋 Лог сохранён: {log_path}")

    print(f"💾 Сохранено: {article_path}")
    print("✅ Готово.")


if __name__ == "__main__":
    main()



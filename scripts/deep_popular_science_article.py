#!/usr/bin/env python3
from __future__ import annotations

import os
import sys as _sys
import traceback as _tb
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
    DraftChunk,
    ArticleTitleLead,
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
    parser.add_argument("--style", type=str, default="article_style_1", help="Article style: article_style_1|article_style_2")
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
    # Select agents by style
    style_key = (args.style or "article_style_1").strip().lower()
    if style_key not in {"article_style_1", "article_style_2"}:
        style_key = "article_style_1"
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
    # content_of_subsections removed in 2‑module pipeline

    user_outline = f"<input>\n<topic>{topic}</topic>\n<lang>{args.lang}</lang>\n</input>"
    outline_agent = build_sections_agent()
    try:
        outline_res = Runner.run_sync(outline_agent, user_outline)
        outline: ArticleOutline = outline_res.final_output  # type: ignore
    except Exception as e:
        print(f"[CLI][OUTLINE_ERR] {type(e).__name__}: {e}", file=_sys.stderr)
        _tb.print_exc()
        raise
    _log_append(logs, "📑 Outline · Sections", f"```json\n{outline.model_dump_json()}\n```")

    # Improvement pass with outline_json
    try:
        improve_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{args.lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            "</input>"
        )
        improved_outline: ArticleOutline = Runner.run_sync(outline_agent, improve_user).final_output  # type: ignore
        if improved_outline and improved_outline.sections:
            outline = improved_outline
            _log_append(logs, "📑 Outline · Improved", f"```json\n{outline.model_dump_json()}\n```")
    except Exception as e:
        print(f"[CLI][OUTLINE_IMPROVE_ERR] {type(e).__name__}: {e}", file=_sys.stderr)

    # Expand content items per subsection
    try:
        expand_user = (
            "<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{args.lang}</lang>\n"
            f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
            f"<expand_content>true</expand_content>\n"
            "</input>"
        )
        expanded_outline: ArticleOutline = Runner.run_sync(outline_agent, expand_user).final_output  # type: ignore
        if expanded_outline and expanded_outline.sections:
            outline = expanded_outline
            _log_append(logs, "📑 Outline · Expanded Content", f"```json\n{outline.model_dump_json()}\n```")
    except Exception as e:
        print(f"[CLI][OUTLINE_EXPAND_ERR] {type(e).__name__}: {e}", file=_sys.stderr)

    # Skip legacy content-of-subsections step (removed)

    # Module 2: Writing (2‑модульная архитектура)

    if style_key == "article_style_2":
        ssw_agent = build_section_writer_agent()
    else:
        ssw_agent = build_subsection_writer_agent()
    drafts_by_subsection: dict[tuple[str, str], DraftChunk] = {}
    for sec in outline.sections:
        if style_key == "article_style_2":
            ssw_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{args.lang}</lang>\n"
                f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                f"<section_id>{sec.id}</section_id>\n"
                f"<content_items_json>{json.dumps([{'id': getattr(ci, 'id', ''), 'point': getattr(ci, 'point', '')} for ci in (getattr(sec, 'content_items', []) or [])], ensure_ascii=False)}</content_items_json>\n"
                "</input>"
            )
            try:
                from schemas.article import SectionDraftChunk as _SDC  # type: ignore
                d: _SDC = Runner.run_sync(ssw_agent, ssw_user).final_output  # type: ignore
            except Exception as e:
                print(f"[CLI][DRAFT_ERR] {sec.id}: {type(e).__name__}: {e}", file=_sys.stderr)
                _tb.print_exc()
                raise
            drafts_by_subsection[(sec.id, "__whole__")] = d  # store per-section
            _log_append(logs, "✍️ Draft · Section", f"{sec.id} → ```json\n{d.model_dump_json()}\n```")
        else:
            for sub in sec.subsections:
                ssw_user = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{args.lang}</lang>\n"
                    f"<outline_json>{outline.model_dump_json()}</outline_json>\n"
                    f"<section_id>{sec.id}</section_id>\n"
                    f"<subsection_id>{sub.id}</subsection_id>\n"
                    f"<content_items_json>{json.dumps([{'id': getattr(ci, 'id', ''), 'point': getattr(ci, 'point', '')} for ci in (getattr(sub, 'content_items', []) or [])], ensure_ascii=False)}</content_items_json>\n"
                    "</input>"
                )
                try:
                    d: DraftChunk = Runner.run_sync(ssw_agent, ssw_user).final_output  # type: ignore
                except Exception as e:
                    print(f"[CLI][DRAFT_ERR] {sec.id}/{sub.id}: {type(e).__name__}: {e}", file=_sys.stderr)
                    _tb.print_exc()
                    raise
                drafts_by_subsection[(sec.id, sub.id)] = d
                _log_append(logs, "✍️ Draft · Subsection", f"{sec.id}/{sub.id} → ```json\n{d.model_dump_json()}\n```")
    # Refine module removed in 2‑module pipeline

    # Assemble final Markdown
    output_dir = ensure_output_dir(args.out)
    base = f"{safe_filename_base(topic)}_article"
    article_path = next_available_filepath(output_dir, base, ".md")

    # Build TOC and content
    def _section_label(idx: int) -> str | None:
        lang_l = (args.lang or "auto").strip().lower()
        if lang_l.startswith("ru"):
            return f"Раздел {idx}."
        if lang_l.startswith("en"):
            return f"Section {idx}."
        # auto/другой → как раньше: без префикса/нумерации
        return None

    def _toc_title() -> str:
        lang_l = (args.lang or "auto").strip().lower()
        if lang_l.startswith("ru"):
            return "Оглавление"
        if lang_l.startswith("en"):
            return "Table of Contents"
        # auto/другой → как раньше
        return "Оглавление"

    toc_lines = [f"## {_toc_title()}"]
    for i, sec in enumerate(outline.sections, start=1):
        toc_lines.append(f"{i}. {sec.title}")
        for j, sub in enumerate(sec.subsections, start=1):
            toc_lines.append(f"  {i}.{j} {sub.title}")

    body_lines: list[str] = []
    # Per-section leads using the same Title&Lead agent (section_id mode)
    atl_agent = build_article_title_lead_writer_agent()
    for idx, sec in enumerate(outline.sections, start=1):
        _lbl = _section_label(idx)
        if body_lines:
            body_lines.append("")
        if _lbl:
            body_lines.append(f"## {_lbl} {sec.title}")
        else:
            body_lines.append(f"## {sec.title}")
        try:
            if style_key == "article_style_2":
                _sd = drafts_by_subsection.get((sec.id, "__whole__"))
                sec_body_text = (getattr(_sd, "markdown", "") or "").strip()
            else:
                sec_md_parts = []
                for sub in sec.subsections:
                    d = drafts_by_subsection.get((sec.id, sub.id))
                    sub_title = d.title if d and d.title else sub.title
                    sub_md = (d.markdown if d else "").strip()
                    sec_md_parts.append(f"### {sub_title}\n\n{sub_md}")
                sec_body_text = "\n\n".join(sec_md_parts)
            sec_user = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{args.lang}</lang>\n"
                f"<article_markdown>{sec.title}\n\n{sec_body_text}</article_markdown>\n"
                f"<section_id>{sec.id}</section_id>\n"
                + (f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n" if style_key == "article_style_2" else "")
                + "</input>"
            )
            sec_lead_obj = Runner.run_sync(atl_agent, sec_user).final_output  # type: ignore
            sec_lead = (getattr(sec_lead_obj, "lead_markdown", "") or "").strip()
            if sec_lead:
                body_lines.append("")
                body_lines.append(sec_lead)
        except Exception as e:
            print(f"[CLI][SECTION_LEAD_ERR] {sec.id}: {type(e).__name__}: {e}", file=_sys.stderr)
        # Append full section body for style 2
        if style_key == "article_style_2" and sec_body_text:
            body_lines.append("")
            body_lines.append(sec_body_text)
        if style_key != "article_style_2":
            for sub in sec.subsections:
                d = drafts_by_subsection.get((sec.id, sub.id))
                sub_title = d.title if d and d.title else sub.title
                sub_md = (d.markdown if d else "").strip()
                body_lines.append("")
                body_lines.append(f"### {sub_title}")
                if sub_md:
                    body_lines.append("")
                    body_lines.append(sub_md)

    toc_text = os.linesep.join(toc_lines)
    body_text = os.linesep.join(body_lines)

    # Title & Lead based on full article content
    atl_agent = build_article_title_lead_writer_agent()
    atl_user = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{args.lang}</lang>\n"
        f"<article_markdown>{toc_text}\n\n{body_text}</article_markdown>\n"
        + (f"<main_idea>{(outline.main_idea or '').strip()}</main_idea>\n" if style_key == "article_style_2" else "")
        + "</input>"
    )
    try:
        atl: ArticleTitleLead = Runner.run_sync(atl_agent, atl_user).final_output  # type: ignore
    except Exception as e:
        print(f"[CLI][TITLE_LEAD_ERR] {type(e).__name__}: {e}", file=_sys.stderr)
        _tb.print_exc()
        raise
    _log_append(logs, "🧾 Title & Lead", f"```json\n{atl.model_dump_json()}\n```")

    title_text = atl.title or (outline.title or topic)
    lead_text = (atl.lead_markdown or "").strip()
    article_md = (
        f"# {title_text}\n\n"
        f"{lead_text}\n\n"
        f"{toc_text}\n\n"
        f"{body_text}\n"
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



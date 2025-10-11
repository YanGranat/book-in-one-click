#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from utils.slug import safe_filename_base
from utils.lang import detect_lang_from_text
from services.providers.runner import ProviderRunner
from llm_agents.memom.extractor import build_meme_extractor_agent
from utils.models import get_model


def _load_system_prompt() -> str:
    p = Path("prompts") / "memom" / "meme_extractor.md"
    if not p.exists():
        raise FileNotFoundError("memom system prompt not found: prompts/memom/meme_extractor.md")
    return p.read_text(encoding="utf-8")


def extract_memes(
    *,
    text: str,
    source_name: str,
    lang: str = "auto",
    provider: str = "openai",
    job_meta: Optional[dict] = None,
    disable_db_record: bool = False,
    return_content: bool = False,
) -> Path | str:
    """
    Run meme extraction with ProviderRunner and persist result + log.

    Returns Path to saved .md (default) or content if return_content=True.
    """
    started_at = datetime.utcnow()
    started_perf = time.perf_counter()

    # Resolve language
    eff_lang = (lang or "auto").strip().lower()
    if eff_lang == "auto":
        try:
            eff_lang = detect_lang_from_text(text or "") or "en"
        except Exception:
            eff_lang = "en"

    # Build system prompt with language instruction based on user settings
    system_prompt = _load_system_prompt()
    
    # Add explicit language instruction based on user language setting
    if eff_lang == "ru":
        system_prompt += "\n\n<output_language>Пиши результат экстракции (названия мемов, их суть, пояснения, обзор) СТРОГО на русском языке, независимо от языка исходного контента.</output_language>"
    elif eff_lang == "en":
        system_prompt += "\n\n<output_language>Write the extraction result (meme names, essence, explanations, overview) STRICTLY in English, regardless of the source content language.</output_language>"
    else:
        # For auto mode - match the source language
        system_prompt += "\n\n<output_language>Write the extraction result in the same language as the source content.</output_language>"

    # Run model
    pnorm = (provider or "openai").strip().lower()
    if pnorm in {"", "auto"}:
        pnorm = "openai"
    used_model = ""
    final_content = ""
    
    if pnorm == "openai":
        # Use OpenAI Agents SDK for multi-agent framework compatibility
        try:
            used_model = get_model("openai", "heavy")
            from agents import Runner  # type: ignore
            agent = build_meme_extractor_agent(model=used_model)
            agent.instructions = system_prompt  # Clean prompt only
            final_content = getattr(Runner.run_sync(agent, text), "final_output", "")
        except Exception:
            # Fallback to provider runner
            runner = ProviderRunner(pnorm)
            used_model = get_model("openai", "heavy")
            final_content = runner.run_text(system_prompt, text, speed="heavy") or ""
    else:
        # Non-OpenAI providers use ProviderRunner
        if pnorm in {"gemini", "google"}:
            used_model = get_model("gemini", "heavy")
        else:
            used_model = get_model("claude", "heavy")
        runner = ProviderRunner(pnorm)
        final_content = runner.run_text(system_prompt, text, speed="heavy") or ""
    # Guarantee non-empty content stored in DB/UI even if model returned empty
    if not final_content.strip():
        final_content = "(no memes extracted)"

    # Persist result file
    output_dir = ensure_output_dir("memes")
    base = f"{safe_filename_base(Path(source_name).stem or 'memes')}_memes"
    result_path = next_available_filepath(output_dir, base, ".md")
    save_markdown(
        result_path,
        title=Path(source_name).name,
        generator=("OpenAI Agents SDK" if (provider or "openai").strip().lower() == "openai" else provider),
        pipeline="MemeExtract",
        content=final_content,
    )

    # Build and persist log (with metadata)
    log_dir = ensure_output_dir("memes")
    log_path = log_dir / f"{safe_filename_base(Path(source_name).stem or 'memes')}_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    header = (
        f"# 🧾 Meme Extraction Log\n\n"
        f"- provider: {pnorm}\n"
        f"- lang: {eff_lang}\n"
        f"- model: {used_model or '?'}\n"
        f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- duration: {duration_s:.1f}s\n"
        f"- source: {Path(source_name).name}\n"
    )
    # For transparency, include first N chars of input (avoid dumping entire large text)
    preview = (text or "")[:4000]
    full_log = (
        header
        + "\n## Input (truncated)\n\n"
        + preview
        + "\n\n## Output (model)\n\n"
        + final_content
        + "\n"
    )
    save_markdown(log_path, title=f"Log: {Path(source_name).name}", generator="bio1c", pipeline="Log", content=full_log)

    # Record in DB (sync, like post pipeline)
    try:
        if os.getenv("DB_URL", "").strip() and not disable_db_record:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            from server.db import JobLog, ResultDoc

            db_url = os.getenv("DB_URL", "").strip()
            sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
            parts = urlsplit(sync_url)
            qs = parse_qs(parts.query or "")
            base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
            cargs = {}
            if "sslmode" not in {k.lower() for k in qs.keys()}:
                cargs["sslmode"] = "require"
            sync_engine = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True, pool_size=15, max_overflow=10, pool_timeout=30)
            SyncSession = sessionmaker(sync_engine)
            with SyncSession() as s:
                # Paths relative to CWD for portability
                try:
                    rel_log = str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path)
                except ValueError:
                    rel_log = str(log_path)
                try:
                    rel_doc = str(result_path.relative_to(Path.cwd())) if result_path.is_absolute() else str(result_path)
                except ValueError:
                    rel_doc = str(result_path)

                # Persist JobLog (content stored in DB)
                jl = JobLog(job_id=int((job_meta or {}).get("job_id", 0) or 0), kind="md", path=rel_log, content=full_log)
                try:
                    s.add(jl)
                    s.flush()
                    s.commit()
                except Exception:
                    s.rollback()

                # Ensure Job link exists (fallback like post pipeline)
                result_job_id = int((job_meta or {}).get("job_id", 0) or 0)
                if result_job_id <= 0:
                    try:
                        from server.db import User, Job as _Job
                        db_user_id = int((job_meta or {}).get("db_user_id", 0) or 0)
                        # If User.id unknown — try resolve by telegram_id
                        if db_user_id <= 0:
                            tg_uid = int((job_meta or {}).get("user_id", 0) or 0)
                            if tg_uid > 0:
                                urow = s.query(User).filter(User.telegram_id == tg_uid).first()
                                if urow is None:
                                    from sqlalchemy.exc import IntegrityError as _IntegrityError
                                    try:
                                        urow = User(telegram_id=tg_uid, credits=0)
                                        s.add(urow)
                                        s.flush()
                                        s.commit()
                                    except _IntegrityError:
                                        s.rollback()
                                        urow = s.query(User).filter(User.telegram_id == tg_uid).first()
                                if urow is not None:
                                    db_user_id = int(getattr(urow, "id", 0) or 0)
                        if db_user_id > 0:
                            import json as _json
                            params = {"source": Path(source_name).name, "lang": eff_lang, "provider": (provider or "openai").strip().lower()}
                            jrow = _Job(user_id=db_user_id, type="meme_extract", status="done", params_json=_json.dumps(params, ensure_ascii=False), cost=0, file_path=str(result_path))
                            s.add(jrow)
                            s.flush()
                            s.commit()
                            result_job_id = int(getattr(jrow, "id", 0) or 0)
                            # Back-link previously saved JobLog to this Job for cascade deletions
                            try:
                                from sqlalchemy import update as _upd
                                if int(getattr(jl, "id", 0) or 0) > 0 and result_job_id > 0:
                                    s.execute(_upd(JobLog).where(JobLog.id == int(jl.id)).values(job_id=result_job_id))
                                    s.commit()
                            except Exception:
                                s.rollback()
                    except Exception:
                        s.rollback()

                # Persist ResultDoc(kind=meme_extract)
                rd = ResultDoc(
                    job_id=result_job_id,
                    kind="meme_extract",
                    path=rel_doc,
                    topic=Path(source_name).name,
                    provider=pnorm,
                    lang=eff_lang,
                    content=final_content,
                    hidden=0,
                )
                try:
                    s.add(rd)
                    s.flush()
                    s.commit()
                except Exception:
                    s.rollback()
            try:
                sync_engine.dispose()
            except Exception:
                pass
    except Exception:
        # Non-fatal if DB not available; files are saved anyway
        pass

    if return_content:
        return final_content
    return result_path



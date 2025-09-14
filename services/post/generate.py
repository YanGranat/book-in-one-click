#!/usr/bin/env python3
"""
Reusable generation function for popular science posts.
Extracted from scripts/Popular_science_post.py and adapted for server/bot usage.

Key guarantees:
- No interactive input; accepts all parameters explicitly
- Can be safely called from a background thread
- Returns a Path to the saved Markdown file
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Type
import os
import asyncio
import time

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.slug import safe_filename_base
from utils.web import build_search_context
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from pipelines.post.pipeline import build_instructions as build_post_instructions


def _try_import_sdk():
    try:
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        raise RuntimeError(
            "Cannot import Agent/Runner from 'agents'. Ensure OpenAI Agents SDK is installed, "
            "and no local package named 'agents' shadows it."
        ) from e


def generate_post(
    topic: str,
    *,
    lang: str = "auto",
    provider: str = "openai",  # openai|gemini|claude
    factcheck: bool = True,
    factcheck_max_items: int = 0,
    research_iterations: int = 2,
    research_concurrency: int = 6,
    output_subdir: str = "post",
    on_progress: Optional[Callable[[str], None]] = None,
    job_meta: Optional[dict] = None,
    return_log_path: bool = False,
) -> Path:
    """
    Generate a popular science post and save it to output/<output_subdir>/.

    Returns the Path to the saved .md file.
    """
    # Ensure project modules are importable and env is loaded
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

    # Ensure this thread has an event loop for libs that expect it
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

    _emit("start:post")
    from datetime import datetime
    started_at = datetime.utcnow()
    started_perf = time.perf_counter()
    log_lines = []
    def log(section: str, body: str):
        log_lines.append(f"## {section}\n\n{body}\n")
    log("🧭 Config", f"provider={_prov}\nlang={lang}")

    instructions = build_post_instructions(topic, lang)

    def _run_openai_with(system: str, user_message_local: str, model: Optional[str] = None) -> str:
        agent = Agent(
            name="Generic Agent",
            instructions=system,
            model=(model or os.getenv("OPENAI_MODEL", "gpt-5")),
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    def _run_openai() -> str:
        agent = Agent(
            name="Popular Science Post Writer",
            instructions=instructions,
            model="gpt-5",
        )
        user_message_local = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{(lang or 'auto').strip()}</lang>\n"
            f"</input>"
        )
        res_local = Runner.run_sync(agent, user_message_local)
        return getattr(res_local, "final_output", "")

    def _run_gemini_with(system: str, user_message_local: str, model_name: Optional[str] = None) -> str:
        import google.generativeai as genai  # type: ignore
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        mname = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
        model = genai.GenerativeModel(model_name=mname, system_instruction=system)
        resp = model.generate_content(user_message_local)
        return (getattr(resp, "text", None) or "").strip()

    def _run_claude_with(system: str, user_message_local: str, model_name: Optional[str] = None) -> str:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        mname = model_name or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-0")
        msg = client.messages.create(
            model=mname,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_message_local}],
        )
        parts = []
        for blk in getattr(msg, "content", []) or []:
            txt = getattr(blk, "text", None)
            if txt:
                parts.append(txt)
        return ("\n\n".join(parts)).strip()

    def run_with_provider(system: str, user_inp: str, speed: str = "heavy") -> str:
        if _prov == "openai":
            model = os.getenv("OPENAI_FAST_MODEL", "gpt-5-mini") if speed == "fast" else os.getenv("OPENAI_MODEL", "gpt-5")
            return _run_openai_with(system, user_inp, model)
        if _prov in {"gemini", "google"}:
            mname = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash") if speed == "fast" else os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
            return _run_gemini_with(system, user_inp, mname)
        # Claude: same model for fast/heavy unless overridden
        cname = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-0")
        return _run_claude_with(system, user_inp, cname)

    def run_json_with_provider(system: str, user_inp: str, cls: Type, speed: str = "fast"):
        import json
        txt = run_with_provider(system, user_inp, speed)
        # strip code fences if any
        if txt.strip().startswith("```") and txt.strip().endswith("```"):
            txt = "\n".join([line for line in txt.strip().splitlines()[1:-1]])
        try:
            data = json.loads(txt)
            return cls.model_validate(data)
        except Exception:
            # try to find first json block
            import re
            m = re.search(r"\{[\s\S]*\}\s*$", txt)
            if m:
                return cls.model_validate(json.loads(m.group(0)))
            raise RuntimeError(f"Failed to parse JSON for {cls.__name__}")

    user_message_local_writer = (
        f"<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{(lang or 'auto').strip()}</lang>\n"
        f"</input>"
    )
    # Log writer input for transparency
    log("⬇️ Writer · Input", f"{user_message_local_writer}")
    content = run_with_provider(instructions, user_message_local_writer, speed="heavy")
    log("✍️ Writer · Output", content[:2000])
    if not content:
        raise RuntimeError("Empty result from writer agent")

    report = None
    if factcheck:
        if _prov == "openai":
            # Preserve original OpenAI Agents SDK flow (with WebSearchTool)
            from llm_agents.post.module_02_review.identify_points import build_identify_points_agent
            from llm_agents.post.module_02_review.iterative_research import build_iterative_research_agent
            from llm_agents.post.module_02_review.recommendation import build_recommendation_agent
            from llm_agents.post.module_02_review.sufficiency import build_sufficiency_agent
            from llm_agents.post.module_02_review.query_synthesizer import build_query_synthesizer_agent
            from utils.config import load_config

            _emit("factcheck:init")
            identify_agent = build_identify_points_agent()
            identify_result = Runner.run_sync(identify_agent, f"<post>\n{content}\n</post>")
            plan = identify_result.final_output  # type: ignore
            points = plan.points or []
            if factcheck_max_items and factcheck_max_items > 0:
                points = points[: factcheck_max_items]
            try:
                log("🔎 Fact-check · Plan (OpenAI)", f"points={len(points)}")
            except Exception:
                pass

            cfg = load_config(__file__)
            pref = (cfg.get("research", {}) or {}).get("preferred_domains", [])
            research_agent = build_iterative_research_agent()
            suff_agent = build_sufficiency_agent()
            rec_agent = build_recommendation_agent()
            synth_agent = build_query_synthesizer_agent()

            async def _run_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
                for i in range(attempts):
                    try:
                        return await Runner.run(agent, inp)
                    except Exception:
                        if i == attempts - 1:
                            raise
                        await asyncio.sleep(base_delay * (2 ** i))

            async def process_point_async(p):
                # Query synthesis
                cfg_pref = ",".join(pref)
                qp_res = await _run_with_retries(
                    synth_agent,
                    f"<input>\n<point>{p.model_dump_json()}</point>\n<preferred_domains>{cfg_pref}</preferred_domains>\n</input>",
                )  # type: ignore
                _ = qp_res.final_output

                notes = []
                for step in range(1, max(1, int(research_iterations)) + 1):
                    rr_input = (
                        "<input>\n"
                        f"<point>{p.model_dump_json()}</point>\n"
                        f"<step>{step}</step>\n"
                        "</input>"
                    )
                    note_res = await _run_with_retries(research_agent, rr_input)  # type: ignore
                    note = note_res.final_output
                    notes.append(note)

                    suff_input = (
                        "<input>\n"
                        f"<point>{p.model_dump_json()}</point>\n"
                        f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                        "</input>"
                    )
                    decision_res = await _run_with_retries(suff_agent, suff_input)  # type: ignore
                    decision = decision_res.final_output
                    if decision.done:
                        break

                class _TmpReport:
                    def __init__(self, point_id, notes):
                        self.point_id = point_id
                        self.notes = notes
                        self.synthesis = ""

                    def model_dump_json(self):
                        import json
                        return json.dumps(
                            {
                                "point_id": self.point_id,
                                "notes": [n.model_dump() for n in self.notes],
                                "synthesis": self.synthesis,
                            },
                            ensure_ascii=False,
                        )

                rr = _TmpReport(p.id, notes)
                rec_res = await _run_with_retries(
                    rec_agent,
                    f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n</input>",
                )  # type: ignore
                rec = rec_res.final_output
                return p, rec, notes

            async def process_all(points_list):
                sem = asyncio.Semaphore(max(1, int(research_concurrency)))

                async def worker(p):
                    async with sem:
                        return await process_point_async(p)

                tasks = [asyncio.create_task(worker(p)) for p in points_list]
                results = []
                for t in asyncio.as_completed(tasks):
                    results.append(await t)
                return results

            results = asyncio.run(process_all(points)) if points else []

            class _SimpleItem:
                def __init__(self, claim_text: str, verdict: str, reason: str, supporting_facts: str):
                    self.claim_text = claim_text
                    self.verdict = verdict
                    self.reason = reason
                    self.supporting_facts = supporting_facts

            simple_items = []
            for (p, r, notes) in results:
                if getattr(r, "action", "keep") == "keep":
                    continue
                if r.action == "clarify":
                    verdict = "uncertain"
                elif r.action == "rewrite" or r.action == "remove":
                    verdict = "fail"
                else:
                    verdict = "fail"
                reason = getattr(r, "explanation", "") or ""
                supporting_facts = " \n".join(getattr(n, "findings", "") for n in (notes or []))
                simple_items.append(_SimpleItem(p.text, verdict, reason, supporting_facts))

            class _SimpleReport:
                def __init__(self, items):
                    self.items = items

                def model_dump_json(self):
                    import json
                    return json.dumps(
                        {
                            "summary": "Issues only for rewrite (exclude confirmed)",
                            "items": [
                                {
                                    "claim_text": i.claim_text,
                                    "verdict": i.verdict,
                                    "reason": i.reason,
                                    "supporting_facts": i.supporting_facts,
                                }
                                for i in self.items
                            ],
                        },
                        ensure_ascii=False,
                    )

            report = _SimpleReport(simple_items) if simple_items else None
            if report is not None:
                log("factcheck_summary", report.model_dump_json())
        else:
            from utils.config import load_config
            from pathlib import Path
            from schemas.research import ResearchPlan, QueryPack, ResearchIterationNote, SufficiencyDecision, Recommendation

            _emit("factcheck:init")
            base = Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_02_review"
            p_ident = (base / "identify_risky_points.md").read_text(encoding="utf-8")
            plan = run_json_with_provider(p_ident, f"<post>\n{content}\n</post>", ResearchPlan, speed="fast")
            points = plan.points or []
            if factcheck_max_items and factcheck_max_items > 0:
                points = points[: factcheck_max_items]
            try:
                log("🔎 Fact-check · Plan", f"```json\n{plan.model_dump_json()}\n```")
            except Exception:
                log("🔎 Fact-check · Plan", f"points={len(points)}")

            cfg = load_config(__file__)
            pref = (cfg.get("research", {}) or {}).get("preferred_domains", [])
            p_iter = (base / "iterative_research.md").read_text(encoding="utf-8")
            p_suff = (base / "sufficiency.md").read_text(encoding="utf-8")
            p_rec = (base / "recommendation.md").read_text(encoding="utf-8")
            p_qs = (base / "query_synthesizer.md").read_text(encoding="utf-8")

        async def _run_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
            for i in range(attempts):
                try:
                    return await Runner.run(agent, inp)
                except Exception:
                    if i == attempts - 1:
                        raise
                    await asyncio.sleep(base_delay * (2 ** i))

        async def process_point_async(p):
            # Query synthesis
            cfg_pref = ",".join(pref)
            qp = run_json_with_provider(
                p_qs,
                f"<input>\n<point>{p.model_dump_json()}</point>\n<preferred_domains>{cfg_pref}</preferred_domains>\n</input>",
                QueryPack,
                speed="fast",
            )
            # Web context build (provider-agnostic)
            queries = getattr(qp, "queries", []) or []
            web_ctx = build_search_context(queries, per_query=2, max_chars=2000)
            if queries:
                log("🌐 Web · Queries", "\n".join([f"- {q}" for q in queries]))
            # Extract sources (best-effort)
            try:
                import re as _re
                urls = _re.findall(r"url=\"([^\"]+)\"", web_ctx)
                if urls:
                    log("🌐 Web · Sources", "\n".join([f"- {u}" for u in urls[:20]]))
            except Exception:
                pass

            notes = []
            for step in range(1, max(1, int(research_iterations)) + 1):
                rr_input = (
                    "<input>\n"
                    f"<point>{p.model_dump_json()}</point>\n"
                    f"<step>{step}</step>\n"
                    f"<web_context>\n{web_ctx}\n</web_context>\n"
                    "</input>"
                )
                note = run_json_with_provider(p_iter, rr_input, ResearchIterationNote, speed="fast")
                notes.append(note)

                suff_input = (
                    "<input>\n"
                    f"<point>{p.model_dump_json()}</point>\n"
                    f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                    "</input>"
                )
                decision = run_json_with_provider(p_suff, suff_input, SufficiencyDecision, speed="fast")
                if decision.done:
                    break

            class _TmpReport:
                def __init__(self, point_id, notes):
                    self.point_id = point_id
                    self.notes = notes
                    self.synthesis = ""

                def model_dump_json(self):
                    import json
                    return json.dumps(
                        {
                            "point_id": self.point_id,
                            "notes": [n.model_dump() for n in self.notes],
                            "synthesis": self.synthesis,
                        },
                        ensure_ascii=False,
                    )

            rr = _TmpReport(p.id, notes)
            rec = run_json_with_provider(
                p_rec,
                f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n</input>",
                Recommendation,
                speed="fast",
            )
            return p, rec, notes

        async def process_all(points_list):
            sem = asyncio.Semaphore(max(1, int(research_concurrency)))

            async def worker(p):
                async with sem:
                    return await process_point_async(p)

            tasks = [asyncio.create_task(worker(p)) for p in points_list]
            results = []
            for t in asyncio.as_completed(tasks):
                results.append(await t)
            return results

        # For non-openai providers concurrency brings little benefit; run sequentially
        if _prov == "openai":
            results = asyncio.run(process_all(points)) if points else []
        else:
            seq_results = []
            for p in points or []:
                seq_results.append(asyncio.run(process_point_async(p)))
            results = seq_results

        class _SimpleItem:
            def __init__(self, claim_text: str, verdict: str, reason: str, supporting_facts: str):
                self.claim_text = claim_text
                self.verdict = verdict
                self.reason = reason
                self.supporting_facts = supporting_facts

        simple_items = []
        for (p, r, notes) in results:
            if getattr(r, "action", "keep") == "keep":
                continue  # confirmed parts не передаём в переписывание
            if r.action == "clarify":
                verdict = "uncertain"
            elif r.action == "rewrite" or r.action == "remove":
                verdict = "fail"
            else:
                verdict = "fail"
            reason = getattr(r, "explanation", "") or ""
            supporting_facts = " \n".join(getattr(n, "findings", "") for n in (notes or []))
            simple_items.append(_SimpleItem(p.text, verdict, reason, supporting_facts))

        class _SimpleReport:
            def __init__(self, items):
                self.items = items

            def model_dump_json(self):
                import json
                return json.dumps(
                    {
                        "summary": "Issues only for rewrite (exclude confirmed)",
                        "items": [
                            {
                                "claim_text": i.claim_text,
                                "verdict": i.verdict,
                                "reason": i.reason,
                                "supporting_facts": i.supporting_facts,
                            }
                            for i in self.items
                        ],
                    },
                    ensure_ascii=False,
                )

        report = _SimpleReport(simple_items) if simple_items else None
        if report is not None:
            log("factcheck_summary", report.model_dump_json())

    # Rewrite and refine
    final_content = content
    if report is not None:
        needs_rewrite = any(i.verdict != "pass" for i in report.items)
        if needs_rewrite:
            _emit("rewrite:init")
            from pathlib import Path
            p_rewrite = (Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_03_rewriting" / "rewrite.md").read_text(encoding="utf-8")
            rw_input = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                f"<post>\n{content}\n</post>\n"
                f"<critique_json>\n{report.model_dump_json()}\n</critique_json>\n"
                "</input>"
            )
            # Log rewrite input and output
            log("⬇️ Rewrite · Input", f"{rw_input}")
            final_content = run_with_provider(p_rewrite, rw_input, speed="heavy") or content
            log("🛠️ Rewrite · Output", final_content[:4000])

    from pathlib import Path
    p_refine = (Path(__file__).resolve().parents[2] / "prompts" / "post" / "module_03_rewriting" / "refine.md").read_text(encoding="utf-8")
    refine_input = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<post>\n{final_content}\n</post>\n"
        "</input>"
    )
    # Log refine input and output
    log("⬇️ Refine · Input", f"{refine_input}")
    final_content = run_with_provider(p_refine, refine_input, speed="heavy") or final_content
    log("✨ Refine · Output", final_content[:4000])

    # Save final
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_post"
    filepath = next_available_filepath(output_dir, base, ".md")
    save_markdown(
        filepath,
        title=topic,
        generator=("OpenAI Agents SDK" if _prov == "openai" else _prov),
        pipeline="PopularSciencePost",
        content=final_content,
    )
    # Save log sidecar .md and register in DB if available
    log_dir = ensure_output_dir(output_subdir)
    log_path = log_dir / f"{safe_filename_base(topic)}_log_{started_at.strftime('%Y%m%d_%H%M%S')}.md"
    finished_at = datetime.utcnow()
    duration_s = max(0.0, time.perf_counter() - started_perf)
    header = (
        f"# 🧾 Generation Log\n\n"
        f"- provider: {_prov}\n"
        f"- lang: {lang}\n"
        f"- model_heavy: {os.getenv('OPENAI_MODEL' if _prov=='openai' else ('GEMINI_MODEL' if _prov in {'gemini','google'} else 'ANTHROPIC_MODEL'))}\n"
        f"- model_fast: {os.getenv('OPENAI_FAST_MODEL' if _prov=='openai' else ('GEMINI_FAST_MODEL' if _prov in {'gemini','google'} else 'ANTHROPIC_MODEL'))}\n"
        f"- started_at: {started_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- finished_at: {finished_at.strftime('%Y-%m-%d %H:%M')}\n"
        f"- duration: {duration_s:.1f}s\n"
        f"- topic: {topic}\n"
        f"- factcheck: {bool(factcheck)}\n"
    )
    save_markdown(log_path, title=f"Log: {topic}", generator="bio1c", pipeline="Log", content=header + "\n".join(log_lines))
    # Record log path in DB if available
    try:
        from server.db import SessionLocal, JobLog
        if SessionLocal is not None:
            # Use sync approach to avoid event loop conflicts in thread executor
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            # Create sync connection from same DB_URL
            db_url = os.getenv("DB_URL", "")
            if db_url:
                # Convert async URL to sync with psycopg2 driver
                sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
                sync_engine = create_engine(sync_url)
                SyncSession = sessionmaker(sync_engine)
                
                with SyncSession() as s:
                    # Import sync model
                    from server.db import JobLog
                    # Store relative path for portability
                    rel_path = str(log_path.relative_to(Path.cwd())) if log_path.is_absolute() else str(log_path)
                    jl = JobLog(job_id=int((job_meta or {}).get("job_id", 0)), kind="md", path=rel_path)
                    s.add(jl)
                    s.commit()
                    print(f"[INFO] Log recorded in DB: id={jl.id}, path={log_path}")
            else:
                print(f"[INFO] No DB_URL configured, log saved to filesystem only: {log_path}")
        else:
            print(f"[INFO] Log saved to filesystem only: {log_path}")
    except Exception as e:
        print(f"[ERROR] Failed to record log in DB: {e}")
        print(f"[INFO] Log available on filesystem: {log_path}")
        # Continue execution - log file is still created even if DB fails
    if return_log_path:
        return log_path
    return filepath



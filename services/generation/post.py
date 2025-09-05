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
from typing import Callable, Optional
import os
import asyncio

from utils.env import ensure_project_root_on_syspath as _ensure_root, load_env_from_root
from utils.slug import safe_filename_base
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
    factcheck: bool = True,
    factcheck_max_items: int = 0,
    research_iterations: int = 2,
    research_concurrency: int = 3,
    output_subdir: str = "popular_science_post",
    on_progress: Optional[Callable[[str], None]] = None,
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

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    # Ensure this thread has an event loop for libs that expect it
    loop_created = False
    try:
        try:
            asyncio.get_event_loop()
        except Exception:
            asyncio.set_event_loop(asyncio.new_event_loop())
            loop_created = True

        Agent, Runner = _try_import_sdk()

    def _emit(stage: str) -> None:
        if on_progress:
            try:
                on_progress(stage)
            except Exception:
                pass

    _emit("start:post")

    agent = Agent(
        name="Popular Science Post Writer",
        instructions=build_post_instructions(topic, lang),
        model="gpt-5",
    )

    user_message = (
        f"<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{(lang or 'auto').strip()}</lang>\n"
        f"</input>"
    )
    res = Runner.run_sync(agent, user_message)
    content = getattr(res, "final_output", "")
    if not content:
        raise RuntimeError("Empty result from writer agent")

    report = None
    if factcheck:
        from llm_agents.review.identify_points import build_identify_points_agent
        from llm_agents.review.iterative_research import build_iterative_research_agent
        from llm_agents.review.recommendation import build_recommendation_agent
        from llm_agents.review.sufficiency import build_sufficiency_agent
        from llm_agents.review.query_synthesizer import build_query_synthesizer_agent
        from utils.config import load_config

        _emit("factcheck:init")
        identify_agent = build_identify_points_agent()
        identify_result = Runner.run_sync(identify_agent, f"<post>\n{content}\n</post>")
        plan = identify_result.final_output  # type: ignore
        points = plan.points or []
        if factcheck_max_items and factcheck_max_items > 0:
            points = points[: factcheck_max_items]

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
            return p, rec

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

        per_point_recs = [r for (_, r) in results]

        class _SimpleItem:
            def __init__(self, claim_text: str, verdict: str, proposed_fix: str | None):
                self.claim_text = claim_text
                self.verdict = verdict
                self.proposed_fix = proposed_fix
                self.evidence = []

        simple_items = []
        for p, r in zip(points, per_point_recs):
            if r.action == "keep":
                verdict = "pass"
                fix = None
            elif r.action == "clarify":
                verdict = "uncertain"
                fix = r.suggestion or ""
            elif r.action == "rewrite":
                verdict = "fail"
                fix = r.suggestion or ""
            else:
                verdict = "fail"
                fix = "Удалить данный фрагмент."
            simple_items.append(_SimpleItem(p.text, verdict, fix))

        class _SimpleReport:
            def __init__(self, items):
                self.items = items

            def model_dump_json(self):
                import json
                return json.dumps(
                    {
                        "summary": "Per-point recommendations",
                        "items": [
                            {
                                "claim_text": i.claim_text,
                                "verdict": i.verdict,
                                "reason": "",
                                "proposed_fix": i.proposed_fix,
                                "evidence": [],
                            }
                            for i in self.items
                        ],
                    },
                    ensure_ascii=False,
                )

        report = _SimpleReport(simple_items) if simple_items else None

    # Rewrite and refine
    final_content = content
    if report is not None:
        needs_rewrite = any(i.verdict != "pass" for i in report.items)
        if needs_rewrite:
            _emit("rewrite:init")
            from llm_agents.writing.rewrite import build_rewrite_agent

            rw_agent = build_rewrite_agent()
            rw_input = (
                "<input>\n"
                f"<topic>{topic}</topic>\n"
                f"<lang>{lang}</lang>\n"
                "<goal>Перепиши пост с учетом критики, сохрани стиль и формат из системного промпта поста.</goal>\n"
                f"<post>\n{content}\n</post>\n"
                f"<critique_json>\n{report.model_dump_json()}\n</critique_json>\n"
                "</input>"
            )

            async def _do_rewrite(inp: str) -> str:
                res = await Runner.run(rw_agent, inp)
                return getattr(res, "final_output", "") or content

            final_content = asyncio.run(_do_rewrite(rw_input))

    from llm_agents.writing.refine import build_refine_agent

    refine_agent = build_refine_agent()
    refine_input = (
        "<input>\n"
        f"<topic>{topic}</topic>\n"
        f"<lang>{lang}</lang>\n"
        f"<post>\n{final_content}\n</post>\n"
        "</input>"
    )

    async def _do_refine(inp: str) -> str:
        res = await Runner.run(refine_agent, inp)
        return getattr(res, "final_output", "") or final_content

    final_content = asyncio.run(_do_refine(refine_input))

    # Save final
    output_dir = ensure_output_dir(output_subdir)
    base = f"{safe_filename_base(topic)}_post"
    filepath = next_available_filepath(output_dir, base, ".md")
    save_markdown(
        filepath,
        title=topic,
        generator="OpenAI Agents SDK",
        pipeline="PopularSciencePost",
        content=final_content,
    )
    return filepath
    finally:
        if loop_created:
            try:
                loop = asyncio.get_event_loop()
                loop.close()
            except Exception:
                pass
            try:
                asyncio.set_event_loop(None)
            except Exception:
                pass



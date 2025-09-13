#!/usr/bin/env python3
"""
Interactive generator for a popular science post.
Saves results to output/post/.
"""
import os
import sys
from pathlib import Path
import argparse
import asyncio

# Ensure project root is on sys.path before importing project modules
_project_root = Path(__file__).resolve().parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from utils.env import ensure_project_root_on_syspath as ensure_root, load_env_from_root
from utils.slug import safe_filename_base
from utils.io import ensure_output_dir, save_markdown, next_available_filepath
from orchestrator import progress
from llm_agents.post.module_02_review.identify_points import build_identify_points_agent
from llm_agents.post.module_02_review.iterative_research import build_iterative_research_agent
from llm_agents.post.module_02_review.recommendation import build_recommendation_agent
from llm_agents.post.module_02_review.sufficiency import build_sufficiency_agent
from llm_agents.post.module_02_review.query_synthesizer import build_query_synthesizer_agent
from llm_agents.post.module_03_rewriting.rewrite import build_rewrite_agent
from llm_agents.post.module_03_rewriting.refine import build_refine_agent
from pipelines.post.pipeline import build_instructions as build_post_instructions


def load_env() -> None:
    load_env_from_root(__file__)


def ensure_project_root_on_syspath() -> None:
    ensure_root(__file__)


def try_import_sdk():
    try:
        # Primary import path used by OpenAI Agents SDK
        from agents import Agent, Runner  # type: ignore
        return Agent, Runner
    except ImportError as e:
        print("❌ ImportError: cannot import Agent/Runner from 'agents'.")
        print("➡️ Likely cause: a local folder named 'agents' shadows the SDK module.")
        print("➡️ Fix: rename local folder to 'llm_agents' (already recommended).")
        print(f"Details: {e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a popular science post")
    parser.add_argument("--topic", type=str, default="", help="Topic to generate about")
    parser.add_argument("--lang", type=str, default="auto", help="Language: auto|ru|en")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider: openai|gemini|claude")
    parser.add_argument("--out", type=str, default="post", help="Output subdirectory")
    parser.add_argument("--no-factcheck", action="store_true", help="Disable fact-check step")
    parser.add_argument("--factcheck-max-items", type=int, default=0, help="Limit number of points to research (0=all)")
    parser.add_argument("--research-iterations", type=int, default=2, help="Max research iterations per point")
    parser.add_argument("--research-concurrency", type=int, default=3, help="Parallel research workers per post")
    args = parser.parse_args()
    ensure_project_root_on_syspath()
    load_env()

    try:
        Agent, Runner = try_import_sdk()
    except ImportError:
        return

    

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле в корне проекта")
        return

    topic = args.topic.strip()
    if not topic:
        print("\n🤔 Введите тему:")
        topic = input("➤ ").strip()
    if not topic:
        print("❌ Тема не может быть пустой")
        return

    print(f"🔄 Генерирую научно-популярный пост: '{topic}'")
    

    try:
        agent = Agent(
            name="Popular Science Post Writer",
            instructions=build_post_instructions(topic, args.lang),
            model="gpt-5",
        )

        lang = (args.lang or "auto").strip()
        user_message = (
            f"<input>\n"
            f"<topic>{topic}</topic>\n"
            f"<lang>{lang}</lang>\n"
            f"</input>"
        )
        result = Runner.run_sync(agent, user_message)
        content = getattr(result, "final_output", "")

        if not content:
            print("❌ Пустой результат от агента")
            return

        # Single-pass output is used; formatting is enforced via the system prompt only

        output_dir = ensure_output_dir(args.out)

        base = f"{safe_filename_base(topic)}_post"

        # Default: run factcheck and rewrite unless disabled
        # For non-OpenAI providers, delegate to services.post.generate for a simple path
        if (args.provider or "openai").strip().lower() != "openai":
            from services.post.generate import generate_post
            path = generate_post(
                topic,
                lang=args.lang,
                provider=args.provider,
                factcheck=not args.no_factcheck,
                research_iterations=args.research_iterations,
                research_concurrency=args.research_concurrency,
                output_subdir=args.out,
            )
            print(f"💾 Сохранено: {path}")
            print("✅ Готово.")
            return
        report = None
        if not args.no_factcheck:
            identify_agent = build_identify_points_agent()
            identify_result = Runner.run_sync(identify_agent, f"<post>\n{content}\n</post>")
            plan = identify_result.final_output  # type: ignore
            points = plan.points or []
            if args.factcheck_max_items and args.factcheck_max_items > 0:
                points = points[: args.factcheck_max_items]
            print(f"📋 Пункты для проверки: {len(points)}")

            # Iterative research per point (with concurrency)
            from utils.config import load_config
            cfg = load_config(__file__)
            pref = (cfg.get("research", {}) or {}).get("preferred_domains", [])
            research_agent = build_iterative_research_agent()
            suff_agent = build_sufficiency_agent()
            rec_agent = build_recommendation_agent()
            synth_agent = build_query_synthesizer_agent()
            per_point_recs = []

            async def _run_with_retries(agent, inp: str, attempts: int = 3, base_delay: float = 1.0):
                for i in range(attempts):
                    try:
                        return await Runner.run(agent, inp)
                    except Exception as e:
                        if i == attempts - 1:
                            raise
                        await asyncio.sleep(base_delay * (2 ** i))

            async def process_point_async(p):
                print(f"\n📡 Пункт {p.id}: {p.text}")
                try:
                    # Query synthesis
                    cfg_pref = ",".join(pref)
                    qp_res = await _run_with_retries(synth_agent, f"<input>\n<point>{p.model_dump_json()}</point>\n<preferred_domains>{cfg_pref}</preferred_domains>\n</input>")  # type: ignore
                    qp = qp_res.final_output
                    if getattr(qp, "queries", None):
                        print(f"   🧩 Синтезированы запросы: {', '.join(qp.queries[:4])}")
                    notes = []
                    for step in range(1, max(1, int(args.research_iterations)) + 1):
                        print(f"   🔎 Итерация {step}: выполняю поиск надёжных источников...")
                        rr_input = (
                            "<input>\n"
                            f"<point>{{\"id\":\"{p.id}\",\"text\":\"{p.text}\"}}</point>\n"
                            f"<step>{step}</step>\n"
                            "</input>"
                        )
                        note_res = await _run_with_retries(research_agent, rr_input)  # type: ignore
                        note = note_res.final_output
                        notes.append(note)
                        try:
                            q = getattr(note, "query", "") or ""
                            qs = getattr(note, "queries", []) or []
                            evn = len(getattr(note, "evidence", []) or [])
                            if qs:
                                print(f"      • Запросы: {', '.join(qs[:3])}...")
                            elif q:
                                print(f"      • Запрос: {q}")
                            print(f"      • Найдено источников: {evn}")
                        except Exception:
                            pass
                        print("   🧪 Оценка достаточности...")
                        suff_input = (
                            "<input>\n"
                            f"<point>{p.model_dump_json()}</point>\n"
                            f"<notes>[{','.join([n.model_dump_json() for n in notes])}]</notes>\n"
                            "</input>"
                        )
                        decision_res = await _run_with_retries(suff_agent, suff_input)  # type: ignore
                        decision = decision_res.final_output
                        if decision.done:
                            print("   ✅ Данных достаточно для выводов.")
                            break
                        else:
                            print("   ➕ Требуются дополнительные данные.")
                    # Build a ResearchReport-like object for recommendation agent
                    class _TmpReport:
                        def __init__(self, point_id, notes):
                            self.point_id = point_id
                            self.notes = notes
                            self.synthesis = ""
                        def model_dump_json(self):
                            import json
                            return json.dumps({
                                "point_id": self.point_id,
                                "notes": [n.model_dump() for n in self.notes],
                                "synthesis": self.synthesis,
                            }, ensure_ascii=False)

                    rr = _TmpReport(p.id, notes)
                    print("   ✍️ Формирую рекомендацию по пункту...")
                    rec_res = await _run_with_retries(rec_agent, f"<input>\n<point>{p.model_dump_json()}</point>\n<report>{rr.model_dump_json()}</report>\n</input>")  # type: ignore
                    rec = rec_res.final_output
                    return p, rec
                except Exception as e:
                    print(f"   ⚠️ Сетевая ошибка по пункту: {e}. Помечу как clarify.")
                    class _TmpRec:
                        pass
                    r = _TmpRec(); r.action = "clarify"; r.explanation = "Temporary clarify due to network error"
                    return p, r

            async def process_all(points_list):
                sem = asyncio.Semaphore(max(1, int(args.research_concurrency)))
                async def worker(p):
                    async with sem:
                        return await process_point_async(p)
                tasks = [asyncio.create_task(worker(p)) for p in points_list]
                results = []
                for t in asyncio.as_completed(tasks):
                    results.append(await t)
                return results

            results = asyncio.run(process_all(points))
            for p, rec in results:
                per_point_recs.append(rec)

            # Build a lightweight FactCheckReport-equivalent summary for rewrite
            # Map recommendations to simple critique items
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
                    fix = ""
                elif r.action == "rewrite":
                    verdict = "fail"
                    fix = ""
                else:
                    verdict = "fail"
                    fix = "Удалить данный фрагмент."
                simple_items.append(_SimpleItem(p.text, verdict, fix))

            class _SimpleReport:
                def __init__(self, items):
                    self.items = items
                def model_dump_json(self):
                    import json
                    return json.dumps({
                        "summary": "Per-point recommendations",
                        "items": [
                            {
                                "claim_text": i.claim_text,
                                "verdict": i.verdict,
                                "reason": "",
                                "proposed_fix": i.proposed_fix,
                                "evidence": [],
                            } for i in self.items
                        ]
                    }, ensure_ascii=False)

            report = _SimpleReport(simple_items)
            print(f"🔎 Исследование пунктов завершено (пул={int(args.research_concurrency)})")

        final_content = content
        if (report is not None):
            needs_rewrite = any(i.verdict != "pass" for i in report.items)
            if needs_rewrite:
                rw_agent = build_rewrite_agent()
                rw_input = (
                    "<input>\n"
                    f"<topic>{topic}</topic>\n"
                    f"<lang>{lang}</lang>\n"
                    f"<post>\n{content}\n</post>\n"
                    f"<critique_json>\n{report.model_dump_json()}\n</critique_json>\n"
                    "</input>"
                )
                async def _do_rewrite(inp: str) -> str:
                    res = await Runner.run(rw_agent, inp)
                    return getattr(res, "final_output", "") or content
                final_content = asyncio.run(_do_rewrite(rw_input))
                print("✍️ Переписывание завершено.")

        # Final style/length refinement pass (always)
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

                # Второй проход факт-чекинга отключён (используем новую систему)

        # Save only final content
        filepath = next_available_filepath(output_dir, base, ".md")
        save_markdown(filepath, title=topic, generator="OpenAI Agents SDK", pipeline="PopularSciencePost", content=final_content)
        print(f"💾 Сохранено: {filepath}")
        print("✅ Готово.")
        preview = final_content[:200] + "..." if len(final_content) > 200 else final_content
        print(f"\n📖 Превью:\n{preview}")
        print("\n📊 Статистика:")
        print(f"   Слов: {len(final_content.split())}")
        print(f"   Символов: {len(final_content)}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()



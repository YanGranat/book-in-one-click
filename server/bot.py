#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, Set
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove

from utils.env import load_env_from_root
# i18n and language detection
from utils.lang import detect_lang_from_text
from services.post.generate import generate_post
from utils.slug import safe_filename_base
from .db import SessionLocal
from .bot_commands import ADMIN_IDS
from .credits import ensure_user_with_credits, charge_credits, charge_credits_kv, get_balance_kv_only
from .kv import set_provider, get_provider, set_logs_enabled, get_logs_enabled, set_incognito, get_incognito
from .kv import set_gen_lang, get_gen_lang
from .kv import set_refine_enabled, get_refine_enabled
from .kv import push_history, get_history, clear_history, rate_allow
# Fact-check settings (KV). Use robust import with fallbacks for older deployments
try:
    from .kv import (
        set_factcheck_enabled,
        get_factcheck_enabled,
        set_factcheck_depth,
        get_factcheck_depth,
    )
except Exception:
    # Provide minimal fallbacks via direct KV access to avoid startup import errors
    try:
        from . import kv as _KV
    except Exception:
        _KV = None  # type: ignore

    async def set_factcheck_enabled(telegram_id: int, enabled: bool) -> None:  # type: ignore
        if _KV is None:
            return
        r = _KV.get_redis()
        await r.set(f"{_KV.kv_prefix()}:fc_enabled:{telegram_id}", "1" if enabled else "0")

    async def get_factcheck_enabled(telegram_id: int) -> bool:  # type: ignore
        if _KV is None:
            return False
        r = _KV.get_redis()
        val = await r.get(f"{_KV.kv_prefix()}:fc_enabled:{telegram_id}")
        try:
            s = (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "0")).strip()
        except Exception:
            s = str(val or "0").strip()
        return s == "1"

    async def set_factcheck_depth(telegram_id: int, depth: int) -> None:  # type: ignore
        if _KV is None:
            return
        try:
            d = int(depth)
        except Exception:
            d = 1
        if d not in (1, 2, 3):
            d = 1
        r = _KV.get_redis()
        await r.set(f"{_KV.kv_prefix()}:fc_depth:{telegram_id}", str(d))

    async def get_factcheck_depth(telegram_id: int) -> int:  # type: ignore
        if _KV is None:
            return 2
        r = _KV.get_redis()
        val = await r.get(f"{_KV.kv_prefix()}:fc_depth:{telegram_id}")
        try:
            s = (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "2")).strip()
            d = int(s)
        except Exception:
            d = 2
        return 2 if d not in (1, 2, 3) else d


def _load_env():
    load_env_from_root(__file__)


_load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


class GenerateStates(StatesGroup):
    ChoosingLanguage = State()
    ChoosingGenLanguage = State()
    ChoosingProvider = State()
    ChoosingLogs = State()
    ChoosingIncognito = State()
    WaitingTopic = State()
    ChoosingFactcheck = State()
    ChoosingDepth = State()
    ChoosingRefine = State()


def _is_ru(ui_lang: str) -> bool:
    # Treat anything except 'en' as Russian UI by default
    return (ui_lang or "").strip().lower() != "en"


# Legacy ReplyKeyboard builders removed; we use inline keyboards everywhere


from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, LabeledPrice

def build_buy_keyboard(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    # Packs: 1, 3, 5, 10, 50 credits at 200 stars per credit
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Купить 1 кредит — 200⭐", callback_data="buy:stars:1"))
        kb.add(InlineKeyboardButton(text="Купить 3 кредита — 600⭐", callback_data="buy:stars:3"))
        kb.add(InlineKeyboardButton(text="Купить 5 кредитов — 1000⭐", callback_data="buy:stars:5"))
        kb.add(InlineKeyboardButton(text="Купить 10 кредитов — 2000⭐", callback_data="buy:stars:10"))
        kb.add(InlineKeyboardButton(text="Купить 50 кредитов — 10000⭐", callback_data="buy:stars:50"))
    else:
        kb.add(InlineKeyboardButton(text="Buy 1 credit — 200⭐", callback_data="buy:stars:1"))
        kb.add(InlineKeyboardButton(text="Buy 3 credits — 600⭐", callback_data="buy:stars:3"))
        kb.add(InlineKeyboardButton(text="Buy 5 credits — 1000⭐", callback_data="buy:stars:5"))
        kb.add(InlineKeyboardButton(text="Buy 10 credits — 2000⭐", callback_data="buy:stars:10"))
        kb.add(InlineKeyboardButton(text="Buy 50 credits — 10000⭐", callback_data="buy:stars:50"))
    return kb


def build_settings_keyboard(ui_lang: str, provider: str, gen_lang: str, refine: bool, logs_enabled: bool, incognito: bool, fc_enabled: bool, fc_depth: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    ru = _is_ru(ui_lang)
    # Provider row
    kb.add(
        InlineKeyboardButton(text=("OpenAI" + (" ✓" if provider == "openai" else "")), callback_data="set:provider:openai"),
        InlineKeyboardButton(text=("Gemini" + (" ✓" if provider == "gemini" else "")), callback_data="set:provider:gemini"),
        InlineKeyboardButton(text=("Claude" + (" ✓" if provider == "claude" else "")), callback_data="set:provider:claude"),
    )
    # Generation language row
    kb.add(
        InlineKeyboardButton(text=("RU" + (" ✓" if gen_lang == "ru" else "")), callback_data="set:gen_lang:ru"),
        InlineKeyboardButton(text=("EN" + (" ✓" if gen_lang == "en" else "")), callback_data="set:gen_lang:en"),
        InlineKeyboardButton(text=(("Авто" if ru else "Auto") + (" ✓" if gen_lang == "auto" else "")), callback_data="set:gen_lang:auto"),
    )
    # Refine row
    kb.add(
        InlineKeyboardButton(text=(("Редактура: вкл" if ru else "Refine: on") + (" ✓" if refine else "")), callback_data="set:refine:yes"),
        InlineKeyboardButton(text=(("Редактура: выкл" if ru else "Refine: off") + (" ✓" if not refine else "")), callback_data="set:refine:no"),
    )
    # Logs row
    kb.add(
        InlineKeyboardButton(text=(("Логи: вкл" if ru else "Logs: on") + (" ✓" if logs_enabled else "")), callback_data="set:logs:enable"),
        InlineKeyboardButton(text=(("Логи: выкл" if ru else "Logs: off") + (" ✓" if not logs_enabled else "")), callback_data="set:logs:disable"),
    )
    # Incognito row
    kb.add(
        InlineKeyboardButton(text=(("Инкогнито: вкл" if ru else "Incognito: on") + (" ✓" if incognito else "")), callback_data="set:incog:enable"),
        InlineKeyboardButton(text=(("Инкогнито: выкл" if ru else "Incognito: off") + (" ✓" if not incognito else "")), callback_data="set:incog:disable"),
    )
    # Fact-check row
    kb.add(
        InlineKeyboardButton(text=(("Факт-чекинг: вкл" if ru else "Fact-check: on") + (" ✓" if fc_enabled else "")), callback_data="set:fc_cmd:enable"),
        InlineKeyboardButton(text=(("Факт-чекинг: выкл" if ru else "Fact-check: off") + (" ✓" if not fc_enabled else "")), callback_data="set:fc_cmd:disable"),
    )
    # Depth row
    kb.add(
        InlineKeyboardButton(text=("D=1" + (" ✓" if fc_depth == 1 else "")), callback_data="set:depth:1"),
        InlineKeyboardButton(text=("D=2" + (" ✓" if fc_depth == 2 else "")), callback_data="set:depth:2"),
        InlineKeyboardButton(text=("D=3" + (" ✓" if fc_depth == 3 else "")), callback_data="set:depth:3"),
    )
    return kb


def build_genlang_inline(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"))
        kb.add(InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"))
        kb.add(InlineKeyboardButton(text="Авто", callback_data="set:gen_lang:auto"))
    else:
        kb.add(InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"))
        kb.add(InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"))
        kb.add(InlineKeyboardButton(text="Auto", callback_data="set:gen_lang:auto"))
    return kb


def build_ui_lang_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="Русский", callback_data="set:ui_lang:ru"))
    kb.add(InlineKeyboardButton(text="English", callback_data="set:ui_lang:en"))
    return kb


def build_provider_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="OpenAI", callback_data="set:provider:openai"))
    kb.add(InlineKeyboardButton(text="Gemini", callback_data="set:provider:gemini"))
    kb.add(InlineKeyboardButton(text="Claude", callback_data="set:provider:claude"))
    return kb


def build_enable_disable_inline(tag: str, ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Включить", callback_data=f"set:{tag}:enable"))
        kb.add(InlineKeyboardButton(text="Отключить", callback_data=f"set:{tag}:disable"))
    else:
        kb.add(InlineKeyboardButton(text="Enable", callback_data=f"set:{tag}:enable"))
        kb.add(InlineKeyboardButton(text="Disable", callback_data=f"set:{tag}:disable"))
    return kb


def build_yesno_inline(tag: str, ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Да", callback_data=f"set:{tag}:yes"))
        kb.add(InlineKeyboardButton(text="Нет", callback_data=f"set:{tag}:no"))
    else:
        kb.add(InlineKeyboardButton(text="Yes", callback_data=f"set:{tag}:yes"))
        kb.add(InlineKeyboardButton(text="No", callback_data=f"set:{tag}:no"))
    return kb


def build_depth_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="1", callback_data="set:depth:1"))
    kb.add(InlineKeyboardButton(text="2", callback_data="set:depth:2"))
    kb.add(InlineKeyboardButton(text="3", callback_data="set:depth:3"))
    return kb


def _stars_enabled() -> bool:
    # Enable Stars only when explicitly configured via env
    flag = os.getenv("TELEGRAM_STARS_ENABLED", "").strip().lower()
    return flag in ("1", "true", "yes")


# Removed legacy ReplyKeyboard-based builders to unify inline-only UI


def create_dispatcher() -> Dispatcher:
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher(bot, storage=MemoryStorage())

    # Simple in-memory guard to avoid duplicate generation per chat
    RUNNING_CHATS: Set[int] = set()
    # Optional global concurrency limit (5–10 parallel jobs target)
    import asyncio
    _sem_capacity = int(os.getenv("BOT_PARALLEL_LIMIT", "12"))
    GLOBAL_SEMAPHORE = asyncio.Semaphore(max(1, _sem_capacity))

    @dp.message_handler(commands=["start"])  # type: ignore
    async def cmd_start(message: types.Message):
        # Mark onboarding flow active
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(onboarding=True)
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_ui_lang_inline(),
        )

    @dp.message_handler(commands=["info"])  # type: ignore
    async def cmd_info(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()

        # Read current user settings from KV (with safe fallbacks)
        prov = (data.get("provider") or "openai").strip().lower()
        gen_lang = (data.get("gen_lang") or "auto").strip().lower()
        incognito = False
        logs_enabled = False
        try:
            if message.from_user:
                try:
                    prov = await get_provider(message.from_user.id)  # type: ignore
                except Exception:
                    prov = prov or "openai"
                try:
                    gen_lang = await get_gen_lang(message.from_user.id)  # type: ignore
                except Exception:
                    gen_lang = gen_lang or "auto"
                try:
                    incognito = await get_incognito(message.from_user.id)  # type: ignore
                except Exception:
                    incognito = False
                try:
                    logs_enabled = await get_logs_enabled(message.from_user.id)  # type: ignore
                except Exception:
                    logs_enabled = False
        except Exception:
            pass

        def _prov_name(p: str) -> str:
            m = {"openai": "OpenAI", "gemini": "Gemini", "claude": "Claude"}
            return m.get((p or "").strip().lower(), p or "openai")

        def _lang_human(l: str, ru: bool) -> str:
            if ru:
                return {"ru": "русский", "en": "английский", "auto": "авто"}.get((l or "auto").lower(), "авто")
            return {"ru": "Russian", "en": "English", "auto": "Auto"}.get((l or "auto").lower(), "Auto")

        # Extra settings
        refine_enabled = False
        fc_enabled = False
        fc_depth = 2
        try:
            if message.from_user:
                refine_enabled = await get_refine_enabled(message.from_user.id)
                fc_enabled = await get_factcheck_enabled(message.from_user.id)
                fc_depth = await get_factcheck_depth(message.from_user.id)
        except Exception:
            pass

        if ui_lang == "ru":
            text = (
                "<b>Как пользоваться:</b>\n"
                "- /generate — отправьте тему, будут использованы ваши настройки по умолчанию.\n"
                "- /start — пройти настройку заново (язык интерфейса, язык генерации, редактура, провайдер, логи, инкогнито, факт‑чекинг, глубина).\n"
                "- /factcheck — задать дефолт факт‑чекинга (вкл/выкл и глубину).\n"
                "- /depth — установить глубину факт‑чекинга по умолчанию.\n"
                "- /refine — включить/выключить финальную редактуру.\n"
                "- /lang, /lang_generate, /provider, /logs, /incognito — точечно поменять настройки.\n\n"
                "<b>Результаты:</b>\n"
                "- Список всех результатов: https://bio1c-bot.onrender.com/results-ui\n"
                "- Каждый результат — Markdown‑файл, который можно скачать и править.\n\n"
                "<b>Кредиты:</b>\n"
                "- 1 генерация = 1 кредит. Посмотреть баланс: /balance. Пополнить: /buy.\n"
                "- /pricing — посмотреть цены по типам.\n\n"
                ""
                "<b>Текущие настройки:</b>\n"
                f"- Провайдер: {_prov_name(prov)}\n"
                f"- Язык генерации: {_lang_human(gen_lang, True)}\n"
                f"- Инкогнито: {'включён' if incognito else 'отключён'}\n"
                f"- Логи: {'включены' if logs_enabled else 'отключены'}\n"
                f"- Финальная редактура: {'включена' if refine_enabled else 'отключена'}\n"
                f"- Факт‑чекинг: {'включён' if fc_enabled else 'отключён'}"
                + (f" (глубина {fc_depth})" if fc_enabled else "")
                + "\n\nGitHub проекта: https://github.com/YanGranat/book-in-one-click"
            )
        else:
            text = (
                "<b>How to use:</b>\n"
                "- /generate — send a topic; your default settings will be used.\n"
                "- /start — run onboarding (UI lang, gen lang, refine, provider, logs, incognito, fact‑check, depth).\n"
                "- /factcheck — set default fact‑check (enable/disable and depth).\n"
                "- /depth — set default fact‑check depth.\n"
                "- /refine — enable/disable final refine step.\n"
                "- /lang, /lang_generate, /provider, /logs, /incognito — tweak settings individually.\n\n"
                "<b>Results:</b>\n"
                "- All results page: https://bio1c-bot.onrender.com/results-ui\n"
                "- Each result is a Markdown file you can download and edit.\n\n"
                "<b>Credits:</b>\n"
                "- 1 generation = 1 credit. Check balance: /balance. Buy: /buy.\n"
                "- /pricing — see pricing per type.\n\n"
                ""
                "<b>Current settings:</b>\n"
                f"- Provider: {_prov_name(prov)}\n"
                f"- Generation language: {_lang_human(gen_lang, False)}\n"
                f"- Incognito: {'enabled' if incognito else 'disabled'}\n"
                f"- Logs: {'enabled' if logs_enabled else 'disabled'}\n"
                f"- Final refine: {'enabled' if refine_enabled else 'disabled'}\n"
                f"- Fact‑check: {'enabled' if fc_enabled else 'disabled'}"
                + (f" (depth {fc_depth})" if fc_enabled else "")
                + "\n\nProject GitHub: https://github.com/YanGranat/book-in-one-click"
            )
        await message.answer(text, disable_web_page_preview=True, parse_mode=types.ParseMode.HTML)


    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:ui_lang:"))  # type: ignore
    async def cb_set_ui_lang(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        ui_lang = "en" if val == "en" else "ru"
        await state.update_data(ui_lang=ui_lang)
        await query.message.edit_reply_markup() if query.message else None
        await query.answer()
        confirm = "Язык интерфейса установлен." if ui_lang == "ru" else "Interface language set."
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, confirm)
        data = await state.get_data()
        onboarding = bool(data.get("onboarding"))
        if onboarding:
            prompt = "Выберите язык генерации:" if _is_ru(ui_lang) else "Choose generation language:"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_genlang_inline(ui_lang))

    @dp.message_handler(commands=["lang_generate"])  # type: ignore
    async def cmd_lang_generate(message: types.Message, state: FSMContext):
        # Single-step: do not enable onboarding
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        await message.answer(
            "Выберите язык генерации:" if _is_ru(ui_lang) else "Choose generation language:",
            reply_markup=build_genlang_inline(ui_lang),
        )
        # Inline only; no FSM step

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:gen_lang:"))  # type: ignore
    async def cb_set_gen_lang(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        gen_lang = val if val in {"ru","en","auto"} else "auto"
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        await state.update_data(gen_lang=gen_lang)
        try:
            if query.from_user:
                await set_gen_lang(query.from_user.id, gen_lang)
        except Exception:
            pass
        await query.answer()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            # Re-render settings panel
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang_cur = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang_cur = gen_lang; refine=False; logs_enabled=False; incognito=False; fc_enabled=False; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang_cur, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Old behavior
            msg = {
                "ru": {
                    "ru": "Язык генерации: русский.",
                    "en": "Язык генерации: английский.",
                    "auto": "Язык генерации: авто (по теме).",
                },
                "en": {
                    "ru": "Generation language: Russian.",
                    "en": "Generation language: English.",
                    "auto": "Generation language: auto (by topic).",
                },
            }
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg.get("ru" if ui_lang == "ru" else "en").get(gen_lang, "OK"))
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))

    @dp.message_handler(commands=["provider"])  # type: ignore
    async def cmd_provider(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Выберите провайдера (OpenAI/Gemini/Claude):" if ui_lang == "ru" else "Choose provider (OpenAI/Gemini/Claude):"
        await message.answer(prompt, reply_markup=build_provider_inline())

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:provider:"))  # type: ignore
    async def cb_set_provider(query: types.CallbackQuery, state: FSMContext):
        prov = (query.data or "").split(":")[-1]
        if prov not in {"openai","gemini","claude"}:
            await query.answer()
            return
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        await state.update_data(provider=prov)
        try:
            if query.from_user:
                await set_provider(query.from_user.id, prov)  # type: ignore
        except Exception:
            pass
        await query.answer()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            # Re-render settings panel in-place
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = prov; gen_lang = (data.get("gen_lang") or "auto"); refine=False; logs_enabled=False; incognito=False; fc_enabled=False; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Old behavior (outside settings panel)
            await query.message.edit_reply_markup() if query.message else None
            ok = "Провайдер установлен." if ui_lang == "ru" else "Provider set."
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ok)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                prompt = "Отправлять логи генерации?" if _is_ru(ui_lang) else "Send generation logs?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_enable_disable_inline("logs", ui_lang))

    @dp.message_handler(commands=["lang"])  # type: ignore
    async def cmd_lang(message: types.Message, state: FSMContext):
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_ui_lang_inline(),
        )
        # Inline only; no FSM step

    @dp.message_handler(commands=["generate"])  # type: ignore
    async def cmd_generate(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Preload default fact-check preferences from KV for /generate flow
        try:
            fc_enabled = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
        except Exception:
            fc_enabled = False
        try:
            fc_depth = await get_factcheck_depth(message.from_user.id) if message.from_user else 2
        except Exception:
            fc_depth = 2
        await state.update_data(factcheck=bool(fc_enabled), research_iterations=(int(fc_depth) if fc_enabled else None), fc_ready=True)
        prompt = "Отправьте тему для поста:" if _is_ru(ui_lang) else "Send a topic for your post:"
        await message.answer(prompt, reply_markup=ReplyKeyboardRemove())
        await GenerateStates.WaitingTopic.set()

    # ---- Settings quick panel ----
    @dp.message_handler(commands=["settings"])  # type: ignore
    async def cmd_settings(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prov = (data.get("provider") or "openai").strip().lower()
        gen_lang = (data.get("gen_lang") or "auto").strip().lower()
        try:
            if message.from_user:
                prov = await get_provider(message.from_user.id)
                gen_lang = await get_gen_lang(message.from_user.id)
                refine = await get_refine_enabled(message.from_user.id)
                logs_enabled = await get_logs_enabled(message.from_user.id)
                incognito = await get_incognito(message.from_user.id)
                fc_enabled = await get_factcheck_enabled(message.from_user.id)
                fc_depth = await get_factcheck_depth(message.from_user.id)
            else:
                refine = False; logs_enabled = False; incognito = False; fc_enabled = False; fc_depth = 2
        except Exception:
            refine = False; logs_enabled = False; incognito = False; fc_enabled = False; fc_depth = 2
        title = (
            "Быстрые настройки (всё — через кнопки, без диалогов):"
            if _is_ru(ui_lang) else
            "Quick settings (all toggles via buttons):"
        )
        await state.update_data(in_settings=True)
        await message.answer(title, reply_markup=build_settings_keyboard(ui_lang, prov, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth)))

    @dp.message_handler(commands=["logs"])  # type: ignore
    async def cmd_logs(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Отправлять логи генерации?" if ui_lang == "ru" else "Send generation logs?"
        await message.answer(prompt, reply_markup=build_enable_disable_inline("logs", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:logs:"))  # type: ignore
    async def cb_set_logs(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        enabled = (val == "enable")
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            if query.from_user:
                await set_logs_enabled(query.from_user.id, enabled)
        except Exception:
            pass
        await query.answer()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            # Re-render settings panel
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang = (data.get("gen_lang") or "auto"); refine=False; logs_enabled=enabled; incognito=False; fc_enabled=False; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Old behavior
            msg = "Логи включены." if (enabled and ui_lang=="ru") else ("Logs enabled." if enabled else ("Логи отключены." if ui_lang=="ru" else "Logs disabled."))
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                prompt = "Инкогнито режим: включить или отключить?" if _is_ru(ui_lang) else "Incognito: Enable or Disable?"
                await dp.bot.send_message(
                    query.message.chat.id if query.message else query.from_user.id,
                    prompt,
                    reply_markup=build_enable_disable_inline("incog", ui_lang),
                )

    @dp.message_handler(commands=["incognito"])  # type: ignore
    async def cmd_incognito(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = (
            "Инкогнито режим (ваши результаты не будут отображаться на публичной странице результатов):"
            if ui_lang == "ru"
            else "Incognito (your results will not appear on the public results page):"
        )
        await message.answer(prompt, reply_markup=build_enable_disable_inline("incog", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:incog:"))  # type: ignore
    async def cb_set_incog(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        enabled = (val == "enable")
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            if query.from_user:
                await set_incognito(query.from_user.id, enabled)
        except Exception:
            pass
        await query.answer()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang = (data.get("gen_lang") or "auto"); refine=False; logs_enabled=False; incognito=enabled; fc_enabled=False; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            msg = "Инкогнито: включён." if (enabled and ui_lang=="ru") else ("Incognito: enabled." if enabled else ("Инкогнито: отключён." if ui_lang=="ru" else "Incognito: disabled."))
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding_flag = bool((await state.get_data()).get("onboarding"))
            if onboarding_flag:
                # After incognito, ask fact-check preference before topic
                prompt = (
                    "Включить факт-чекинг?"
                    if _is_ru(ui_lang)
                    else "Enable fact-checking?"
                )
                await dp.bot.send_message(
                    query.message.chat.id if query.message else query.from_user.id,
                    prompt,
                    reply_markup=build_yesno_inline("fc", ui_lang),
                )

    @dp.message_handler(commands=["refine"])  # type: ignore
    async def cmd_refine(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
        await message.answer(prompt, reply_markup=build_yesno_inline("refine", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:refine:"))  # type: ignore
    async def cb_set_refine(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        enabled = (val == "yes")
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            if query.from_user:
                await set_refine_enabled(query.from_user.id, enabled)
        except Exception:
            pass
        await query.answer()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang = (data.get("gen_lang") or "auto"); refine=enabled; logs_enabled=False; incognito=False; fc_enabled=False; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            msg = "Финальная редактура: включена." if (enabled and ui_lang=="ru") else ("Final refine: enabled." if enabled else ("Финальная редактура: отключена." if ui_lang=="ru" else "Final refine: disabled."))
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                prompt = "Выберите провайдера (OpenAI/Gemini/Claude):" if _is_ru(ui_lang) else "Choose provider (OpenAI/Gemini/Claude):"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_provider_inline())

    @dp.message_handler(commands=["cancel"])  # type: ignore
    async def cmd_cancel(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        done = "Отменено." if ui_lang == "ru" else "Cancelled."
        await state.finish()
        await message.answer(done, reply_markup=ReplyKeyboardRemove())

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc:"))  # type: ignore
    async def cb_set_fc(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        enabled = (val == "yes")
        await state.update_data(factcheck=enabled)
        try:
            if query.from_user:
                await set_factcheck_enabled(query.from_user.id, enabled)
        except Exception:
            pass
        await query.message.edit_reply_markup() if query.message else None
        await query.answer()
        onboarding = bool((await state.get_data()).get("onboarding"))
        if enabled:
            prompt = "Выберите глубину проверки (1–3):" if _is_ru(ui_lang) else "Select research depth (1–3):"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_depth_inline())
        else:
            # Mark FC decision as done in onboarding to avoid asking again on topic
            await state.update_data(factcheck=False, research_iterations=None, fc_ready=True)
            if onboarding:
                prompt = "Отправьте тему для поста:" if _is_ru(ui_lang) else "Send a topic for your post:"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt)
                await GenerateStates.WaitingTopic.set()

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:depth:"))  # type: ignore
    async def cb_set_depth(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        try:
            depth = int(val)
        except Exception:
            depth = 1
        # Finalize FC choices
        await state.update_data(factcheck=True, research_iterations=depth, fc_ready=True)
        try:
            if query.from_user:
                await set_factcheck_depth(query.from_user.id, depth)
        except Exception:
            pass
        await query.answer()
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang = (data.get("gen_lang") or "auto"); refine=False; logs_enabled=False; incognito=False; fc_enabled=True; fc_depth=depth
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            onboarding = bool(data.get("onboarding"))
            if onboarding:
                prompt = "Отправьте тему для поста:" if _is_ru(ui_lang) else "Send a topic for your post:"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt)
                await GenerateStates.WaitingTopic.set()
            else:
                # Standalone /factcheck flow: confirm and stay
                msg = "Глубина факт-чекинга сохранена." if _is_ru(ui_lang) else "Fact-check depth saved."
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)

    # ---- Fact-check settings command ----
    @dp.message_handler(commands=["factcheck"])  # type: ignore
    async def cmd_factcheck(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Факт-чекинг?" if _is_ru(ui_lang) else "Fact-check?"
        await message.answer(prompt, reply_markup=build_enable_disable_inline("fc_cmd", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc_cmd:"))  # type: ignore
    async def cb_fc_cmd_toggle(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        enabled = (val == "enable")
        await set_factcheck_enabled(query.from_user.id, enabled)
        await query.answer()
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        in_settings = bool((await state.get_data()).get("in_settings"))
        if in_settings and query.message:
            try:
                user_id = query.from_user.id
                prov_cur = await get_provider(user_id)
                gen_lang = await get_gen_lang(user_id)
                refine = await get_refine_enabled(user_id)
                logs_enabled = await get_logs_enabled(user_id)
                incognito = await get_incognito(user_id)
                fc_enabled = await get_factcheck_enabled(user_id)
                fc_depth = await get_factcheck_depth(user_id)
            except Exception:
                prov_cur = (data.get("provider") or "openai"); gen_lang = (data.get("gen_lang") or "auto"); refine=False; logs_enabled=False; incognito=False; fc_enabled=enabled; fc_depth=2
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth))
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            if enabled:
                prompt = "Выберите глубину проверки (1–3):" if _is_ru(ui_lang) else "Select research depth (1–3):"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_depth_inline())
            else:
                msg = "Факт-чекинг: отключён." if _is_ru(ui_lang) else "Fact-check: disabled."
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)

    # ---- Dedicated depth command ----
    @dp.message_handler(commands=["depth"])  # type: ignore
    async def cmd_depth(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Глубина факт-чекинга (1–3):" if _is_ru(ui_lang) else "Fact-check depth (1–3):"
        await message.answer(prompt, reply_markup=build_depth_inline())


    # Helper for results link
    RESULTS_ORIGIN = os.getenv("RESULTS_UI_ORIGIN", "https://bio1c-bot.onrender.com").rstrip("/")

    def _result_url(res_id: int) -> str:
        return f"{RESULTS_ORIGIN}/results-ui/id/{int(res_id)}"

    @dp.message_handler(state=GenerateStates.WaitingTopic, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def topic_received(message: types.Message, state: FSMContext):
        text_raw = (message.text or "").strip()
        data = await state.get_data()
        ui_lang = data.get("ui_lang", "ru")
        if text_raw.lower() in {"/cancel"}:
            done = "Отменено." if ui_lang == "ru" else "Cancelled."
            await message.answer(done, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            return
        topic = text_raw
        if not topic:
            msg = "Тема не может быть пустой. Отправьте тему:" if ui_lang == "ru" else "Topic cannot be empty. Send a topic:"
            await message.answer(msg)
            return
        await state.update_data(topic=topic)
        # Decide whether to ask fact-check during onboarding only; otherwise use defaults and start
        onboarding = bool(data.get("onboarding"))
        fc_ready = bool(data.get("fc_ready"))
        if onboarding and not fc_ready:
            prompt = "Включить факт-чекинг?" if _is_ru(ui_lang) else "Enable fact-checking?"
            await message.answer(prompt, reply_markup=build_yesno_inline("fc", ui_lang))
            return

        # Rate limiting (admins bypass)
        if not (message.from_user and message.from_user.id in ADMIN_IDS):
            allowed = True
            try:
                allowed = await rate_allow(message.from_user.id, scope="gen", per_hour=int(os.getenv("RATE_HOURLY", "20")), per_day=int(os.getenv("RATE_DAILY", "100")))
            except Exception:
                allowed = True
            if not allowed:
                await message.answer("Превышен лимит запросов. Попробуйте позже." if _is_ru(ui_lang) else "Rate limit exceeded. Try later.")
                return
            # Provider-specific limits
            try:
                prov_scope = f"prov:{(data.get('provider') or '').strip().lower() or 'openai'}"
                ph = int(os.getenv(f"RATE_HOURLY_{prov_scope.split(':')[-1].upper()}", os.getenv("RATE_HOURLY_PROVIDER", "30")))
                pd = int(os.getenv(f"RATE_DAILY_{prov_scope.split(':')[-1].upper()}", os.getenv("RATE_DAILY_PROVIDER", "200")))
                if not await rate_allow(message.from_user.id, scope=prov_scope, per_hour=ph, per_day=pd):
                    await message.answer("Провайдер: превышен лимит. Попробуйте позже." if _is_ru(ui_lang) else "Provider rate limit exceeded. Try later.")
                    return
            except Exception:
                pass

        # Use current state (or KV defaults) to start generation immediately
        # Resolve provider, language, refine, incognito
        chat_id = message.chat.id
        if chat_id in RUNNING_CHATS:
            return
        RUNNING_CHATS.add(chat_id)

        # Optional confirmation with price (skip for admins)
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if not is_admin:
            try:
                ui_lang_local = (data.get("ui_lang") or "ru").strip()
                confirm_txt = (
                    "Будет списано 1 кредит. Подтвердить?"
                    if _is_ru(ui_lang_local) else
                    "It will cost 1 credit. Proceed?"
                )
                kb = InlineKeyboardMarkup()
                kb.add(InlineKeyboardButton(text=("Подтвердить" if _is_ru(ui_lang_local) else "Confirm"), callback_data="confirm:charge:yes"))
                kb.add(InlineKeyboardButton(text=("Отмена" if _is_ru(ui_lang_local) else "Cancel"), callback_data="confirm:charge:no"))
                await message.answer(confirm_txt, reply_markup=kb)
                # Save pending topic and wait for callback
                await state.update_data(pending_topic=topic)
                return
            except Exception:
                pass

        # Charge 1 credit before starting (DB if configured, else Redis KV)
        from sqlalchemy.exc import SQLAlchemyError
        try:
            charged = False
            # Admins generate for free
            if message.from_user and message.from_user.id in ADMIN_IDS:
                charged = True
            if SessionLocal is not None and message.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, message.from_user.id)
                    ok, remaining = await charge_credits(session, user, 1, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged and message.from_user:
                ok, remaining = await charge_credits_kv(message.from_user.id, 1)
                if not ok:
                    warn = "Недостаточно кредитов" if _is_ru(ui_lang) else "Insufficient credits"
                    await message.answer(warn, reply_markup=ReplyKeyboardRemove())
                    try:
                        await message.answer(
                            ("Купить кредиты за ⭐? Один кредит = 200⭐" if _is_ru(ui_lang) else "Buy credits with ⭐? One credit = 200⭐"),
                            reply_markup=build_buy_keyboard(ui_lang),
                        )
                    except Exception:
                        pass
                    await state.finish()
                    RUNNING_CHATS.discard(chat_id)
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if ui_lang == "ru" else "Temporary DB error. Try later."
            await message.answer(warn, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            RUNNING_CHATS.discard(chat_id)
            return

        # Create Job row (running)
        job_id = 0
        try:
            if SessionLocal is not None and message.from_user:
                async with SessionLocal() as session:
                    from .db import Job
                    from .db import get_or_create_user as _get_or_create_user
                    import json as _json
                    db_user = await _get_or_create_user(session, message.from_user.id)
                    params = {
                        "topic": topic,
                        "lang": (data.get("gen_lang") or "auto"),
                        "provider": (data.get("provider") or "openai"),
                        "factcheck": bool(data.get("factcheck")),
                        "depth": int(data.get("research_iterations") or 0) if bool(data.get("factcheck")) else 0,
                        "refine": bool(await get_refine_enabled(message.from_user.id) if message.from_user else False),
                    }
                    j = Job(user_id=db_user.id, type="post", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=1)
                    session.add(j)
                    await session.flush()
                    job_id = int(j.id)
                    await session.commit()
        except Exception:
            job_id = 0

        # Light progress notes before long run
        try:
            notes = [
                "Формирую план…" if _is_ru(ui_lang) else "Planning…",
                "Пишу черновик…" if _is_ru(ui_lang) else "Drafting…",
            ]
            # If FC/refine are expected, inform once up-front
            try:
                ref_pref = await get_refine_enabled(message.from_user.id) if message.from_user else False
            except Exception:
                ref_pref = False
            if bool(data.get("factcheck")):
                notes.append("Проведу факт‑чекинг…" if _is_ru(ui_lang) else "Will run fact‑check…")
            if ref_pref:
                notes.append("Применю финальную редактуру…" if _is_ru(ui_lang) else "Will apply final refine…")
            await message.answer("\n".join(notes), reply_markup=ReplyKeyboardRemove())
        except Exception:
            working = "Генерирую. Это может занять несколько минут..." if _is_ru(ui_lang) else "Working on it. This may take a few minutes..."
            await message.answer(working, reply_markup=ReplyKeyboardRemove())

        # Run generation
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            # Resolve provider and language
            prov = (data.get("provider") or "").strip().lower()
            if not prov and message.from_user:
                try:
                    prov = await get_provider(message.from_user.id)  # type: ignore
                except Exception:
                    prov = "openai"
            # Ensure gen_lang from KV if FSM lost it
            persisted_gen_lang = None
            try:
                if message.from_user:
                    persisted_gen_lang = await get_gen_lang(message.from_user.id)
            except Exception:
                persisted_gen_lang = None
            gen_lang = (data.get("gen_lang") or persisted_gen_lang or "auto")
            eff_lang = gen_lang
            if (gen_lang or "auto").strip().lower() == "auto":
                try:
                    det = (detect_lang_from_text(topic) or "").lower()
                    eff_lang = "en" if det.startswith("en") else "ru"
                except Exception:
                    eff_lang = "ru"

            # Refine preference
            refine_enabled = False
            try:
                if message.from_user:
                    refine_enabled = await get_refine_enabled(message.from_user.id)
            except Exception:
                refine_enabled = False

            # Fact-check decision
            fc_enabled_state = data.get("factcheck")
            if fc_enabled_state is None and message.from_user:
                try:
                    fc_enabled_state = await get_factcheck_enabled(message.from_user.id)
                except Exception:
                    fc_enabled_state = False
            fc_enabled_state = bool(fc_enabled_state)
            depth = data.get("research_iterations")
            if fc_enabled_state and depth is None and message.from_user:
                try:
                    depth = await get_factcheck_depth(message.from_user.id)
                except Exception:
                    depth = 2
            if not fc_enabled_state:
                depth = None

            async with GLOBAL_SEMAPHORE:
                # Prepare job metadata for logging
                job_meta = {
                    "user_id": message.from_user.id if message.from_user else 0,
                    "chat_id": message.chat.id,
                    "topic": topic,
                    "provider": prov or "openai",
                    "lang": eff_lang,
                    "incognito": (await get_incognito(message.from_user.id)) if message.from_user else False,
                    "refine": refine_enabled,
                }
                if job_id:
                    job_meta["job_id"] = job_id
            stages = []
            def _on_progress(stage: str) -> None:
                stages.append(stage)
                try:
                    # Send lightweight hints on main milestones only
                    if stage in {"start:post","factcheck:init","rewrite:init","refine:init","save:init","done"}:
                        txt = {
                            "start:post": "Начинаю…" if _is_ru(ui_lang) else "Starting…",
                            "factcheck:init": "Факт‑чекинг…" if _is_ru(ui_lang) else "Fact‑checking…",
                            "rewrite:init": "Переписываю проблемные места…" if _is_ru(ui_lang) else "Rewriting issues…",
                            "refine:init": "Финальная редактура…" if _is_ru(ui_lang) else "Final refine…",
                            "save:init": "Сохраняю результат…" if _is_ru(ui_lang) else "Saving…",
                            "done": "Готово." if _is_ru(ui_lang) else "Done.",
                        }.get(stage)
                        if txt:
                            # Fire-and-forget
                            asyncio.create_task(message.answer(txt))
                except Exception:
                    pass

            timeout_s = int(os.getenv("GEN_TIMEOUT_S", "900"))
            if fc_enabled_state:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=eff_lang,
                        provider=(prov or "openai"),
                        factcheck=True,
                        research_iterations=int(depth or 2),
                        job_meta=job_meta,
                        on_progress=_on_progress,
                        use_refine=refine_enabled,
                    ),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            else:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=eff_lang,
                        provider=(prov or "openai"),
                        factcheck=False,
                        job_meta=job_meta,
                        on_progress=_on_progress,
                        use_refine=refine_enabled,
                    ),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            # Send main result
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if _is_ru(ui_lang) else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
            # Mark job done
            try:
                if job_id and SessionLocal is not None:
                    from sqlalchemy import update as _upd
                    from datetime import datetime as _dt
                    async with SessionLocal() as session:
                        from .db import Job
                        await session.execute(_upd(Job).where(Job.id == job_id).values(status="done", finished_at=_dt.utcnow(), file_path=str(path)))
                        await session.commit()
            except Exception:
                pass

            # Send logs if enabled
            try:
                if message.from_user:
                    logs_enabled = await get_logs_enabled(message.from_user.id)
                    if logs_enabled:
                        topic_base = safe_filename_base(topic)
                        log_files = list(path.parent.glob(f"{topic_base}_log_*.md"))
                        if log_files:
                            log_path = log_files[0]
                            with open(log_path, "rb") as log_f:
                                log_cap = f"Лог: {log_path.name}" if _is_ru(ui_lang) else f"Log: {log_path.name}"
                                await message.answer_document(log_f, caption=log_cap)
            except Exception:
                pass
            # Record history (KV) + try to attach result_id for clickable link
            try:
                res_id: int | None = None
                if SessionLocal is not None:
                    from sqlalchemy import select
                    async with SessionLocal() as session:
                        from .db import ResultDoc, Job
                        res = await session.execute(select(ResultDoc).where(ResultDoc.path == str(path)).order_by(ResultDoc.created_at.desc()))
                        row = res.scalars().first()
                        if row is not None:
                            res_id = int(row.id)
                            # mark job done if we have job_id
                            try:
                                jid = int(job_meta.get("job_id", 0)) if isinstance(job_meta, dict) else 0
                            except Exception:
                                jid = 0
                            if jid:
                                from sqlalchemy import update as _upd
                                from datetime import datetime as _dt
                                await session.execute(_upd(Job).where(Job.id == jid).values(status="done", finished_at=_dt.utcnow(), file_path=str(path)))
                                await session.commit()
                payload = {"topic": topic, "path": str(path), "created_at": datetime.utcnow().isoformat()}
                if res_id is not None:
                    # fetch hidden flag to decide link visibility
                    if SessionLocal is not None:
                        async with SessionLocal() as session:
                            from sqlalchemy import select
                            from .db import ResultDoc
                            r2 = await session.execute(select(ResultDoc).where(ResultDoc.id == int(res_id)))
                            rr = r2.scalar_one_or_none()
                            hidden = int(getattr(rr, "hidden", 0) or 0) if rr is not None else 0
                    else:
                        hidden = 0
                    payload["result_id"] = int(res_id)
                    if hidden == 0:
                        payload["url"] = _result_url(int(res_id))
                if message.from_user:
                    await push_history(message.from_user.id, payload)
            except Exception:
                pass
        except Exception as e:
            err = f"Ошибка: {e}" if ui_lang == "ru" else f"Error: {e}"
            await message.answer(err)
            # Mark job error
            try:
                if SessionLocal is not None:
                    from sqlalchemy import update as _upd
                    async with SessionLocal() as session:
                        from .db import Job
                        try:
                            jid = int((job_meta or {}).get("job_id", 0))
                        except Exception:
                            jid = 0
                        if jid:
                            await session.execute(_upd(Job).where(Job.id == jid).values(status="error"))
                            await session.commit()
            except Exception:
                pass

        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    @dp.callback_query_handler(lambda c: c.data in {"confirm:charge:yes","confirm:charge:no"})  # type: ignore
    async def cb_confirm_charge(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if query.data.endswith(":no"):
            await query.answer()
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, "Отменено." if _is_ru(ui_lang) else "Cancelled.")
            await state.finish()
            return
        # proceed and charge then run generation using saved topic
        topic = (data.get("pending_topic") or "").strip()
        if not topic:
            await query.answer()
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, "Тема не найдена, отправьте заново /generate" if _is_ru(ui_lang) else "Topic missing, send /generate again")
            await state.finish()
            return
        chat_id = query.message.chat.id if query.message else query.from_user.id
        if chat_id in RUNNING_CHATS:
            await query.answer()
            return
        RUNNING_CHATS.add(chat_id)
        # Charge
        from sqlalchemy.exc import SQLAlchemyError
        try:
            charged = False
            if SessionLocal is not None and query.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, query.from_user.id)
                    ok, remaining = await charge_credits(session, user, 1, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged and query.from_user:
                ok, remaining = await charge_credits_kv(query.from_user.id, 1)
                if not ok:
                    warn = "Недостаточно кредитов" if _is_ru(ui_lang) else "Insufficient credits"
                    await dp.bot.send_message(chat_id, warn)
                    RUNNING_CHATS.discard(chat_id)
                    await state.finish()
                    await query.answer()
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later."
            await dp.bot.send_message(chat_id, warn)
            RUNNING_CHATS.discard(chat_id)
            await state.finish()
            await query.answer()
            return

        await query.answer()
        await query.message.edit_reply_markup() if query.message else None
        # Reuse generation code path by simulating message with topic — call inner function via helper
        # To avoid duplication, inline the core after charging:
        # Resolve provider/lang/refine/fc and run generate_post (same as in topic_received)
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            prov = (data.get("provider") or "").strip().lower()
            if not prov and query.from_user:
                try:
                    prov = await get_provider(query.from_user.id)
                except Exception:
                    prov = "openai"
            persisted_gen_lang = None
            try:
                if query.from_user:
                    persisted_gen_lang = await get_gen_lang(query.from_user.id)
            except Exception:
                persisted_gen_lang = None
            gen_lang = (data.get("gen_lang") or persisted_gen_lang or "auto")
            eff_lang = gen_lang
            if (gen_lang or "auto").strip().lower() == "auto":
                try:
                    det = (detect_lang_from_text(topic) or "").lower()
                    eff_lang = "en" if det.startswith("en") else "ru"
                except Exception:
                    eff_lang = "ru"
            refine_enabled = False
            try:
                if query.from_user:
                    refine_enabled = await get_refine_enabled(query.from_user.id)
            except Exception:
                refine_enabled = False
            fc_enabled_state = data.get("factcheck")
            if fc_enabled_state is None and query.from_user:
                try:
                    fc_enabled_state = await get_factcheck_enabled(query.from_user.id)
                except Exception:
                    fc_enabled_state = False
            fc_enabled_state = bool(fc_enabled_state)
            depth = data.get("research_iterations")
            if fc_enabled_state and depth is None and query.from_user:
                try:
                    depth = await get_factcheck_depth(query.from_user.id)
                except Exception:
                    depth = 2
            if not fc_enabled_state:
                depth = None
            # Create Job (running)
            job_id = 0
            try:
                if SessionLocal is not None and query.from_user:
                    async with SessionLocal() as session:
                        from .db import Job
                        from .db import get_or_create_user as _get_or_create_user
                        import json as _json
                        db_user = await _get_or_create_user(session, query.from_user.id)
                        params = {
                            "topic": topic,
                            "lang": eff_lang,
                            "provider": prov or "openai",
                            "factcheck": bool(fc_enabled_state),
                            "depth": int(depth or 0) if fc_enabled_state else 0,
                            "refine": bool(refine_enabled),
                        }
                        j = Job(user_id=db_user.id, type="post", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=1)
                        session.add(j)
                        await session.flush()
                        job_id = int(j.id)
                        await session.commit()
            except Exception:
                job_id = 0

            job_meta = {
                "user_id": query.from_user.id if query.from_user else 0,
                "chat_id": chat_id,
                "topic": topic,
                "provider": prov or "openai",
                "lang": eff_lang,
                "incognito": (await get_incognito(query.from_user.id)) if query.from_user else False,
                "refine": refine_enabled,
            }
            if job_id:
                job_meta["job_id"] = job_id
            await dp.bot.send_message(chat_id, "Генерирую…" if _is_ru(ui_lang) else "Working…")
            timeout_s = int(os.getenv("GEN_TIMEOUT_S", "900"))
            if fc_enabled_state:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(topic, lang=eff_lang, provider=(prov or "openai"), factcheck=True, research_iterations=int(depth or 2), job_meta=job_meta, use_refine=refine_enabled),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            else:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(topic, lang=eff_lang, provider=(prov or "openai"), factcheck=False, job_meta=job_meta, use_refine=refine_enabled),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if _is_ru(ui_lang) else f"Done: {path.name}"
                await dp.bot.send_message(chat_id, cap)
                await dp.bot.send_document(chat_id, f)
            # Mark job done
            try:
                if job_id and SessionLocal is not None:
                    from sqlalchemy import update as _upd
                    from datetime import datetime as _dt
                    async with SessionLocal() as session:
                        from .db import Job
                        await session.execute(_upd(Job).where(Job.id == job_id).values(status="done", finished_at=_dt.utcnow(), file_path=str(path)))
                        await session.commit()
            except Exception:
                pass
            try:
                if query.from_user:
                    # Resolve public link if not hidden
                    res_id = None
                    if SessionLocal is not None:
                        from sqlalchemy import select
                        async with SessionLocal() as session:
                            from .db import ResultDoc
                            rs = await session.execute(select(ResultDoc).where(ResultDoc.path == str(path)).order_by(ResultDoc.created_at.desc()))
                            rrow = rs.scalars().first()
                            if rrow is not None:
                                res_id = int(rrow.id)
                                hidden = int(getattr(rrow, "hidden", 0) or 0)
                                payload = {"topic": topic, "path": str(path), "created_at": datetime.utcnow().isoformat(), "result_id": res_id}
                                if hidden == 0:
                                    payload["url"] = _result_url(res_id)
                                await push_history(query.from_user.id, payload)
                    else:
                        await push_history(query.from_user.id, {"topic": topic, "path": str(path), "created_at": datetime.utcnow().isoformat()})
            except Exception:
                pass
        except Exception as e:
            await dp.bot.send_message(chat_id, f"Ошибка: {e}" if _is_ru(ui_lang) else f"Error: {e}")
            # Mark job error
            try:
                if job_id and SessionLocal is not None:
                    from sqlalchemy import update as _upd
                    async with SessionLocal() as session:
                        from .db import Job
                        await session.execute(_upd(Job).where(Job.id == job_id).values(status="error"))
                        await session.commit()
            except Exception:
                pass
        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    # ---- History command ----
    @dp.message_handler(commands=["history"])  # type: ignore
    async def cmd_history(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        parts = (message.text or "").split()
        if len(parts) > 1 and parts[1].lower() == "clear":
            # Backward compatibility: /history clear
            try:
                await clear_history(message.from_user.id)
            except Exception:
                pass
            await message.answer("История очищена." if _is_ru(ui_lang) else "History cleared.")
            return
        try:
            hist = await get_history(message.from_user.id, limit=20)
        except Exception:
            hist = []
        if not hist:
            await message.answer("История пуста." if _is_ru(ui_lang) else "No history yet.")
            return
        # Build clickable list: only link when explicitly public
        lines = []
        for it in hist:
            topic = str(it.get("topic") or "(no topic)")
            path = str(it.get("path") or "")
            url = it.get("url") or ""
            if not url and it.get("result_id"):
                # Fallback only if DB confirms not hidden
                try:
                    if SessionLocal is not None:
                        from sqlalchemy import select as _select
                        from .db import ResultDoc as _RD
                        async with SessionLocal() as _s:
                            r = await _s.execute(_select(_RD.hidden).where(_RD.id == int(it.get("result_id"))))
                            hidden_val = r.scalar_one_or_none()
                            if hidden_val is not None and int(hidden_val or 0) == 0:
                                url = _result_url(int(it.get("result_id")))
                except Exception:
                    url = ""
            if url:
                lines.append(f"• <a href='{url}'>{topic}</a>")
            else:
                lines.append(f"• {topic}")
        prefix = "История генераций (последние):\n" if _is_ru(ui_lang) else "Your recent generations:\n"
        lines.append("\n" + ("Очистить: /history_clear" if _is_ru(ui_lang) else "Clear: /history_clear"))
        await message.answer(prefix + "\n".join(lines), parse_mode=types.ParseMode.HTML, disable_web_page_preview=True)

    @dp.message_handler(commands=["history_clear"])  # type: ignore
    async def cmd_history_clear(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            await clear_history(message.from_user.id)
        except Exception:
            pass
        await message.answer("История очищена." if _is_ru(ui_lang) else "History cleared.")

    # Legacy state handler removed: fact-check choices are inline-only now

    # Legacy depth state handler removed: depth is inline-only now

    # ---- Balance and purchasing with Telegram Stars ----
    @dp.message_handler(commands=["balance"])  # type: ignore
    async def cmd_balance(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            from .kv import get_balance_kv
            bal = await get_balance_kv(message.from_user.id) if message.from_user else 0
        except Exception:
            bal = 0
        txt = (f"Баланс: {bal} кредит(ов)." if _is_ru(ui_lang) else f"Balance: {bal} credits.")
        if _stars_enabled():
            try:
                await message.answer(
                    txt + ("\nХотите купить кредиты за ⭐?" if _is_ru(ui_lang) else "\nWant to buy credits with ⭐?"),
                    reply_markup=build_buy_keyboard(ui_lang),
                )
            except Exception:
                await message.answer(txt)
        else:
            await message.answer(txt)

    @dp.message_handler(commands=["buy"])  # type: ignore
    async def cmd_buy(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if _stars_enabled():
            await message.answer(
                ("Выберите пакет кредитов:" if _is_ru(ui_lang) else "Choose a credits pack:"),
                reply_markup=build_buy_keyboard(ui_lang),
            )
        else:
            await message.answer("Покупка через ⭐ недоступна." if _is_ru(ui_lang) else "Buying with ⭐ is unavailable.")

    @dp.message_handler(commands=["pricing"])  # type: ignore
    async def cmd_pricing(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if _is_ru(ui_lang):
            await message.answer(
                "Цены:\n- Пост: 1 кредит\n- Статья: 3 кредита\n- Серия: 1×N кредитов\n- Книга: по главам\nОплата: Telegram Stars (если включено)."
            )
        else:
            await message.answer(
                "Pricing:\n- Post: 1 credit\n- Article: 3 credits\n- Series: 1×N credits\n- Book: per chapter\nPayments: Telegram Stars (if enabled)."
            )

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("buy:stars:"))  # type: ignore
    async def cb_buy_stars(query: types.CallbackQuery, state: FSMContext):
        credits_map = {"1": 1, "3": 3, "5": 5, "10": 10, "50": 50}
        parts = (query.data or "").split(":")
        pack = credits_map.get(parts[-1], 1)
        user_id = query.from_user.id
        chat_id = query.message.chat.id if query.message else user_id
        # Env gating
        import os as _os
        enable_stars = _stars_enabled()
        provider_token = _os.getenv("TELEGRAM_STARS_PROVIDER_TOKEN", "").strip()
        if not enable_stars:
            await query.answer("Not configured", show_alert=True)
            return
        try:
            prices = [LabeledPrice(label=f"Credits x{pack}", amount=pack * 200)]
            payload = f"credits={pack}&stars={pack*200}"
            title = f"Credits x{pack}"
            description = "Buy generation credits using Telegram Stars"
            await dp.bot.send_invoice(
                chat_id=chat_id,
                title=title,
                description=description,
                payload=payload,
                provider_token=provider_token,
                currency="XTR",
                prices=prices,
                start_parameter=f"buy_{pack}",
            )
            await query.answer()
        except Exception as e:
            await query.answer(str(e)[:180], show_alert=True)

    @dp.pre_checkout_query_handler(lambda q: True)  # type: ignore
    async def process_pre_checkout_query(pre_checkout_q: types.PreCheckoutQuery):
        try:
            await dp.bot.answer_pre_checkout_query(pre_checkout_q.id, ok=True)
        except Exception:
            pass

    @dp.message_handler(content_types=types.ContentTypes.SUCCESSFUL_PAYMENT)  # type: ignore
    async def got_payment(message: types.Message):
        try:
            payload = (message.successful_payment.invoice_payload or "")
            # payload format: credits=N&stars=S
            num = 0
            try:
                for kv in payload.split("&"):
                    k, v = kv.split("=", 1)
                    if k == "credits":
                        num = int(v)
            except Exception:
                num = 0
            if num > 0 and message.from_user:
                from .kv import topup_kv
                await topup_kv(message.from_user.id, num)
            try:
                st = await dp.current_state(user=message.from_user.id, chat=message.chat.id).get_data()
                ui_lang = (st.get("ui_lang") or "ru").strip()
            except Exception:
                ui_lang = "ru"
            await message.answer("Спасибо! Кредиты начислены." if _is_ru(ui_lang) else "Thank you! Credits added.")
        except Exception:
            pass

    return dp




#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
try:
    from aiogram.contrib.fsm_storage.redis import RedisStorage2  # type: ignore
except Exception:
    RedisStorage2 = None  # type: ignore
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardRemove

from utils.env import load_env_from_root
# i18n and language detection
from utils.lang import detect_lang_from_text
from services.post.generate import generate_post
from services.post_series.generate_series import generate_series
from services.article.generate_article import generate_article
from utils.slug import safe_filename_base
from .db import SessionLocal
from .bot_commands import ADMIN_IDS, SUPER_ADMIN_ID
from .credits import ensure_user_with_credits, charge_credits, charge_credits_kv, get_balance_kv_only, refund_credits, refund_credits_kv
from .kv import set_provider, get_provider, set_logs_enabled, get_logs_enabled, set_incognito, get_incognito
from .kv import set_gen_lang, get_gen_lang
from .kv import set_refine_enabled, get_refine_enabled
from .kv import push_history, get_history, clear_history, rate_allow
from .kv import chat_clear
# Chat runner
from services.chat.run import run_chat_message, build_system_prompt
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
    ChoosingGenType = State()
    ChoosingSeriesPreset = State()
    ChoosingSeriesCount = State()


class ChatStates(StatesGroup):
    Active = State()


class HistoryStates(StatesGroup):
    WaitingDeleteIds = State()


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


def build_settings_keyboard(ui_lang: str, provider: str, gen_lang: str, refine: bool, logs_enabled: bool, incognito: bool, fc_enabled: bool, fc_depth: int, *, is_admin: bool = True, is_superadmin: bool = False) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    ru = _is_ru(ui_lang)
    # Provider row (superadmin only)
    if is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Авто" if ru else "Auto") + (" ✓" if provider == "auto" else "")), callback_data="set:provider:auto"),
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
    # Refine row (superadmin only)
    if is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Редактура: вкл" if ru else "Refine: on") + (" ✓" if refine else "")), callback_data="set:refine:yes"),
            InlineKeyboardButton(text=(("Редактура: выкл" if ru else "Refine: off") + (" ✓" if not refine else "")), callback_data="set:refine:no"),
        )
    # Logs row (admins/superadmins only)
    if is_admin or is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Логи: вкл" if ru else "Logs: on") + (" ✓" if logs_enabled else "")), callback_data="set:logs:enable"),
            InlineKeyboardButton(text=(("Логи: выкл" if ru else "Logs: off") + (" ✓" if not logs_enabled else "")), callback_data="set:logs:disable"),
        )
    # Public results row
    kb.add(
        InlineKeyboardButton(text=(("Публично: да" if ru else "Public: yes") + (" ✓" if not incognito else "")), callback_data="set:incog:disable"),
        InlineKeyboardButton(text=(("Публично: нет" if ru else "Public: no") + (" ✓" if incognito else "")), callback_data="set:incog:enable"),
    )
    # Fact-check row (superadmin only); depth selection removed from settings
    if is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Факт-чекинг: вкл" if ru else "Fact-check: on") + (" ✓" if fc_enabled else "")), callback_data="set:fc_cmd:enable"),
            InlineKeyboardButton(text=(("Факт-чекинг: выкл" if ru else "Fact-check: off") + (" ✓" if not fc_enabled else "")), callback_data="set:fc_cmd:disable"),
        )
        # Depth is no longer configurable via settings (asked only during onboarding/generation)
    return kb


def build_genlang_inline(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(
            InlineKeyboardButton(text="Авто", callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"),
            InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"),
        )
    else:
        kb.add(
            InlineKeyboardButton(text="Auto", callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"),
            InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"),
        )
    return kb


def build_ui_lang_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="Русский", callback_data="set:ui_lang:ru"))
    kb.add(InlineKeyboardButton(text="English", callback_data="set:ui_lang:en"))
    return kb


def build_provider_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    # Default prompt elsewhere is RU/EN; keep button text simple
    kb.add(InlineKeyboardButton(text="Auto", callback_data="set:provider:auto"))
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


def build_gen_depth_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="1", callback_data="set:gen_depth:1"))
    kb.add(InlineKeyboardButton(text="2", callback_data="set:gen_depth:2"))
    kb.add(InlineKeyboardButton(text="3", callback_data="set:gen_depth:3"))
    return kb


def build_gentype_keyboard(ui_lang: str, *, allow_series: bool) -> InlineKeyboardMarkup:
    ru = _is_ru(ui_lang)
    kb = InlineKeyboardMarkup()
    kb.add(
        InlineKeyboardButton(text=("Пост" if ru else "Post"), callback_data="set:gentype:post"),
        InlineKeyboardButton(text=(("Серия постов" if ru else "Series of posts")), callback_data="set:gentype:series") if allow_series else InlineKeyboardButton(text=("Статья" if ru else "Article"), callback_data="set:gentype:article"),
    )
    if allow_series:
        kb.add(InlineKeyboardButton(text=("Статья" if ru else "Article"), callback_data="set:gentype:article"))
    return kb


def _stars_enabled() -> bool:
    # Enable Stars only when explicitly configured via env
    flag = os.getenv("TELEGRAM_STARS_ENABLED", "").strip().lower()
    return flag in ("1", "true", "yes")


# Removed legacy ReplyKeyboard-based builders to unify inline-only UI


def _try_build_redis_storage():
    url = os.getenv("REDIS_URL", "").strip()
    if not url or RedisStorage2 is None:
        return None
    try:
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        host = parts.hostname or "localhost"
        port = int(parts.port or 6379)
        db = 0
        try:
            # e.g. redis://:pwd@host:port/2
            if parts.path and parts.path.strip("/"):
                db = int(parts.path.strip("/"))
        except Exception:
            db = 0
        password = parts.password
        return RedisStorage2(host=host, port=port, db=db, password=password)
    except Exception:
        return None


def create_dispatcher() -> Dispatcher:
    bot = Bot(token=TELEGRAM_TOKEN)
    storage = _try_build_redis_storage() or MemoryStorage()
    dp = Dispatcher(bot, storage=storage)

    # Simple in-memory guard to avoid duplicate generation per chat
    RUNNING_CHATS: Set[int] = set()
    # Optional global concurrency limit (5–10 parallel jobs target)
    import asyncio
    _sem_capacity = int(os.getenv("BOT_PARALLEL_LIMIT", "12"))
    GLOBAL_SEMAPHORE = asyncio.Semaphore(max(1, _sem_capacity))

    @dp.message_handler(commands=["start"])  # type: ignore
    async def cmd_start(message: types.Message):
        # Mark onboarding flow active and reset settings panel state
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(
            onboarding=True,
            in_settings=False,
            fc_ready=False,
            series_mode=None,
            series_count=None,
        )
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_ui_lang_inline(),
        )

        # Update command menu: full set in private chat; admin-only extras
        try:
            if message.from_user and message.chat and message.chat.type == "private":
                is_admin = message.from_user.id in ADMIN_IDS
                # RU set (user default)
                base_ru = [
                    types.BotCommand("start", "Начать"),
                    types.BotCommand("generate", "Сгенерировать"),
                    types.BotCommand("settings", "Настройки"),
                    types.BotCommand("balance", "Баланс"),
                    types.BotCommand("pricing", "Цены"),
                    types.BotCommand("history", "История"),
                    types.BotCommand("info", "Инфо"),
                    types.BotCommand("lang", "Язык интерфейса"),
                    types.BotCommand("lang_generate", "Язык генерации"),
                    types.BotCommand("public", "Публичность"),
                ]
                # EN set (user default)
                base_en = [
                    types.BotCommand("start", "Start"),
                    types.BotCommand("generate", "Generate"),
                    types.BotCommand("settings", "Settings"),
                    types.BotCommand("balance", "Balance"),
                    types.BotCommand("pricing", "Pricing"),
                    types.BotCommand("history", "History"),
                    types.BotCommand("info", "Info"),
                    types.BotCommand("lang", "Language"),
                    types.BotCommand("lang_generate", "Gen language"),
                    types.BotCommand("public", "Public"),
                ]
                # Admin extras: chat only
                if is_admin:
                    # For admins, expose /logs explicitly
                    base_ru.insert(9, types.BotCommand("logs", "Логи генерации"))
                    base_en.insert(9, types.BotCommand("logs", "Logs"))
                    base_ru = base_ru + [
                        types.BotCommand("chat", "Чат с ИИ"),
                        types.BotCommand("endchat", "Завершить чат"),
                    ]
                    base_en = base_en + [
                        types.BotCommand("chat", "Chat with AI"),
                        types.BotCommand("endchat", "End chat"),
                    ]
                # Superadmin extras
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
                if is_superadmin:
                    base_ru.insert(9, types.BotCommand("provider", "Провайдер"))
                    base_en.insert(9, types.BotCommand("provider", "Provider"))
                    base_ru.append(types.BotCommand("depth", "Глубина"))
                    base_en.append(types.BotCommand("depth", "Depth"))
                # Cancel at the very bottom
                base_ru.append(types.BotCommand("cancel", "Отмена"))
                base_en.append(types.BotCommand("cancel", "Cancel"))
                try:
                    await dp.bot.set_my_commands(base_en, scope=types.BotCommandScopeChat(message.chat.id))
                except Exception:
                    pass
                try:
                    await dp.bot.set_my_commands(base_ru, scope=types.BotCommandScopeChat(message.chat.id), language_code="ru")
                except Exception:
                    pass
        except Exception:
            pass

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
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
                if is_superadmin:
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

        def _prov_name(p: str, ru: bool) -> str:
            key = (p or "").strip().lower()
            if key == "auto":
                return "Авто" if ru else "Auto"
            if key in {"openai","gemini","claude"}:
                return key.capitalize()
            return p or ("Авто" if ru else "Auto")

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
                "- /generate — выбрать Пост, Серию или Статью; серия: пресеты 2/5/авто/кастом.\n"
                "- /start — пройти настройку заново (язык интерфейса, язык генерации, публикация, факт‑чекинг).\n"
                "- /history — показать историю генераций, выборочно удалить или очистить всё.\n"
                "- /history_clear — очистить историю.\n"
                "- /factcheck — задать дефолт факт‑чекинга (вкл/выкл).\n"
                "- /refine — включить/выключить финальную редактуру.\n"
                "- /lang, /lang_generate, /incognito — точечно поменять настройки.\n"
                "- /chat, /endchat — чат с ИИ (для админов).\n\n"
                "<b>Результаты:</b>\n"
                f"- <a href='{RESULTS_ORIGIN}/results-ui'>Список всех результатов</a>\n"
                "- Каждый результат — Markdown‑файл, который можно скачать и править; серия — один агрегатный .md.\n\n"
                "<b>Кредиты:</b>\n"
                "- Пост: 1 кредит; факт‑чек +1 (лёгкий) или +3 (глубокий ≥3); редактура +1.\n"
                "- Серия: предоплата min(30, баланс) в авто или N×стоимость в fixed; остаток возвращаем.\n"
                "- Посмотреть баланс: /balance. Пополнить: /buy.\n"
                "- /pricing — посмотреть цены по типам.\n\n"
                ""
                "<b>Текущие настройки:</b>\n"
                f"- Провайдер: {_prov_name(prov, True)}\n"
                f"- Язык генерации: {_lang_human(gen_lang, True)}\n"
                f"- Публичные результаты: {'да' if not incognito else 'нет'}\n"
                # logs removed from user info
                f"- Финальная редактура: {'включена' if refine_enabled else 'отключена'}\n"
                f"- Факт‑чекинг: {'включён' if fc_enabled else 'отключён'}"
                + "\n\n<a href='https://github.com/YanGranat/book-in-one-click'>GitHub проекта</a>"
            )
        else:
            text = (
                "<b>How to use:</b>\n"
                "- /generate — choose Post, Series or Article; series: presets 2/5/auto/custom.\n"
                "- /start — onboarding (UI lang, gen lang, publishing, fact‑check).\n"
                "- /history — show history, delete selected or clear all.\n"
                "- /history_clear — clear history.\n"
                "- /factcheck — set default fact‑check (enable/disable).\n"
                "- /refine — enable/disable final refine step.\n"
                "- /lang, /lang_generate, /incognito — tweak settings individually.\n"
                "- /chat, /endchat — chat with AI (admins).\n\n"
                "<b>Results:</b>\n"
                f"- <a href='{RESULTS_ORIGIN}/results-ui'>All results page</a>\n"
                "- Each result is a Markdown file; series arrive as one aggregate .md.\n\n"
                "<b>Credits:</b>\n"
                "- Post: 1 credit; fact‑check +1 (light) or +3 (deep ≥3); refine +1.\n"
                "- Series: prepay min(30, balance) in auto or N×price in fixed; refund remainder.\n"
                "- Check balance: /balance. Buy: /buy.\n"
                "- /pricing — see pricing per type.\n\n"
                ""
                "<b>Current settings:</b>\n"
                f"- Provider: {_prov_name(prov, False)}\n"
                f"- Generation language: {_lang_human(gen_lang, False)}\n"
                f"- Public results: {'yes' if not incognito else 'no'}\n"
                # logs removed from user info
                f"- Final refine: {'enabled' if refine_enabled else 'disabled'}\n"
                f"- Fact‑check: {'enabled' if fc_enabled else 'disabled'}"
                + "\n\n<a href='https://github.com/YanGranat/book-in-one-click'>Project GitHub</a>"
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            from .bot_commands import SUPER_ADMIN_ID
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang_cur, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Onboarding continues: after generation language → provider (superadmin only); others → logs
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
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
                is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
                if is_superadmin:
                    prompt = "Выберите провайдера:" if _is_ru(ui_lang) else "Choose provider:"
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_provider_inline())
                elif is_admin_local:
                    # Admins: still offer logs toggle
                    prompt = ("Отправлять логи генерации?" if _is_ru(ui_lang) else "Send generation logs?")
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_enable_disable_inline("logs", ui_lang))
                else:
                    # Regular users: skip logs question → go directly to public results question
                    prompt = ("Сделать результаты публичными?" if _is_ru(ui_lang) else "Make results public?")
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("incog", ui_lang))

    @dp.message_handler(commands=["provider"])  # type: ignore
    async def cmd_provider(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        if not is_superadmin:
            await message.answer("Недоступно." if ui_lang == "ru" else "Not available.")
            return
        prompt = "Выберите провайдера:" if ui_lang == "ru" else "Choose provider:"
        await message.answer(prompt, reply_markup=build_provider_inline())

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:provider:"))  # type: ignore
    async def cb_set_provider(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID)):
            await query.answer()
            return
        prov = (query.data or "").split(":")[-1]
        # Accept 'auto' as a stored preference; runtime mapping happens at call sites
        if prov not in {"auto", "openai", "gemini", "claude"}:
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            from .bot_commands import SUPER_ADMIN_ID
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Onboarding continues: after provider (superadmin only)
            await query.message.edit_reply_markup() if query.message else None
            ok = "Провайдер установлен." if ui_lang == "ru" else "Provider set."
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ok)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                # After provider (superadmin), go directly to public results question
                prompt = ("Сделать результаты публичными?" if _is_ru(ui_lang) else "Make results public?")
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("incog", ui_lang))

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
        # Ask what to generate; users cannot generate series
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        allow_series = bool(is_admin or is_superadmin)
        ru = _is_ru(ui_lang)
        kb = build_gentype_keyboard(ui_lang, allow_series=allow_series)
        await message.answer(("Что генерировать?" if ru else "What to generate?"), reply_markup=kb)
        await GenerateStates.ChoosingGenType.set()
    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:gentype:"), state=GenerateStates.ChoosingGenType)  # type: ignore
    async def cb_gen_type(query: types.CallbackQuery, state: FSMContext):
        kind = (query.data or "").split(":")[-1]
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        await query.answer()
        # For non-admins, block series
        is_admin = bool(query.from_user and query.from_user.id in ADMIN_IDS)
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
        if kind == "series" and not (is_admin or is_superadmin):
            # silently ignore or send message
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Недоступно." if ru else "Not available."))
            return
        if kind == "post":
            # Superadmin: ask FC (with depth), then refine, then topic
            if is_superadmin:
                await state.update_data(series_mode=None, series_count=None, active_flow="post", next_after_fc="post")
                prompt = "Включить факт-чекинг?" if ru else "Enable fact-checking?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("fc", ui_lang))
                return
            # Admins/users: ask topic directly
            await state.update_data(series_mode=None, series_count=None)
            prompt = "Отправьте тему для поста:" if ru else "Send a topic for your post:"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=ReplyKeyboardRemove())
            await GenerateStates.WaitingTopic.set()
            return
        if kind == "article":
            await state.update_data(series_mode=None, series_count=None, gen_article=True)
            prompt = "Отправьте тему для статьи:" if ru else "Send a topic for your article:"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=ReplyKeyboardRemove())
            await GenerateStates.WaitingTopic.set()
            return
        # Series branch
        if is_superadmin:
            # Ask FC (with depth), then refine, then how many posts, then topic
            await state.update_data(series_mode=None, series_count=None, active_flow="series", next_after_fc="series")
            prompt = "Включить факт-чекинг?" if ru else "Enable fact-checking?"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("fc", ui_lang))
            return
        # Admins: ask preset 2/5/auto/custom immediately
        kb = InlineKeyboardMarkup()
        kb.add(
            InlineKeyboardButton(text="2", callback_data="set:series_preset:2"),
            InlineKeyboardButton(text="5", callback_data="set:series_preset:5"),
        )
        kb.add(
            InlineKeyboardButton(text=("Авто" if ru else "Auto"), callback_data="set:series_preset:auto"),
            InlineKeyboardButton(text=("Кастом" if ru else "Custom"), callback_data="set:series_preset:custom"),
        )
        await state.update_data(series_mode=None, series_count=None)
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Сколько постов?" if ru else "How many posts?"), reply_markup=kb)
        await GenerateStates.ChoosingSeriesPreset.set()

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:series_preset:"), state=GenerateStates.ChoosingSeriesPreset)  # type: ignore
    async def cb_series_preset(query: types.CallbackQuery, state: FSMContext):
        val = (query.data or "").split(":")[-1]
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        await query.answer()
        if val == "auto":
            await state.update_data(series_mode="auto", series_count=None)
            prompt = ("Отправьте тему для серии (авто)." if ru else "Send a topic for the series (auto).")
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt)
            await GenerateStates.WaitingTopic.set()
            return
        if val in {"2","5"}:
            await state.update_data(series_mode="fixed", series_count=int(val))
            prompt = (f"Отправьте тему для серии (ровно {val})." if ru else f"Send a topic for the series (exactly {val}).")
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt)
            await GenerateStates.WaitingTopic.set()
            return
        # custom
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Введите число постов:" if ru else "Enter number of posts:"))
        await GenerateStates.ChoosingSeriesCount.set()

    @dp.message_handler(state=GenerateStates.ChoosingSeriesCount, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def series_custom_count(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        txt = (message.text or "").strip()
        try:
            n = int(txt)
        except Exception:
            await message.answer(("Введите целое число > 0:" if ru else "Enter integer > 0:"))
            return
        if n <= 0:
            await message.answer(("Введите целое число > 0:" if ru else "Enter integer > 0:"))
            return
        await state.update_data(series_mode="fixed", series_count=int(n))
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if is_admin:
            await message.answer(("Ок. Отправьте тему для серии." if ru else "OK. Send a topic for the series."))
            await GenerateStates.WaitingTopic.set()
            return
        # Compute unit cost based on prefs
        try:
            refine_pref = await get_refine_enabled(message.from_user.id) if message.from_user else False
        except Exception:
            refine_pref = False
        try:
            fc_enabled_pref = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
        except Exception:
            fc_enabled_pref = False
        try:
            fc_depth_pref = await get_factcheck_depth(message.from_user.id) if message.from_user else 2
        except Exception:
            fc_depth_pref = 2
        fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
        unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
        total = unit_cost * int(n)
        kb = InlineKeyboardMarkup()
        kb.add(
            InlineKeyboardButton(text=("Подтвердить" if ru else "Confirm"), callback_data="confirm:series_custom:yes"),
            InlineKeyboardButton(text=("Отмена" if ru else "Cancel"), callback_data="confirm:series_custom:no"),
        )
        await state.update_data(pending_series_count=int(n), pending_series_cost=int(total))
        await message.answer((f"Будет списано {total} кредит(ов). Начать?" if ru else f"It will cost {total} credits. Start?"), reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data in {"confirm:series_custom:yes","confirm:series_custom:no"}, state=GenerateStates.ChoosingSeriesCount)  # type: ignore
    async def cb_series_custom_confirm(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        if query.data.endswith(":no"):
            await query.answer()
            await query.message.edit_reply_markup() if query.message else None
            # Silent cancel: no extra messages
            await state.finish()
            return
        # Precharge now
        total = int(data.get("pending_series_cost") or 0)
        count = int(data.get("pending_series_count") or 0)
        if total <= 0 or count <= 0:
            await query.answer()
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Ошибка параметров." if ru else "Invalid parameters."))
            await state.finish()
            return
        from sqlalchemy.exc import SQLAlchemyError
        try:
            charged = False
            if SessionLocal is not None and query.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, query.from_user.id)
                    ok, remaining = await charge_credits(session, user, int(total), reason="post_series_fixed_prepay")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged and query.from_user:
                ok, remaining = await charge_credits_kv(query.from_user.id, int(total))
                if not ok:
                    await query.answer()
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Недостаточно кредитов" if ru else "Insufficient credits"))
                    await state.finish()
                    return
        except SQLAlchemyError:
            await query.answer()
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Временная ошибка БД. Попробуйте позже." if ru else "Temporary DB error. Try later."))
            await state.finish()
            return
        # Mark precharged and ask for topic
        await state.update_data(series_mode="fixed", series_count=int(count), series_precharged_amount=int(total))
        await query.answer()
        await query.message.edit_reply_markup() if query.message else None
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Ок. Отправьте тему для серии." if ru else "OK. Send a topic for the series."))
        await GenerateStates.WaitingTopic.set()

    # ---- Series command ----
    @dp.message_handler(commands=["series"])  # type: ignore
    async def cmd_series(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        txt = (
            "Отправьте тему для серии постов (авторежим, максимум 30 кредитов).\n"
            "Используйте /series_fixed N — чтобы задать точное число постов."
            if _is_ru(ui_lang) else
            "Send a topic for a series (auto mode, up to 30 credits).\nUse /series_fixed N to request exact number of posts."
        )
        await message.answer(txt)
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(series_mode="auto")
        await GenerateStates.WaitingTopic.set()

    @dp.message_handler(commands=["series_fixed"])  # type: ignore
    async def cmd_series_fixed(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        parts = (message.text or "").split()
        n = 0
        if len(parts) >= 2:
            try:
                n = int(parts[1])
            except Exception:
                n = 0
        if n <= 0:
            await message.answer("Укажите число постов: /series_fixed 8" if _is_ru(ui_lang) else "Specify posts count: /series_fixed 8")
            return
        await message.answer((f"Отправьте тему для серии (ровно {n} постов)." if _is_ru(ui_lang) else f"Send a topic (exactly {n} posts)."))
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(series_mode="fixed", series_count=n)
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
        title = ("Настройки" if _is_ru(ui_lang) else "Settings")
        await state.update_data(in_settings=True)
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        await message.answer(title, reply_markup=build_settings_keyboard(ui_lang, prov, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin, is_superadmin=is_superadmin))

    @dp.message_handler(commands=["logs"])  # type: ignore
    async def cmd_logs(message: types.Message, state: FSMContext):
        # Admin/superadmin only; hide for regular users
        from .bot_commands import SUPER_ADMIN_ID
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        if not (is_admin or is_superadmin):
            return
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Onboarding continues: after logs → incognito toggle
            msg = "Логи включены." if (enabled and ui_lang=="ru") else ("Logs enabled." if enabled else ("Логи отключены." if ui_lang=="ru" else "Logs disabled."))
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                prompt = (
                    "Сделать результаты публичными?"
                    if _is_ru(ui_lang)
                    else "Make results public?"
                )
                await dp.bot.send_message(
                    query.message.chat.id if query.message else query.from_user.id,
                    prompt,
                    reply_markup=build_yesno_inline("incog", ui_lang),
                )

    @dp.message_handler(commands=["public"])  # type: ignore
    async def cmd_incognito(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = (
            "Сделать результаты публичными?"
            if ui_lang == "ru"
            else "Make results public?"
        )
        await message.answer(prompt, reply_markup=build_yesno_inline("incog", ui_lang))

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
            is_admin = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            # Affirmative/negative public state
            msg = (
                ("Публичные результаты: да." if ui_lang=="ru" else "Public results: yes.")
                if not enabled  # enabled means incognito ON -> public NO
                else ("Публичные результаты: нет." if ui_lang=="ru" else "Public results: no.")
            )
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding_flag = bool((await state.get_data()).get("onboarding"))
            if onboarding_flag:
                # After incognito decision → ask what to generate (role-based series availability)
                is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
                kb = build_gentype_keyboard(ui_lang, allow_series=bool(is_admin_local or is_superadmin))
                await dp.bot.send_message(
                    query.message.chat.id if query.message else query.from_user.id,
                    ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"),
                    reply_markup=kb,
                )
                await GenerateStates.ChoosingGenType.set()

    @dp.message_handler(commands=["refine"])  # type: ignore
    async def cmd_refine(message: types.Message, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID)):
            await message.answer("Недоступно.")
            return
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
        await message.answer(prompt, reply_markup=build_yesno_inline("refine", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:refine:"))  # type: ignore
    async def cb_set_refine(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID)):
            await query.answer()
            return
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            msg = "Финальная редактура: включена." if (enabled and ui_lang=="ru") else ("Final refine: enabled." if enabled else ("Финальная редактура: отключена." if ui_lang=="ru" else "Final refine: disabled."))
            await query.message.edit_reply_markup() if query.message else None
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding and (query.from_user and query.from_user.id in ADMIN_IDS):
                # After refine in onboarding → ask fact-check (admins only)
                prompt = "Включить факт-чекинг?" if _is_ru(ui_lang) else "Enable fact-checking?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("fc", ui_lang))

    @dp.message_handler(lambda m: (m.text or "").strip().lower() in {"cancel","отмена"}, state="*")  # type: ignore
    @dp.message_handler(commands=["cancel"], state="*")  # type: ignore
    async def cmd_cancel(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Silent cancel: no message text
        try:
            # Ensure any running job gates are released for this chat
            RUNNING_CHATS.discard(message.chat.id)
        except Exception:
            pass
        # Reset transient flags
        try:
            await state.update_data(
                onboarding=False,
                in_settings=False,
                fc_ready=False,
                series_mode=None,
                series_count=None,
                pending_topic=None,
            )
        except Exception:
            pass
        # Reset to neutral state and prompt minimal hint so next command works
        try:
            await state.reset_state(with_data=False)
        except Exception:
            try:
                await state.finish()
            except Exception:
                pass
        try:
            await message.edit_reply_markup()  # best-effort remove keyboard if applicable
        except Exception:
            pass
        # Single cancel message
        try:
            done = "Отменено." if _is_ru(ui_lang) else "Cancelled."
            await message.answer(done, reply_markup=ReplyKeyboardRemove())
        except Exception:
            pass

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc:"))  # type: ignore
    async def cb_set_fc(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        is_super = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
        if not query.from_user or (query.from_user.id not in ADMIN_IDS and not is_super):
            await query.answer()
            return
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
        # Always answer callback first to avoid long spinner in Telegram
        try:
            await query.answer()
        except Exception:
            pass
        # Best-effort attempt to remove the inline keyboard; ignore edit errors
        try:
            if query.message:
                await query.message.edit_reply_markup()
        except Exception:
            pass
        # Small acknowledgment to show immediate feedback
        try:
            ack = (
                ("Факт-чекинг: включён." if _is_ru(ui_lang) else "Fact-check: enabled.")
                if enabled else
                ("Факт-чекинг: отключён." if _is_ru(ui_lang) else "Fact-check: disabled.")
            )
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ack)
        except Exception:
            pass
        onboarding = bool((await state.get_data()).get("onboarding"))
        if enabled:
            # Depth selection only for superadmin
            from .bot_commands import SUPER_ADMIN_ID
            if query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID):
                prompt = "Выберите глубину проверки (1–3):" if _is_ru(ui_lang) else "Select research depth (1–3):"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_depth_inline())
            else:
                await state.update_data(factcheck=True, research_iterations=2, fc_ready=True)
                # Continue onboarding if applicable
                onboarding2 = bool((await state.get_data()).get("onboarding"))
                if onboarding2:
                    kb = build_gentype_keyboard(ui_lang, allow_series=False)
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"), reply_markup=kb)
                    await GenerateStates.ChoosingGenType.set()
        else:
            # Mark FC decision as done in onboarding to avoid asking again on topic
            await state.update_data(factcheck=False, research_iterations=None, fc_ready=True)
            if onboarding:
                sd = await state.get_data()
                active_flow = (sd.get("active_flow") or "").strip().lower()
                if active_flow in {"post", "series"}:
                    # Proceed to refine question in flow
                    prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))
                else:
                    # After disabling FC in onboarding → ask what to generate (series disabled)
                    kb = build_gentype_keyboard(ui_lang, allow_series=False)
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"), reply_markup=kb)
                    # Ensure callback is routed correctly
                    await GenerateStates.ChoosingGenType.set()

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:depth:"))  # type: ignore
    async def cb_set_depth(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID)):
            await query.answer()
            return
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            from .bot_commands import SUPER_ADMIN_ID
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            onboarding = bool(data.get("onboarding"))
            if onboarding:
                sd = await state.get_data()
                active_flow = (sd.get("active_flow") or "").strip().lower()
                if active_flow == "post":
                    # Next ask refine for post
                    prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))
                elif active_flow == "series":
                    # Next ask refine for series, then proceed to count
                    prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))
                else:
                    # Fallback: what to generate
                    kb = build_gentype_keyboard(ui_lang, allow_series=False)
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"), reply_markup=kb)
                    await GenerateStates.ChoosingGenType.set()
            else:
                # Standalone /factcheck flow: confirm and stay
                msg = "Глубина факт-чекинга сохранена." if _is_ru(ui_lang) else "Fact-check depth saved."
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)

    # ---- Fact-check settings command ----
    @dp.message_handler(commands=["factcheck"])  # type: ignore
    async def cmd_factcheck(message: types.Message, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID)):
            await message.answer("Недоступно.")
            return
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Факт-чекинг?" if _is_ru(ui_lang) else "Fact-check?"
        await message.answer(prompt, reply_markup=build_enable_disable_inline("fc_cmd", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc_cmd:"))  # type: ignore
    async def cb_fc_cmd_toggle(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        if not (query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID)):
            await query.answer()
            return
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
            is_admin_local = bool(query.from_user and query.from_user.id in ADMIN_IDS)
            from .bot_commands import SUPER_ADMIN_ID
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
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
        from .bot_commands import SUPER_ADMIN_ID
        if not (message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID)):
            await message.answer("Недоступно." if ((await state.get_data()).get("ui_lang") or "ru") == "ru" else "Not available.")
            return
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
            try:
                RUNNING_CHATS.discard(message.chat.id)
            except Exception:
                pass
            try:
                await state.update_data(pending_topic=None)
            except Exception:
                pass
            try:
                await message.edit_reply_markup()
            except Exception:
                pass
            # Single cancel message
            try:
                done = "Отменено." if _is_ru(ui_lang) else "Cancelled."
                await message.answer(done, reply_markup=ReplyKeyboardRemove())
            except Exception:
                pass
            await state.finish()
            return
        topic = text_raw
        if not topic:
            msg = "Тема не может быть пустой. Отправьте тему:" if ui_lang == "ru" else "Topic cannot be empty. Send a topic:"
            await message.answer(msg)
            return
        await state.update_data(topic=topic)

        # If article mode is active, branch into article generation flow
        if bool(data.get("gen_article")):
            chat_id = message.chat.id
            if chat_id in RUNNING_CHATS:
                return
            RUNNING_CHATS.add(chat_id)
            try:
                # Resolve provider/lang
                prov = (data.get("provider") or "openai").strip().lower()
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

                # Create Job row (article)
                job_id = 0
                try:
                    if SessionLocal is not None and message.from_user:
                        async with SessionLocal() as session:
                            from .db import Job
                            import json as _json
                            from .db import get_or_create_user as _get_or_create_user
                            db_user = await _get_or_create_user(session, message.from_user.id)
                            params = {"topic": topic, "lang": eff_lang, "provider": prov or "openai"}
                            j = Job(user_id=db_user.id, type="article", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=3)
                            session.add(j)
                            await session.flush()
                            job_id = int(j.id)
                            await session.commit()
                except Exception:
                    job_id = 0

                # Charge 3 credits for article (admins free)
                is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
                if not is_admin:
                    from sqlalchemy.exc import SQLAlchemyError
                    try:
                        charged = False
                        if SessionLocal is not None and message.from_user:
                            async with SessionLocal() as session:
                                from .db import get_or_create_user
                                user = await get_or_create_user(session, message.from_user.id)
                                ok, remaining = await charge_credits(session, user, 3, reason="article")
                                if ok:
                                    await session.commit()
                                    charged = True
                        if not charged and message.from_user:
                            ok, remaining = await charge_credits_kv(message.from_user.id, 3)  # type: ignore
                            if not ok:
                                need = max(0, 3 - int(remaining))
                                warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /buy для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /buy to top up.")
                                await message.answer(warn)
                                await state.finish()
                                RUNNING_CHATS.discard(chat_id)
                                return
                    except SQLAlchemyError:
                        await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                        await state.finish()
                        RUNNING_CHATS.discard(chat_id)
                        return

                await message.answer("Генерирую статью…" if _is_ru(ui_lang) else "Generating article…")
                import asyncio as _asyncio
                loop = _asyncio.get_running_loop()
                # 10 часов по умолчанию (можно переопределить GEN_TIMEOUT_S)
                timeout_s = int(os.getenv("GEN_TIMEOUT_S", "36000"))
                # Resolve incognito flag upfront (cannot use await inside lambda)
                try:
                    inc_flag = (await get_incognito(message.from_user.id)) if message.from_user else False
                except Exception:
                    inc_flag = False

                # Resolve feature flags upfront (cannot use await inside lambda)
                try:
                    fc_flag = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    fc_flag = False
                try:
                    refine_flag = await get_refine_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    refine_flag = False
                # Only superadmin can run article fact-check/refine
                try:
                    is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
                except Exception:
                    is_superadmin = False
                if not is_superadmin:
                    fc_flag = False
                    refine_flag = False

                fut = loop.run_in_executor(
                    None,
                    lambda: generate_article(
                        topic,
                        lang=eff_lang,
                        provider=((prov if prov != "auto" else "openai") or "openai"),
                        output_subdir="deep_article",
                        job_meta={"user_id": message.from_user.id if message.from_user else 0, "chat_id": message.chat.id, "job_id": job_id, "incognito": inc_flag},
                        enable_research=bool(fc_flag),
                        enable_refine=bool(refine_flag),
                    ),
                )
                try:
                    article_path = await _asyncio.wait_for(fut, timeout=timeout_s)
                except _asyncio.TimeoutError:
                    warn = (
                        f"Превышено время ожидания ({int(timeout_s/60)} мин). Генерация продолжается в фоне; проверьте /results-ui позже."
                        if _is_ru(ui_lang) else
                        f"Timeout ({int(timeout_s/60)} min). Generation continues in background; check /results-ui later."
                    )
                    await message.answer(warn)
                    await state.finish()
                    RUNNING_CHATS.discard(chat_id)
                    return
                # Send main result
                try:
                    with open(article_path, "rb") as f:
                        cap = ("Готово (статья): " + Path(article_path).name) if _is_ru(ui_lang) else ("Done (article): " + Path(article_path).name)
                        await message.answer_document(f, caption=cap)
                except Exception:
                    pass
                # Send log if enabled
                try:
                    if message.from_user:
                        logs_enabled = await get_logs_enabled(message.from_user.id)
                        if logs_enabled:
                            from utils.slug import safe_filename_base as _sfb
                            base = _sfb(topic)
                            log_files = list(Path(article_path).parent.glob(f"{base}_article_log_*.md"))
                            if log_files:
                                lp = log_files[0]
                                with open(lp, "rb") as log_f:
                                    log_cap = f"Лог статьи: {lp.name}" if _is_ru(ui_lang) else f"Article log: {lp.name}"
                                    await message.answer_document(log_f, caption=log_cap)
                except Exception:
                    pass
                # Mark job done
                try:
                    if job_id and SessionLocal is not None:
                        from sqlalchemy import update as _upd
                        from datetime import datetime as _dt
                        async with SessionLocal() as session:
                            from .db import Job
                            await session.execute(_upd(Job).where(Job.id == job_id).values(status="done", finished_at=_dt.utcnow(), file_path=str(article_path)))
                            await session.commit()
                except Exception:
                    pass
            except Exception as e:
                await message.answer((f"Ошибка: {e}" if _is_ru(ui_lang) else f"Error: {e}"))
            finally:
                await state.finish()
                RUNNING_CHATS.discard(chat_id)
            return

        # If series mode is active, branch into series generation flow
        series_mode = (data.get("series_mode") or "").strip().lower()
        if series_mode in {"auto", "fixed"}:
            chat_id = message.chat.id
            if chat_id in RUNNING_CHATS:
                return
            RUNNING_CHATS.add(chat_id)
            try:
                # Resolve provider and language
                prov = (data.get("provider") or "").strip().lower()
                if not prov and message.from_user:
                    try:
                        prov = await get_provider(message.from_user.id)  # type: ignore
                    except Exception:
                        prov = "openai"
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

                # Preferences
                try:
                    refine_pref = await get_refine_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    refine_pref = False
                try:
                    fc_enabled_pref = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    fc_enabled_pref = False
                try:
                    fc_depth_pref = await get_factcheck_depth(message.from_user.id) if message.from_user else 2
                except Exception:
                    fc_depth_pref = 2
                # Pricing
                fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
                unit_cost = 1 + fc_extra + (1 if refine_pref else 0)

                is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)

                # Precharge logic
                precharged = 0
                target_count = 0
                if series_mode == "fixed":
                    series_count = int(data.get("series_count") or 0)
                    if series_count <= 0:
                        await message.answer("Некорректное число постов." if _is_ru(ui_lang) else "Invalid posts count.")
                        await state.finish()
                        RUNNING_CHATS.discard(chat_id)
                        return
                    target_count = series_count
                    total_cost = 0 if is_admin else series_count * unit_cost
                    if not is_admin:
                        # Charge exact cost
                        from sqlalchemy.exc import SQLAlchemyError
                        try:
                            if SessionLocal is not None and message.from_user:
                                async with SessionLocal() as session:
                                    user = await ensure_user_with_credits(session, message.from_user.id)
                                    ok, remaining = await charge_credits(session, user, int(total_cost), reason="post_series_fixed_prepay")
                                    if not ok:
                                        await message.answer(("Недостаточно кредитов. Используйте /buy для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /buy to top up."))
                                        await state.finish()
                                        RUNNING_CHATS.discard(chat_id)
                                        return
                                    await session.commit()
                                    precharged = int(total_cost)
                            else:
                                ok, remaining = await charge_credits_kv(message.from_user.id, int(total_cost))  # type: ignore
                                if not ok:
                                    await message.answer(("Недостаточно кредитов. Используйте /buy для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /buy to top up."))
                                    await state.finish()
                                    RUNNING_CHATS.discard(chat_id)
                                    return
                                precharged = int(total_cost)
                        except SQLAlchemyError:
                            await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                            await state.finish()
                            RUNNING_CHATS.discard(chat_id)
                            return
                else:  # auto
                    if is_admin:
                        target_count = 0  # unlimited for admin
                    else:
                        # Determine prepay budget = min(30, balance)
                        balance = 0
                        if SessionLocal is not None and message.from_user:
                            try:
                                async with SessionLocal() as session:
                                    from .db import get_or_create_user
                                    user = await get_or_create_user(session, message.from_user.id)
                                    balance = int(user.credits)
                            except Exception:
                                balance = 0
                        if balance <= 0 and message.from_user:
                            try:
                                balance = int(await get_balance_kv_only(message.from_user.id))
                            except Exception:
                                balance = 0
                        prepay_budget = min(30, balance)
                        if prepay_budget <= 0:
                            await message.answer(("Недостаточно кредитов. Используйте /buy для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /buy to top up."))
                            await state.finish()
                            RUNNING_CHATS.discard(chat_id)
                            return
                        # Compute max posts affordable with current options
                        if unit_cost <= 0:
                            unit_cost = 1
                        target_count = prepay_budget // unit_cost
                        if target_count <= 0:
                            await message.answer(
                                ("Недостаточно кредитов для текущих настроек (стоимость поста слишком высока). Отключите факт‑чек/редактуру или пополните баланс."
                                 if _is_ru(ui_lang) else
                                 "Insufficient credits for current settings (post cost too high). Disable fact‑check/refine or top up.")
                            )
                            await state.finish()
                            RUNNING_CHATS.discard(chat_id)
                            return
                        # Precharge full budget (30 or balance)
                        from sqlalchemy.exc import SQLAlchemyError
                        try:
                            if SessionLocal is not None and message.from_user:
                                async with SessionLocal() as session:
                                    from .db import get_or_create_user
                                    user = await get_or_create_user(session, message.from_user.id)
                                    ok, remaining = await charge_credits(session, user, int(prepay_budget), reason="post_series_auto_prepay")
                                    if not ok:
                                        await message.answer(("Недостаточно кредитов. Используйте /buy для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /buy to top up."))
                                        await state.finish()
                                        RUNNING_CHATS.discard(chat_id)
                                        return
                                    await session.commit()
                                    precharged = int(prepay_budget)
                            else:
                                ok, remaining = await charge_credits_kv(message.from_user.id, int(prepay_budget))  # type: ignore
                                if not ok:
                                    await message.answer(("Недостаточно кредитов. Используйте /buy для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /buy to top up."))
                                    await state.finish()
                                    RUNNING_CHATS.discard(chat_id)
                                    return
                                precharged = int(prepay_budget)
                        except SQLAlchemyError:
                            await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                            await state.finish()
                            RUNNING_CHATS.discard(chat_id)
                            return

                # Create Job row (series)
                job_id = 0
                try:
                    if SessionLocal is not None and message.from_user:
                        async with SessionLocal() as session:
                            from .db import Job
                            import json as _json
                            params = {
                                "topic": topic,
                                "lang": eff_lang,
                                "provider": prov or "openai",
                                "mode": series_mode,
                                "count": int(target_count or 0),
                                "factcheck": bool(fc_enabled_pref),
                                "depth": int(fc_depth_pref or 0) if fc_enabled_pref else 0,
                                "refine": bool(refine_pref),
                            }
                            # Normalize to internal User.id for consistency with single-post flow
                            from .db import get_or_create_user as _get_or_create_user
                            db_user = await _get_or_create_user(session, message.from_user.id)
                            j = Job(user_id=db_user.id, type="post_series", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=int(precharged))
                            session.add(j)
                            await session.flush()
                            job_id = int(j.id)
                            await session.commit()
                except Exception:
                    job_id = 0

                # Run series generation (sequential)
                import asyncio as _asyncio
                loop = _asyncio.get_running_loop()
                await message.answer("Генерирую серию…" if _is_ru(ui_lang) else "Generating series…")
                from services.post_series.generate_series import generate_series as _gen_series
                timeout_s = int(os.getenv("GEN_TIMEOUT_S", "1800"))
                mode_param = ("auto" if is_admin and series_mode == "auto" else ("fixed" if target_count else series_mode))
                count_param = (0 if mode_param == "auto" else int(target_count or 0))
                fut = loop.run_in_executor(
                    None,
                    lambda: _gen_series(
                        topic,
                        lang=eff_lang,
                        provider=((prov if prov != "auto" else "openai") or "openai"),
                        mode=mode_param,
                        count=count_param,
                        max_iterations=int(os.getenv("SERIES_MAX_ITER", "1")),
                        sufficiency_heavy_after=3,
                        output_mode="single",
                        output_subdir="post_series",
                        factcheck=bool(fc_enabled_pref),
                        research_iterations=int(fc_depth_pref or 2),
                        refine=bool(refine_pref),
                        job_meta={"user_id": message.from_user.id if message.from_user else 0, "chat_id": message.chat.id, "job_id": job_id},
                    ),
                )
                aggregate_path = await _asyncio.wait_for(fut, timeout=timeout_s)

                # Send aggregate result
                try:
                    with open(aggregate_path, "rb") as f:
                        cap = ("Готово (серия): " + Path(aggregate_path).name) if _is_ru(ui_lang) else ("Done (series): " + Path(aggregate_path).name)
                        await message.answer_document(f, caption=cap)
                except Exception:
                    pass

                # Try compute number of posts generated for refund calculation
                n_generated = 0
                try:
                    import re as _re
                    text = Path(aggregate_path).read_text(encoding="utf-8")
                    # Prefer strict pattern for post sections like: '## t01. Title'
                    m = _re.findall(r"(?m)^##\s+[tT]\d+\.", text)
                    n_generated = len(m)
                    if n_generated == 0:
                        # Fallback: count generic '## ' and subtract header '## Список тем' if present
                        n_generic = sum(1 for line in text.splitlines() if line.strip().startswith("## "))
                        n_generated = max(0, n_generic - 1)
                except Exception:
                    n_generated = int(target_count or 0)

                # Send series log if user enabled logs
                try:
                    if message.from_user:
                        logs_enabled = await get_logs_enabled(message.from_user.id)
                        if logs_enabled:
                            try:
                                from utils.slug import safe_filename_base as _sfb
                            except Exception:
                                _sfb = lambda s: (s or "topic").lower().replace(" ", "_")
                            base = _sfb(topic)
                            log_files = list(Path(aggregate_path).parent.glob(f"{base}_series_log_*.md"))
                            if log_files:
                                lp = log_files[0]
                                with open(lp, "rb") as log_f:
                                    log_cap = f"Лог серии: {lp.name}" if _is_ru(ui_lang) else f"Series log: {lp.name}"
                                    await message.answer_document(log_f, caption=log_cap)
                except Exception:
                    pass

                # Refund unused credits
                if not is_admin and precharged > 0:
                    used = int((n_generated or 0) * unit_cost)
                    refund = max(0, int(precharged) - used)
                    if refund > 0 and message.from_user:
                        try:
                            if SessionLocal is not None:
                                async with SessionLocal() as session:
                                    from .db import get_or_create_user
                                    user = await get_or_create_user(session, message.from_user.id)
                                    await refund_credits(session, user, refund, reason="post_series_refund")
                                    await session.commit()
                            else:
                                await refund_credits_kv(message.from_user.id, refund)  # type: ignore
                        except Exception:
                            pass

                # Mark job done
                try:
                    if job_id and SessionLocal is not None:
                        from sqlalchemy import update as _upd
                        from datetime import datetime as _dt
                        async with SessionLocal() as session:
                            from .db import Job
                            await session.execute(_upd(Job).where(Job.id == job_id).values(status="done", finished_at=_dt.utcnow(), file_path=str(aggregate_path)))
                            await session.commit()
                except Exception:
                    pass

            except Exception as e:
                await message.answer((f"Ошибка: {e}" if _is_ru(ui_lang) else f"Error: {e}"))
            finally:
                await state.finish()
                RUNNING_CHATS.discard(chat_id)
            return
        # Decide whether to ask fact-check during onboarding only; otherwise use defaults and start
        onboarding = bool(data.get("onboarding"))
        fc_ready = bool(data.get("fc_ready"))
        if onboarding and not fc_ready:
            # Ask FC only for superadmin; others proceed without FC
            from .bot_commands import SUPER_ADMIN_ID
            is_super = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
            if is_super:
                prompt = "Включить факт-чекинг?" if _is_ru(ui_lang) else "Enable fact-checking?"
                await message.answer(prompt, reply_markup=build_yesno_inline("fc", ui_lang))
                return
            else:
                try:
                    await state.update_data(fc_ready=True, factcheck=False, research_iterations=None)
                except Exception:
                    pass

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
                # Compute dynamic price for single post
                try:
                    refine_pref = await get_refine_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    refine_pref = False
                try:
                    fc_enabled_pref = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
                except Exception:
                    fc_enabled_pref = False
                try:
                    fc_depth_pref = await get_factcheck_depth(message.from_user.id) if message.from_user else 2
                except Exception:
                    fc_depth_pref = 2
                fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
                unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
                confirm_txt = (
                    f"Будет списано {unit_cost} кредит(ов). Подтвердить?"
                    if _is_ru(ui_lang_local) else
                    f"It will cost {unit_cost} credit(s). Proceed?"
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

        # Charge dynamic price before starting (DB if configured, else Redis KV)
        from sqlalchemy.exc import SQLAlchemyError
        try:
            charged = False
            # Admins generate for free
            if message.from_user and message.from_user.id in ADMIN_IDS:
                charged = True
            # Compute price based on current preferences
            try:
                refine_pref = await get_refine_enabled(message.from_user.id) if message.from_user else False
            except Exception:
                refine_pref = False
            try:
                fc_enabled_pref = await get_factcheck_enabled(message.from_user.id) if message.from_user else False
            except Exception:
                fc_enabled_pref = False
            try:
                fc_depth_pref = await get_factcheck_depth(message.from_user.id) if message.from_user else 2
            except Exception:
                fc_depth_pref = 2
            fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
            unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
            # Only attempt DB charge when not already marked as charged (non-admin)
            if (not charged) and SessionLocal is not None and message.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, message.from_user.id)
                    ok, remaining = await charge_credits(session, user, unit_cost, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged and message.from_user:
                ok, remaining = await charge_credits_kv(message.from_user.id, unit_cost)
                if not ok:
                    need = max(0, int(unit_cost) - int(remaining))
                    warn = (
                        f"Недостаточно кредитов. Не хватает: {need}. Используйте /buy для пополнения."
                        if _is_ru(ui_lang)
                        else f"Insufficient credits. Need: {need}. Use /buy to top up."
                    )
                    await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                    try:
                        await dp.bot.send_message(
                            ("Купить кредиты за ⭐? Один кредит = 200⭐" if _is_ru(ui_lang) else "Buy credits with ⭐? One credit = 200⭐"),
                            reply_markup=build_buy_keyboard(ui_lang),
                        )
                    except Exception:
                        pass
                    await state.finish()
                    RUNNING_CHATS.discard(chat_id)
                    return
        except SQLAlchemyError:
            # For admin path we do not attempt DB charge; this error indicates a true DB failure for non-admin charge
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
                        provider=((prov if prov != "auto" else "openai") or "openai"),
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
                        provider=((prov if prov != "auto" else "openai") or "openai"),
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

    @dp.callback_query_handler(lambda c: c.data in {"confirm:charge:yes","confirm:charge:no"}, state="*")  # type: ignore
    async def cb_confirm_charge(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if query.data.endswith(":no"):
            await query.answer()
            await query.message.edit_reply_markup() if query.message else None
            chat_id = query.message.chat.id if query.message else (query.from_user.id if query.from_user else 0)
            try:
                RUNNING_CHATS.discard(chat_id)
            except Exception:
                pass
            try:
                await state.update_data(pending_topic=None)
            except Exception:
                pass
            # Single cancel message
            try:
                await dp.bot.send_message(chat_id, ("Отменено." if _is_ru(ui_lang) else "Cancelled."))
            except Exception:
                pass
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
                    # Compute unit price again
                    try:
                        refine_pref = await get_refine_enabled(query.from_user.id)
                    except Exception:
                        refine_pref = False
                    try:
                        fc_enabled_pref = await get_factcheck_enabled(query.from_user.id)
                    except Exception:
                        fc_enabled_pref = False
                    try:
                        fc_depth_pref = await get_factcheck_depth(query.from_user.id)
                    except Exception:
                        fc_depth_pref = 2
                    fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
                    unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
                    ok, remaining = await charge_credits(session, user, unit_cost, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged and query.from_user:
                # KV fallback
                try:
                    refine_pref = await get_refine_enabled(query.from_user.id)
                except Exception:
                    refine_pref = False
                try:
                    fc_enabled_pref = await get_factcheck_enabled(query.from_user.id)
                except Exception:
                    fc_enabled_pref = False
                try:
                    fc_depth_pref = await get_factcheck_depth(query.from_user.id)
                except Exception:
                    fc_depth_pref = 2
                fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
                unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
                ok, remaining = await charge_credits_kv(query.from_user.id, unit_cost)
                if not ok:
                    need = max(0, int(unit_cost) - int(remaining))
                    warn = (
                        f"Недостаточно кредитов. Не хватает: {need}. Используйте /buy для пополнения."
                        if _is_ru(ui_lang)
                        else f"Insufficient credits. Need: {need}. Use /buy to top up."
                    )
                    await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                    try:
                        await dp.bot.send_message(
                            ("Купить кредиты за ⭐? Один кредит = 200⭐" if _is_ru(ui_lang) else "Buy credits with ⭐? One credit = 200⭐"),
                            reply_markup=build_buy_keyboard(ui_lang),
                        )
                    except Exception:
                        pass
                    await state.finish()
                    RUNNING_CHATS.discard(chat_id)
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
                    lambda: generate_post(topic, lang=eff_lang, provider=((prov if prov != "auto" else "openai") or "openai"), factcheck=True, research_iterations=int(depth or 2), job_meta=job_meta, use_refine=refine_enabled),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            else:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(topic, lang=eff_lang, provider=((prov if prov != "auto" else "openai") or "openai"), factcheck=False, job_meta=job_meta, use_refine=refine_enabled),
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
    async def _list_deletable_ids_for_user(telegram_user_id: int, ids: Optional[list[int]] | None = None, *, only_visible: bool = False) -> list[int]:
        """Return list of ResultDoc IDs belonging to the telegram user.
        If ids provided, intersect with them. If only_visible, filter hidden=0/NULL.
        """
        if SessionLocal is None:
            return []
        from sqlalchemy import select as _select
        from sqlalchemy import or_ as _or
        from sqlalchemy import join as _join
        async with SessionLocal() as _s:
            from .db import ResultDoc, Job, User
            # Map telegram -> DB user.id when possible
            db_uid: Optional[int] = None
            try:
                uq = await _s.execute(_select(User).where(User.telegram_id == int(telegram_user_id)))
                urow = uq.scalars().first()
                if urow is not None:
                    db_uid = int(urow.id)
            except Exception:
                db_uid = None
            cond = Job.user_id == int(telegram_user_id)
            if db_uid is not None:
                cond = _or(cond, Job.user_id == db_uid)
            jn = _join(ResultDoc, Job, ResultDoc.job_id == Job.id)
            sel = _select(ResultDoc.id).select_from(jn).where(cond)
            if ids:
                try:
                    _ids = [int(x) for x in ids if isinstance(x, int)]
                except Exception:
                    _ids = []
                if _ids:
                    sel = sel.where(ResultDoc.id.in_(_ids))
                else:
                    return []
            else:
                if only_visible:
                    from sqlalchemy import or_ as __or
                    sel = sel.where(__or(ResultDoc.hidden == 0, ResultDoc.hidden.is_(None)))
            res = await _s.execute(sel)
            return [int(row[0]) for row in res.fetchall() if row and row[0]]

    async def _delete_results_for_user(telegram_user_id: int, ids: Optional[list[int]] | None = None, *, only_visible: bool = False) -> int:
        """Delete ResultDoc rows by IDs (or all) that belong to this Telegram user.
        Returns number of deleted rows.
        """
        if SessionLocal is None:
            return 0
        from sqlalchemy import delete as _delete
        allowed_ids = await _list_deletable_ids_for_user(telegram_user_id, ids=ids, only_visible=only_visible)
        if not allowed_ids:
            return 0
        async with SessionLocal() as _s:
            from .db import ResultDoc
            await _s.execute(_delete(ResultDoc).where(ResultDoc.id.in_(allowed_ids)))
            await _s.commit()
        return len(allowed_ids)

    @dp.message_handler(commands=["history"])  # type: ignore
    async def cmd_history(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        parts = (message.text or "").split()
        if len(parts) > 1 and parts[1].lower() == "clear":
            # Backward compatibility: /history clear
            try:
                from .kv import set_history_cleared_at
                await clear_history(message.from_user.id)
                await set_history_cleared_at(message.from_user.id)
                # Also remove user's results from Results page (including hidden)
                if message.from_user:
                    try:
                        await _delete_results_for_user(message.from_user.id, only_visible=False)
                    except Exception:
                        pass
            except Exception:
                pass
            await message.answer("История очищена." if _is_ru(ui_lang) else "History cleared.")
            return
        # Show unified history from DB Results (not only posts)
        items = []
        try:
            if SessionLocal is not None and message.from_user:
                from sqlalchemy import select as _select
                from sqlalchemy import join as _join
                async with SessionLocal() as _s:
                    from sqlalchemy import or_ as _or
                    from .db import ResultDoc, Job, User
                    # Resolve both legacy (Job.user_id == telegram_id) and normalized (Job.user_id == User.id)
                    db_uid = None
                    try:
                        uq = await _s.execute(_select(User).where(User.telegram_id == int(message.from_user.id)))
                        urow = uq.scalars().first()
                        if urow is not None:
                            db_uid = int(urow.id)
                    except Exception:
                        db_uid = None
                    jn = _join(ResultDoc, Job, ResultDoc.job_id == Job.id)
                    cond = Job.user_id == int(message.from_user.id)
                    if db_uid is not None:
                        cond = _or(cond, Job.user_id == db_uid)
                    res = await _s.execute(
                        _select(ResultDoc, Job.user_id)
                        .select_from(jn)
                        .where(cond)
                        .order_by(ResultDoc.created_at.desc())
                        .limit(50)
                    )
                    rows = res.fetchall()
                    # Filter by last clear ts if exists
                    cleared_ts = None
                    try:
                        from .kv import get_history_cleared_at
                        cleared_ts = await get_history_cleared_at(message.from_user.id)
                    except Exception:
                        cleared_ts = None
                    for rdoc, _uid in rows:
                        try:
                            if cleared_ts is not None:
                                # Skip results created before clear mark
                                if getattr(rdoc, "created_at", None) is not None and float(cleared_ts) > 0:
                                    if rdoc.created_at.timestamp() < float(cleared_ts):
                                        continue
                        except Exception:
                            pass
                        items.append({
                            "id": int(rdoc.id),
                            "kind": getattr(rdoc, "kind", "") or "",
                            "topic": getattr(rdoc, "topic", "") or "",
                            "hidden": int(getattr(rdoc, "hidden", 0) or 0),
                        })
        except Exception:
            items = []
        if not items:
            await message.answer("История пуста." if _is_ru(ui_lang) else "No history yet.")
            return
        lines = []
        for it in items:
            topic = (it.get("topic") or "(no topic)")
            kind = (it.get("kind") or "").lower()
            url = ""
            try:
                if it.get("hidden") == 0 and it.get("id"):
                    url = _result_url(int(it.get("id")))
            except Exception:
                url = ""
            tag = {"post":"post","post_series":"series","article":"article","summary":"summary"}.get(kind, kind or "result")
            idtxt = f" [id: {int(it.get('id'))}]" if it.get("id") else ""
            if url:
                lines.append(f"• [{tag}] <a href='{url}'>{topic}</a>{idtxt}")
            else:
                lines.append(f"• [{tag}] {topic}{idtxt}")
        prefix = "История генераций (последние):\n\n" if _is_ru(ui_lang) else "Your recent generations:\n\n"
        # Inline clear button for clarity
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton(text=("Очистить историю" if _is_ru(ui_lang) else "Clear history"), callback_data="history:clear"))
        kb.add(types.InlineKeyboardButton(text=("Удалить выборочно" if _is_ru(ui_lang) else "Delete selected"), callback_data="history:delete"))
        await message.answer(prefix + "\n".join(lines), parse_mode=types.ParseMode.HTML, disable_web_page_preview=True, reply_markup=kb)

    @dp.message_handler(commands=["history_clear"])  # type: ignore
    async def cmd_history_clear(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            await clear_history(message.from_user.id)
            try:
                from .kv import set_history_cleared_at
                await set_history_cleared_at(message.from_user.id)
            except Exception:
                pass
            # Remove user's results from Results page as well (including hidden)
            if message.from_user:
                try:
                    await _delete_results_for_user(message.from_user.id, only_visible=False)
                except Exception:
                    pass
        except Exception:
            pass
        await message.answer("История очищена." if _is_ru(ui_lang) else "History cleared.")

    @dp.callback_query_handler(lambda c: c.data == "history:clear")  # type: ignore
    async def cb_history_clear(query: types.CallbackQuery, state: FSMContext):
        try:
            from .kv import set_history_cleared_at
            await clear_history(query.from_user.id)
            await set_history_cleared_at(query.from_user.id)
            # Also delete all results for this user from Results page (including hidden)
            if query.from_user:
                try:
                    await _delete_results_for_user(query.from_user.id, only_visible=False)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
        except Exception:
            ui_lang = "ru"
        await query.answer("Очищено" if _is_ru(ui_lang) else "Cleared", show_alert=False)
        await dp.bot.edit_message_text(
            chat_id=query.message.chat.id,
            message_id=query.message.message_id,
            text=("История очищена." if _is_ru(ui_lang) else "History cleared."),
        )

    @dp.callback_query_handler(lambda c: c.data == "history:delete")  # type: ignore
    async def cb_history_delete(query: types.CallbackQuery, state: FSMContext):
        try:
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
        except Exception:
            ui_lang = "ru"
        prompt_ru = "Напишите через запятую id результатов, которые нужно удалить."
        prompt_en = "Send comma-separated result IDs to delete."
        await HistoryStates.WaitingDeleteIds.set()
        try:
            await query.answer()
        except Exception:
            pass
        # Show Cancel button while waiting for IDs
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton(text=("Отмена" if _is_ru(ui_lang) else "Cancel"), callback_data="history:delete_cancel"))
        await dp.bot.send_message(
            chat_id=query.message.chat.id,
            text=(prompt_ru if _is_ru(ui_lang) else prompt_en),
            reply_markup=kb,
        )

    @dp.message_handler(state=HistoryStates.WaitingDeleteIds, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def history_delete_ids_received(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        raw = (message.text or "").strip()
        # Textual cancel fallback
        if raw.lower() in {"отмена", "cancel"}:
            await state.finish()
            await message.answer("Удаление отменено." if _is_ru(ui_lang) else "Deletion cancelled.")
            return
        # Parse comma/space separated integers
        import re as _re
        parts = [_p.strip() for _p in _re.split(r"[,\s]+", raw) if _p.strip()]
        ids: list[int] = []
        for p in parts:
            try:
                ids.append(int(p))
            except Exception:
                continue
        if not ids:
            await message.answer("Не распознал id. Отправьте, например: 36, 34" if _is_ru(ui_lang) else "No IDs found. For example: 36, 34")
            return
        # Build confirmation: which ids are allowed vs skipped
        allowed: list[int] = []
        try:
            if message.from_user:
                allowed = await _list_deletable_ids_for_user(message.from_user.id, ids=ids)
        except Exception:
            allowed = []
        allowed_set = set(allowed)
        skipped = [i for i in ids if i not in allowed_set]
        if not allowed:
            await message.answer("Нечего удалять: не найдено ваших результатов с такими ID." if _is_ru(ui_lang) else "Nothing to delete: no matching results owned by you.")
            await state.finish()
            return
        # Save pending list and show confirm/cancel
        await state.update_data(pending_delete_ids=allowed)
        kb = types.InlineKeyboardMarkup()
        kb.add(types.InlineKeyboardButton(text=("Подтвердить" if _is_ru(ui_lang) else "Confirm"), callback_data="history:delete_confirm"))
        kb.add(types.InlineKeyboardButton(text=("Отмена" if _is_ru(ui_lang) else "Cancel"), callback_data="history:delete_cancel"))
        lines = []
        lines.append(("Будут удалены id: " if _is_ru(ui_lang) else "Will delete ids: ") + ", ".join(str(x) for x in allowed))
        if skipped:
            lines.append(("Пропущены (не ваши/не найдены): " if _is_ru(ui_lang) else "Skipped (not yours/not found): ") + ", ".join(str(x) for x in skipped))
        await message.answer("\n".join(lines), reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data == "history:delete_cancel", state=HistoryStates.WaitingDeleteIds)  # type: ignore
    async def cb_history_delete_cancel(query: types.CallbackQuery, state: FSMContext):
        try:
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
        except Exception:
            ui_lang = "ru"
        try:
            await query.answer()
        except Exception:
            pass
        try:
            await state.finish()
        except Exception:
            pass
        await dp.bot.send_message(
            chat_id=query.message.chat.id,
            text=("Удаление отменено." if _is_ru(ui_lang) else "Deletion cancelled."),
        )

    @dp.callback_query_handler(lambda c: c.data == "history:delete_confirm", state=HistoryStates.WaitingDeleteIds)  # type: ignore
    async def cb_history_delete_confirm(query: types.CallbackQuery, state: FSMContext):
        try:
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
            ids = [int(x) for x in (data.get("pending_delete_ids") or [])]
        except Exception:
            ui_lang = "ru"
            ids = []
        deleted = 0
        try:
            if query.from_user and ids:
                deleted = await _delete_results_for_user(query.from_user.id, ids=ids)
        except Exception:
            deleted = 0
        try:
            await state.finish()
        except Exception:
            pass
        try:
            await query.answer()
        except Exception:
            pass
        await dp.bot.send_message(
            chat_id=query.message.chat.id,
            text=((f"Удалено: {deleted}" if _is_ru(ui_lang) else f"Deleted: {deleted}") if deleted > 0 else ("Ничего не удалено." if _is_ru(ui_lang) else "Nothing deleted.")),
        )

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
                "Цены:\n"
                "- Пост: 1 кредит\n"
                "- Серия: 1×N кредитов\n"
                "- Статья: 3 кредита\n- Книга: по главам\n"
                "Оплата: Telegram Stars (если включено)."
            )
        else:
            await message.answer(
                "Pricing:\n"
                "- Post: 1 credit\n"
                "- Series: 1×N credits\n"
                "- Article: 3 credits\n- Book: per chapter\n"
                "Payments: Telegram Stars (if enabled)."
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

    # ---- Chat mode ----
    @dp.message_handler(commands=["chat"])  # type: ignore
    async def cmd_chat(message: types.Message, state: FSMContext):
        if not message.from_user or message.from_user.id not in ADMIN_IDS:
            await message.answer("Недоступно.")
            return
        # Parse args
        parts = (message.text or "").strip().split()
        req_id = None
        if len(parts) >= 2:
            if parts[1].lower() == "id" and len(parts) >= 3 and parts[2].isdigit():
                req_id = int(parts[2])
            elif parts[1].isdigit():
                req_id = int(parts[1])
        # Resolve context strictly from DB
        content = None
        kind = "result"
        src_id = None
        if SessionLocal is None:
            await message.answer("БД недоступна.")
            return
        try:
            from sqlalchemy import select
            async with SessionLocal() as s:
                from .db import ResultDoc, Job, User
                # Determine current admin's DB uid
                db_uid = None
                try:
                    uq = await s.execute(select(User).where(User.telegram_id == int(message.from_user.id)))
                    urow = uq.scalars().first()
                    if urow is not None:
                        db_uid = int(urow.id)
                except Exception:
                    db_uid = None
                if req_id is not None:
                    q = await s.execute(
                        select(ResultDoc, Job.user_id)
                        .select_from(ResultDoc.__table__.join(Job.__table__, ResultDoc.job_id == Job.id))
                        .where(ResultDoc.id == int(req_id))
                    )
                    row = q.first()
                    if not row:
                        await message.answer("Результат не найден.")
                        return
                    rdoc, owner_uid = row
                    if db_uid is not None and int(owner_uid) != db_uid and int(owner_uid) != int(message.from_user.id):
                        await message.answer("Доступ запрещён.")
                        return
                    content = getattr(rdoc, "content", None)
                    kind = getattr(rdoc, "kind", "result") or "result"
                    src_id = int(getattr(rdoc, "id", 0) or 0)
                else:
                    # last result for this admin
                    from sqlalchemy import or_ as _or
                    cond = (Job.user_id == int(message.from_user.id))
                    if db_uid is not None:
                        cond = _or(cond, Job.user_id == db_uid)
                    q = await s.execute(
                        select(ResultDoc, Job.user_id)
                        .select_from(ResultDoc.__table__.join(Job.__table__, ResultDoc.job_id == Job.id))
                        .where(cond)
                        .order_by(ResultDoc.created_at.desc())
                        .limit(1)
                    )
                    row = q.first()
                    if not row:
                        await message.answer("Нет результатов для контекста чата.")
                        return
                    rdoc, _ = row
                    content = getattr(rdoc, "content", None)
                    kind = getattr(rdoc, "kind", "result") or "result"
                    src_id = int(getattr(rdoc, "id", 0) or 0)
        except Exception:
            await message.answer("Ошибка при получении контекста.")
            return

        await state.update_data(chat_context={
            "kind": kind,
            "content": content or "",
            "result_id": src_id,
        })
        await ChatStates.Active.set()
        await message.answer((f"Чат активирован. Источник: id {src_id}" if src_id else "Чат активирован"))

    @dp.message_handler(commands=["endchat"], state=ChatStates.Active)  # type: ignore
    async def cmd_endchat(message: types.Message, state: FSMContext):
        try:
            # Clear provider-native session and KV history for this chat
            prov = "openai"
            if message.from_user:
                try:
                    prov = await get_provider(message.from_user.id)
                except Exception:
                    prov = "openai"
            try:
                from services.chat.run import clear_provider_sessions
                sid = f"{message.chat.id}:{message.from_user.id}:{prov}"
                clear_provider_sessions(prov, sid)
            except Exception:
                pass
            try:
                await chat_clear(message.from_user.id, message.chat.id, prov)
            except Exception:
                pass
        except Exception:
            pass
        await state.finish()
        await message.answer("Чат завершён.")

    # Generic commands while chat active: finish state and re-dispatch the same update
    @dp.message_handler(lambda m: (m.text or "").startswith("/"), state=ChatStates.Active, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def cmd_any_in_chat(message: types.Message, state: FSMContext):
        try:
            await state.finish()
        except Exception:
            pass
        # Direct dispatch to the proper command handler to avoid timing races
        try:
            raw = (message.text or "").strip()
            cmd = raw.split()[0].lstrip("/").split("@")[0].lower()
            handlers = {
                "start": cmd_start,
                "info": cmd_info,
                "generate": cmd_generate,
                "series": cmd_series,
                "series_fixed": cmd_series_fixed,
                "settings": cmd_settings,
                "history": cmd_history,
                "history_clear": cmd_history_clear,
                "balance": cmd_balance,
                "buy": cmd_buy,
                "pricing": cmd_pricing,
                "lang": cmd_lang,
                "lang_generate": cmd_lang_generate,
                "provider": cmd_provider,
                "public": cmd_incognito,
                "refine": cmd_refine,
                "factcheck": cmd_factcheck,
                "depth": cmd_depth,
                "chat": cmd_chat,
                "endchat": cmd_endchat,
                "cancel": cmd_cancel,
                "logs": cmd_logs,
            }
            h = handlers.get(cmd)
            if h is not None:
                try:
                    await h(message, state)  # type: ignore[arg-type]
                except TypeError:
                    # Some handlers accept only (message)
                    await h(message)  # type: ignore[misc]
        except Exception:
            pass
        return

    # Generic commands while ANY FSM state (except active chat): finish state and re-dispatch
    @dp.message_handler(lambda m: (m.text or "").startswith("/"), state="*", content_types=types.ContentTypes.TEXT)  # type: ignore
    async def cmd_any_during_fsm(message: types.Message, state: FSMContext):
        try:
            cur = await state.get_state()
        except Exception:
            cur = None
        # Do not shadow chat-specific handler; let it handle commands in chat state
        try:
            if cur == ChatStates.Active.state:
                return
        except Exception:
            pass
        # Finish any other state to avoid blocking commands
        try:
            await state.finish()
        except Exception:
            pass
        # Directly dispatch the command to its handler to avoid timing races
        try:
            raw = (message.text or "").strip()
            cmd = raw.split()[0].lstrip("/").split("@")[0].lower()
            handlers = {
                "start": cmd_start,
                "info": cmd_info,
                "generate": cmd_generate,
                "series": cmd_series,
                "series_fixed": cmd_series_fixed,
                "settings": cmd_settings,
                "history": cmd_history,
                "history_clear": cmd_history_clear,
                "balance": cmd_balance,
                "buy": cmd_buy,
                "pricing": cmd_pricing,
                "lang": cmd_lang,
                "lang_generate": cmd_lang_generate,
                "provider": cmd_provider,
                "public": cmd_incognito,
                "refine": cmd_refine,
                "factcheck": cmd_factcheck,
                "depth": cmd_depth,
                "chat": cmd_chat,
                "endchat": cmd_endchat,
                "cancel": cmd_cancel,
                "logs": cmd_logs,
            }
            h = handlers.get(cmd)
            if h is not None:
                try:
                    await h(message, state)  # type: ignore[arg-type]
                except TypeError:
                    # Some handlers accept only (message)
                    await h(message)  # type: ignore[misc]
        except Exception:
            pass
        return

    @dp.message_handler(lambda m: not (getattr(m.from_user, "is_bot", False) or (m.text or "").startswith("/")), state=ChatStates.Active, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def chat_active_message(message: types.Message, state: FSMContext):
        txt = (message.text or "").strip()
        # Ignore bot/self messages handled via decorator predicate; no-op here
        # Deduplicate: avoid processing the same message twice (webhook retries)
        try:
            sd = await state.get_data()
            last_id = int(sd.get("last_handled_message_id") or 0)
            if last_id == int(message.message_id):
                return
            await state.update_data(last_handled_message_id=int(message.message_id))
        except Exception:
            pass
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ctx = data.get("chat_context") or {}
        kind = (ctx.get("kind") or "result")
        full_content = (ctx.get("content") or "")
        # Detect language from user input or UI
        try:
            det = (detect_lang_from_text(message.text or "") or "").lower()
            chat_lang = "en" if det.startswith("en") else ("ru" if _is_ru(ui_lang) else "en")
        except Exception:
            chat_lang = "ru" if _is_ru(ui_lang) else "en"
        # Keep rolling history in state; if too long, build a recap to keep prompt short
        history = (data.get("chat_history") or [])
        try:
            if len(history) > 60:
                prov = await get_provider(message.from_user.id) if message.from_user else "openai"
                from services.chat.run import run_chat_summary
                # prepare plain text transcript
                transcript = "\n".join(f"[{r}]: {c}" for r, c in history)
                recap = run_chat_summary(prov, transcript)
                history = [("recap", recap)] + history[-40:]
                await state.update_data(chat_history=history)
        except Exception:
            pass
        # Construct compact user_message: include last turns
        def _format_history(items):
            out = []
            for role, content in items[-40:]:
                out.append(f"[{role}] {content}")
            return "\n".join(out)
        user_payload = ("\n\n" + _format_history(history) + "\n\n[user] " + (message.text or "")).strip()
        try:
            prov = "openai"
            if message.from_user:
                try:
                    prov = await get_provider(message.from_user.id)
                except Exception:
                    prov = "openai"
            system = build_system_prompt(chat_lang=chat_lang, kind=kind, full_content=full_content)
            import asyncio as _aio
            _loop = _aio.get_running_loop()
            eff_prov = (prov if prov != "auto" else "openai")
            sid = f"{message.chat.id}:{message.from_user.id}:{eff_prov}"
            reply = await _loop.run_in_executor(None, lambda: run_chat_message(eff_prov, system, user_payload, session_id=sid))
        except Exception as e:
            await message.answer(f"Ошибка: {e}")
            return
        text = (reply or "").strip()
        # md_output wrapper -> send file
        import re
        blocks = list(re.finditer(r"<md_output([^>]*)>([\s\S]*?)</md_output>", text))
        if blocks:
            from io import BytesIO
            for idx, bm in enumerate(blocks, start=1):
                attrs = bm.group(1) or ""
                body = bm.group(2) or ""
                doc = body if body.endswith("\n") else (body + "\n")
                buf = BytesIO(doc.encode("utf-8"))
                title = None
                mt = re.search(r"title=\"([^\"]+)\"", attrs)
                if mt:
                    title = mt.group(1)
                base = safe_filename_base(title) if title else f"result_{idx}"
                buf.name = f"{base}.md"
                await message.answer_document(buf, caption=("Готово" if _is_ru(ui_lang) else "Done"))
            return
        # Otherwise chunked messages
        def _chunks(s: str, limit: int = 4000):
            parts, buf = [], []
            for para in (s or "").split("\n\n"):
                cur = ("\n\n".join(buf) if buf else "")
                if len(cur) + (2 if cur else 0) + len(para) <= limit:
                    buf.append(para)
                else:
                    if buf:
                        parts.append("\n\n".join(buf))
                        buf = []
                    if len(para) <= limit:
                        buf = [para]
                    else:
                        for i in range(0, len(para), limit):
                            parts.append(para[i:i+limit])
            if buf:
                parts.append("\n\n".join(buf))
            return parts
        # Update history
        try:
            history.append(("user", message.text or ""))
            history.append(("assistant", text))
            await state.update_data(chat_history=history)
        except Exception:
            pass
        for chunk in _chunks(text):
            await message.answer(chunk)

    # ---- Auto-chat entry for admins on plain text (no command) ----
    @dp.message_handler(lambda m: not getattr(m.from_user, "is_bot", False), content_types=types.ContentTypes.TEXT, state="*")  # type: ignore
    async def auto_chat_entry(message: types.Message, state: FSMContext):
        txt = (message.text or "").strip()
        # Ignore commands or empty
        if not txt or txt.startswith("/"):
            return
        # Deduplicate: avoid processing the same message twice (webhook retries)
        try:
            sd = await state.get_data()
            last_id = int(sd.get("last_handled_message_id") or 0)
            if last_id == int(message.message_id):
                return
            await state.update_data(last_handled_message_id=int(message.message_id))
        except Exception:
            pass
        # Admins only and private chat only
        if not message.from_user or message.from_user.id not in ADMIN_IDS:
            return
        try:
            if message.chat and getattr(message.chat, "type", "") != "private":
                return
        except Exception:
            pass
        # Do not interfere with generation/topic states or active chat
        try:
            cur = await state.get_state()
        except Exception:
            cur = None
        try:
            if cur in {GenerateStates.WaitingTopic.state}:
                return
        except Exception:
            pass
        try:
            if cur == ChatStates.Active.state:
                return
        except Exception:
            pass

        # Resolve last result for this admin
        content = None
        kind = "result"
        src_id = None
        if SessionLocal is None:
            return
        try:
            from sqlalchemy import select
            async with SessionLocal() as s:
                from .db import ResultDoc, Job, User
                db_uid = None
                try:
                    uq = await s.execute(select(User).where(User.telegram_id == int(message.from_user.id)))
                    urow = uq.scalars().first()
                    if urow is not None:
                        db_uid = int(urow.id)
                except Exception:
                    db_uid = None
                from sqlalchemy import or_ as _or
                cond = (Job.user_id == int(message.from_user.id))
                if db_uid is not None:
                    cond = _or(cond, Job.user_id == db_uid)
                q = await s.execute(
                    select(ResultDoc, Job.user_id)
                    .select_from(ResultDoc.__table__.join(Job.__table__, ResultDoc.job_id == Job.id))
                    .where(cond)
                    .order_by(ResultDoc.created_at.desc())
                    .limit(1)
                )
                row = q.first()
                if not row:
                    await message.answer("Нет результатов для контекста чата. Сначала сгенерируйте пост.")
                    return
                rdoc, _ = row
                content = getattr(rdoc, "content", None)
                kind = getattr(rdoc, "kind", "result") or "result"
                src_id = int(getattr(rdoc, "id", 0) or 0)
        except Exception:
            return

        # Initialize chat state and context
        await state.update_data(chat_context={
            "kind": kind,
            "content": content or "",
            "result_id": src_id,
            "chat_history": [],
        })
        await ChatStates.Active.set()

        # Detect language
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        try:
            det = (detect_lang_from_text(txt) or "").lower()
            chat_lang = "en" if det.startswith("en") else ("ru" if _is_ru(ui_lang) else "en")
        except Exception:
            chat_lang = "ru" if _is_ru(ui_lang) else "en"

        # Prepare payload (no prior history yet)
        try:
            prov = "openai"
            if message.from_user:
                try:
                    prov = await get_provider(message.from_user.id)
                except Exception:
                    prov = "openai"
            system = build_system_prompt(chat_lang=chat_lang, kind=kind, full_content=(content or ""))
            import asyncio as _aio
            _loop = _aio.get_running_loop()
            eff_prov = (prov if prov != "auto" else "openai")
            sid = f"{message.chat.id}:{message.from_user.id}:{eff_prov}"
            reply = await _loop.run_in_executor(None, lambda: run_chat_message(eff_prov, system, txt, session_id=sid))
        except Exception as e:
            await message.answer(f"Ошибка: {e}")
            return

        # Send result (md_output or chunks) and store history
        text = (reply or "").strip()
        import re
        blocks2 = list(re.finditer(r"<md_output([^>]*)>([\s\S]*?)</md_output>", text))
        if blocks2:
            from io import BytesIO
            for idx, bm in enumerate(blocks2, start=1):
                attrs = bm.group(1) or ""
                body = bm.group(2) or ""
                doc = body if body.endswith("\n") else (body + "\n")
                buf = BytesIO(doc.encode("utf-8"))
                title = None
                mt = re.search(r"title=\"([^\"]+)\"", attrs)
                if mt:
                    title = mt.group(1)
                base = safe_filename_base(title) if title else f"result_{idx}"
                buf.name = f"{base}.md"
                await message.answer_document(buf, caption=("Готово" if _is_ru(ui_lang) else "Done"))
            # Update history with concatenated body text for simplicity
            try:
                joined = "\n\n".join(bm.group(2) or "" for bm in blocks2)
                hist = (data.get("chat_history") or [])
                hist.append(("user", txt))
                hist.append(("assistant", joined))
                await state.update_data(chat_history=hist)
            except Exception:
                pass
            return

        # Chunked send
        def _chunks(s: str, limit: int = 4000):
            parts, buf = [], []
            for para in (s or "").split("\n\n"):
                cur = ("\n\n".join(buf) if buf else "")
                if len(cur) + (2 if cur else 0) + len(para) <= limit:
                    buf.append(para)
                else:
                    if buf:
                        parts.append("\n\n".join(buf))
                        buf = []
                    if len(para) <= limit:
                        buf = [para]
                    else:
                        for i in range(0, len(para), limit):
                            parts.append(para[i:i+limit])
            if buf:
                parts.append("\n\n".join(buf))
            return parts

        for chunk in _chunks(text):
            await message.answer(chunk)
        try:
            hist = (data.get("chat_history") or [])
            hist.append(("user", txt))
            hist.append(("assistant", text))
            await state.update_data(chat_history=hist)
        except Exception:
            pass

    return dp




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
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, LabeledPrice

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
from .kv import is_chat_running, mark_chat_running, unmark_chat_running
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
    ChoosingPostStyle = State()
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
    # Packs: 1, 5, 10, 50, 100, 500 credits at 50 stars per credit
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Купить 1 кредит — 50⭐", callback_data="buy:stars:1"))
        kb.add(InlineKeyboardButton(text="Купить 5 кредитов — 250⭐", callback_data="buy:stars:5"))
        kb.add(InlineKeyboardButton(text="Купить 10 кредитов — 500⭐", callback_data="buy:stars:10"))
        kb.add(InlineKeyboardButton(text="Купить 50 кредитов — 2500⭐", callback_data="buy:stars:50"))
        kb.add(InlineKeyboardButton(text="Купить 100 кредитов — 5000⭐", callback_data="buy:stars:100"))
        kb.add(InlineKeyboardButton(text="Купить 500 кредитов — 25000⭐", callback_data="buy:stars:500"))
    else:
        kb.add(InlineKeyboardButton(text="Buy 1 credit — 50⭐", callback_data="buy:stars:1"))
        kb.add(InlineKeyboardButton(text="Buy 5 credits — 250⭐", callback_data="buy:stars:5"))
        kb.add(InlineKeyboardButton(text="Buy 10 credits — 500⭐", callback_data="buy:stars:10"))
        kb.add(InlineKeyboardButton(text="Buy 50 credits — 2500⭐", callback_data="buy:stars:50"))
        kb.add(InlineKeyboardButton(text="Buy 100 credits — 5000⭐", callback_data="buy:stars:100"))
        kb.add(InlineKeyboardButton(text="Buy 500 credits — 25000⭐", callback_data="buy:stars:500"))
    return kb


def build_settings_keyboard(ui_lang: str, provider: str, gen_lang: str, refine: bool, logs_enabled: bool, incognito: bool, fc_enabled: bool, fc_depth: int, *, is_admin: bool = True, is_superadmin: bool = False) -> InlineKeyboardMarkup:
    # Legacy builder kept for compatibility; not used by new menu
    kb = InlineKeyboardMarkup()
    ru = _is_ru(ui_lang)
    if is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Авто" if ru else "Auto") + (" ✓" if provider == "auto" else "")), callback_data="set:provider:auto"),
            InlineKeyboardButton(text=("OpenAI" + (" ✓" if provider == "openai" else "")), callback_data="set:provider:openai"),
            InlineKeyboardButton(text=("Gemini" + (" ✓" if provider == "gemini" else "")), callback_data="set:provider:gemini"),
            InlineKeyboardButton(text=("Claude" + (" ✓" if provider == "claude" else "")), callback_data="set:provider:claude"),
        )
    kb.add(
        InlineKeyboardButton(text=(("Авто" if ru else "Auto") + (" ✓" if gen_lang == "auto" else "")), callback_data="set:gen_lang:auto"),
        InlineKeyboardButton(text=("EN" + (" ✓" if gen_lang == "en" else "")), callback_data="set:gen_lang:en"),
        InlineKeyboardButton(text=("RU" + (" ✓" if gen_lang == "ru" else "")), callback_data="set:gen_lang:ru"),
    )
    if is_admin or is_superadmin:
        kb.add(
            InlineKeyboardButton(text=(("Логи: вкл" if ru else "Logs: on") + (" ✓" if logs_enabled else "")), callback_data="set:logs:enable"),
            InlineKeyboardButton(text=(("Логи: выкл" if ru else "Logs: off") + (" ✓" if not logs_enabled else "")), callback_data="set:logs:disable"),
        )
    kb.add(
        InlineKeyboardButton(text=(("Публично: да" if ru else "Public: yes") + (" ✓" if not incognito else "")), callback_data="set:incog:disable"),
        InlineKeyboardButton(text=(("Публично: нет" if ru else "Public: no") + (" ✓" if incognito else "")), callback_data="set:incog:enable"),
    )
    return kb

def build_settings_main_menu(ui_lang: str, *, is_admin: bool, is_superadmin: bool) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    ru = _is_ru(ui_lang)
    # Generation language for all
    kb.add(InlineKeyboardButton(text=("Язык генерации" if ru else "Generation language"), callback_data="settings:open:gen_lang"))
    # Logs for admins/superadmins
    if is_admin or is_superadmin:
        kb.add(InlineKeyboardButton(text=("Логи генерации" if ru else "Generation logs"), callback_data="settings:open:logs"))
    # Public results for all
    kb.add(InlineKeyboardButton(text=("Публичные результаты" if ru else "Public results"), callback_data="settings:open:public"))
    # Provider only for superadmin
    if is_superadmin:
        kb.add(InlineKeyboardButton(text=("Провайдер" if ru else "Provider"), callback_data="settings:open:provider"))
    return kb

def build_back_only(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
    return kb


def build_genlang_inline(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    # Order: Auto | EN | RU
    if _is_ru(ui_lang):
        kb.add(
            InlineKeyboardButton(text="Авто", callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"),
            InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"),
        )
    else:
        kb.add(
            InlineKeyboardButton(text="Auto", callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="EN", callback_data="set:gen_lang:en"),
            InlineKeyboardButton(text="RU", callback_data="set:gen_lang:ru"),
        )
    return kb


def build_genlang_inline_with_check(current: str, ui_lang: str) -> InlineKeyboardMarkup:
    """Build generation language keyboard with checkmark on current selection and Back button."""
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(
            InlineKeyboardButton(text="Авто" + (" ✓" if current == "auto" else ""), callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="EN" + (" ✓" if current == "en" else ""), callback_data="set:gen_lang:en"),
            InlineKeyboardButton(text="RU" + (" ✓" if current == "ru" else ""), callback_data="set:gen_lang:ru"),
        )
    else:
        kb.add(
            InlineKeyboardButton(text="Auto" + (" ✓" if current == "auto" else ""), callback_data="set:gen_lang:auto"),
            InlineKeyboardButton(text="EN" + (" ✓" if current == "en" else ""), callback_data="set:gen_lang:en"),
            InlineKeyboardButton(text="RU" + (" ✓" if current == "ru" else ""), callback_data="set:gen_lang:ru"),
        )
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
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


def build_provider_inline_with_check(current: str, ui_lang: str) -> InlineKeyboardMarkup:
    """Build provider keyboard with checkmark on current selection and Back button."""
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="Auto" + (" ✓" if current == "auto" else ""), callback_data="set:provider:auto"))
    kb.add(InlineKeyboardButton(text="OpenAI" + (" ✓" if current == "openai" else ""), callback_data="set:provider:openai"))
    kb.add(InlineKeyboardButton(text="Gemini" + (" ✓" if current == "gemini" else ""), callback_data="set:provider:gemini"))
    kb.add(InlineKeyboardButton(text="Claude" + (" ✓" if current == "claude" else ""), callback_data="set:provider:claude"))
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
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


def build_enable_disable_inline_with_check(current: bool, ui_lang: str) -> InlineKeyboardMarkup:
    """Build enable/disable keyboard with checkmark on current selection and Back button."""
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Включить" + (" ✓" if current else ""), callback_data="set:logs:enable"))
        kb.add(InlineKeyboardButton(text="Отключить" + (" ✓" if not current else ""), callback_data="set:logs:disable"))
    else:
        kb.add(InlineKeyboardButton(text="Enable" + (" ✓" if current else ""), callback_data="set:logs:enable"))
        kb.add(InlineKeyboardButton(text="Disable" + (" ✓" if not current else ""), callback_data="set:logs:disable"))
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
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


def build_yesno_inline_with_check(current: bool, ui_lang: str) -> InlineKeyboardMarkup:
    """Build yes/no keyboard with checkmark on current selection and Back button."""
    kb = InlineKeyboardMarkup()
    if _is_ru(ui_lang):
        kb.add(InlineKeyboardButton(text="Да" + (" ✓" if current else ""), callback_data="set:incognito:yes"))
        kb.add(InlineKeyboardButton(text="Нет" + (" ✓" if not current else ""), callback_data="set:incognito:no"))
    else:
        kb.add(InlineKeyboardButton(text="Yes" + (" ✓" if current else ""), callback_data="set:incognito:yes"))
        kb.add(InlineKeyboardButton(text="No" + (" ✓" if not current else ""), callback_data="set:incognito:no"))
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
    return kb


def build_depth_inline() -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="1", callback_data="set:depth:1"))
    kb.add(InlineKeyboardButton(text="2", callback_data="set:depth:2"))
    kb.add(InlineKeyboardButton(text="3", callback_data="set:depth:3"))
    return kb


def build_depth_inline_with_check(current: int, ui_lang: str) -> InlineKeyboardMarkup:
    """Build depth keyboard with checkmark on current selection and Back button."""
    kb = InlineKeyboardMarkup()
    kb.add(InlineKeyboardButton(text="1" + (" ✓" if current == 1 else ""), callback_data="set:depth:1"))
    kb.add(InlineKeyboardButton(text="2" + (" ✓" if current == 2 else ""), callback_data="set:depth:2"))
    kb.add(InlineKeyboardButton(text="3" + (" ✓" if current == 3 else ""), callback_data="set:depth:3"))
    kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
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


def build_post_style_keyboard(ui_lang: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardMarkup()
    ru = _is_ru(ui_lang)
    kb.add(
        InlineKeyboardButton(text=("Стиль 1" if ru else "Style 1"), callback_data="set:post_style:post_style_1"),
        InlineKeyboardButton(text=("Стиль 2" if ru else "Style 2"), callback_data="set:post_style:post_style_2"),
    )
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

    # Running chats are now tracked in Redis for multi-worker coordination
    # Global concurrency limit: increased to 30 for better multi-user performance
    # Each user generation may spawn up to ~6 parallel workers, so 30 allows ~5 concurrent users
    import asyncio
    _sem_capacity = int(os.getenv("BOT_PARALLEL_LIMIT", "30"))
    GLOBAL_SEMAPHORE = asyncio.Semaphore(max(1, _sem_capacity))

    @dp.message_handler(commands=["start"])  # type: ignore
    async def cmd_start(message: types.Message):
        # Prevent starting onboarding during active generation
        if await is_chat_running(message.chat.id):
            state = dp.current_state(user=message.from_user.id, chat=message.chat.id)
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
            warn = "⏳ Генерация уже выполняется. Дождитесь завершения или используйте /cancel." if _is_ru(ui_lang) else "⏳ Generation in progress. Wait for completion or use /cancel."
            await message.answer(warn)
            return
        # Mark onboarding flow active and reset settings panel state
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(
            onboarding=True,
            in_settings=False,
            fc_ready=False,
            series_mode=None,
            series_count=None,
        )
        await message.answer(
            "Choose interface language / Выберите язык интерфейса:",
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
                    types.BotCommand("history", "История"),
                    types.BotCommand("info", "Инфо"),
                    types.BotCommand("interface_lang", "Язык интерфейса"),
                ]
                # EN set (user default)
                base_en = [
                    types.BotCommand("start", "Start"),
                    types.BotCommand("generate", "Generate"),
                    types.BotCommand("settings", "Settings"),
                    types.BotCommand("history", "History"),
                    types.BotCommand("info", "Info"),
                    types.BotCommand("interface_lang", "Interface language"),
                ]
                # Admin extras: chat and meme_extract
                if is_admin:
                    base_ru = base_ru + [
                        types.BotCommand("chat", "Чат с ИИ"),
                        types.BotCommand("endchat", "Завершить чат"),
                        types.BotCommand("meme_extract", "Экстракция мемов"),
                    ]
                    base_en = base_en + [
                        types.BotCommand("chat", "Chat with AI"),
                        types.BotCommand("endchat", "End chat"),
                        types.BotCommand("meme_extract", "Meme extraction"),
                    ]
                # Regular user extras: credits
                if (not is_admin):
                    base_ru = base_ru + [types.BotCommand("credits", "Кредиты")]
                    base_en = base_en + [types.BotCommand("credits", "Credits")]
                # Superadmin extras: ensure chat/endchat/meme_extract present
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
                if is_superadmin:
                    ru_cmds = {c.command for c in base_ru}
                    en_cmds = {c.command for c in base_en}
                    if "chat" not in ru_cmds:
                        base_ru.append(types.BotCommand("chat", "Чат с ИИ"))
                    if "endchat" not in ru_cmds:
                        base_ru.append(types.BotCommand("endchat", "Завершить чат"))
                    if "meme_extract" not in ru_cmds:
                        base_ru.append(types.BotCommand("meme_extract", "Экстракция мемов"))
                    if "chat" not in en_cmds:
                        base_en.append(types.BotCommand("chat", "Chat with AI"))
                    if "endchat" not in en_cmds:
                        base_en.append(types.BotCommand("endchat", "End chat"))
                    if "meme_extract" not in en_cmds:
                        base_en.append(types.BotCommand("meme_extract", "Meme extraction"))
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
        post_style = (data.get("post_style") or "post_style_1").strip().lower()
        try:
            if message.from_user:
                refine_enabled = await get_refine_enabled(message.from_user.id)
                fc_enabled = await get_factcheck_enabled(message.from_user.id)
                fc_depth = await get_factcheck_depth(message.from_user.id)
        except Exception:
            pass

        # Role flags
        try:
            from .bot_commands import SUPER_ADMIN_ID, ADMIN_IDS
            is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and str(message.from_user.id) == str(SUPER_ADMIN_ID))
            is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        except Exception:
            is_superadmin = False
            is_admin = False

        # Intro with title
        intro_ru = (
            "<b>Добро пожаловать в Book in one click!</b>\n\n"
            "Это бот для генерации научно‑популярного контента.\n\n"
        )
        intro_en = (
            "<b>Welcome to Book in one click!</b>\n\n"
            "This bot generates popular‑science content.\n\n"
        )

        if ui_lang == "ru":
            # Commands by role
            if is_superadmin:
                commands_block = (
                    "<b>Команды:</b>\n"
                    "- /start — онбординг.\n"
                    "- /generate — Пост / Серия / Статья.\n"
                    "- /settings — настройки.\n"
                    "- /history — история.\n"
                    "- /info — справка.\n"
                    "- /interface_lang — язык интерфейса.\n"
                    "- /chat, /endchat — чат с ИИ.\n"
                    "- /meme_extract — экстракция мемов.\n\n"
                )
            elif is_admin:
                commands_block = (
                    "<b>Команды:</b>\n"
                    "- /start — онбординг.\n"
                    "- /generate — Пост / Серия / Статья.\n"
                    "- /settings — настройки.\n"
                    "- /history — история.\n"
                    "- /info — справка.\n"
                    "- /interface_lang — язык интерфейса.\n"
                    "- /chat, /endchat — чат с ИИ.\n"
                    "- /meme_extract — экстракция мемов.\n\n"
                )
            else:
                commands_block = (
                    "<b>Команды:</b>\n"
                    "- /start — онбординг.\n"
                    "- /generate — Пост / Статья.\n"
                    "- /settings — настройки.\n"
                    "- /history — история.\n"
                    "- /info — справка.\n"
                    "- /credits — баланс/покупка/цены.\n"
                    "- /interface_lang — язык интерфейса.\n\n"
                )

            if is_superadmin or is_admin:
                results_block = (
                    "<b>Результаты:</b>\n"
                    f"- <a href='{RESULTS_ORIGIN}/results-ui'>Список всех результатов</a>\n"
                    f"- <a href='{RESULTS_ORIGIN}/memes-ui'>Экстракции мемов</a>\n\n"
                )
            else:
                results_block = (
                    "<b>Результаты:</b>\n"
                    f"- <a href='{RESULTS_ORIGIN}/results-ui'>Список всех результатов</a>\n\n"
                )
            provider_line_ru = (f"- Провайдер: {_prov_name(prov, True)}\n" if is_superadmin else "")
            logs_line_ru = (f"- Логи генерации: {'вкл' if logs_enabled else 'выкл'}\n" if (is_superadmin or is_admin) else "")
            style_line_ru = f"- Стиль поста: {'Стиль 1' if post_style == 'post_style_1' else 'Стиль 2'}\n"
            settings_block = (
                "<b>Текущие настройки:</b>\n"
                f"{provider_line_ru}"
                f"{logs_line_ru}"
                f"{style_line_ru}"
                f"- Язык генерации: {_lang_human(gen_lang, True)}\n"
                f"- Публичные результаты: {'да' if not incognito else 'нет'}"
                + "\n\n<a href='https://github.com/YanGranat/book-in-one-click'>GitHub проекта</a>"
            )
            text = intro_ru + commands_block + results_block + settings_block
        else:
            if is_superadmin:
                commands_block = (
                    "<b>Commands:</b>\n"
                    "- /start — onboarding.\n"
                    "- /generate — Post / Series / Article.\n"
                    "- /settings — settings.\n"
                    "- /history — history.\n"
                    "- /info — info.\n"
                    "- /interface_lang — interface language.\n"
                    "- /chat, /endchat — chat with AI.\n"
                    "- /meme_extract — meme extraction.\n\n"
                )
            elif is_admin:
                commands_block = (
                    "<b>Commands:</b>\n"
                    "- /start — onboarding.\n"
                    "- /generate — Post / Series / Article.\n"
                    "- /settings — settings.\n"
                    "- /history — history.\n"
                    "- /info — info.\n"
                    "- /interface_lang — interface language.\n"
                    "- /chat, /endchat — chat with AI.\n"
                    "- /meme_extract — meme extraction.\n\n"
                )
            else:
                commands_block = (
                    "<b>Commands:</b>\n"
                    "- /start — onboarding.\n"
                    "- /generate — Post / Article.\n"
                    "- /settings — settings.\n"
                    "- /history — history.\n"
                    "- /info — info.\n"
                    "- /credits — balance/buy/pricing.\n"
                    "- /interface_lang — interface language.\n\n"
                )

            if is_superadmin or is_admin:
                results_block = (
                    "<b>Results:</b>\n"
                    f"- <a href='{RESULTS_ORIGIN}/results-ui'>All results page</a>\n"
                    f"- <a href='{RESULTS_ORIGIN}/memes-ui'>Meme extractions</a>\n\n"
                )
            else:
                results_block = (
                    "<b>Results:</b>\n"
                    f"- <a href='{RESULTS_ORIGIN}/results-ui'>All results page</a>\n\n"
                )
            provider_line_en = (f"- Provider: {_prov_name(prov, False)}\n" if is_superadmin else "")
            logs_line_en = (f"- Generation logs: {'on' if logs_enabled else 'off'}\n" if (is_superadmin or is_admin) else "")
            style_line_en = f"- Post style: {'Style 1' if post_style == 'post_style_1' else 'Style 2'}\n"
            settings_block = (
                "<b>Current settings:</b>\n"
                f"{provider_line_en}"
                f"{logs_line_en}"
                f"{style_line_en}"
                f"- Generation language: {_lang_human(gen_lang, False)}\n"
                f"- Public results: {'yes' if not incognito else 'no'}"
                + "\n\n<a href='https://github.com/YanGranat/book-in-one-click'>Project GitHub</a>"
            )
            text = intro_en + commands_block + results_block + settings_block
        await message.answer(text, disable_web_page_preview=True, parse_mode=types.ParseMode.HTML)


    # ===== Meme Extraction (admins and superadmin) =====
    @dp.message_handler(commands=["meme_extract"])  # type: ignore
    async def cmd_meme_extract(message: types.Message, state: FSMContext):
        # Admin or Superadmin gate
        from .bot_commands import SUPER_ADMIN_ID, ADMIN_IDS
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if not (is_superadmin or is_admin):
            return
        await dp.current_state(user=message.from_user.id, chat=message.chat.id).update_data(meme_waiting_file=True)
        await message.answer("Пришлите .txt или .md файл для экстракции мемов.")

    @dp.message_handler(lambda m: getattr(m.from_user, 'is_bot', False) is False, content_types=types.ContentTypes.DOCUMENT, state='*')  # type: ignore
    async def meme_file_received(message: types.Message, state: FSMContext):
        try:
            data = await state.get_data()
        except Exception:
            data = {}
        if not bool(data.get("meme_waiting_file")):
            return
        # Admin/Superadmin gate again
        from .bot_commands import SUPER_ADMIN_ID, ADMIN_IDS
        if not (message.from_user and (
            (SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
            or (message.from_user.id in ADMIN_IDS)
        )):
            return
        # Deduplicate by message id
        try:
            last_id = int(data.get("meme_last_doc_msg_id") or 0)
        except Exception:
            last_id = 0
        try:
            cur_id = int(message.message_id)
        except Exception:
            cur_id = 0
        if cur_id and last_id and cur_id == last_id:
            return
        try:
            await state.update_data(meme_last_doc_msg_id=cur_id)
        except Exception:
            pass
        # Chat-level lock to avoid duplicate runs across workers
        chat_id = message.chat.id
        if not await mark_chat_running(chat_id):
            return
        # Stop waiting for more files immediately
        try:
            await state.update_data(meme_waiting_file=False)
        except Exception:
            pass
        doc = message.document
        if not doc:
            await unmark_chat_running(chat_id)
            return
        name = (doc.file_name or "document").lower()
        if not (name.endswith('.txt') or name.endswith('.md')):
            await message.answer("Поддерживаются только .txt или .md")
            await unmark_chat_running(chat_id)
            return
        # Download file
        try:
            f = await dp.bot.get_file(doc.file_id)
            import aiohttp
            import io as _io
            url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{f.file_path}"
            async with aiohttp.ClientSession() as sess:
                async with sess.get(url) as resp:
                    buf = await resp.read()
            text = buf.decode('utf-8', errors='replace')
        except Exception:
            await message.answer("Не удалось загрузить файл.")
            try:
                await state.update_data(meme_waiting_file=False)
            except Exception:
                pass
            await unmark_chat_running(chat_id)
            return

        # Resolve provider/lang
        data = await state.get_data()
        gen_lang = (data.get("gen_lang") or "auto").strip().lower()
        # KV fallback for gen_lang
        try:
            if message.from_user and (not gen_lang or gen_lang == 'auto'):
                from .kv import get_gen_lang as _get_gen_lang
                kv_lang = await _get_gen_lang(message.from_user.id)
                if kv_lang:
                    gen_lang = kv_lang
        except Exception:
            pass
        prov = (data.get("provider") or "openai").strip().lower()
        # Resolve from KV only for superadmin
        try:
            if message.from_user:
                from .bot_commands import SUPER_ADMIN_ID as _SA
                if _SA is not None and int(message.from_user.id) == int(_SA):
                    from .kv import get_provider as _get_provider
                    prov = await _get_provider(message.from_user.id)
        except Exception:
            pass

        # Run extraction in thread pool
        await message.answer("Обрабатываю…")
        import asyncio as _asyncio
        from services.memom.extract import extract_memes
        loop = _asyncio.get_running_loop()
        try:
            # Pass job_meta with user ids for DB linkage
            job_meta = {"user_id": int(message.from_user.id) if message.from_user else 0}
            def _run():
                return extract_memes(text=text, source_name=name, lang=gen_lang or 'auto', provider=prov or 'openai', job_meta=job_meta)
            result = await loop.run_in_executor(None, _run)
        except Exception:
            await message.answer("Ошибка при экстракции мемов.")
            try:
                await state.update_data(meme_waiting_file=False)
            except Exception:
                pass
            await unmark_chat_running(chat_id)
            return

        # Read content back and send as .md
        try:
            from io import BytesIO
            if hasattr(result, 'read_text'):
                content = result.read_text(encoding='utf-8')  # type: ignore
                fname = Path(result).name  # type: ignore
            else:
                # if return_content mode was used
                content = str(result)
                from utils.slug import safe_filename_base as _sfb
                fname = f"{_sfb(Path(name).stem)}_memes.md"
            buf = BytesIO((content if content.endswith('\n') else content + '\n').encode('utf-8'))
            buf.name = fname
            await message.answer_document(buf, caption="Готово")
        except Exception:
            await message.answer("Готово (но не удалось отправить файл).")
        finally:
            try:
                await state.update_data(meme_waiting_file=False)
            except Exception:
                pass
            await unmark_chat_running(chat_id)


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
        # Always ack to stop Telegram spinner
        try:
            await query.answer()
        except Exception:
            pass
        cur = await state.get_data()
        in_settings = bool(cur.get("in_settings"))
        settings_view = cur.get("settings_view")
        if in_settings and query.message:
            # When in new settings UI, update keyboard with checkmark on new selection
            if settings_view == "gen_lang":
                try:
                    await query.message.edit_reply_markup(reply_markup=build_genlang_inline_with_check(gen_lang, ui_lang))
                except Exception:
                    pass
                return
            # Legacy inline panel refresh fallback
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
            try:
                await query.message.edit_reply_markup(reply_markup=kb)
            except Exception:
                pass
            return
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
                    prompt = ("Отправлять логи генерации?" if _is_ru(ui_lang) else "Send generation logs?")
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_enable_disable_inline("logs", ui_lang))
                else:
                    # Regular users: go directly to public results question
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
        kb = InlineKeyboardMarkup()
        for row in build_provider_inline().inline_keyboard:
            try:
                kb.row(*row)
            except Exception:
                for b in row:
                    kb.add(b)
        kb.add(InlineKeyboardButton(text=("⬅ Назад" if _is_ru(ui_lang) else "⬅ Back"), callback_data="settings:back"))
        await state.update_data(settings_view="provider")
        await message.answer(prompt, reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:provider:"))  # type: ignore
    async def cb_set_provider(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        # Accept string/int equality for SUPER_ADMIN_ID; also allow during onboarding for resilience
        onboarding_flag = False
        try:
            onboarding_flag = bool((await state.get_data()).get("onboarding"))
        except Exception:
            onboarding_flag = False
        if not (query.from_user and SUPER_ADMIN_ID is not None and str(query.from_user.id) == str(SUPER_ADMIN_ID)):
            if onboarding_flag:
                await query.answer()
                return
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
        cur = await state.get_data()
        in_settings = bool(cur.get("in_settings"))
        settings_view = cur.get("settings_view")
        if in_settings and query.message:
            if settings_view == "provider":
                try:
                    await query.message.edit_reply_markup(reply_markup=build_provider_inline_with_check(prov, ui_lang))
                except Exception:
                    pass
            else:
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
            # Onboarding continues after provider
            await query.message.edit_reply_markup() if query.message else None
            ok = "Провайдер установлен." if ui_lang == "ru" else "Provider set."
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ok)
            onboarding = bool((await state.get_data()).get("onboarding"))
            if onboarding:
                # Superadmin: ask logs next
                prompt = ("Отправлять логи генерации?" if _is_ru(ui_lang) else "Send generation logs?")
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_enable_disable_inline("logs", ui_lang))

    @dp.message_handler(commands=["interface_lang"])  # type: ignore
    async def cmd_lang(message: types.Message, state: FSMContext):
        await message.answer(
            "Choose interface language / Выберите язык интерфейса:",
            reply_markup=build_ui_lang_inline(),
        )
        # Inline only; no FSM step

    @dp.message_handler(commands=["generate"])  # type: ignore
    async def cmd_generate(message: types.Message, state: FSMContext):
        # Prevent starting new generation during active generation
        if await is_chat_running(message.chat.id):
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
            warn = "⏳ Генерация уже выполняется. Дождитесь завершения или используйте /cancel." if _is_ru(ui_lang) else "⏳ Generation in progress. Wait for completion or use /cancel."
            await message.answer(warn)
            return
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Reset transient mode flags to avoid leakage from previous flows (including in_settings!)
        try:
            await state.update_data(gen_article=False, series_mode=None, series_count=None, active_flow=None, next_after_fc=None, in_settings=False, settings_view=None)
        except Exception:
            pass
        # Note: FC/refine will be asked through flow for superadmin (not preloaded from KV)
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
            # Ask for post style for all roles first
            await state.update_data(gen_article=False, series_mode=None, series_count=None, active_flow="post")
            prompt = (
                "Выберите стиль поста:\n"
                "- Стиль 1 — структурированный образовательный; есть факт‑чек и редактура.\n"
                "- Стиль 2 — динамичный публицистический; без факт‑чека и редактуры."
                if ru else
                "Choose post style:\n"
                "- Style 1 — structured educational; fact‑check and refine available.\n"
                "- Style 2 — dynamic opinionated; no fact‑check, no refine."
            )
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_post_style_keyboard(ui_lang))
            await GenerateStates.ChoosingPostStyle.set()
            return
    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:post_style:"), state=GenerateStates.ChoosingPostStyle)  # type: ignore
    async def cb_post_style(query: types.CallbackQuery, state: FSMContext):
        style = (query.data or "").split(":")[-1]
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        await query.answer()
        # Persist style in FSM
        await state.update_data(post_style=style)
        # Role checks
        is_admin = bool(query.from_user and query.from_user.id in ADMIN_IDS)
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
        # For Style 1 and superadmin: ask FC -> Depth -> Refine -> Topic
        if style == "post_style_1" and is_superadmin:
            await state.update_data(next_after_fc="post")
            prompt = "Включить факт-чекинг?" if ru else "Enable fact-checking?"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("fc", ui_lang))
            return
        # For Style 2 or non-superadmin: go directly to topic
        prompt = "Отправьте тему для поста:" if ru else "Send a topic for your post:"
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=ReplyKeyboardRemove())
        await GenerateStates.WaitingTopic.set()

        # end of post flow branch; other branches below
        return

        if kind == "article":
            # Articles only support OpenAI for now (due to OpenAI Agents SDK dependency)
            await state.update_data(series_mode=None, series_count=None, gen_article=True, active_flow=None, next_after_fc=None, provider="openai")
            prompt = "Отправьте тему для статьи:" if ru else "Send a topic for your article:"
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=ReplyKeyboardRemove())
            await GenerateStates.WaitingTopic.set()
            return
        # Series branch: no FC/Refine for series
        if is_superadmin:
            await state.update_data(gen_article=False, series_mode=None, series_count=None)
            kb = InlineKeyboardMarkup()
            kb.add(
                InlineKeyboardButton(text="2", callback_data="set:series_preset:2"),
                InlineKeyboardButton(text="5", callback_data="set:series_preset:5"),
            )
            kb.add(
                InlineKeyboardButton(text=("Авто" if ru else "Auto"), callback_data="set:series_preset:auto"),
                InlineKeyboardButton(text=("Кастом" if ru else "Custom"), callback_data="set:series_preset:custom"),
            )
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Сколько постов?" if ru else "How many posts?"), reply_markup=kb)
            await GenerateStates.ChoosingSeriesPreset.set()
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
            chat_id_series = query.message.chat.id if query.message else query.from_user.id
            if SessionLocal is not None and query.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, query.from_user.id)
                    ok, remaining = await charge_credits(session, user, int(total), reason="post_series_fixed_prepay")
                    if ok:
                        await session.commit()
                        charged = True
                    else:
                        # Not enough credits in DB
                        need = max(0, int(total) - int(remaining))
                        warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if ru else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                        await query.answer()
                        await dp.bot.send_message(chat_id_series, warn)
                        if _stars_enabled():
                            try:
                                await dp.bot.send_message(chat_id_series, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if ru else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard("ru" if ru else "en"))
                            except Exception:
                                pass
                        await state.finish()
                        return
            if not charged and query.from_user:
                ok, remaining = await charge_credits_kv(query.from_user.id, int(total))
                if not ok:
                    need = max(0, int(total) - int(remaining))
                    warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if ru else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                    await query.answer()
                    await dp.bot.send_message(chat_id_series, warn)
                    if _stars_enabled():
                        try:
                            await dp.bot.send_message(chat_id_series, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if ru else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard("ru" if ru else "en"))
                        except Exception:
                            pass
                    await state.finish()
                    return
        except SQLAlchemyError:
            await query.answer()
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Временная ошибка БД. Попробуйте позже." if ru else "Temporary DB error. Try later."))
            await state.finish()
            return
        # Mark precharged and ask for topic
        await state.update_data(series_mode="fixed", series_count=int(count), series_precharged_amount=int(total))
        # Be defensive: Telegram may raise "Message is not modified" if markup already cleared
        try:
            await query.answer()
        except Exception:
            pass
        try:
            if query.message:
                await query.message.edit_reply_markup()
        except Exception:
            pass
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Ок. Отправьте тему для серии." if ru else "OK. Send a topic for the series."))
        await GenerateStates.WaitingTopic.set()

    # ---- Series command ----
    @dp.message_handler(commands=["series"])  # type: ignore
    async def cmd_series(message: types.Message, state: FSMContext):
        # Admin/superadmin only
        from .bot_commands import SUPER_ADMIN_ID
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        if not (is_admin or is_superadmin):
            return
        # Prevent starting new generation during active generation
        if await is_chat_running(message.chat.id):
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
            warn = "⏳ Генерация уже выполняется. Дождитесь завершения или используйте /cancel." if _is_ru(ui_lang) else "⏳ Generation in progress. Wait for completion or use /cancel."
            await message.answer(warn)
            return
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
        # Admin/superadmin only
        from .bot_commands import SUPER_ADMIN_ID
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        if not (is_admin or is_superadmin):
            return
        # Prevent starting new generation during active generation
        if await is_chat_running(message.chat.id):
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
            warn = "⏳ Генерация уже выполняется. Дождитесь завершения или используйте /cancel." if _is_ru(ui_lang) else "⏳ Generation in progress. Wait for completion or use /cancel."
            await message.answer(warn)
            return
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
        await state.update_data(in_settings=True, settings_view="main")
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(message.from_user and SUPER_ADMIN_ID is not None and int(message.from_user.id) == int(SUPER_ADMIN_ID))
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        ru = _is_ru(ui_lang)
        # Button-based settings main menu
        kb = build_settings_main_menu(ui_lang, is_admin=is_admin, is_superadmin=is_superadmin)
        await message.answer(("Настройки" if ru else "Settings"), reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("settings:open:"))  # type: ignore
    async def cb_settings_open(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        section = (query.data or "").split(":")[-1]
        await query.answer()
        # Open submenus with a Back button; actual toggles reuse existing callbacks
        if section == "provider":
            from .bot_commands import SUPER_ADMIN_ID
            if not (query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID)):
                return
            # Get current provider to show checkmark
            try:
                current_prov = await get_provider(query.from_user.id) if query.from_user else "openai"
            except Exception:
                current_prov = "openai"
            kb = build_provider_inline_with_check(current_prov, ui_lang)
            await state.update_data(settings_view="provider")
            if query.message:
                await query.message.edit_text(("Выберите провайдера:" if _is_ru(ui_lang) else "Choose provider:"), reply_markup=kb)
            return
        if section == "gen_lang":
            # Get current language to show checkmark
            try:
                current_lang = await get_gen_lang(query.from_user.id) if query.from_user else "auto"
            except Exception:
                current_lang = "auto"
            kb = build_genlang_inline_with_check(current_lang, ui_lang)
            await state.update_data(settings_view="gen_lang")
            if query.message:
                await query.message.edit_text(("Выберите язык генерации:" if _is_ru(ui_lang) else "Choose generation language:"), reply_markup=kb)
            return
        if section == "logs":
            # Get current logs setting to show checkmark
            try:
                current_logs = await get_logs_enabled(query.from_user.id) if query.from_user else False
            except Exception:
                current_logs = False
            kb = build_enable_disable_inline_with_check(current_logs, ui_lang)
            await state.update_data(settings_view="logs")
            if query.message:
                await query.message.edit_text(("Отправлять логи генерации?" if _is_ru(ui_lang) else "Send generation logs?"), reply_markup=kb)
            return
        if section == "public":
            # Get current incognito setting to show checkmark
            try:
                current_incog = await get_incognito(query.from_user.id) if query.from_user else False
            except Exception:
                current_incog = False
            # Note: incognito=True means private, so we invert for "public yes/no" question
            kb = build_yesno_inline_with_check(not current_incog, ui_lang)
            await state.update_data(settings_view="public")
            if query.message:
                await query.message.edit_text(("Сделать результаты публичными?" if _is_ru(ui_lang) else "Make results public?"), reply_markup=kb)
            return

    @dp.callback_query_handler(lambda c: c.data == "settings:back")  # type: ignore
    async def cb_settings_back(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
        is_admin = bool(query.from_user and query.from_user.id in ADMIN_IDS)
        await state.update_data(settings_view="main")
        kb = build_settings_main_menu(ui_lang, is_admin=is_admin, is_superadmin=is_superadmin)
        await query.answer()
        if query.message:
            await query.message.edit_text(("Настройки" if _is_ru(ui_lang) else "Settings"), reply_markup=kb)

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
        cur = await state.get_data()
        in_settings = bool(cur.get("in_settings"))
        settings_view = cur.get("settings_view")
        if in_settings and query.message:
            # If inside new logs submenu, show updated keyboard with checkmark
            if settings_view == "logs":
                try:
                    kb = build_enable_disable_inline_with_check(enabled, ui_lang)
                    await query.message.edit_reply_markup(reply_markup=kb)
                except Exception:
                    pass
            else:
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
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin_local = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
                kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=False, is_superadmin=is_superadmin_local)
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
        cur = await state.get_data()
        in_settings = bool(cur.get("in_settings"))
        settings_view = cur.get("settings_view")
        if in_settings and query.message:
            if settings_view == "public":
                try:
                    # Note: incognito=True means private, so we invert for "public yes/no"
                    kb = build_yesno_inline_with_check(not enabled, ui_lang)
                    await query.message.edit_reply_markup(reply_markup=kb)
                except Exception:
                    pass
            else:
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
                # After incognito decision → ask what to generate (series allowed for admins/superadmins)
                from .bot_commands import SUPER_ADMIN_ID
                is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
                kb = build_gentype_keyboard(ui_lang, allow_series=bool(is_superadmin or (query.from_user and query.from_user.id in ADMIN_IDS)))
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
            # Hide from admins/users completely
            try:
                await message.delete()
            except Exception:
                pass
            return
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
        await message.answer(prompt, reply_markup=build_yesno_inline("refine", ui_lang))

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:refine:"), state="*")  # type: ignore
    async def cb_set_refine(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        # Superadmin-only control; block admins/users entirely
        if not (query.from_user and SUPER_ADMIN_ID is not None and str(query.from_user.id) == str(SUPER_ADMIN_ID)):
            try:
                await query.answer()
            except Exception:
                pass
            return
        val = (query.data or "").split(":")[-1]
        enabled = (val == "yes")
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Always answer first to stop spinner
        try:
            await query.answer()
        except Exception:
            pass
        try:
            if query.from_user:
                await set_refine_enabled(query.from_user.id, enabled)
        except Exception:
            pass
        # Best-effort remove inline keyboard
        try:
            if query.message:
                await query.message.edit_reply_markup()
        except Exception:
            pass
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
            # Do not display refine/fact-check rows for admins
            is_admin_local = False
            is_superadmin = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
            kb = build_settings_keyboard(ui_lang, prov_cur, gen_lang, refine, logs_enabled, incognito, fc_enabled, int(fc_depth), is_admin=is_admin_local, is_superadmin=is_superadmin)
            await query.message.edit_reply_markup(reply_markup=kb)
        else:
            msg = (
                "Финальная редактура: включена." if (enabled and ui_lang=="ru") else (
                "Final refine: enabled." if enabled else (
                "Финальная редактура: отключена." if ui_lang=="ru" else "Final refine: disabled."
                ))
            )
            try:
                await query.message.edit_reply_markup() if query.message else None
            except Exception:
                pass
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, msg)
            # Continue flow by active_flow regardless of onboarding flag (superadmin only)
            sd_all = await state.get_data()
            from .bot_commands import SUPER_ADMIN_ID as _SA
            if query.from_user and _SA is not None and int(query.from_user.id) == int(_SA):
                active_flow = (sd_all.get("active_flow") or "").strip().lower()
                ru = _is_ru(ui_lang)
                if active_flow == "post":
                    prompt = "Отправьте тему для поста:" if ru else "Send a topic for your post:"
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=ReplyKeyboardRemove())
                    await GenerateStates.WaitingTopic.set()
                elif active_flow == "series":
                    kb = InlineKeyboardMarkup()
                    kb.add(
                        InlineKeyboardButton(text="2", callback_data="set:series_preset:2"),
                        InlineKeyboardButton(text="5", callback_data="set:series_preset:5"),
                    )
                    kb.add(
                        InlineKeyboardButton(text=("Авто" if ru else "Auto"), callback_data="set:series_preset:auto"),
                        InlineKeyboardButton(text=("Кастом" if ru else "Custom"), callback_data="set:series_preset:custom"),
                    )
                    await dp.bot.send_message(
                        query.message.chat.id if query.message else query.from_user.id,
                        ("Сколько постов?" if ru else "How many posts?"),
                        reply_markup=kb,
                    )
                    await GenerateStates.ChoosingSeriesPreset.set()

    @dp.message_handler(lambda m: (m.text or "").strip().lower() in {"cancel","отмена"}, state="*")  # type: ignore
    @dp.message_handler(commands=["cancel"], state="*")  # type: ignore
    async def cmd_cancel(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Silent cancel: no message text
        try:
            # Ensure any running job gates are released for this chat
            await unmark_chat_running(message.chat.id)
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

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc:"), state="*")  # type: ignore
    async def cb_set_fc(query: types.CallbackQuery, state: FSMContext):
        from .bot_commands import SUPER_ADMIN_ID
        # Superadmin only
        if not (query.from_user and SUPER_ADMIN_ID is not None and str(query.from_user.id) == str(SUPER_ADMIN_ID)):
            try:
                await query.answer()
            except Exception:
                pass
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
            if query.from_user and SUPER_ADMIN_ID is not None and str(query.from_user.id) == str(SUPER_ADMIN_ID):
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
            # Mark FC decision as done; continue flow if applicable
            await state.update_data(factcheck=False, research_iterations=None, fc_ready=True)
            sd2 = await state.get_data()
            active_flow2 = (sd2.get("active_flow") or "").strip().lower()
            if active_flow2 == "post":
                prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))
            elif onboarding:
                # Onboarding fallback: ask what to generate
                kb = build_gentype_keyboard(ui_lang, allow_series=False)
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"), reply_markup=kb)
                await GenerateStates.ChoosingGenType.set()

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:depth:"), state="*")  # type: ignore
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
            # Continue flow regardless of onboarding flag
            sd = await state.get_data()
            active_flow = (sd.get("active_flow") or "").strip().lower()
            if active_flow == "post":
                prompt = "Финальная редактура?" if _is_ru(ui_lang) else "Final refine?"
                await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, prompt, reply_markup=build_yesno_inline("refine", ui_lang))
            else:
                # If onboarding is active, fallback to choose what to generate; else just acknowledge
                if bool(sd.get("onboarding")):
                    kb = build_gentype_keyboard(ui_lang, allow_series=False)
                    await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Что генерировать?" if _is_ru(ui_lang) else "What to generate?"), reply_markup=kb)
                    await GenerateStates.ChoosingGenType.set()
                else:
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

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("set:fc_cmd:"), state="*")  # type: ignore
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
        # Diagnostics: entrypoint
        try:
            import sys as _sys
            print(f"[FLOW][topic_received] uid={getattr(getattr(message,'from_user',None),'id',None)} chat={getattr(getattr(message,'chat',None),'id',None)} mid={getattr(message,'message_id',None)}", file=_sys.stderr, flush=True)
        except Exception:
            pass
        text_raw = (message.text or "").strip()
        data = await state.get_data()
        try:
            import sys as _sys
            print(f"[FLOW][topic_received] text_len={len(text_raw)} keys={list(data.keys())[:10]}", file=_sys.stderr, flush=True)
        except Exception:
            pass
        ui_lang = data.get("ui_lang", "ru")
        # Mark this message as handled to prevent auto-chat from reusing it on webhook retries
        try:
            await state.update_data(last_handled_message_id=int(message.message_id))
        except Exception:
            pass
        # If user typed a command while waiting for topic — treat it as a command, not a topic
        if text_raw.startswith("/"):
            # Finish topic-waiting state and re-dispatch the command to its handler
            try:
                await state.finish()
            except Exception:
                pass
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
                    "interface_lang": cmd_lang,
                    "credits": cmd_credits,
                    "chat": cmd_chat,
                    "endchat": cmd_endchat,
                    "cancel": cmd_cancel,
                "meme_extract": cmd_meme_extract,
                }
                h = handlers.get(cmd)
                if h is not None:
                    try:
                        await h(message, state)  # type: ignore[arg-type]
                    except TypeError:
                        await h(message)  # type: ignore[misc]
            except Exception:
                pass
            return
        if text_raw.lower() in {"/cancel"}:
            try:
                await unmark_chat_running(message.chat.id)
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
            # Atomic check-and-set to prevent race conditions between workers
            if not await mark_chat_running(chat_id):
                return  # Another worker already started generation
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
                # Preserve 'auto' to let prompts choose language by topic; don't coerce to ru/en here
                eff_lang = ("auto" if (gen_lang or "auto").strip().lower() == "auto" else gen_lang)

                # Create Job row (article) and ensure User exists
                job_id = 0
                db_user_id = None
                try:
                    if SessionLocal is not None and message.from_user:
                        async with SessionLocal() as session:
                            from .db import Job
                            import json as _json
                            from .db import get_or_create_user as _get_or_create_user
                            db_user = await _get_or_create_user(session, message.from_user.id)
                            db_user_id = int(db_user.id)
                            params = {"topic": topic, "lang": eff_lang, "provider": prov or "openai"}
                            j = Job(user_id=db_user.id, type="article", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=100)
                            session.add(j)
                            await session.flush()
                            job_id = int(j.id)
                            await session.commit()
                except Exception:
                    job_id = 0

                # Charge 100 credits for article (admins free)
                is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
                if not is_admin:
                    from sqlalchemy.exc import SQLAlchemyError
                    try:
                        charged = False
                        if SessionLocal is not None and message.from_user:
                            async with SessionLocal() as session:
                                from .db import get_or_create_user
                                user = await get_or_create_user(session, message.from_user.id)
                                ok, remaining = await charge_credits(session, user, 100, reason="article")
                                if ok:
                                    await session.commit()
                                    charged = True
                                else:
                                    # Not enough credits in DB
                                    need = max(0, 100 - int(remaining))
                                    warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                    await message.answer(warn)
                                    if _stars_enabled():
                                        try:
                                            await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                        except Exception:
                                            pass
                                    await state.finish()
                                    await unmark_chat_running(chat_id)
                                    return
                        if not charged and message.from_user:
                            ok, remaining = await charge_credits_kv(message.from_user.id, 100)  # type: ignore
                            if not ok:
                                need = max(0, 100 - int(remaining))
                                warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                await message.answer(warn)
                                if _stars_enabled():
                                    try:
                                        await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                    except Exception:
                                        pass
                                await state.finish()
                                await unmark_chat_running(chat_id)
                                return
                    except SQLAlchemyError:
                        await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                        await state.finish()
                        await unmark_chat_running(chat_id)
                        return

                await message.answer("Генерирую…" if _is_ru(ui_lang) else "Generating…")
                try:
                    print(f"[BOT][ARTICLE][START] uid={message.from_user.id if message.from_user else 0} chat={message.chat.id} topic_len={len(topic)} prov={prov} lang={eff_lang}", file=sys.stderr)
                except Exception:
                    pass
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
                # Articles: disable research/refine modules by default per new spec
                fc_flag = False
                refine_flag = False

                fut = loop.run_in_executor(
                    None,
                    lambda: generate_article(
                        topic,
                        lang=eff_lang,
                        provider=((prov if prov != "auto" else "openai") or "openai"),
                        output_subdir="deep_article",
                        job_meta={"user_id": message.from_user.id if message.from_user else 0, "db_user_id": db_user_id, "chat_id": message.chat.id, "job_id": job_id, "incognito": inc_flag},
                        enable_research=False,
                        enable_refine=False,
                    ),
                )
                try:
                    article_path = await _asyncio.wait_for(fut, timeout=timeout_s)
                    try:
                        print(f"[BOT][ARTICLE][DONE] path={article_path}", file=sys.stderr)
                    except Exception:
                        pass
                except _asyncio.TimeoutError:
                    warn = (
                        f"Превышено время ожидания ({int(timeout_s/60)} мин). Генерация продолжается в фоне; проверьте /results-ui позже."
                        if _is_ru(ui_lang) else
                        f"Timeout ({int(timeout_s/60)} min). Generation continues in background; check /results-ui later."
                    )
                    await message.answer(warn)
                    await state.finish()
                    await unmark_chat_running(chat_id)
                    return
                # Send main result
                try:
                    with open(article_path, "rb") as f:
                        cap = ("Готово (статья): " + Path(article_path).name) if _is_ru(ui_lang) else ("Done (article): " + Path(article_path).name)
                        await message.answer_document(f, caption=cap)
                except Exception as e:
                    try:
                        import traceback as _tb
                        print(f"[BOT][ARTICLE][SEND_ERR] {type(e).__name__}: {e}", file=sys.stderr)
                        _tb.print_exc()
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
                except Exception as e:
                    try:
                        import traceback as _tb
                        print(f"[BOT][ARTICLE][LOG_SEND_ERR] {type(e).__name__}: {e}", file=sys.stderr)
                        _tb.print_exc()
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
                try:
                    import traceback as _tb
                    print(f"[BOT][ARTICLE][ERR] {type(e).__name__}: {e}", file=sys.stderr)
                    _tb.print_exc()
                except Exception:
                    pass
            finally:
                await state.finish()
                await unmark_chat_running(chat_id)
            return

        # If series mode is active, branch into series generation flow
        series_mode = (data.get("series_mode") or "").strip().lower()
        if series_mode in {"auto", "fixed"}:
            chat_id = message.chat.id
            # Atomic check-and-set to prevent race conditions between workers
            if not await mark_chat_running(chat_id):
                return  # Another worker already started generation
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
                # Preserve 'auto' to let prompts choose language by topic; don't coerce to ru/en here
                eff_lang = ("auto" if (gen_lang or "auto").strip().lower() == "auto" else gen_lang)

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
                # Only superadmin may use refine/fact-check
                try:
                    from .bot_commands import SUPER_ADMIN_ID as _SA
                    is_superadmin_local = bool(message.from_user and _SA is not None and int(message.from_user.id) == int(_SA))
                except Exception:
                    is_superadmin_local = False
                if not is_superadmin_local:
                    refine_pref = False
                    fc_enabled_pref = False
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
                        await unmark_chat_running(chat_id)
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
                                        need = max(0, int(total_cost) - int(remaining))
                                        warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                        await message.answer(warn)
                                        if _stars_enabled():
                                            try:
                                                await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                            except Exception:
                                                pass
                                        await state.finish()
                                        await unmark_chat_running(chat_id)
                                        return
                                    await session.commit()
                                    precharged = int(total_cost)
                            else:
                                ok, remaining = await charge_credits_kv(message.from_user.id, int(total_cost))  # type: ignore
                                if not ok:
                                    need = max(0, int(total_cost) - int(remaining))
                                    warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                    await message.answer(warn)
                                    if _stars_enabled():
                                        try:
                                            await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                        except Exception:
                                            pass
                                    await state.finish()
                                    await unmark_chat_running(chat_id)
                                    return
                                precharged = int(total_cost)
                        except SQLAlchemyError:
                            await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                            await state.finish()
                            await unmark_chat_running(chat_id)
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
                            await message.answer(("Недостаточно кредитов. Используйте /credits для пополнения." if _is_ru(ui_lang) else "Insufficient credits. Use /credits to top up."))
                            if _stars_enabled():
                                try:
                                    await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                except Exception:
                                    pass
                            await state.finish()
                            await unmark_chat_running(chat_id)
                            return
                        # Compute max posts affordable with current options
                        if unit_cost <= 0:
                            unit_cost = 1
                        target_count = prepay_budget // unit_cost
                        if target_count <= 0:
                            await message.answer(
                                ("Недостаточно кредитов для текущих настроек (стоимость поста слишком высока). Отключите факт‑чек/редактуру или пополните баланс через /credits."
                                 if _is_ru(ui_lang) else
                                 "Insufficient credits for current settings (post cost too high). Disable fact‑check/refine or top up via /credits.")
                            )
                            if _stars_enabled():
                                try:
                                    await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                except Exception:
                                    pass
                            await state.finish()
                            await unmark_chat_running(chat_id)
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
                                        need = max(0, int(prepay_budget) - int(remaining))
                                        warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                        await message.answer(warn)
                                        if _stars_enabled():
                                            try:
                                                await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                            except Exception:
                                                pass
                                        await state.finish()
                                        await unmark_chat_running(chat_id)
                                        return
                                    await session.commit()
                                    precharged = int(prepay_budget)
                            else:
                                ok, remaining = await charge_credits_kv(message.from_user.id, int(prepay_budget))  # type: ignore
                                if not ok:
                                    need = max(0, int(prepay_budget) - int(remaining))
                                    warn = (f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения." if _is_ru(ui_lang) else f"Insufficient credits. Need: {need}. Use /credits to top up.")
                                    await message.answer(warn)
                                    if _stars_enabled():
                                        try:
                                            await message.answer(("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                                        except Exception:
                                            pass
                                    await state.finish()
                                    await unmark_chat_running(chat_id)
                                    return
                                precharged = int(prepay_budget)
                        except SQLAlchemyError:
                            await message.answer("Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later.")
                            await state.finish()
                            await unmark_chat_running(chat_id)
                            return

                # Create Job row (series)
                job_id = 0
                db_user_id = None
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
                            db_user_id = int(db_user.id)
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
                timeout_s = int(os.getenv("GEN_TIMEOUT_S", "3600"))
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
                        job_meta={"user_id": message.from_user.id if message.from_user else 0, "db_user_id": db_user_id, "chat_id": message.chat.id, "job_id": job_id},
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
                await unmark_chat_running(chat_id)
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

        # Optional confirmation with price (skip for admins)
        # NOTE: Don't mark chat as running yet - only after confirmation
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
                # Style-dependent pricing: Style 2 ignores FC/Refine
                try:
                    post_style = (data.get("post_style") or "post_style_1").strip().lower()
                except Exception:
                    post_style = "post_style_1"
                if post_style == "post_style_2":
                    refine_pref = False
                    fc_enabled_pref = False
                    fc_depth_pref = 0
                fc_extra = (3 if int(fc_depth_pref or 0) >= 3 else 1) if fc_enabled_pref else 0
                unit_cost = 1 + fc_extra + (1 if refine_pref else 0)
                try:
                    import sys as _sys
                    print(f"[FLOW][confirm_show] uid={getattr(getattr(message,'from_user',None),'id',None)} cost={unit_cost} refine={refine_pref} fc={fc_enabled_pref} depth={fc_depth_pref} style={post_style}", file=_sys.stderr, flush=True)
                except Exception:
                    pass
                confirm_txt = (
                    f"Будет списано {unit_cost} кредит(ов). Подтвердить?"
                    if _is_ru(ui_lang_local) else
                    f"It will cost {unit_cost} credit(s). Proceed?"
                )
                kb = InlineKeyboardMarkup()
                kb.add(InlineKeyboardButton(text=("Подтвердить" if _is_ru(ui_lang_local) else "Confirm"), callback_data="confirm:charge:yes"))
                kb.add(InlineKeyboardButton(text=("Отмена" if _is_ru(ui_lang_local) else "Cancel"), callback_data="confirm:charge:no"))
                await message.answer(confirm_txt, reply_markup=kb)
                # Save pending topic and wait for callback (chat will be marked running in cb_confirm_charge)
                await state.update_data(pending_topic=topic)
                try:
                    import sys as _sys
                    print(f"[FLOW][confirm_saved] uid={getattr(getattr(message,'from_user',None),'id',None)} topic_saved_len={len(topic)}", file=_sys.stderr, flush=True)
                except Exception:
                    pass
                return
            except Exception:
                pass
        
        # Atomic check-and-set to prevent race conditions between workers
        # (only for admins who skip confirmation, or after exception in confirmation flow)
        if not await mark_chat_running(chat_id):
            return  # Another worker already started generation

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
            # Style-dependent pricing: Style 2 ignores FC/Refine
            try:
                post_style = (data.get("post_style") or "post_style_1").strip().lower()
            except Exception:
                post_style = "post_style_1"
            if post_style == "post_style_2":
                refine_pref = False
                fc_enabled_pref = False
                fc_depth_pref = 0
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
                    else:
                        # Not enough credits in DB
                        need = max(0, int(unit_cost) - int(remaining))
                        warn = (
                            f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения."
                            if _is_ru(ui_lang)
                            else f"Insufficient credits. Need: {need}. Use /credits to top up."
                        )
                        await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                        if _stars_enabled():
                            try:
                                await dp.bot.send_message(chat_id, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                            except Exception:
                                pass
                        await state.finish()
                        await unmark_chat_running(chat_id)
                        return
            if not charged and message.from_user:
                ok, remaining = await charge_credits_kv(message.from_user.id, unit_cost)
                if not ok:
                    need = max(0, int(unit_cost) - int(remaining))
                    warn = (
                        f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения."
                        if _is_ru(ui_lang)
                        else f"Insufficient credits. Need: {need}. Use /credits to top up."
                    )
                    await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                    if _stars_enabled():
                        try:
                            await dp.bot.send_message(chat_id, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                        except Exception:
                            pass
                    await state.finish()
                    await unmark_chat_running(chat_id)
                    return
        except SQLAlchemyError:
            # For admin path we do not attempt DB charge; this error indicates a true DB failure for non-admin charge
            warn = "Временная ошибка БД. Попробуйте позже." if ui_lang == "ru" else "Temporary DB error. Try later."
            await message.answer(warn, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            await unmark_chat_running(chat_id)
            return

        # Create Job row (running) and ensure User exists
        job_id = 0
        db_user_id = None  # Track User.id for job_meta
        try:
            if SessionLocal is not None and message.from_user:
                async with SessionLocal() as session:
                    from .db import Job
                    from .db import get_or_create_user as _get_or_create_user
                    import json as _json
                    db_user = await _get_or_create_user(session, message.from_user.id)
                    db_user_id = int(db_user.id)  # Capture User.id
                    print(f"[DEBUG] User resolved: telegram_id={message.from_user.id}, User.id={db_user_id}")
                    params = {
                        "topic": topic,
                        "lang": (data.get("gen_lang") or "auto"),
                        "provider": (data.get("provider") or "openai"),
                        "style": (data.get("post_style") or "post_style_1"),
                        "factcheck": bool(data.get("factcheck")),
                        "depth": int(data.get("research_iterations") or 0) if bool(data.get("factcheck")) else 0,
                        "refine": bool(await get_refine_enabled(message.from_user.id) if message.from_user else False),
                    }
                    j = Job(user_id=db_user.id, type="post", status="running", params_json=_json.dumps(params, ensure_ascii=False), cost=1)
                    session.add(j)
                    await session.flush()
                    job_id = int(j.id)
                    await session.commit()
                    print(f"[DEBUG] Job created: Job.id={job_id}, User.id={db_user_id}")
        except Exception as e:
            print(f"[ERROR] Job creation FAILED for telegram_id={message.from_user.id if message.from_user else 'N/A'}, User.id={db_user_id}: {type(e).__name__}: {str(e)[:300]}")
            job_id = 0

        # Light progress notes before long run
        try:
            working = "Генерирую…" if _is_ru(ui_lang) else "Generating…"
            await message.answer(working, reply_markup=ReplyKeyboardRemove())
        except Exception:
            pass

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
            # Preserve 'auto' to let prompts choose language by topic; don't coerce to ru/en here
            eff_lang = ("auto" if (gen_lang or "auto").strip().lower() == "auto" else gen_lang)

            # Refine preference (superadmin only)
            refine_enabled = False
            try:
                if message.from_user:
                    refine_enabled = await get_refine_enabled(message.from_user.id)
            except Exception:
                refine_enabled = False
            try:
                from .bot_commands import SUPER_ADMIN_ID as _SA
                is_superadmin_local = bool(message.from_user and _SA is not None and int(message.from_user.id) == int(_SA))
            except Exception:
                is_superadmin_local = False
            if not is_superadmin_local:
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
            # Only superadmin may use fact-check
            if not is_superadmin_local:
                fc_enabled_state = False
                depth = None

            # Prepare job metadata for logging (removed GLOBAL_SEMAPHORE to prevent blocking)
            job_meta = {
                "user_id": message.from_user.id if message.from_user else 0,  # telegram_id for backward compat
                "db_user_id": db_user_id,  # User.id for accurate fallback Job creation
                "chat_id": message.chat.id,
                "topic": topic,
                "provider": prov or "openai",
                "lang": eff_lang,
                "style": (data.get("post_style") or "post_style_1"),
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
                            # Schedule send on main loop safely from executor thread
                            try:
                                import asyncio as _aio
                                loop_main = _aio.get_running_loop()
                                loop_main.create_task(message.answer(txt))
                            except Exception:
                                pass
                except Exception:
                    pass

            timeout_s = int(os.getenv("GEN_TIMEOUT_S", "2400"))
            # Respect style: force fc off for style 2
            post_style = (data.get("post_style") or "post_style_1").strip().lower()
            if post_style == "post_style_2":
                fc_enabled_state = False
                depth = None

            if fc_enabled_state:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=eff_lang,
                        provider=((prov if prov != "auto" else "openai") or "openai"),
                        style=post_style,
                        factcheck=True,
                        factcheck_max_items=int(os.getenv("FC_MAX_ITEMS", "7")),
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
                        style=post_style,
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
        except asyncio.TimeoutError:
            warn = (
                "Превышено время ожидания генерации. Уменьшите глубину факт-чекинга, отключите финальную редактуру или попробуйте позже."
                if _is_ru(ui_lang) else
                "Generation timed out. Reduce fact-check depth, disable final refine, or try again later."
            )
            await message.answer(warn)
        except Exception as e:
            err = f"Ошибка: {e}" if _is_ru(ui_lang) else f"Error: {e}"
            await message.answer(err)
            # Persist minimal failure log to DB so it appears in /logs-ui even on errors
            try:
                from datetime import datetime as _dt
                import json as _json
                if os.getenv("DB_URL", "").strip():
                    # Build minimal log content
                    started_at_str = _dt.utcnow().strftime('%Y-%m-%d %H:%M')
                    stages_text = "\n".join([f"- {s}" for s in stages]) if stages else "(no stages recorded)"
                    header = (
                        f"# 🧾 Generation Log (failure)\n\n"
                        f"- provider: {prov if prov else 'openai'}\n"
                        f"- lang: {eff_lang}\n"
                        f"- started_at: {started_at_str}\n"
                        f"- topic: {topic}\n"
                        f"- stages:\n{stages_text}\n\n"
                        f"- error: {str(e)}\n"
                    )
                    full_log_content = header
                    # Write JobLog row synchronously
                    from sqlalchemy import create_engine
                    from sqlalchemy.orm import sessionmaker
                    from urllib.parse import urlsplit, urlunsplit, parse_qs
                    db_url = os.getenv("DB_URL", "")
                    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
                    parts = urlsplit(sync_url)
                    qs = parse_qs(parts.query or "")
                    base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
                    cargs = {}
                    if "sslmode" not in {k.lower() for k in qs.keys()}:
                        cargs["sslmode"] = "require"
                    sync_engine = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True, pool_size=2, max_overflow=0)
                    SyncSession = sessionmaker(sync_engine)
                    with SyncSession() as s:
                        try:
                            from .db import JobLog
                            jid = 0
                            try:
                                jid = int((job_meta or {}).get("job_id", 0)) if isinstance(job_meta, dict) else 0
                            except Exception:
                                jid = 0
                            jl = JobLog(job_id=jid, kind="md", path=None, content=full_log_content)
                            s.add(jl)
                            s.flush()
                            s.commit()
                        except Exception:
                            s.rollback()
                    try:
                        sync_engine.dispose()
                    except Exception:
                        pass
            except Exception:
                pass
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
        finally:
            await state.finish()
            await unmark_chat_running(chat_id)

    @dp.callback_query_handler(lambda c: c.data in {"confirm:charge:yes","confirm:charge:no"}, state="*")  # type: ignore
    async def cb_confirm_charge(query: types.CallbackQuery, state: FSMContext):
        try:
            import sys as _sys
            print(f"[FLOW][cb_confirm_charge] from={getattr(getattr(query,'from_user',None),'id',None)} chat={getattr(getattr(getattr(query,'message',None),'chat',None),'id',None)} data={(getattr(query,'data',None))}", file=_sys.stderr, flush=True)
        except Exception:
            pass
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        # Always ack and try to remove inline keyboard to avoid hanging spinner
        try:
            await query.answer()
        except Exception:
            pass
        try:
            if query.message:
                await query.message.edit_reply_markup()
        except Exception:
            pass
        if query.data.endswith(":no"):
            chat_id = query.message.chat.id if query.message else (query.from_user.id if query.from_user else 0)
            # No need to unmark - chat was never marked as running (marking happens only on ":yes")
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
        try:
            import sys as _sys
            print(f"[FLOW][cb_confirm_charge] pending_topic_len={len(topic)}", file=_sys.stderr, flush=True)
        except Exception:
            pass
        if not topic:
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, "Тема не найдена, отправьте заново /generate" if _is_ru(ui_lang) else "Topic missing, send /generate again")
            await state.finish()
            return
        chat_id = query.message.chat.id if query.message else query.from_user.id
        # Atomic check-and-set to prevent race conditions between workers
        if not await mark_chat_running(chat_id):
            try:
                import sys as _sys
                print(f"[FLOW][cb_confirm_charge] already_running chat={chat_id}", file=_sys.stderr, flush=True)
            except Exception:
                pass
            await query.answer()
            return  # Another worker already started generation
        # Charge
        from sqlalchemy.exc import SQLAlchemyError
        # Only superadmin may use fact-check/refine; verify before computing price
        from .bot_commands import SUPER_ADMIN_ID
        is_superadmin_local = bool(query.from_user and SUPER_ADMIN_ID is not None and int(query.from_user.id) == int(SUPER_ADMIN_ID))
        try:
            charged = False
            if SessionLocal is not None and query.from_user:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, query.from_user.id)
                    # Compute unit price again (respect superadmin-only features)
                    refine_pref = False
                    fc_enabled_pref = False
                    fc_depth_pref = 2
                    if is_superadmin_local:
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
                    try:
                        import sys as _sys
                        print(f"[FLOW][charge_db] uid={getattr(getattr(query,'from_user',None),'id',None)} cost={unit_cost}", file=_sys.stderr, flush=True)
                    except Exception:
                        pass
                    ok, remaining = await charge_credits(session, user, unit_cost, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
                        try:
                            import sys as _sys
                            print(f"[FLOW][charge_db_ok] remaining={remaining}", file=_sys.stderr, flush=True)
                        except Exception:
                            pass
                    else:
                        # Not enough credits in DB
                        need = max(0, int(unit_cost) - int(remaining))
                        warn = (
                            f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения."
                            if _is_ru(ui_lang)
                            else f"Insufficient credits. Need: {need}. Use /credits to top up."
                        )
                        await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                        if _stars_enabled():
                            try:
                                await dp.bot.send_message(chat_id, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                            except Exception:
                                pass
                        await state.finish()
                        await unmark_chat_running(chat_id)
                        return
            if not charged and query.from_user:
                # KV fallback (respect superadmin-only features)
                refine_pref = False
                fc_enabled_pref = False
                fc_depth_pref = 2
                if is_superadmin_local:
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
                try:
                    import sys as _sys
                    print(f"[FLOW][charge_kv] uid={getattr(getattr(query,'from_user',None),'id',None)} cost={unit_cost}", file=_sys.stderr, flush=True)
                except Exception:
                    pass
                ok, remaining = await charge_credits_kv(query.from_user.id, unit_cost)
                if not ok:
                    try:
                        import sys as _sys
                        print(f"[FLOW][charge_kv_fail] bal={remaining}", file=_sys.stderr, flush=True)
                    except Exception:
                        pass
                    need = max(0, int(unit_cost) - int(remaining))
                    warn = (
                        f"Недостаточно кредитов. Не хватает: {need}. Используйте /credits для пополнения."
                        if _is_ru(ui_lang)
                        else f"Insufficient credits. Need: {need}. Use /credits to top up."
                    )
                    await dp.bot.send_message(chat_id, warn, reply_markup=ReplyKeyboardRemove())
                    if _stars_enabled():
                        try:
                            await dp.bot.send_message(chat_id, ("Купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "Buy credits with ⭐? (1 credit = 50⭐)"), reply_markup=build_buy_keyboard(ui_lang))
                        except Exception:
                            pass
                    await state.finish()
                    await unmark_chat_running(chat_id)
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if _is_ru(ui_lang) else "Temporary DB error. Try later."
            await dp.bot.send_message(chat_id, warn)
            await unmark_chat_running(chat_id)
            await state.finish()
            await query.answer()
            return

        await query.answer()
        try:
            if query.message:
                await query.message.edit_reply_markup()
        except Exception:
            pass
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
            # Preserve 'auto' to let prompts choose language by topic; don't coerce to ru/en here
            eff_lang = ("auto" if (gen_lang or "auto").strip().lower() == "auto" else gen_lang)
            try:
                import sys as _sys
                print(f"[FLOW][gen_start] uid={getattr(getattr(query,'from_user',None),'id',None)} topic_len={len(topic)} lang={eff_lang} prov={prov} style={post_style}", file=_sys.stderr, flush=True)
            except Exception:
                pass
            
            # Refine preference (superadmin only)
            refine_enabled = False
            if is_superadmin_local:
                try:
                    if query.from_user:
                        refine_enabled = await get_refine_enabled(query.from_user.id)
                except Exception:
                    refine_enabled = False
            # Style selection (mandatory before topic)
            post_style = (data.get("post_style") or "post_style_1").strip().lower()
            # Fact-check decision (superadmin only) and style-aware
            fc_enabled_state = False
            depth = None
            if is_superadmin_local:
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
            # Force off for style 2
            if post_style == "post_style_2":
                refine_enabled = False
                fc_enabled_state = False
                depth = None
            # Create Job (running) and ensure User exists
            job_id = 0
            db_user_id = None  # Track User.id for job_meta
            try:
                if SessionLocal is not None and query.from_user:
                    async with SessionLocal() as session:
                        from .db import Job
                        from .db import get_or_create_user as _get_or_create_user
                        import json as _json
                        db_user = await _get_or_create_user(session, query.from_user.id)
                        db_user_id = int(db_user.id)  # Capture User.id
                        params = {
                            "topic": topic,
                            "lang": eff_lang,
                            "provider": prov or "openai",
                            "style": post_style,
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
                "user_id": query.from_user.id if query.from_user else 0,  # telegram_id for backward compat
                "db_user_id": db_user_id,  # User.id for accurate fallback Job creation
                "chat_id": chat_id,
                "topic": topic,
                "provider": prov or "openai",
                "lang": eff_lang,
                "style": post_style,
                "incognito": (await get_incognito(query.from_user.id)) if query.from_user else False,
                "refine": refine_enabled,
            }
            if job_id:
                job_meta["job_id"] = job_id
            await dp.bot.send_message(chat_id, "Генерирую…" if _is_ru(ui_lang) else "Working…")
            timeout_s = int(os.getenv("GEN_TIMEOUT_S", "3600"))
            if fc_enabled_state:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(topic, lang=eff_lang, provider=((prov if prov != "auto" else "openai") or "openai"), style=post_style, factcheck=True, factcheck_max_items=int(os.getenv("FC_MAX_ITEMS", "3")), research_iterations=int(depth or 2), job_meta=job_meta, use_refine=refine_enabled),
                )
                path = await asyncio.wait_for(fut, timeout=timeout_s)
            else:
                fut = loop.run_in_executor(
                    None,
                    lambda: generate_post(topic, lang=eff_lang, provider=((prov if prov != "auto" else "openai") or "openai"), style=post_style, factcheck=False, job_meta=job_meta, use_refine=refine_enabled),
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
        finally:
            await state.finish()
            await unmark_chat_running(chat_id)

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
            # Map telegram -> DB user.id (Job.user_id always stores User.id)
            db_uid: Optional[int] = None
            try:
                uq = await _s.execute(_select(User).where(User.telegram_id == int(telegram_user_id)))
                urow = uq.scalars().first()
                if urow is not None:
                    db_uid = int(urow.id)
            except Exception:
                db_uid = None
            if db_uid is None:
                return []
            jn = _join(ResultDoc, Job, ResultDoc.job_id == Job.id)
            sel = _select(ResultDoc.id).select_from(jn).where(Job.user_id == db_uid)
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
                    # Job.user_id ALWAYS stores User.id, not telegram_id directly
                    # So we MUST resolve telegram_id -> User.id first
                    db_uid = None
                    try:
                        uq = await _s.execute(_select(User).where(User.telegram_id == int(message.from_user.id)))
                        urow = uq.scalars().first()
                        if urow is not None:
                            db_uid = int(urow.id)
                    except Exception:
                        db_uid = None
                    
                    # If User not found, history will be empty (as it should be - no generations yet)
                    if db_uid is None:
                        items = []
                    else:
                        jn = _join(ResultDoc, Job, ResultDoc.job_id == Job.id)
                        # Job.user_id always stores User.id (normalized schema)
                        res = await _s.execute(
                            _select(ResultDoc, Job.user_id)
                            .select_from(jn)
                            .where(Job.user_id == db_uid)
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
        except Exception as e:
            print(f"[ERROR] History fetch failed: {e}")
            items = []
        if not items:
            await message.answer("История пуста." if _is_ru(ui_lang) else "No history yet.")
            return
        lines = []
        for it in items:
            topic = (it.get("topic") or "(no topic)")
            kind = (it.get("kind") or "").lower()
            # Skip meme_extract items from user-visible history
            if kind == "meme_extract":
                continue
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
        ru = _is_ru(ui_lang)
        kb = types.InlineKeyboardMarkup()
        kb.add(
            types.InlineKeyboardButton(text=("Да, удалить" if ru else "Yes, delete"), callback_data="history:confirm_clear:yes"),
            types.InlineKeyboardButton(text=("Отмена" if ru else "Cancel"), callback_data="history:confirm_clear:no"),
        )
        await message.answer(("Очистить всю историю? Это удалит и результаты на странице результатов." if ru else "Clear all history? This will also delete results from the results page."), reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data == "history:clear")  # type: ignore
    async def cb_history_clear(query: types.CallbackQuery, state: FSMContext):
        # Show confirmation instead of immediate deletion
        try:
            data = await state.get_data()
            ui_lang = (data.get("ui_lang") or "ru").strip()
        except Exception:
            ui_lang = "ru"
        ru = _is_ru(ui_lang)
        kb = types.InlineKeyboardMarkup()
        kb.add(
            types.InlineKeyboardButton(text=("Да, удалить" if ru else "Yes, delete"), callback_data="history:confirm_clear:yes"),
            types.InlineKeyboardButton(text=("Отмена" if ru else "Cancel"), callback_data="history:confirm_clear:no"),
        )
        try:
            await query.answer()
        except Exception:
            pass
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Очистить всю историю? Это удалит и результаты на странице результатов." if ru else "Clear all history? This will also delete results from the results page."), reply_markup=kb)

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("history:confirm_clear:"))  # type: ignore
    async def cb_history_confirm_clear(query: types.CallbackQuery, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        ru = _is_ru(ui_lang)
        if (query.data or "").endswith(":no"):
            try:
                await query.answer()
            except Exception:
                pass
            if query.message:
                try:
                    await query.message.edit_reply_markup()
                except Exception:
                    pass
            await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("Отменено." if ru else "Cancelled."))
            return
        # yes
        try:
            await clear_history(query.from_user.id)
            from .kv import set_history_cleared_at
            await set_history_cleared_at(query.from_user.id)
            await _delete_results_for_user(query.from_user.id, only_visible=False)
        except Exception:
            pass
        try:
            await query.answer("OK")
        except Exception:
            pass
        if query.message:
            try:
                await query.message.edit_reply_markup()
            except Exception:
                pass
        await dp.bot.send_message(query.message.chat.id if query.message else query.from_user.id, ("История очищена." if ru else "History cleared."))

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
            chat_id=query.message.chat.id if query.message else query.from_user.id,
            text=(prompt_ru if _is_ru(ui_lang) else prompt_en),
            reply_markup=kb,
        )

    @dp.message_handler(state=HistoryStates.WaitingDeleteIds, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def history_delete_ids_received(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        raw = (message.text or "").strip()
        # If user typed a command while waiting for IDs — treat it as a command, exit state
        if raw.startswith("/"):
            try:
                await state.finish()
            except Exception:
                pass
            try:
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
                    "interface_lang": cmd_lang,
                    "credits": cmd_credits,
                    "chat": cmd_chat,
                    "endchat": cmd_endchat,
                    "cancel": cmd_cancel,
                    "meme_extract": cmd_meme_extract,
                }
                h = handlers.get(cmd)
                if h is not None:
                    try:
                        await h(message, state)  # type: ignore[arg-type]
                    except TypeError:
                        await h(message)  # type: ignore[misc]
            except Exception:
                pass
            return
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
            chat_id=query.message.chat.id if query.message else query.from_user.id,
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
            chat_id=query.message.chat.id if query.message else query.from_user.id,
            text=((f"Удалено: {deleted}" if _is_ru(ui_lang) else f"Deleted: {deleted}") if deleted > 0 else ("Ничего не удалено." if _is_ru(ui_lang) else "Nothing deleted.")),
        )

    # Legacy state handler removed: fact-check choices are inline-only now

    # Legacy depth state handler removed: depth is inline-only now

    # ---- Balance and purchasing with Telegram Stars ----
    @dp.message_handler(commands=["balance"])  # type: ignore
    async def cmd_balance(message: types.Message, state: FSMContext):
        """Redirect to /credits for regular users, keep for backward compatibility"""
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if not is_admin:
            # Regular users should use /credits instead
            await cmd_credits(message, state)
            return
        
        # For admins, show balance info
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
                    txt + ("\nХотите купить кредиты за ⭐? (1 кредит = 50⭐)" if _is_ru(ui_lang) else "\nWant to buy credits with ⭐? (1 credit = 50⭐)"),
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
                (("Выберите пакет кредитов (1 кредит = 50⭐):" if _is_ru(ui_lang) else "Choose a credits pack (1 credit = 50⭐):")),
                reply_markup=build_buy_keyboard(ui_lang),
            )
        else:
            await message.answer("Покупка через ⭐ недоступна." if _is_ru(ui_lang) else "Buying with ⭐ is unavailable.")

    @dp.message_handler(commands=["pricing"])  # type: ignore
    async def cmd_pricing(message: types.Message, state: FSMContext):
        """Show pricing. For regular users, redirect to /credits"""
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if not is_admin:
            # Regular users should use /credits instead (which includes pricing)
            await cmd_credits(message, state)
            return
        
        # For admins, show detailed pricing
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if _is_ru(ui_lang):
            await message.answer(
                "Цены:\n"
                "- Пост: 1 кредит\n"
                "- Серия: 1×N кредитов\n"
                "- Статья: 100 кредитов\n"
            )
        else:
            await message.answer(
                "Pricing:\n"
                "- Post: 1 credit\n"
                "- Series: 1×N credits\n"
                "- Article: 100 credits\n"
            )

    @dp.callback_query_handler(lambda c: c.data and c.data.startswith("buy:stars:"))  # type: ignore
    async def cb_buy_stars(query: types.CallbackQuery, state: FSMContext):
        credits_map = {"1": 1, "5": 5, "10": 10, "50": 50, "100": 100, "500": 500}
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
            prices = [LabeledPrice(label=f"Credits x{pack}", amount=pack * 50)]
            payload = f"credits={pack}&stars={pack*50}"
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
    _allowed_cmds = {"start","info","generate","series","series_fixed","settings","history","history_clear","interface_lang","credits","pricing","chat","endchat","cancel"}
    @dp.message_handler(lambda m: (m.text or "").startswith("/") and ((m.text or "").split()[0].lstrip("/").split("@")[0].lower() in _allowed_cmds), state=ChatStates.Active, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def cmd_any_in_chat(message: types.Message, state: FSMContext):
        try:
            import sys as _sys
            raw = (message.text or "").strip()
            cmd = raw.split()[0].lstrip("/").split("@")[0].lower()
            print(f"[CMD_ANY_CHAT] cmd={cmd} allowed={cmd in _allowed_cmds}", file=_sys.stderr, flush=True)
        except Exception:
            pass
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
                "interface_lang": cmd_lang,
                "credits": cmd_credits,
                "pricing": cmd_pricing,
                "chat": cmd_chat,
                "endchat": cmd_endchat,
                "cancel": cmd_cancel,
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
    @dp.message_handler(lambda m: (m.text or "").startswith("/") and ((m.text or "").split()[0].lstrip("/").split("@")[0].lower() in _allowed_cmds), state="*", content_types=types.ContentTypes.TEXT)  # type: ignore
    async def cmd_any_during_fsm(message: types.Message, state: FSMContext):
        try:
            import sys as _sys
            raw = (message.text or "").strip()
            cmd = raw.split()[0].lstrip("/").split("@")[0].lower()
            print(f"[CMD_ANY_FSM] cmd={cmd} allowed={cmd in _allowed_cmds}", file=_sys.stderr, flush=True)
        except Exception:
            pass
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
                "interface_lang": cmd_lang,
                "credits": cmd_credits,
                "pricing": cmd_pricing,
                "chat": cmd_chat,
                "endchat": cmd_endchat,
                "cancel": cmd_cancel,
                # /topup lives in bot_commands; aiogram dispatcher will invoke it via registration
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
        # Block while generation is running or pending (race-safe)
        try:
            if message.chat and await is_chat_running(message.chat.id):
                return
        except Exception:
            pass
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
        # Block auto-chat during ANY step of generation wizard
        try:
            gen_states = {
                GenerateStates.ChoosingLanguage.state,
                GenerateStates.ChoosingGenLanguage.state,
                GenerateStates.ChoosingProvider.state,
                GenerateStates.ChoosingLogs.state,
                GenerateStates.ChoosingIncognito.state,
                GenerateStates.ChoosingGenType.state,
                GenerateStates.WaitingTopic.state,
                GenerateStates.ChoosingSeriesPreset.state,
                GenerateStates.ChoosingSeriesCount.state,
                GenerateStates.ChoosingFactcheck.state,
                GenerateStates.ChoosingDepth.state,
                GenerateStates.ChoosingRefine.state,
            }
            if cur in gen_states or cur == ChatStates.Active.state:
                return
        except Exception:
            pass
        # Block when generation flow has pending flags (confirmation/payment or flow markers)
        try:
            sd2 = await state.get_data()
            if sd2.get("pending_topic") or sd2.get("active_flow") or sd2.get("series_mode") or sd2.get("series_count") or sd2.get("fc_ready") or sd2.get("in_settings"):
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
                
                # Job.user_id always stores User.id (normalized schema)
                if db_uid is None:
                    await message.answer("Нет результатов для контекста чата. Сначала сгенерируйте пост.")
                    return
                
                q = await s.execute(
                    select(ResultDoc, Job.user_id)
                    .select_from(ResultDoc.__table__.join(Job.__table__, ResultDoc.job_id == Job.id))
                    .where(Job.user_id == db_uid)
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

    @dp.message_handler(commands=["credits"])  # type: ignore
    async def cmd_credits(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if is_admin:
            await message.answer("Админ: генерации бесплатны." if _is_ru(ui_lang) else "Admin: generation is free.")
            return
        # Prefer DB balance when available; fallback to KV
        bal = 0
        try:
            from .db import SessionLocal as _Sess, get_or_create_user as _gcu
            if _Sess is not None and message.from_user:
                async with _Sess() as s:
                    user = await _gcu(s, int(message.from_user.id))
                    bal = int(getattr(user, "credits", 0) or 0)
            else:
                bal = await get_balance_kv_only(message.from_user.id) if message.from_user else 0
        except Exception:
            try:
                bal = await get_balance_kv_only(message.from_user.id) if message.from_user else 0
            except Exception:
                bal = 0
        if _is_ru(ui_lang):
            txt = [f"Баланс: {int(bal)} кредит(ов).", "", "Цены:", "- Пост: 1 кредит", "- Статья: 100 кредитов", "", "Купить за ⭐ (1 кредит = 50⭐):"]
        else:
            txt = [f"Balance: {int(bal)} credits.", "", "Pricing:", "- Post: 1 credit", "- Article: 100 credits", "", "Buy with ⭐ (1 credit = 50⭐):"]
        try:
            await message.answer("\n".join(txt), reply_markup=build_buy_keyboard(ui_lang))
        except Exception:
            await message.answer("\n".join(txt))

    # /topup хендлер вынесен в server/bot_commands.py, чтобы избежать дублирования регистрации

    return dp




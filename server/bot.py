#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict, Set

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

from utils.env import load_env_from_root
from utils.lang import detect_lang_from_text
from services.post.generate import generate_post
from utils.slug import safe_filename_base
from .db import SessionLocal
from .bot_commands import ADMIN_IDS
from .credits import ensure_user_with_credits, charge_credits, charge_credits_kv, get_balance_kv_only
from .kv import set_provider, get_provider, set_logs_enabled, get_logs_enabled, set_incognito, get_incognito
from .kv import set_gen_lang, get_gen_lang
from .kv import set_refine_enabled, get_refine_enabled


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


def build_lang_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("English"))
    kb.add(KeyboardButton("Русский"))
    kb.add(KeyboardButton("Other / Auto"))
    return kb


def build_yesno_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Yes"), KeyboardButton("No"))
    return kb


def build_depth_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("1"), KeyboardButton("2"), KeyboardButton("3"))
    return kb


def build_genlang_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Auto"), KeyboardButton("RU"), KeyboardButton("EN"))
    return kb
def build_provider_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("OpenAI"), KeyboardButton("Gemini"), KeyboardButton("Claude"))
    return kb


def build_logs_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Включить"), KeyboardButton("Отключить"))
    return kb

def build_incognito_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Включить"), KeyboardButton("Отключить"))
    return kb


# Removed unused build_cancel_keyboard - we use /cancel command instead


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
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_lang_keyboard(),
        )
        await GenerateStates.ChoosingLanguage.set()

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

        if ui_lang == "ru":
            text = (
                "Этот бот генерирует научно-популярные посты.\n"
                "1) Выберите язык генерации: /lang_generate (Auto/RU/EN).\n"
                "2) Нажмите /generate и отправьте тему.\n"
                "На выходе получите Markdown-файл с постом.\n"
                "GitHub проекта: https://github.com/YanGranat/book-in-one-click\n\n"
                "Текущие настройки:\n"
                f"- Провайдер: {_prov_name(prov)}\n"
                f"- Язык генерации: {_lang_human(gen_lang, True)}\n"
                f"- Инкогнито: {'включён' if incognito else 'отключён'}\n"
                f"- Логи: {'включены' if logs_enabled else 'отключены'}"
            )
        else:
            text = (
                "This bot generates popular-science posts.\n"
                "1) Pick generation language: /lang_generate (Auto/RU/EN).\n"
                "2) Press /generate and send a topic.\n"
                "You will get a Markdown file with the post.\n"
                "Project GitHub: https://github.com/YanGranat/book-in-one-click\n\n"
                "Current settings:\n"
                f"- Provider: {_prov_name(prov)}\n"
                f"- Generation language: {_lang_human(gen_lang, False)}\n"
                f"- Incognito: {'enabled' if incognito else 'disabled'}\n"
                f"- Logs: {'enabled' if logs_enabled else 'disabled'}"
            )
        await message.answer(text)


    @dp.message_handler(state=GenerateStates.ChoosingLanguage)  # type: ignore
    async def choose_language(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        if text.startswith("english"):
            ui_lang = "en"
        elif text.startswith("рус"):
            ui_lang = "ru"
        else:
            ui_lang = "auto"
        await state.update_data(ui_lang=ui_lang)
        confirm = "Язык интерфейса установлен." if ui_lang == "ru" else "Interface language set."
        await message.answer(confirm, reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["lang_generate"])  # type: ignore
    async def cmd_lang_generate(message: types.Message, state: FSMContext):
        await message.answer(
            "Выберите язык генерации / Choose generation language:",
            reply_markup=build_genlang_keyboard(),
        )
        await GenerateStates.ChoosingGenLanguage.set()

    @dp.message_handler(state=GenerateStates.ChoosingGenLanguage)  # type: ignore
    async def choose_gen_language(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        if text.startswith("ru"):
            gen_lang = "ru"
        elif text.startswith("en"):
            gen_lang = "en"
        else:
            gen_lang = "auto"
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        await state.update_data(gen_lang=gen_lang)
        try:
            if message.from_user:
                await set_gen_lang(message.from_user.id, gen_lang)  # persist per-user
        except Exception:
            pass
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
        await message.answer(msg.get("ru" if ui_lang == "ru" else "en").get(gen_lang, "OK"), reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["provider"])  # type: ignore
    async def cmd_provider(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Выберите провайдера (OpenAI/Gemini/Claude):" if ui_lang == "ru" else "Choose provider (OpenAI/Gemini/Claude):"
        await message.answer(prompt, reply_markup=build_provider_keyboard())
        await GenerateStates.ChoosingProvider.set()

    @dp.message_handler(state=GenerateStates.ChoosingProvider)  # type: ignore
    async def choose_provider(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        prov_map = {"openai": "openai", "gemini": "gemini", "claude": "claude"}
        prov = prov_map.get(text, None)
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if not prov:
            msg = "Пожалуйста, выберите: OpenAI, Gemini или Claude." if ui_lang == "ru" else "Please choose: OpenAI, Gemini or Claude."
            await message.answer(msg, reply_markup=build_provider_keyboard())
            return
        await state.update_data(provider=prov)
        try:
            if message.from_user:
                await set_provider(message.from_user.id, prov)  # type: ignore
        except Exception:
            pass
        ok = "Провайдер установлен." if ui_lang == "ru" else "Provider set."
        await message.answer(ok, reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["lang"])  # type: ignore
    async def cmd_lang(message: types.Message, state: FSMContext):
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_lang_keyboard(),
        )
        await GenerateStates.ChoosingLanguage.set()

    @dp.message_handler(commands=["logs"])  # type: ignore
    async def cmd_logs(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Отправлять логи генерации?" if ui_lang == "ru" else "Send generation logs?"
        await message.answer(prompt, reply_markup=build_logs_keyboard())
        await GenerateStates.ChoosingLogs.set()

    @dp.message_handler(state=GenerateStates.ChoosingLogs)  # type: ignore
    async def choose_logs(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        
        if text.startswith("включ") or text.startswith("enable"):
            enabled = True
            msg = "Логи включены." if ui_lang == "ru" else "Logs enabled."
        elif text.startswith("отключ") or text.startswith("disable"):
            enabled = False
            msg = "Логи отключены." if ui_lang == "ru" else "Logs disabled."
        else:
            prompt = "Выберите: Включить или Отключить." if ui_lang == "ru" else "Choose: Enable or Disable."
            await message.answer(prompt, reply_markup=build_logs_keyboard())
            return
            
        try:
            if message.from_user:
                await set_logs_enabled(message.from_user.id, enabled)
        except Exception:
            pass
        await message.answer(msg, reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["incognito"])  # type: ignore
    async def cmd_incognito(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Инкогнито режим: Включить или Отключить?" if ui_lang == "ru" else "Incognito: Enable or Disable?"
        await message.answer(prompt, reply_markup=build_incognito_keyboard())
        await GenerateStates.ChoosingIncognito.set()

    @dp.message_handler(state=GenerateStates.ChoosingIncognito)  # type: ignore
    async def choose_incognito(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if text.startswith("включ") or text.startswith("enable"):
            enabled = True
            msg = "Инкогнито: включён." if ui_lang == "ru" else "Incognito: enabled."
        elif text.startswith("отключ") or text.startswith("disable"):
            enabled = False
            msg = "Инкогнито: отключён." if ui_lang == "ru" else "Incognito: disabled."
        else:
            prompt = "Выберите: Включить или Отключить." if ui_lang == "ru" else "Choose: Enable or Disable."
            await message.answer(prompt, reply_markup=build_incognito_keyboard())
            return
        try:
            if message.from_user:
                await set_incognito(message.from_user.id, enabled)
        except Exception:
            pass
        await message.answer(msg, reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["refine"])  # type: ignore
    async def cmd_refine(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = (
            "Финальная редактура: включить или отключить?"
            if ui_lang == "ru"
            else "Final refine step: Enable or Disable?"
        )
        await message.answer(prompt, reply_markup=build_logs_keyboard())  # reuse yes/no style RU buttons
        await GenerateStates.ChoosingRefine.set()

    @dp.message_handler(state=GenerateStates.ChoosingRefine)  # type: ignore
    async def choose_refine(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        if text.startswith("включ") or text.startswith("enable"):
            enabled = True
            msg = "Финальная редактура: включена." if ui_lang == "ru" else "Final refine: enabled."
        elif text.startswith("отключ") or text.startswith("disable"):
            enabled = False
            msg = "Финальная редактура: отключена." if ui_lang == "ru" else "Final refine: disabled."
        else:
            prompt = (
                "Выберите: Включить или Отключить."
                if ui_lang == "ru"
                else "Choose: Enable or Disable."
            )
            await message.answer(prompt, reply_markup=build_logs_keyboard())
            return
        try:
            if message.from_user:
                await set_refine_enabled(message.from_user.id, enabled)
        except Exception:
            pass
        await message.answer(msg, reply_markup=ReplyKeyboardRemove())
        await state.finish()

    @dp.message_handler(commands=["cancel"])  # type: ignore
    async def cmd_cancel(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        done = "Отменено." if ui_lang == "ru" else "Cancelled."
        await state.finish()
        await message.answer(done, reply_markup=ReplyKeyboardRemove())

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
        q = "Включить факт-чекинг?" if ui_lang == "ru" else "Enable fact-checking?"
        await message.answer(q, reply_markup=build_yesno_keyboard())
        await GenerateStates.ChoosingFactcheck.set()

    @dp.message_handler(state=GenerateStates.ChoosingFactcheck)  # type: ignore
    async def choose_factcheck(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        data = await state.get_data()
        topic = data.get("topic", "")
        ui_lang = data.get("ui_lang", "ru")
        if text in {"/cancel"}:
            done = "Отменено." if ui_lang == "ru" else "Cancelled."
            await message.answer(done, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            return
        # yes in en = y/yes; in ru = д/да
        factcheck = text.startswith("y") or text.startswith("д")

        # If fact-checking is enabled, ask for depth first
        if factcheck:
            await state.update_data(factcheck=True)
            prompt = "Выберите глубину проверки (1–3):" if ui_lang == "ru" else "Select research depth (1–3):"
            await message.answer(prompt, reply_markup=build_depth_keyboard())
            await GenerateStates.ChoosingDepth.set()
            return

        chat_id = message.chat.id
        if chat_id in RUNNING_CHATS:
            # Silently ignore duplicate start while previous job is running
            return
        RUNNING_CHATS.add(chat_id)

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
                    warn = "Недостаточно кредитов" if ui_lang == "ru" else "Insufficient credits"
                    await message.answer(warn, reply_markup=ReplyKeyboardRemove())
                    await state.finish()
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if ui_lang == "ru" else "Temporary DB error. Try later."
            await message.answer(warn, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            RUNNING_CHATS.discard(chat_id)
            return

        working = "Генерирую. Это может занять несколько минут..." if ui_lang == "ru" else "Working on it. This may take a few minutes..."
        await message.answer(working, reply_markup=ReplyKeyboardRemove())

        # Run generation in a thread to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            # Resolve provider before jumping into executor
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
            # Refine preference
            refine_enabled = False
            try:
                if message.from_user:
                    refine_enabled = await get_refine_enabled(message.from_user.id)
            except Exception:
                refine_enabled = False
            async with GLOBAL_SEMAPHORE:
                # Prepare job metadata for logging
                job_meta = {
                    "user_id": message.from_user.id if message.from_user else 0,
                    "chat_id": message.chat.id,
                    "topic": topic,
                    "provider": prov or "openai",
                    "lang": gen_lang,
                    "incognito": (await get_incognito(message.from_user.id)) if message.from_user else False,
                    "refine": refine_enabled,
                }
                path = await loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=gen_lang,
                        provider=(prov or "openai"),
                        factcheck=False,
                        job_meta=job_meta,
                        use_refine=refine_enabled,
                    ),
                )
            # Send main result
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if ui_lang == "ru" else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
            
            # Send logs if enabled
            try:
                if message.from_user:
                    logs_enabled = await get_logs_enabled(message.from_user.id)
                    if logs_enabled:
                        # Find corresponding log file by scanning directory
                        topic_base = safe_filename_base(topic)
                        log_files = list(path.parent.glob(f"{topic_base}_log_*.md"))
                        if log_files:
                            log_path = log_files[0]  # Take first match
                            with open(log_path, "rb") as log_f:
                                log_cap = f"Лог: {log_path.name}" if ui_lang == "ru" else f"Log: {log_path.name}"
                                await message.answer_document(log_f, caption=log_cap)
            except Exception:
                pass
        except Exception as e:
            err = f"Ошибка: {e}" if ui_lang == "ru" else f"Error: {e}"
            await message.answer(err)

        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    @dp.message_handler(state=GenerateStates.ChoosingDepth)  # type: ignore
    async def choose_depth(message: types.Message, state: FSMContext):
        txt = (message.text or "").strip()
        data = await state.get_data()
        ui_lang = data.get("ui_lang", "ru")
        topic = data.get("topic", "")
        if txt.lower() in {"/cancel"}:
            done = "Отменено." if ui_lang == "ru" else "Cancelled."
            await message.answer(done, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            return
        if txt not in {"1", "2", "3"}:
            prompt = "Выберите 1, 2 или 3:" if ui_lang == "ru" else "Please choose 1, 2 or 3:"
            await message.answer(prompt, reply_markup=build_depth_keyboard())
            return
        depth = int(txt)
        await state.update_data(research_iterations=depth)

        chat_id = message.chat.id
        if chat_id in RUNNING_CHATS:
            return
        RUNNING_CHATS.add(chat_id)

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
                    warn = "Недостаточно кредитов" if ui_lang == "ru" else "Insufficient credits"
                    await message.answer(warn, reply_markup=ReplyKeyboardRemove())
                    await state.finish()
                    RUNNING_CHATS.discard(chat_id)
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if ui_lang == "ru" else "Temporary DB error. Try later."
            await message.answer(warn, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            RUNNING_CHATS.discard(chat_id)
            return

        working = "Генерирую. Это может занять несколько минут..." if ui_lang == "ru" else "Working on it. This may take a few minutes..."
        await message.answer(working, reply_markup=ReplyKeyboardRemove())

        # Run generation in a thread to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            # Resolve provider before jumping into executor
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
            # Refine preference
            refine_enabled = False
            try:
                if message.from_user:
                    refine_enabled = await get_refine_enabled(message.from_user.id)
            except Exception:
                refine_enabled = False
            async with GLOBAL_SEMAPHORE:
                # Prepare job metadata for logging
                job_meta = {
                    "user_id": message.from_user.id if message.from_user else 0,
                    "chat_id": message.chat.id,
                    "topic": topic,
                    "provider": prov or "openai",
                    "lang": gen_lang,
                    "incognito": (await get_incognito(message.from_user.id)) if message.from_user else False,
                    "refine": refine_enabled,
                }
                path = await loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=gen_lang,
                        provider=(prov or "openai"),
                        factcheck=True,
                        research_iterations=depth,
                        job_meta=job_meta,
                        use_refine=refine_enabled,
                    ),
                )
            # Send main result
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if ui_lang == "ru" else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
            
            # Send logs if enabled
            try:
                if message.from_user:
                    logs_enabled = await get_logs_enabled(message.from_user.id)
                    if logs_enabled:
                        # Find corresponding log file by scanning directory
                        topic_base = safe_filename_base(topic)
                        log_files = list(path.parent.glob(f"{topic_base}_log_*.md"))
                        if log_files:
                            log_path = log_files[0]  # Take first match
                            with open(log_path, "rb") as log_f:
                                log_cap = f"Лог: {log_path.name}" if ui_lang == "ru" else f"Log: {log_path.name}"
                                await message.answer_document(log_f, caption=log_cap)
            except Exception:
                pass
        except Exception as e:
            err = f"Ошибка: {e}" if ui_lang == "ru" else f"Error: {e}"
            await message.answer(err)

        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    return dp




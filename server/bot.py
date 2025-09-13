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
from .db import SessionLocal
from .bot_commands import ADMIN_IDS
from .credits import ensure_user_with_credits, charge_credits, charge_credits_kv, get_balance_kv_only


def _load_env():
    load_env_from_root(__file__)


_load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


class GenerateStates(StatesGroup):
    ChoosingLanguage = State()
    ChoosingGenLanguage = State()
    WaitingTopic = State()
    ChoosingFactcheck = State()
    ChoosingDepth = State()


def build_lang_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("English"))
    kb.add(KeyboardButton("Русский"))
    kb.add(KeyboardButton("Other / Auto"))
    return kb


def build_yesno_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Yes"), KeyboardButton("No"))
    kb.add(KeyboardButton("Cancel"), KeyboardButton("Отмена"))
    return kb


def build_depth_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("1"), KeyboardButton("2"), KeyboardButton("3"))
    return kb


def build_genlang_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Auto"), KeyboardButton("RU"), KeyboardButton("EN"))
    return kb


def build_cancel_keyboard() -> ReplyKeyboardMarkup:
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("Cancel"), KeyboardButton("Отмена"))
    return kb


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
        if ui_lang == "ru":
            text = (
                "Этот бот генерирует научно-популярные посты.\n"
                "1) Выберите язык генерации: /lang_generate (Auto/RU/EN).\n"
                "2) Нажмите /generate и отправьте тему.\n"
                "На выходе получите Markdown-файл с постом."
            )
        else:
            text = (
                "This bot generates popular-science posts.\n"
                "1) Pick generation language: /lang_generate (Auto/RU/EN).\n"
                "2) Press /generate and send a topic.\n"
                "You will get a Markdown file with the post."
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

    @dp.message_handler(commands=["generate"])  # type: ignore
    async def cmd_generate(message: types.Message, state: FSMContext):
        data = await state.get_data()
        ui_lang = (data.get("ui_lang") or "ru").strip()
        prompt = "Отправьте тему для поста:" if ui_lang == "ru" else "Send a topic for your post:"
        await message.answer(prompt, reply_markup=build_cancel_keyboard())
        await GenerateStates.WaitingTopic.set()

    @dp.message_handler(commands=["lang"])  # type: ignore
    async def cmd_lang(message: types.Message, state: FSMContext):
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_lang_keyboard(),
        )
        await GenerateStates.ChoosingLanguage.set()

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
        if text_raw.lower() in {"cancel", "отмена"}:
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
        if text in {"cancel", "отмена"}:
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
            if SessionLocal is not None:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, message.from_user.id)  # type: ignore
                    ok, remaining = await charge_credits(session, user, 1, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged:
                ok, remaining = await charge_credits_kv(message.from_user.id, 1)  # type: ignore
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
            async with GLOBAL_SEMAPHORE:
                path = await loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=(data.get("gen_lang") or "auto"),
                        factcheck=False,
                    ),
                )
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if ui_lang == "ru" else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
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
        if txt.lower() in {"cancel", "отмена"}:
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
            if SessionLocal is not None:
                async with SessionLocal() as session:
                    user = await ensure_user_with_credits(session, message.from_user.id)  # type: ignore
                    ok, remaining = await charge_credits(session, user, 1, reason="post")
                    if ok:
                        await session.commit()
                        charged = True
            if not charged:
                ok, remaining = await charge_credits_kv(message.from_user.id, 1)  # type: ignore
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
            async with GLOBAL_SEMAPHORE:
                path = await loop.run_in_executor(
                    None,
                    lambda: generate_post(
                        topic,
                        lang=(data.get("gen_lang") or "auto"),
                        factcheck=True,
                        research_iterations=depth,
                    ),
                )
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if ui_lang == "ru" else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
        except Exception as e:
            err = f"Ошибка: {e}" if ui_lang == "ru" else f"Error: {e}"
            await message.answer(err)

        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    return dp




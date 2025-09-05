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
from services.generation.post import generate_post
from .db import SessionLocal
from .bot_commands import ADMIN_IDS
from .credits import ensure_user_with_credits, charge_credits, charge_credits_kv, get_balance_kv_only


def _load_env():
    load_env_from_root(__file__)


_load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


class GenerateStates(StatesGroup):
    ChoosingLanguage = State()
    WaitingTopic = State()
    ChoosingFactcheck = State()


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


def create_dispatcher() -> Dispatcher:
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher(bot, storage=MemoryStorage())

    # Simple in-memory guard to avoid duplicate generation per chat
    RUNNING_CHATS: Set[int] = set()

    @dp.message_handler(commands=["start", "help"])  # type: ignore
    async def cmd_start(message: types.Message):
        await message.answer(
            "Выберите язык интерфейса / Choose interface language:",
            reply_markup=build_lang_keyboard(),
        )
        await GenerateStates.ChoosingLanguage.set()


    @dp.message_handler(state=GenerateStates.ChoosingLanguage)  # type: ignore
    async def choose_language(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        if text.startswith("english"):
            lang = "en"
        elif text.startswith("рус"):
            lang = "ru"
        else:
            lang = "auto"
        await state.update_data(lang=lang)
        prompt = "Отправьте тему для поста:" if lang == "ru" else "Send a topic for your post:"
        await message.answer(prompt, reply_markup=ReplyKeyboardRemove())
        await GenerateStates.WaitingTopic.set()

    @dp.message_handler(state=GenerateStates.WaitingTopic, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def topic_received(message: types.Message, state: FSMContext):
        topic = (message.text or "").strip()
        if not topic:
            data = await state.get_data()
            lang = data.get("lang", "auto")
            msg = "Тема не может быть пустой. Отправьте тему:" if lang == "ru" else "Topic cannot be empty. Send a topic:"
            await message.answer(msg)
            return
        await state.update_data(topic=topic)
        data = await state.get_data()
        lang = data.get("lang", "auto")
        q = "Включить факт-чекинг?" if lang == "ru" else "Enable fact-checking?"
        await message.answer(q, reply_markup=build_yesno_keyboard())
        await GenerateStates.ChoosingFactcheck.set()

    @dp.message_handler(state=GenerateStates.ChoosingFactcheck)  # type: ignore
    async def choose_factcheck(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        # yes in en = y/yes; in ru = д/да
        factcheck = text.startswith("y") or text.startswith("д")
        data = await state.get_data()
        topic = data.get("topic", "")
        lang = data.get("lang", "auto")

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
                    warn = "Недостаточно кредитов. Попросите админа выполнить /topup." if lang == "ru" else "Insufficient credits. Ask admin to /topup."
                    await message.answer(warn, reply_markup=ReplyKeyboardRemove())
                    await state.finish()
                    return
        except SQLAlchemyError:
            warn = "Временная ошибка БД. Попробуйте позже." if lang == "ru" else "Temporary DB error. Try later."
            await message.answer(warn, reply_markup=ReplyKeyboardRemove())
            await state.finish()
            RUNNING_CHATS.discard(chat_id)
            return

        working = "Генерирую. Это может занять несколько минут..." if lang == "ru" else "Working on it. This may take a few minutes..."
        await message.answer(working, reply_markup=ReplyKeyboardRemove())

        # Run generation in a thread to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            path = await loop.run_in_executor(
                None,
                lambda: generate_post(
                    topic,
                    lang=lang,
                    factcheck=factcheck,
                ),
            )
            with open(path, "rb") as f:
                cap = f"Готово: {path.name}" if lang == "ru" else f"Done: {path.name}"
                await message.answer_document(f, caption=cap)
        except Exception as e:
            err = f"Ошибка: {e}" if lang == "ru" else f"Error: {e}"
            await message.answer(err)

        await state.finish()
        RUNNING_CHATS.discard(chat_id)

    return dp




#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict

from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

from utils.env import load_env_from_root
from utils.lang import detect_lang_from_text
from services.generation.post import generate_post
from .db import SessionLocal
from .credits import ensure_user_with_credits, charge_credits


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

    @dp.message_handler(commands=["start", "help"])  # type: ignore
    async def cmd_start(message: types.Message):
        await message.answer(
            "Welcome! Choose interface language:",
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
        await message.answer("Send a topic for your post:")
        await GenerateStates.WaitingTopic.set()

    @dp.message_handler(state=GenerateStates.WaitingTopic, content_types=types.ContentTypes.TEXT)  # type: ignore
    async def topic_received(message: types.Message, state: FSMContext):
        topic = (message.text or "").strip()
        if not topic:
            await message.answer("Topic cannot be empty. Send a topic:")
            return
        await state.update_data(topic=topic)
        await message.answer("Enable fact-checking?", reply_markup=build_yesno_keyboard())
        await GenerateStates.ChoosingFactcheck.set()

    @dp.message_handler(state=GenerateStates.ChoosingFactcheck)  # type: ignore
    async def choose_factcheck(message: types.Message, state: FSMContext):
        text = (message.text or "").strip().lower()
        factcheck = text.startswith("y")
        data = await state.get_data()
        topic = data.get("topic", "")
        lang = data.get("lang", "auto")

        # Charge 1 credit before starting
        from sqlalchemy.exc import SQLAlchemyError
        try:
            if SessionLocal is None:
                await message.answer("Service misconfigured: DB_URL not set.")
                await state.finish()
                return
            async with SessionLocal() as session:
                user = await ensure_user_with_credits(session, message.from_user.id)  # type: ignore
                ok, remaining = await charge_credits(session, user, 1, reason="post")
                if not ok:
                    await message.answer("Insufficient credits. Ask admin to /topup.")
                    await state.finish()
                    return
                await session.commit()
        except SQLAlchemyError as e:
            await message.answer("Temporary DB error. Try later.")
            await state.finish()
            return

        await message.answer("Working on it. This may take a few minutes...")

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
                await message.answer_document(f, caption=f"Done: {path.name}")
        except Exception as e:
            await message.answer(f"Error: {e}")

        await state.finish()

    return dp




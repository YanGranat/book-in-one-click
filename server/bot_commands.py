#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import List

from aiogram import Dispatcher, types
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import SessionLocal, init_db
from .credits import topup_credits, topup_credits_kv, get_balance_kv_only


ADMIN_IDS: List[int] = []
_env = os.getenv("BOT_ADMIN_IDS", "")
if _env:
    for tok in _env.replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            ADMIN_IDS.append(int(tok))

# Series-related defaults and limits (used by bot)
SERIES_AUTO_MAX_DEFAULT = 30  # max posts in auto mode for non-admin
SERIES_MAX_ITERATIONS_DEFAULT = 1
SERIES_SUFF_HEAVY_AFTER = 3


def register_admin_commands(dp: Dispatcher, session_factory: async_sessionmaker):
    @dp.message_handler(commands=["balance"])  # type: ignore
    async def balance_cmd(message: types.Message):
        is_admin = bool(message.from_user and message.from_user.id in ADMIN_IDS)
        if not message.from_user:
            await message.answer("Error: user information not available")
            return
            
        if session_factory is None:
            _ = await get_balance_kv_only(message.from_user.id)
            if is_admin:
                await message.answer("Admin: generation is free.")
            else:
                await message.answer(f"Your balance: {_} credits")
            return
        async with session_factory() as session:
            from .db import get_or_create_user
            user = await get_or_create_user(session, message.from_user.id)
            if is_admin:
                await message.answer("Admin: generation is free.")
            else:
                await message.answer(f"Your balance: {user.credits} credits")

    @dp.message_handler(commands=["topup"])  # type: ignore
    async def topup_cmd(message: types.Message):
        if message.from_user is None or message.from_user.id not in ADMIN_IDS:
            await message.answer("Forbidden")
            return
        parts = (message.text or "").split()
        if len(parts) < 3 or not parts[1].isdigit() or not parts[2].isdigit():
            await message.answer("Usage: /topup <telegram_id> <amount>")
            return
        telegram_id = int(parts[1])
        amount = int(parts[2])
        if session_factory is None:
            new_balance = await topup_credits_kv(telegram_id, amount)
            await message.answer(f"OK. New balance for {telegram_id}: {new_balance}")
            return
        async with session_factory() as session:
            new_balance = await topup_credits(session, telegram_id, amount)
            await session.commit()
        await message.answer(f"OK. New balance for {telegram_id}: {new_balance}")


async def ensure_db_ready():
    await init_db()



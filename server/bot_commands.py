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
# Ensure requested admin id present
# Built-in admins
for _aid in (452623935, 152840069):
    if _aid not in ADMIN_IDS:
        ADMIN_IDS.append(_aid)


def register_admin_commands(dp: Dispatcher, session_factory: async_sessionmaker):
    @dp.message_handler(commands=["balance"])  # type: ignore
    async def balance_cmd(message: types.Message):
        if session_factory is None:
            bal = await get_balance_kv_only(message.from_user.id)  # type: ignore
            await message.answer(f"Your balance: {bal} credits")
            return
        async with session_factory() as session:
            from .db import get_or_create_user
            user = await get_or_create_user(session, message.from_user.id)  # type: ignore
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



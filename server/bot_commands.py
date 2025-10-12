#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import List, Optional

from aiogram import Dispatcher, types
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import SessionLocal, init_db
from .credits import topup_credits, topup_credits_kv, get_balance_kv_only


ADMIN_IDS: List[int] = []
# Superadmin: full access. Read from SUPER_ADMIN_ID (preferred) or BOT_SUPER_ADMIN_ID (legacy)
SUPER_ADMIN_ID_ENV = os.getenv("SUPER_ADMIN_ID", os.getenv("BOT_SUPER_ADMIN_ID", "")).strip()
SUPER_ADMIN_ID: Optional[int] = int(SUPER_ADMIN_ID_ENV) if SUPER_ADMIN_ID_ENV.isdigit() else None
_env = os.getenv("BOT_ADMIN_IDS", "")
if _env:
    for tok in _env.replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            ADMIN_IDS.append(int(tok))
# Ensure superadmin is always included in ADMIN_IDS for free generation/bypass checks
if SUPER_ADMIN_ID is not None and SUPER_ADMIN_ID not in ADMIN_IDS:
    ADMIN_IDS.append(SUPER_ADMIN_ID)

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
        # Superadmin-only topup
        if message.from_user is None or SUPER_ADMIN_ID is None or int(message.from_user.id) != int(SUPER_ADMIN_ID):
            await message.answer("Forbidden")
            return
        parts = (message.text or "").split()
        if len(parts) < 3 or not parts[1].isdigit() or not parts[2].isdigit():
            await message.answer("Usage: /topup <telegram_id> <amount>")
            return
        telegram_id = int(parts[1])
        amount = int(parts[2])
        # Always top up KV (source of truth for chat/UI); also mirror to DB if available
        new_balance_kv = await topup_credits_kv(telegram_id, amount)
        # Try to notify the user regardless of DB presence
        try:
            await message.bot.send_message(
                telegram_id,
                f"🎁 Вам начислено {amount} кредит(ов)!\nВаш баланс: {new_balance_kv} кредитов."
            )
        except Exception:
            pass
        if session_factory is None:
            await message.answer(f"OK. New balance for {telegram_id}: {new_balance_kv}")
            return
        async with session_factory() as session:
            try:
                from .db import get_or_create_user
                # Mirror to DB ledger
                await topup_credits(session, telegram_id, amount)
                await session.commit()
                # Read DB balance for confirmation
                user = await get_or_create_user(session, telegram_id)
                db_balance = int(getattr(user, "credits", 0) or 0)
            except Exception:
                db_balance = new_balance_kv
        await message.answer(f"OK. New balance for {telegram_id}: {db_balance}")


async def ensure_db_ready():
    await init_db()



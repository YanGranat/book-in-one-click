#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .db import User, Tx, get_or_create_user
from .kv import topup_kv, charge_kv, get_balance_kv


async def ensure_user_with_credits(session: AsyncSession, telegram_id: int) -> User:
    user = await get_or_create_user(session, telegram_id)
    return user


async def charge_credits(session: AsyncSession | None, user: User | None, amount: int, reason: str) -> Tuple[bool, int]:
    # Prefer DB when session and user provided, else fall back to KV
    if session is not None and user is not None:
        if user.credits < amount:
            return False, user.credits
        user.credits -= amount
        tx = Tx(user_id=user.id, delta=-amount, reason=reason)
        session.add(tx)
        await session.flush()
        return True, user.credits
    # KV fallback
    from aiogram.types import User as TgUser  # type: ignore
    # Here 'user' may be None; we require telegram_id via reason field is not suitable.
    raise NotImplementedError("KV charge requires explicit telegram_id; use charge_credits_kv")


async def topup_credits(session: AsyncSession, telegram_id: int, amount: int, reason: str = "admin_topup") -> int:
    user = await get_or_create_user(session, telegram_id)
    user.credits += amount
    tx = Tx(user_id=user.id, delta=amount, reason=reason)
    session.add(tx)
    await session.flush()
    return user.credits


# KV-first helpers for cases when DB is not configured
async def topup_credits_kv(telegram_id: int, amount: int) -> int:
    return await topup_kv(telegram_id, amount)


async def charge_credits_kv(telegram_id: int, amount: int) -> Tuple[bool, int]:
    return await charge_kv(telegram_id, amount)


async def get_balance_kv_only(telegram_id: int) -> int:
    return await get_balance_kv(telegram_id)


async def refund_credits(session: AsyncSession | None, user: User | None, amount: int, reason: str = "refund_failed") -> None:
    if session is None or user is None:
        return
    user.credits += amount
    tx = Tx(user_id=user.id, delta=amount, reason=reason)
    session.add(tx)
    await session.flush()


async def refund_credits_kv(telegram_id: int, amount: int) -> None:
    # Same as topup in KV model
    await topup_kv(telegram_id, amount)



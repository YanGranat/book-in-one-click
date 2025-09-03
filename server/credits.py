#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .db import User, Tx, get_or_create_user


async def ensure_user_with_credits(session: AsyncSession, telegram_id: int) -> User:
    user = await get_or_create_user(session, telegram_id)
    return user


async def charge_credits(session: AsyncSession, user: User, amount: int, reason: str) -> Tuple[bool, int]:
    if user.credits < amount:
        return False, user.credits
    user.credits -= amount
    tx = Tx(user_id=user.id, delta=-amount, reason=reason)
    session.add(tx)
    await session.flush()
    return True, user.credits


async def topup_credits(session: AsyncSession, telegram_id: int, amount: int, reason: str = "admin_topup") -> int:
    user = await get_or_create_user(session, telegram_id)
    user.credits += amount
    tx = Tx(user_id=user.id, delta=amount, reason=reason)
    session.add(tx)
    await session.flush()
    return user.credits



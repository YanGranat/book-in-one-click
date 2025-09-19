#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional

try:
    from redis import asyncio as redis  # type: ignore
except Exception as e:  # pragma: no cover
    redis = None  # type: ignore


_redis_client: Optional["redis.Redis"] = None


def kv_prefix() -> str:
    # Reuse DB_TABLE_PREFIX for key prefix if available
    pref = os.getenv("DB_TABLE_PREFIX", "bio1c_")
    pref = pref.rstrip("_ ")
    return pref or "bio1c"


def get_redis() -> "redis.Redis":
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    url = os.getenv("REDIS_URL", "")
    if not url:
        raise RuntimeError("REDIS_URL is not set; provide Internal Key Value URL from Render")
    if redis is None:
        raise RuntimeError("redis package not installed; add 'redis' to requirements.txt")
    _redis_client = redis.from_url(url)
    return _redis_client


async def get_balance_kv(telegram_id: int) -> int:
    r = get_redis()
    key = f"{kv_prefix()}:credits:{telegram_id}"
    val = await r.get(key)
    try:
        return int(val) if val is not None else 0
    except Exception:
        return 0


async def topup_kv(telegram_id: int, amount: int) -> int:
    r = get_redis()
    key = f"{kv_prefix()}:credits:{telegram_id}"
    new_balance = await r.incrby(key, int(amount))
    return int(new_balance)


_CHARGE_LUA = """
local key = KEYS[1]
local cost = tonumber(ARGV[1])
local bal = tonumber(redis.call('GET', key) or '0')
if bal >= cost then
  local newbal = redis.call('DECRBY', key, cost)
  return newbal
else
  return -1
end
"""


async def charge_kv(telegram_id: int, amount: int) -> tuple[bool, int]:
    r = get_redis()
    key = f"{kv_prefix()}:credits:{telegram_id}"
    try:
        new_balance = await r.eval(_CHARGE_LUA, 1, key, int(amount))
        if isinstance(new_balance, bytes):
            new_balance = int(new_balance)
        if int(new_balance) >= 0:
            return True, int(new_balance)
        return False, await get_balance_kv(telegram_id)
    except Exception:
        # Fallback: manual check-decr (not strictly atomic). Acceptable for MVP.
        bal = await get_balance_kv(telegram_id)
        if bal >= amount:
            await r.decrby(key, int(amount))
            return True, bal - amount
        return False, bal


# ---- User preferences (provider, logs, etc.) ----

async def set_provider(telegram_id: int, provider: str) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:provider:{telegram_id}"
    await r.set(key, (provider or "openai").strip().lower())


async def get_provider(telegram_id: int) -> str:
    r = get_redis()
    key = f"{kv_prefix()}:provider:{telegram_id}"
    val = await r.get(key)
    try:
        prov = (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "")).strip().lower()
    except Exception:
        prov = str(val or "").strip().lower()
    return prov or "openai"


async def set_logs_enabled(telegram_id: int, enabled: bool) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:logs:{telegram_id}"
    await r.set(key, "1" if enabled else "0")


async def get_logs_enabled(telegram_id: int) -> bool:
    r = get_redis()
    key = f"{kv_prefix()}:logs:{telegram_id}"
    val = await r.get(key)
    try:
        return (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "0")).strip() == "1"
    except Exception:
        return False


# ---- Incognito preference ----

async def set_incognito(telegram_id: int, enabled: bool) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:incognito:{telegram_id}"
    await r.set(key, "1" if enabled else "0")


async def get_incognito(telegram_id: int) -> bool:
    r = get_redis()
    key = f"{kv_prefix()}:incognito:{telegram_id}"
    val = await r.get(key)
    try:
        return (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "0")).strip() == "1"
    except Exception:
        return False


# ---- Generation language preference ----

async def set_gen_lang(telegram_id: int, gen_lang: str) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:gen_lang:{telegram_id}"
    # normalize to ru/en/auto
    text = (gen_lang or "").strip().lower()
    if text.startswith("ru"):
        norm = "ru"
    elif text.startswith("en"):
        norm = "en"
    else:
        norm = "auto"
    await r.set(key, norm)


async def get_gen_lang(telegram_id: int) -> str:
    r = get_redis()
    key = f"{kv_prefix()}:gen_lang:{telegram_id}"
    val = await r.get(key)
    try:
        s = (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "")).strip().lower()
    except Exception:
        s = str(val or "").strip().lower()
    if s in {"ru", "en", "auto"}:
        return s
    return "auto"



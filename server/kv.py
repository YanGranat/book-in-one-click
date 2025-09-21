#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
import time
import json

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
# ---- Simple history storage (KV fallback) ----

async def push_history(telegram_id: int, item: Dict[str, Any]) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:history:{telegram_id}"
    try:
        await r.lpush(key, json.dumps(item, ensure_ascii=False))
        await r.ltrim(key, 0, 49)
        await r.expire(key, 60 * 60 * 24 * 60)  # keep 60 days
    except Exception:
        pass


async def get_history(telegram_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    r = get_redis()
    key = f"{kv_prefix()}:history:{telegram_id}"
    try:
        vals = await r.lrange(key, 0, max(0, int(limit) - 1))
        out: List[Dict[str, Any]] = []
        for v in vals:
            try:
                s = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []


async def clear_history(telegram_id: int) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:history:{telegram_id}"
    try:
        await r.delete(key)
    except Exception:
        pass


async def set_history_cleared_at(telegram_id: int, ts: Optional[float] = None) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:history_cleared_at:{telegram_id}"
    try:
        val = int(ts) if ts is not None else int(time.time())
        await r.set(key, str(val))
        # Keep marker for up to a year
        await r.expire(key, 60 * 60 * 24 * 365)
    except Exception:
        pass


async def get_history_cleared_at(telegram_id: int) -> Optional[float]:
    r = get_redis()
    key = f"{kv_prefix()}:history_cleared_at:{telegram_id}"
    try:
        val = await r.get(key)
        if val is None:
            return None


# ---- Chat session history (provider-agnostic) ----

def _chat_key(telegram_id: int, chat_id: int, provider: str) -> str:
    return f"{kv_prefix()}:chat:{int(telegram_id)}:{int(chat_id)}:{(provider or 'openai').strip().lower()}"


async def chat_append(telegram_id: int, chat_id: int, provider: str, role: str, content: str, *, max_items: int = 200) -> None:
    r = get_redis()
    key = _chat_key(telegram_id, chat_id, provider)
    try:
        item = {"role": (role or "user"), "content": content or ""}
        import json as _json
        await r.rpush(key, _json.dumps(item, ensure_ascii=False))
        await r.ltrim(key, max(-1, -int(max_items)), -1)
        await r.expire(key, 60 * 60 * 24 * 30)
    except Exception:
        pass


async def chat_get(telegram_id: int, chat_id: int, provider: str, *, limit: int = 200) -> List[Dict[str, Any]]:
    r = get_redis()
    key = _chat_key(telegram_id, chat_id, provider)
    out: List[Dict[str, Any]] = []
    try:
        vals = await r.lrange(key, max(0, -int(limit)), -1)
        for v in vals or []:
            try:
                s = v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                import json as _json
                obj = _json.loads(s)
                if isinstance(obj, dict) and (obj.get("role") and obj.get("content") is not None):
                    out.append({"role": str(obj.get("role")), "content": str(obj.get("content"))})
            except Exception:
                continue
    except Exception:
        return []
    return out


async def chat_clear(telegram_id: int, chat_id: int, provider: str) -> None:
    r = get_redis()
    key = _chat_key(telegram_id, chat_id, provider)
    try:
        await r.delete(key)
    except Exception:
        pass
        try:
            sval = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
        except Exception:
            sval = str(val)
        return float(sval)
    except Exception:
        return None


# ---- Simple rate limiting via Redis counters ----

async def rate_allow(telegram_id: int, scope: str = "gen", per_hour: int = 10, per_day: int = 50) -> bool:
    r = get_redis()
    scope = (scope or "gen").strip().lower()
    hkey = f"{kv_prefix()}:rate:{scope}:h:{telegram_id}"
    dkey = f"{kv_prefix()}:rate:{scope}:d:{telegram_id}"
    try:
        h = await r.incr(hkey)
        if int(h) == 1:
            await r.expire(hkey, 3600)
        d = await r.incr(dkey)
        if int(d) == 1:
            await r.expire(dkey, 86400)
        return (int(h) <= int(per_hour)) and (int(d) <= int(per_day))
    except Exception:
        # On Redis issues, allow to avoid false negatives
        return True


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

# ---- Refine preference ----

async def set_refine_enabled(telegram_id: int, enabled: bool) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:refine:{telegram_id}"
    await r.set(key, "1" if enabled else "0")


async def get_refine_enabled(telegram_id: int) -> bool:
    r = get_redis()
    key = f"{kv_prefix()}:refine:{telegram_id}"
    val = await r.get(key)
    try:
        return (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "0")).strip() == "1"
    except Exception:
        return False


# ---- Fact-check preference ----

async def set_factcheck_enabled(telegram_id: int, enabled: bool) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:fc_enabled:{telegram_id}"
    await r.set(key, "1" if enabled else "0")


async def get_factcheck_enabled(telegram_id: int) -> bool:
    r = get_redis()
    key = f"{kv_prefix()}:fc_enabled:{telegram_id}"
    val = await r.get(key)
    try:
        return (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "0")).strip() == "1"
    except Exception:
        return False


async def set_factcheck_depth(telegram_id: int, depth: int) -> None:
    r = get_redis()
    key = f"{kv_prefix()}:fc_depth:{telegram_id}"
    try:
        d = int(depth)
    except Exception:
        d = 1
    if d not in (1, 2, 3):
        d = 1
    await r.set(key, str(d))


async def get_factcheck_depth(telegram_id: int) -> int:
    r = get_redis()
    key = f"{kv_prefix()}:fc_depth:{telegram_id}"
    val = await r.get(key)
    try:
        s = (val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val or "2")).strip()
        d = int(s)
    except Exception:
        d = 2
    return 2 if d not in (1, 2, 3) else d


#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from aiogram import types, Bot, Dispatcher

from utils.env import load_env_from_root
from .bot import create_dispatcher
from .bot_commands import register_admin_commands, ensure_db_ready
from .db import SessionLocal, JobLog, Job


def _load_env():
    load_env_from_root(__file__)


_load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

app = FastAPI(title="Book in One Click Bot API")

# Create a single dispatcher/bot for webhook handling
DP = create_dispatcher()
register_admin_commands(DP, SessionLocal)


@app.on_event("startup")
async def _startup():
    # Initialize DB if configured; safe no-op otherwise
    await ensure_db_ready()
    # Ensure enough worker threads for concurrent generations (5–10 users)
    import asyncio
    import concurrent.futures
    try:
        max_workers = int(os.getenv("BOT_WORKERS", "16"))
    except Exception:
        max_workers = 16
    try:
        loop = asyncio.get_event_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=max_workers))
    except Exception:
        # Non-fatal: fallback to default executor
        pass
    # Register bot commands and enable command menu button
    try:
        await DP.bot.set_my_commands([
            types.BotCommand(command="start", description="Start / Начать"),
            types.BotCommand(command="balance", description="Balance / Баланс"),
            types.BotCommand(command="info", description="Info / Инфо"),
            types.BotCommand(command="lang", description="Language / Язык"),
            types.BotCommand(command="lang_generate", description="Gen Language / Язык генерации"),
            types.BotCommand(command="generate", description="Generate / Сгенерировать"),
            types.BotCommand(command="provider", description="Provider / Провайдер"),
            types.BotCommand(command="cancel", description="Cancel / Отмена"),
        ])
        try:
            await DP.bot.set_chat_menu_button(menu_button=types.MenuButtonCommands())
        except Exception:
            pass
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"ok": True}
@app.get("/logs")
async def list_logs():
    items = []
    # Fallback: if no SQL DB configured, read logs from filesystem
    if SessionLocal is None:
        base = Path("output")
        files = list(base.glob("**/*_log.md")) if base.exists() else []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for idx, p in enumerate(files, start=1):
            try:
                ts = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
            except Exception:
                ts = ""
            items.append({
                "id": idx,
                "job_id": 0,
                "kind": "md",
                "path": str(p),
                "created_at": ts,
                "source": "fs",
            })
        return {"items": items}
    async with SessionLocal() as s:
        from sqlalchemy import select
        res = await s.execute(select(JobLog).order_by(JobLog.id.desc()).limit(200))
        rows = res.scalars().all()
        for r in rows:
            items.append({
                "id": r.id,
                "job_id": r.job_id,
                "kind": r.kind,
                "path": r.path,
                "created_at": str(r.created_at),
            })
    return {"items": items}


@app.get("/logs/{log_id}")
async def get_log(log_id: int):
    if SessionLocal is None:
        # Fallback: map log_id to Nth recent file in output/**/*_log.md
        base = Path("output")
        files = list(base.glob("**/*_log.md")) if base.exists() else []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if log_id <= 0 or log_id > len(files):
            return {"error": "not found"}
        p = files[log_id - 1]
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
        try:
            created = datetime.fromtimestamp(p.stat().st_mtime).isoformat()
        except Exception:
            created = ""
        return {
            "id": log_id,
            "job_id": 0,
            "path": str(p),
            "created_at": created,
            "content": content,
            "source": "fs",
        }
    async with SessionLocal() as s:
        from sqlalchemy import select
        res = await s.execute(select(JobLog).where(JobLog.id == log_id))
        row = res.scalar_one_or_none()
        if row is None:
            return {"error": "not found"}
        try:
            with open(row.path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            content = ""
        # try resolve basic job info
        topic = ""
        provider = ""
        created = str(row.created_at)
        return {
            "id": row.id,
            "job_id": row.job_id,
            "path": row.path,
            "created_at": created,
            "content": content,
        }


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if not TELEGRAM_TOKEN:
        raise HTTPException(status_code=500, detail="Bot not configured")
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        data = await request.json()
        # Aiogram v2 expects Update constructed from dict via kwargs
        update = types.Update(**data)
        # Ensure aiogram context is set in webhook execution
        Bot.set_current(DP.bot)
        Dispatcher.set_current(DP)
        await DP.process_update(update)
        return JSONResponse({"ok": True})
    except Exception as e:
        # Avoid Telegram retries storm; log and return ok
        print(f"webhook error: {e}")
        return JSONResponse({"ok": True})




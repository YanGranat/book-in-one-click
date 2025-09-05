#!/usr/bin/env python3
from __future__ import annotations

import os
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from aiogram import types

from utils.env import load_env_from_root
from .bot import create_dispatcher
from .bot_commands import register_admin_commands, ensure_db_ready
from .db import SessionLocal


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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if not TELEGRAM_TOKEN:
        raise HTTPException(status_code=500, detail="Bot not configured")
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    # Aiogram v2 expects Update constructed from dict via kwargs
    update = types.Update(**data)
    await DP.process_update(update)
    return JSONResponse({"ok": True})




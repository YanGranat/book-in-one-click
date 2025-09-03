#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# Bot uses aiogram's Dispatcher separate process ideally, but for MVP we'll keep webhook handling simple
from aiogram import Bot, types
from aiogram.dispatcher.webhook import WebhookRequestHandler

from utils.env import load_env_from_root


def _load_env():
    load_env_from_root(__file__)


_load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

if not TELEGRAM_TOKEN:
    # App can still start; but endpoint will raise until configured
    pass

bot: Optional[Bot] = Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

app = FastAPI(title="Book in One Click Bot API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/webhook/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if not bot:
        raise HTTPException(status_code=500, detail="Bot not configured")
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden")

    data = await request.json()
    update = types.Update.to_object(data)
    # For MVP: reply with a basic message to confirm delivery
    if update.message and update.message.text:
        chat_id = update.message.chat.id
        await bot.send_message(chat_id, "✅ Webhook received. Please use the main bot entry point (soon)")
    return JSONResponse({"ok": True})




#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
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


def _extract_meta_from_text(md: str) -> dict:
    meta = {}
    try:
        for line in md.splitlines()[:50]:
            line = line.strip()
            if line.startswith("- provider:"):
                v = line.split("`", 2)
                if len(v) >= 2:
                    meta["provider"] = v[1]
            elif line.startswith("- lang:"):
                v = line.split("`", 2)
                if len(v) >= 2:
                    meta["lang"] = v[1]
            elif line.startswith("- model_heavy:"):
                v = line.split("`", 2)
                if len(v) >= 2:
                    meta["model_heavy"] = v[1]
            elif line.startswith("- model_fast:"):
                v = line.split("`", 2)
                if len(v) >= 2:
                    meta["model_fast"] = v[1]
    except Exception:
        pass
    return meta


@app.get("/logs-ui", response_class=HTMLResponse)
async def logs_ui():
    # Reuse /logs data
    data = await list_logs()
    items = data.get("items", [])
    # Derive topic from filename
    for it in items:
        stem = Path(it.get("path", "")).name
        if stem.endswith("_log.md"):
            stem = stem[:-7]
        it["topic"] = stem
    html_rows = []
    for it in items:
        html_rows.append(
            f"<tr><td>{it.get('id')}</td><td><a href='/logs-ui/{it.get('id')}'>{it.get('topic')}</a></td>"
            f"<td>{it.get('created_at','')}</td><td>{it.get('kind','')}</td></tr>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>Logs</title>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;padding:20px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #333;padding:8px;text-align:left}"
        "th{background:#111;color:#eee}tr:nth-child(even){background:#1a1a1a;color:#eee}a{color:#6cf}</style></head><body>"
        "<h1>Generation Logs</h1>"
        f"<p>Total: {len(items)}</p>"
        "<table><thead><tr><th>ID</th><th>Topic</th><th>Created</th><th>Kind</th></tr></thead><tbody>"
        + ("".join(html_rows) or "<tr><td colspan='4'>No logs yet</td></tr>")
        + "</tbody></table>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/logs-ui/{log_id}", response_class=HTMLResponse)
async def log_view_ui(log_id: int):
    # Reuse JSON endpoint to load content
    data = await get_log(log_id)
    if "error" in data:
        return HTMLResponse(f"<h1>Not found</h1><p>{data['error']}</p>", status_code=404)
    content = data.get("content", "")
    meta = _extract_meta_from_text(content)
    title = Path(data.get("path", "")).name
    html = (
        "<html><head><meta charset='utf-8'><title>Log View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0}"
        "header{background:#111;color:#eee;padding:12px 16px;display:flex;gap:16px;align-items:center}"
        "main{padding:16px}#content{max-width:1000px;margin:0 auto}a{color:#6cf}"
        "code,pre{background:#0f0f0f;color:#ddd;padding:4px 6px;border-radius:4px}"
        "</style></head><body>"
        f"<header><a href='/logs-ui'>← Back</a><div>{title}</div>"
        f"<div style='margin-left:auto;opacity:.8'>provider={meta.get('provider','?')} | lang={meta.get('lang','?')}</div>"
        "</header>"
        "<main>"
        "<div id='content'></div>"
        "<textarea id='md' style='display:none'></textarea>"
        "</main>"
        "<script>document.getElementById('md').value = atob('" + content.encode('utf-8').hex() + "');</script>"
        "<script>const hex= document.getElementById('md').value;"
        "function hexToStr(h){let s='';for(let i=0;i<h.length;i+=2){s+=String.fromCharCode(parseInt(h.substr(i,2),16));}return s;}"
        "document.getElementById('content').innerHTML = marked.parse(hexToStr(hex));</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


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




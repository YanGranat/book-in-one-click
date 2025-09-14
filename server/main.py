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
            types.BotCommand(command="logs", description="Logs / Логи генерации"),
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
    
    # Read only from DB
    if SessionLocal is not None:
        try:
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
        except Exception as e:
            print(f"[ERROR] DB read failed: {e}")
    
    return {"items": items}


@app.get("/logs/{log_id}")
async def get_log(log_id: int):
    # Read only from DB
    if SessionLocal is None:
        return {"error": "db is not configured"}
    
    try:
        async with SessionLocal() as s:
            from sqlalchemy import select
            res = await s.execute(select(JobLog).where(JobLog.id == log_id))
            row = res.scalar_one_or_none()
            if row is None:
                return {"error": "not found"}
            
            try:
                # Try relative path first, then absolute
                log_file_path = Path(row.path)
                if not log_file_path.is_absolute():
                    log_file_path = Path.cwd() / log_file_path
                with open(log_file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                print(f"[ERROR] Cannot read log file {row.path}: {e}")
                content = f"Error reading log file: {e}"
            
            return {
                "id": row.id,
                "job_id": row.job_id,
                "path": row.path,
                "created_at": str(row.created_at),
                "content": content,
            }
    except Exception as e:
        print(f"[ERROR] DB get_log failed: {e}")
        return {"error": f"database error: {e}"}


@app.api_route("/logs-seed", methods=["GET", "POST"])
async def logs_seed():
    if SessionLocal is None:
        return {"error": "db is not configured"}
    try:
        async with SessionLocal() as s:
            item = JobLog(job_id=0, kind="seed", path="seed")
            s.add(item)
            await s.commit()
            return {"ok": True, "id": item.id}
    except Exception as e:
        return {"error": str(e)}


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
            elif line.startswith("- topic:"):
                v = line.split("`", 2)
                if len(v) >= 2:
                    meta["topic"] = v[1]
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
    # Extract topic from log content and format timestamp
    for it in items:
        # Try to get topic from log file content first
        topic_from_content = ""
        try:
            path_str = it.get("path", "")
            if path_str:
                # Handle both relative and absolute paths
                log_file_path = Path(path_str)
                if not log_file_path.is_absolute():
                    log_file_path = Path.cwd() / log_file_path
                if log_file_path.exists():
                    with open(log_file_path, "r", encoding="utf-8") as f:
                        content = f.read(2000)  # Read first 2KB to find topic
                    meta = _extract_meta_from_text(content)
                    topic_from_content = meta.get("topic", "")
        except Exception:
            pass
        
        # Use topic from content, fallback to cleaned filename
        if topic_from_content:
            it["topic"] = topic_from_content
        else:
            stem = Path(it.get("path", "")).name
            if stem.endswith(".md"):
                stem = stem[:-3]
            # Remove _log and timestamp suffix
            import re
            stem = re.sub(r"_log(_\d{8}_\d{6})?$", "", stem)
            it["topic"] = stem
            
        # Format timestamp as YYYY-MM-DD HH:MM
        try:
            from datetime import datetime
            ts_str = it.get("created_at", "")
            if ts_str:
                if "T" in ts_str:  # ISO format
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:  # Already formatted
                    dt = datetime.fromisoformat(ts_str)
                it["created_at"] = dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            pass
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
    import base64
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    html = (
        "<html><head><meta charset='utf-8'><title>Log View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;line-height:1.6}"
        "header{background:#111;color:#eee;padding:12px 16px;display:flex;gap:16px;align-items:center}"
        "main{padding:16px}#content{max-width:1000px;margin:0 auto}a{color:#6cf}"
        "h1{font-size:1.8em;margin:1em 0 0.5em}h2{font-size:1.4em;margin:1em 0 0.5em}h3{font-size:1.2em;margin:0.8em 0 0.4em;color:#333}"
        "code{background:none;color:inherit;padding:0}pre{background:none;color:inherit;padding:0;white-space:pre-wrap;word-wrap:break-word}"
        "p{margin:0.5em 0}ul,ol{margin:0.5em 0}li{margin:0.2em 0}"
        "</style></head><body>"
        f"<header><a href='/logs-ui'>← Back</a><div>{title}</div>"
        f"<div style='margin-left:auto;opacity:.8'>provider={meta.get('provider','?')} | lang={meta.get('lang','?')}</div>"
        "</header>"
        "<main>"
        "<div id='content'></div>"
        "</main>"
        f"<script>const b64='{b64}';"
        "const bin=atob(b64);const bytes=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++){bytes[i]=bin.charCodeAt(i);}"
        "const text=new TextDecoder('utf-8').decode(bytes);"
        "const escaped=text.replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\\n/g,'<br>');"
        "document.getElementById('content').innerHTML = marked.parse(escaped);</script>"
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




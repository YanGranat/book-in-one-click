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
                    # Extract topic from stored content if available
                    topic = ""
                    try:
                        if getattr(r, "content", None):
                            meta = _extract_meta_from_text((r.content or "")[:2000])
                            topic = meta.get("topic", "")
                    except Exception:
                        pass
                    if not topic:
                        # Fallback to filename-based topic
                        stem = Path(getattr(r, "path", "")).name
                        if stem.endswith(".md"):
                            stem = stem[:-3]
                        import re
                        stem = re.sub(r"_log(_\d{8}_\d{6})?$", "", stem)
                        topic = stem
                    items.append({
                        "id": r.id,
                        "job_id": r.job_id,
                        "kind": r.kind,
                        "path": r.path,
                        "created_at": str(r.created_at),
                        "topic": topic,
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
            
            # DB-only: return content from DB or report absence
            if row.content:
                content = row.content
            else:
                return {
                    "id": row.id,
                    "job_id": row.job_id,
                    "path": row.path,
                    "created_at": str(row.created_at),
                    "content": "",
                    "error": "no content stored for this log"
                }
            
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
        for line in md.splitlines()[:80]:
            line = line.strip()
            def _val(s: str) -> str:
                return s.split(":", 1)[1].strip() if ":" in s else ""
            if line.startswith("- provider:"):
                meta["provider"] = _val(line)
            elif line.startswith("- lang:"):
                meta["lang"] = _val(line)
            elif line.startswith("- topic:"):
                meta["topic"] = _val(line)
            elif line.startswith("- model_heavy:"):
                meta["model_heavy"] = _val(line)
            elif line.startswith("- model_fast:"):
                meta["model_fast"] = _val(line)
    except Exception:
        pass
    return meta


@app.get("/logs-ui", response_class=HTMLResponse)
async def logs_ui():
    # Reuse /logs data
    data = await list_logs()
    items = data.get("items", [])
    # Format timestamp as YYYY-MM-DD HH:MM
    for it in items:
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
    # Try to get initial content for instant render and title
    title = f"Log #{log_id}"
    initial_content = ""
    try:
        data = await get_log(log_id)
        if "path" in data and data["path"]:
            title = Path(data["path"]).name
        initial_content = data.get("content", "") or ""
    except Exception:
        pass
    from html import escape as _esc
    _raw = (_esc(initial_content or "").replace("</textarea>", "&lt;/textarea&gt;") if initial_content else "")
    html = (
        "<html><head><meta charset='utf-8'><title>Log View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;line-height:1.5;font-size:12px}"
        "header{background:#111;color:#eee;padding:6px 10px;display:flex;gap:10px;align-items:center;font-size:12px}"
        "main{padding:10px}#content{max-width:1000px;margin:0 auto}a{color:#6cf}"
        "h1{font-size:1.8em;margin:0.6em 0 0.3em;font-weight:700}h2{font-size:1.5em;margin:0.6em 0 0.3em;color:#333;font-weight:700}h3{font-size:1.25em;margin:0.5em 0 0.25em;color:#555;font-weight:600}"
        "code{background:#0b0b0b0a;color:inherit;padding:0;font-size:12px}pre{background:#0b0b0b0a;color:inherit;padding:8px;border-radius:6px;white-space:pre-wrap;word-wrap:break-word;font-size:12px}"
        "p{margin:0.25em 0;font-size:12px}ul,ol{margin:0.25em 0}li{margin:0.1em 0;font-size:12px}"
        "</style></head><body>"
        f"<header><a href='/logs-ui'>← Back</a><div>{title}</div>"
        f"<div id='meta' style='margin-left:auto;opacity:.8'>provider=? | lang=?</div>"
        "</header>"
        "<main>"
        "<div id='content'></div>"
        f"<textarea id='raw' style='display:none'>{_raw}</textarea>"
        "</main>"
        f"<script>const LOG_ID={log_id};"
        "const TAGS=['input','topic','lang','post','critique_json'];"
        "function escapeOutsideCode(md){const lines=md.split('\\n');let inCode=false;for(let i=0;i<lines.length;i++){const t=lines[i].trim();if(t.startsWith('```')){inCode=!inCode;continue;}if(!inCode){let s=lines[i];for(const tag of TAGS){s=s.replace(new RegExp('<'+tag+'>','g'),'&lt;'+tag+'&gt;').replace(new RegExp('</'+tag+'>','g'),'&lt;/'+tag+'&gt;');}lines[i]=s;}}return lines.join('\\n');}"
        "function extractMeta(md){const out={};const lines=md.split('\\n');for(let i=0;i<Math.min(lines.length,80);i++){const line=lines[i].trim();if(line.startsWith('- provider:')){out.provider=line.split(':').slice(1).join(':').trim();}else if(line.startsWith('- lang:')){out.lang=line.split(':').slice(1).join(':').trim();}else if(line.startsWith('- topic:')){out.topic=line.split(':').slice(1).join(':').trim();}}return out;}"
        "let text=(document.getElementById('raw')?document.getElementById('raw').value:'');"
        "function render(md){const escaped=escapeOutsideCode(md||'');const fenced=escaped.replace(/&lt;input&gt;[\\s\\S]*?&lt;\\/input&gt;/g, m=> '```\\n'+m+'\\n```');"
        "let html='';try{html=(window.marked?window.marked.parse(fenced):'');}catch(e){html='';}"
        "if(!html||html.trim()===''){const safe=fenced.replace(/</g,'&lt;').replace(/>/g,'&gt;');html='<pre>'+safe+'</pre>'; }"
        "const m=extractMeta(md||'');const metaEl=document.getElementById('meta');if(metaEl){metaEl.textContent=`provider=${m.provider||'?'} | lang=${m.lang||'?'}`;}"
        "document.getElementById('content').innerHTML = html;}"
        "render(text);"
        "async function refresh(){try{const r=await fetch('/logs/'+LOG_ID,{cache:'no-store'});const j=await r.json();if(j&&j.content&&(j.content.length>0)){text=j.content;render(text);}}catch(_){/* ignore */}}"
        "refresh();</script>"
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




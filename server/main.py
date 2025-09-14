#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
import asyncio
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from aiogram import types, Bot, Dispatcher
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import base64

from utils.env import load_env_from_root
# Load .env BEFORE importing DB module to ensure DB_URL/DB_TABLE_PREFIX are visible
load_env_from_root(__file__)
from .bot import create_dispatcher
from .bot_commands import register_admin_commands, ensure_db_ready
from .db import SessionLocal, JobLog, Job, ResultDoc


def _load_env():
    load_env_from_root(__file__)


# Already loaded above
# _load_env()

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
                # Order by created_at if available; otherwise id desc
                res = await s.execute(select(JobLog).order_by(JobLog.created_at.desc(), JobLog.id.desc()).limit(200))
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
    # Fallback: sync read in case async engine/session isn't available
    if (not items):
        try:
            from sqlalchemy import create_engine, text as _sqtext
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            db_url = os.getenv("DB_URL", "").strip()
            if db_url:
                sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
                parts = urlsplit(sync_url)
                qs = parse_qs(parts.query or "")
                base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
                cargs = {"sslmode": "require"} if not any(k.lower()=="sslmode" for k in qs.keys()) else {}
                eng = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True)
                table = JobLog.__tablename__
                with eng.connect() as conn:
                    res = conn.execute(_sqtext(f"SELECT id, job_id, kind, path, content, created_at FROM {table} ORDER BY created_at DESC NULLS LAST, id DESC LIMIT :lim"), {"lim": 200})
                    for r in res.fetchall():
                        rid, rjob, rkind, rpath, rcont, rts = r
                        topic = ""
                        try:
                            if rcont:
                                meta = _extract_meta_from_text((rcont or "")[:2000])
                                topic = meta.get("topic", "")
                        except Exception:
                            pass
                        if not topic:
                            stem = Path(rpath or "").name
                            if stem.endswith(".md"):
                                stem = stem[:-3]
                            import re as _re
                            stem = _re.sub(r"_log(_\d{8}_\d{6})?$", "", stem)
                            topic = stem
                        items.append({
                            "id": rid,
                            "job_id": rjob,
                            "kind": rkind,
                            "path": rpath,
                            "created_at": str(rts) if rts is not None else "",
                            "topic": topic,
                        })
                try:
                    eng.dispose()
                except Exception:
                    pass
        except Exception as e:
            print(f"[ERROR] Sync fallback read failed: {e}")
    return {"items": items}


@app.get("/logs/{log_id}")
async def get_log(log_id: int):
    # Read only from DB
    if SessionLocal is None:
        # Try sync fallback
        try:
            from sqlalchemy import create_engine, text as _sqtext
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            db_url = os.getenv("DB_URL", "").strip()
            if not db_url:
                return {"error": "db is not configured"}
            sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
            parts = urlsplit(sync_url)
            qs = parse_qs(parts.query or "")
            base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
            cargs = {"sslmode": "require"} if not any(k.lower()=="sslmode" for k in qs.keys()) else {}
            eng = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True)
            table = JobLog.__tablename__
            with eng.connect() as conn:
                res = conn.execute(_sqtext(f"SELECT id, job_id, kind, path, content, created_at FROM {table} WHERE id=:id"), {"id": int(log_id)})
                row = res.fetchone()
            try:
                eng.dispose()
            except Exception:
                pass
            if not row:
                return {"error": "not found"}
            rid, rjob, rkind, rpath, rcont, rts = row
            return {"id": rid, "job_id": rjob, "path": rpath, "created_at": str(rts) if rts else "", "content": rcont or ""}
        except Exception as e:
            return {"error": f"db not configured: {e}"}
    
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
                # Try sync fallback read (in case async row lacks column due to migration lag)
                try:
                    from sqlalchemy import create_engine, text as _sqtext
                    from urllib.parse import urlsplit, urlunsplit, parse_qs
                    db_url = os.getenv("DB_URL", "").strip()
                    sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
                    parts = urlsplit(sync_url)
                    qs = parse_qs(parts.query or "")
                    base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
                    cargs = {"sslmode": "require"} if not any(k.lower()=="sslmode" for k in qs.keys()) else {}
                    eng = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True)
                    table = JobLog.__tablename__
                    with eng.connect() as conn:
                        res2 = conn.execute(_sqtext(f"SELECT content FROM {table} WHERE id=:id"), {"id": int(log_id)})
                        r2 = res2.fetchone()
                    try:
                        eng.dispose()
                    except Exception:
                        pass
                    content = r2[0] if r2 and r2[0] else ""
                except Exception:
                    content = ""
            
            return {
                "id": row.id,
                "job_id": row.job_id,
                "path": row.path,
                "created_at": str(row.created_at),
                "content": content,
            }
    except Exception as e:
        print(f"[ERROR] DB get_log failed: {e}")
        # Try sync fallback as last resort
        try:
            from sqlalchemy import create_engine, text as _sqtext
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            db_url = os.getenv("DB_URL", "").strip()
            if not db_url:
                return {"error": f"database error: {e}"}
            sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
            parts = urlsplit(sync_url)
            qs = parse_qs(parts.query or "")
            base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
            cargs = {"sslmode": "require"} if not any(k.lower()=="sslmode" for k in qs.keys()) else {}
            eng = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True)
            table = JobLog.__tablename__
            with eng.connect() as conn:
                res = conn.execute(_sqtext(f"SELECT id, job_id, kind, path, content, created_at FROM {table} WHERE id=:id"), {"id": int(log_id)})
                row = res.fetchone()
            try:
                eng.dispose()
            except Exception:
                pass
            if not row:
                return {"error": "not found"}
            rid, rjob, rkind, rpath, rcont, rts = row
            return {"id": rid, "job_id": rjob, "path": rpath, "created_at": str(rts) if rts else "", "content": rcont or ""}
        except Exception as e2:
            return {"error": f"database error: {e} / fallback: {e2}"}


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


# ----- Admin UI auth (HTTP Basic) -----
_basic = HTTPBasic()

def require_admin(creds: HTTPBasicCredentials = Depends(_basic)) -> bool:
    user = os.getenv("ADMIN_UI_USER", "admin")
    pwd = os.getenv("ADMIN_UI_PASSWORD", "")
    if not pwd:
        # Explicitly deny when password is not configured
        raise HTTPException(status_code=503, detail="ADMIN_UI_PASSWORD not configured")
    if secrets.compare_digest(creds.username, user) and secrets.compare_digest(creds.password, pwd):
        return True
    raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})


@app.get("/logs-ui", response_class=HTMLResponse)
async def logs_ui(_: bool = Depends(require_admin)):
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
            f"<tr>"
            f"<td><input type='checkbox' class='sel' value='{it.get('id')}'></td>"
            f"<td>{it.get('id')}</td>"
            f"<td><a href='/logs-ui/{it.get('id')}'>{it.get('topic')}</a></td>"
            f"<td>{it.get('created_at','')}</td>"
            f"<td>{it.get('kind','')}</td>"
            f"<td><button class='delBtn' data-id='{it.get('id')}'>Delete</button></td>"
            f"</tr>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>Logs</title>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;padding:20px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #333;padding:8px;text-align:left}"
        "th{background:#111;color:#eee}tr:nth-child(even){background:#1a1a1a;color:#eee}a{color:#6cf}button{padding:4px 8px}</style></head><body>"
        "<h1>Generation Logs</h1>"
        f"<p>Total: {len(items)}</p>"
        "<button id='delSel'>Delete selected</button>"
        "<table><thead><tr><th></th><th>ID</th><th>Topic</th><th>Created</th><th>Kind</th><th>Action</th></tr></thead><tbody>"
        + ("".join(html_rows) or "<tr><td colspan='4'>No logs yet</td></tr>")
        + "</tbody></table>"
        "<script>document.addEventListener('click',async(e)=>{if(e.target.matches('.delBtn')){const id=e.target.getAttribute('data-id');if(confirm('Delete log '+id+'?')){const r=await fetch('/logs/'+id,{method:'DELETE'});const j=await r.json();if(j&&j.ok){location.reload();}}}});document.getElementById('delSel').onclick=async()=>{const ids=[...document.querySelectorAll('.sel:checked')].map(x=>parseInt(x.value));if(!ids.length)return;if(!confirm('Delete '+ids.length+' logs?'))return;const r=await fetch('/logs/purge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});const j=await r.json();if(j&&j.ok){location.reload();}};</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/logs-ui/{log_id}", response_class=HTMLResponse)
async def log_view_ui(log_id: int, _: bool = Depends(require_admin)):
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
        "function render(md){const raw=(md||'');const fencedRaw=raw.replace(/<input>[\\s\\S]*?<\\/input>/g, m=> '```\\n'+m+'\\n```');const escaped=escapeOutsideCode(fencedRaw);"
        "let html='';try{html=(window.marked?window.marked.parse(escaped):'');}catch(e){html='';}"
        "if(!html||html.trim()===''){const safe=escaped.replace(/</g,'&lt;').replace(/>/g,'&gt;');html='<pre>'+safe+'</pre>'; }"
        "const m=extractMeta(md||'');const metaEl=document.getElementById('meta');if(metaEl){metaEl.textContent=`provider=${m.provider||'?'} | lang=${m.lang||'?'}`;}"
        "document.getElementById('content').innerHTML = html;}"
        "render(text);"
        "async function refresh(){try{const r=await fetch('/logs/'+LOG_ID,{cache:'no-store'});const j=await r.json();if(j&&j.content&&(j.content.length>0)){text=j.content;render(text);}}catch(_){/* ignore */}}"
        "refresh();</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.delete("/logs/{log_id}")
async def delete_log_api(log_id: int, _: bool = Depends(require_admin)):
    if SessionLocal is None:
        raise HTTPException(status_code=500, detail="DB is not configured")
    async with SessionLocal() as s:
        obj = await s.get(JobLog, log_id)
        if obj is None:
            return {"ok": False, "error": "not found"}
        # Also delete results for same job_id
        try:
            if getattr(obj, "job_id", 0):
                from sqlalchemy import delete as _sqdel
                await s.execute(_sqdel(ResultDoc).where(ResultDoc.job_id == obj.job_id))
        except Exception:
            pass
        await s.delete(obj)
        await s.commit()
        return {"ok": True, "deleted_id": log_id}


@app.post("/logs/purge")
async def purge_logs(payload: dict, _: bool = Depends(require_admin)):
    if SessionLocal is None:
        raise HTTPException(status_code=500, detail="DB is not configured")
    ids = payload.get("ids") or []
    if not isinstance(ids, list) or not all(isinstance(x, int) for x in ids):
        return {"ok": False, "error": "ids must be list[int]"}
    deleted = 0
    async with SessionLocal() as s:
        if ids:
            from sqlalchemy import delete, select
            # Collect affected job_ids first
            res = await s.execute(select(JobLog.job_id).where(JobLog.id.in_(ids)))
            job_ids = [jid for (jid,) in (res.fetchall() or []) if isinstance(jid, int) and jid > 0]
            if job_ids:
                await s.execute(delete(ResultDoc).where(ResultDoc.job_id.in_(job_ids)))
            stmt = delete(JobLog).where(JobLog.id.in_(ids))
            res2 = await s.execute(stmt)
            await s.commit()
            deleted = res2.rowcount or 0
    return {"ok": True, "deleted": int(deleted)}


# ------------------- Results (final outputs on filesystem) -------------------

async def _list_result_files() -> list[dict]:
    # Prefer DB (ResultDoc), fallback to filesystem if empty
    items: list[dict] = []
    if SessionLocal is not None:
        try:
            async with SessionLocal() as s:
                from sqlalchemy import select
                res = await s.execute(select(ResultDoc).where(ResultDoc.hidden == 0).order_by(ResultDoc.created_at.desc(), ResultDoc.id.desc()))
                rows = res.scalars().all()
                for r in rows:
                    items.append({
                        "id": r.id,
                        "path": r.path,
                        "name": Path(r.path).name,
                        "created_at": str(r.created_at),
                        "kind": r.kind,
                        "topic": getattr(r, "topic", "") or "",
                        "provider": getattr(r, "provider", "") or "",
                        "lang": getattr(r, "lang", "") or "",
                    })
        except Exception:
            items = []
    if items:
        return items
    # Fallback: scan filesystem
    fs_items: list[dict] = []
    root = Path("output")
    if not root.exists():
        return fs_items
    for p in root.rglob("*.md"):
        if "_log_" in p.name or p.name.endswith("_log.md"):
            continue
        try:
            stat = p.stat()
            created = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            fs_items.append({
                "path": str(p),
                "name": p.name,
                "created_at": created,
                "kind": p.parent.name,
            })
        except Exception:
            continue
    fs_items.sort(key=lambda x: x["created_at"], reverse=True)
    return fs_items


@app.get("/results")
async def list_results():
    return {"items": await _list_result_files()}


@app.get("/results-ui", response_class=HTMLResponse)
async def results_ui():
    items = await _list_result_files()
    rows = []
    for it in items:
        try:
            if "id" in it and isinstance(it.get("id"), int):
                link = f"/results-ui/id/{it['id']}"
            else:
                b64 = base64.urlsafe_b64encode((it.get("path") or "").encode("utf-8")).decode("ascii")
                link = f"/results-ui/file/{b64}"
            rows.append(
                f"<tr><td><a href='{link}'>{it.get('name','result.md')}</a></td>"
                f"<td>{it.get('created_at','')}</td><td>{it.get('kind','')}</td><td>{it.get('provider','')}</td><td>{it.get('lang','')}</td><td>{it.get('topic','')}</td></tr>"
            )
        except Exception:
            continue
    html = (
        "<html><head><meta charset='utf-8'><title>Results</title>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;padding:20px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #333;padding:8px;text-align:left}"
        "th{background:#111;color:#eee}tr:nth-child(even){background:#1a1a1a;color:#eee}a{color:#6cf}</style></head><body>"
        "<h1>Generated Results</h1>"
        f"<p>Total: {len(items)}</p>"
        "<table><thead><tr><th>Name</th><th>Created</th><th>Kind</th><th>Provider</th><th>Lang</th><th>Topic</th></tr></thead><tbody>"
        + ("".join(rows) or "<tr><td colspan='6'>No results yet</td></tr>")
        + "</tbody></table>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/results/{res_id}")
async def get_result(res_id: int):
    if SessionLocal is None:
        # Sync fallback
        try:
            from sqlalchemy import create_engine, text as _sqtext
            from urllib.parse import urlsplit, urlunsplit, parse_qs
            db_url = os.getenv("DB_URL", "").strip()
            if not db_url:
                return {"error": "db is not configured"}
            sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg2://").replace("postgresql://", "postgresql+psycopg2://")
            parts = urlsplit(sync_url)
            qs = parse_qs(parts.query or "")
            base_sync_url = urlunsplit((parts.scheme, parts.netloc, parts.path, "", parts.fragment))
            cargs = {"sslmode": "require"} if not any(k.lower()=="sslmode" for k in qs.keys()) else {}
            eng = create_engine(base_sync_url, connect_args=cargs, pool_pre_ping=True)
            table = ResultDoc.__tablename__
            with eng.connect() as conn:
                res = conn.execute(_sqtext(f"SELECT id, path, content, created_at, kind, provider, lang, topic FROM {table} WHERE id=:id"), {"id": int(res_id)})
                row = res.fetchone()
            try:
                eng.dispose()
            except Exception:
                pass
            if not row:
                return {"error": "not found"}
            rid, rpath, rcont, rts, rkind, rprov, rlang, rtopic = row
            return {"id": rid, "path": rpath, "content": rcont or "", "created_at": str(rts) if rts else "", "kind": rkind, "provider": rprov, "lang": rlang, "topic": rtopic}
        except Exception as e:
            return {"error": f"db not configured: {e}"}
    try:
        async with SessionLocal() as s:
            from sqlalchemy import select
            res = await s.execute(select(ResultDoc).where(ResultDoc.id == res_id))
            row = res.scalar_one_or_none()
            if row is None:
                return {"error": "not found"}
            return {"id": row.id, "path": row.path, "content": row.content or "", "created_at": str(row.created_at), "kind": row.kind, "provider": getattr(row, "provider", None), "lang": getattr(row, "lang", None), "topic": getattr(row, "topic", None)}
    except Exception as e:
        return {"error": f"database error: {e}"}


@app.get("/results-ui/id/{res_id}", response_class=HTMLResponse)
async def result_view_ui_id(res_id: int):
    # Fetch initial content
    data = await get_result(res_id)
    # Hide incognito docs from UI
    try:
        if isinstance(data, dict) and data.get("error") is None:
            # need DB check for hidden flag
            if SessionLocal is not None:
                async with SessionLocal() as s:
                    from sqlalchemy import select
                    res = await s.execute(select(ResultDoc).where(ResultDoc.id == res_id))
                    row = res.scalar_one_or_none()
                    if row is not None and getattr(row, "hidden", 0) == 1:
                        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    except Exception:
        pass
    content = data.get("content", "") if isinstance(data, dict) else ""
    title = f"Result #{res_id}"
    from html import escape as _esc
    _raw = (_esc(content or "").replace("</textarea>", "&lt;/textarea&gt;") if content else "")
    html = (
        "<html><head><meta charset='utf-8'><title>Result View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;line-height:1.5;font-size:14px}"
        "header{background:#111;color:#eee;padding:6px 10px;display:flex;gap:10px;align-items:center}"
        "main{padding:10px}#content{max-width:1000px;margin:0 auto}a{color:#6cf}"
        "h1,h2,h3{margin:0.6em 0 0.3em;font-weight:700}pre{white-space:pre-wrap;word-wrap:break-word}"
        "</style></head><body>"
        f"<header><a href='/results-ui'>← Back</a><div>{title}</div></header>"
        "<main>"
        "<div id='content'></div>"
        f"<textarea id='raw' style='display:none'>{_raw}</textarea>"
        "</main>"
        f"<script>const RES_ID={res_id};let text=(document.getElementById('raw')?document.getElementById('raw').value:'');"
        "function render(md){let html='';try{html=(window.marked?window.marked.parse(md):'');}catch(e){html='';}if(!html||html.trim()===''){const safe=md.replace(/</g,'&lt;').replace(/>/g,'&gt;');html='<pre>'+safe+'</pre>';}document.getElementById('content').innerHTML=html;}"
        "render(text);async function refresh(){try{const r=await fetch('/results/'+RES_ID,{cache:'no-store'});const j=await r.json();if(j&&j.content&&(j.content.length>0)){text=j.content;render(text);}}catch(_){}}refresh();</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)


@app.get("/results-ui/file/{b64}", response_class=HTMLResponse)
async def result_view_ui_file(b64: str):
    try:
        path_str = base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8")
    except Exception:
        return HTMLResponse("<h1>Bad request</h1>", status_code=400)
    p = Path(path_str)
    # Security: ensure under output/
    try:
        if not p.resolve().is_relative_to(Path("output").resolve()):
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
    except Exception:
        root = str(Path("output").resolve())
        if root not in str(p.resolve()):
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)
    if not p.exists():
        return HTMLResponse("<h1>Not found</h1>", status_code=404)
    content = p.read_text(encoding="utf-8", errors="ignore")
    title = p.name
    html = (
        "<html><head><meta charset='utf-8'><title>Result View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>body{font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;line-height:1.5;font-size:14px}"
        "header{background:#111;color:#eee;padding:6px 10px;display:flex;gap:10px;align-items:center}"
        "main{padding:10px}#content{max-width:1000px;margin:0 auto}a{color:#6cf}"
        "h1,h2,h3{margin:0.6em 0 0.3em;font-weight:700}pre{white-space:pre-wrap;word-wrap:break-word}"
        "</style></head><body>"
        f"<header><a href='/results-ui'>← Back</a><div>{title}</div></header>"
        "<main><div id='content'></div></main>"
        "<script>const txt=`" + content.replace("`", "\\`") + "`;document.getElementById('content').innerHTML=marked.parse(txt);</script>"
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




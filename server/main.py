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
from .kv import get_redis, kv_prefix
from .bot_commands import register_admin_commands, ensure_db_ready
from .db import SessionLocal, JobLog, Job, ResultDoc


def _load_env():
    load_env_from_root(__file__)


# Already loaded above
# _load_env()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

app = FastAPI(title="Book in One Click Bot API")


async def _rate_allow_download(key_base: str, ip: str, *, per_min: int, per_hour: int, per_day: int) -> bool:
    if not ip:
        return True
    try:
        r = get_redis()
    except Exception:
        return True
    try:
        kb = f"{kv_prefix()}:dl:{key_base}:{ip}"
        m = await r.incr(kb+":m")  # type: ignore
        if int(m) == 1:
            await r.expire(kb+":m", 60)  # type: ignore
        h = await r.incr(kb+":h")  # type: ignore
        if int(h) == 1:
            await r.expire(kb+":h", 3600)  # type: ignore
        d = await r.incr(kb+":d")  # type: ignore
        if int(d) == 1:
            await r.expire(kb+":d", 86400)  # type: ignore
        if (per_min and int(m) > per_min) or (per_hour and int(h) > per_hour) or (per_day and int(d) > per_day):
            return False
        return True
    except Exception:
        return True

# Create a single dispatcher/bot for webhook handling
DP = create_dispatcher()
register_admin_commands(DP, SessionLocal)


@app.on_event("startup")
async def _startup():
    # Initialize DB if configured; safe no-op otherwise
    await ensure_db_ready()
    # Extra: ensure results table exists proactively (defense in depth)
    try:
        from .db import ResultDoc
        if SessionLocal is not None:
            async with SessionLocal() as s:
                await s.run_sync(ResultDoc.__table__.create, checkfirst=True)
    except Exception:
        pass
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
            types.BotCommand(command="generate", description="Generate / Сгенерировать"),
            types.BotCommand(command="settings", description="Settings / Настройки"),
            types.BotCommand(command="balance", description="Balance / Баланс"),
            types.BotCommand(command="pricing", description="Pricing / Цены"),
            types.BotCommand(command="history", description="History / История"),
            types.BotCommand(command="info", description="Info / Инфо"),
            types.BotCommand(command="lang", description="Language / Язык"),
            types.BotCommand(command="lang_generate", description="Gen Language / Язык генерации"),
            types.BotCommand(command="provider", description="Provider / Провайдер"),
            types.BotCommand(command="logs", description="Logs / Логи генерации"),
            types.BotCommand(command="incognito", description="Incognito / Инкогнито"),
            types.BotCommand(command="factcheck", description="Fact-check / Факт-чекинг"),
            types.BotCommand(command="depth", description="Depth / Глубина"),
            types.BotCommand(command="refine", description="Refine Step / Финальная редактура"),
            types.BotCommand(command="cancel", description="Cancel / Отмена"),
        ])
        # RU-localized menu
        try:
            await DP.bot.set_my_commands([
                types.BotCommand(command="start", description="Начать"),
                types.BotCommand(command="generate", description="Сгенерировать"),
                types.BotCommand(command="settings", description="Настройки"),
                types.BotCommand(command="balance", description="Баланс"),
                types.BotCommand(command="pricing", description="Цены"),
                types.BotCommand(command="history", description="История"),
                types.BotCommand(command="info", description="Инфо"),
                types.BotCommand(command="lang", description="Язык интерфейса"),
                types.BotCommand(command="lang_generate", description="Язык генерации"),
                types.BotCommand(command="provider", description="Провайдер"),
                types.BotCommand(command="logs", description="Логи генерации"),
                types.BotCommand(command="incognito", description="Инкогнито"),
                types.BotCommand(command="factcheck", description="Факт-чекинг"),
                types.BotCommand(command="depth", description="Глубина"),
                types.BotCommand(command="refine", description="Финальная редактура"),
                types.BotCommand(command="cancel", description="Отмена"),
            ], language_code="ru")
        except Exception:
            pass
        try:
            await DP.bot.set_chat_menu_button(menu_button=types.MenuButtonCommands())
        except Exception:
            pass
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/ip")
async def ip_endpoint(request: Request):
    try:
        ip = request.client.host if request and request.client else ""
    except Exception:
        ip = ""
    return {"ip": ip or "unknown"}

@app.get("/allow-download")
async def allow_download(key: str, ip: str):
    # Limits can be tuned via env; safe defaults
    per_min = int(os.getenv("DL_PER_MIN", "5"))
    per_hour = int(os.getenv("DL_PER_HOUR", "60"))
    per_day = int(os.getenv("DL_PER_DAY", "300"))
    ok = await _rate_allow_download(key, ip, per_min=per_min, per_hour=per_hour, per_day=per_day)
    return {"allow": bool(ok)}


@app.get("/")
async def root():
    return {"ok": True}
@app.get("/logs")
async def list_logs(_: bool = Depends(require_admin)):
    items = []
    
    # Read only from DB
    if SessionLocal is not None:
        try:
            async with SessionLocal() as s:
                from sqlalchemy import select, outerjoin
                # join Job to fetch material type and status; join ResultDoc to fetch result id
                jn = outerjoin(JobLog, Job, Job.id == JobLog.job_id)
                jn2 = outerjoin(jn, ResultDoc, ResultDoc.job_id == Job.id)
                res = await s.execute(
                    select(JobLog, Job.type, Job.status, ResultDoc.id)
                    .select_from(jn2)
                    .order_by(JobLog.created_at.desc(), JobLog.id.desc())
                    .limit(200)
                )
                rows = res.all()
                # Build dict to strictly keep one item per JobLog.id
                items_by_id: dict[int, dict] = {}
                for jl, jtype, jstatus, resid in rows:
                    rid_val = int(getattr(jl, "id", 0) or 0)
                    if rid_val in items_by_id:
                        # Already captured this log; prefer to keep existing (or update result_id if missing)
                        if items_by_id[rid_val].get("result_id") is None and resid is not None:
                            try:
                                items_by_id[rid_val]["result_id"] = int(resid)
                            except Exception:
                                pass
                        continue
                    # Extract topic from stored content if available
                    topic = ""
                    try:
                        if getattr(jl, "content", None):
                            meta = _extract_meta_from_text((jl.content or "")[:2000])
                            topic = meta.get("topic", "")
                    except Exception:
                        pass
                    if not topic:
                        # Fallback to filename-based topic
                        stem = Path(getattr(jl, "path", "")).name
                        if stem.endswith(".md"):
                            stem = stem[:-3]
                        import re
                        stem = re.sub(r"_log(_\d{8}_\d{6})?$", "", stem)
                        topic = stem
                    # Material type: prefer Job.type, else infer from path (e.g., output/post/... -> post)
                    mtype = (jtype or "")
                    try:
                        if not mtype and getattr(jl, "path", None):
                            p = Path(jl.path)
                            mtype = p.parent.name or ""
                    except Exception:
                        pass
                    items_by_id[rid_val] = {
                        "id": jl.id,
                        "job_id": jl.job_id,
                        "kind": jl.kind,
                        "path": jl.path,
                        "created_at": str(jl.created_at),
                        "topic": topic,
                        "mtype": mtype,
                        "status": (jstatus or ""),
                        "result_id": (int(resid) if resid is not None else None),
                    }
                # Materialize into list preserving DB order by iterating rows again
                items = [items_by_id[int(getattr(jl, "id", 0) or 0)] for jl, *_ in rows if int(getattr(jl, "id", 0) or 0) in items_by_id]
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
                jtable = Job.__tablename__
                with eng.connect() as conn:
                    res = conn.execute(_sqtext(
                        f"SELECT jl.id, jl.job_id, jl.kind, jl.path, jl.content, jl.created_at, j.type as mtype "
                        f"FROM {table} jl LEFT JOIN {jtable} j ON j.id = jl.job_id "
                        f"ORDER BY jl.created_at DESC NULLS LAST, jl.id DESC LIMIT :lim"
                    ), {"lim": 200})
                    for r in res.fetchall():
                        rid, rjob, rkind, rpath, rcont, rts, rmtype = r
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
                        # Material type fallback from path
                        mtype = rmtype or ""
                        try:
                            if not mtype and rpath:
                                mtype = Path(rpath).parent.name or ""
                        except Exception:
                            pass
                        items.append({
                            "id": rid,
                            "job_id": rjob,
                            "kind": rkind,
                            "path": rpath,
                            "created_at": str(rts) if rts is not None else "",
                            "topic": topic,
                            "mtype": mtype,
                            "status": "",
                            "result_id": None,
                        })
                try:
                    eng.dispose()
                except Exception:
                    pass
        except Exception as e:
            print(f"[ERROR] Sync fallback read failed: {e}")
    return {"items": items}


@app.get("/logs/{log_id}")
async def get_log(log_id: int, _: bool = Depends(require_admin)):
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

async def require_admin(request: Request, creds: HTTPBasicCredentials = Depends(_basic)) -> bool:
    user = os.getenv("ADMIN_UI_USER", "admin")
    pwd = os.getenv("ADMIN_UI_PASSWORD", "")
    if not pwd:
        # Explicitly deny when password is not configured
        raise HTTPException(status_code=503, detail="ADMIN_UI_PASSWORD not configured")
    # Brute-force lock: IP-based attempt counter via Redis
    client_ip = None
    try:
        client_ip = request.client.host if request and request.client else None
    except Exception:
        client_ip = None
    lock_key = None
    r = None
    try:
        r = get_redis()
        if client_ip:
            lock_key = f"{kv_prefix()}:adminui:fail:{client_ip}"
            val = None
            try:
                val = await r.get(lock_key)  # type: ignore
            except Exception:
                val = None
            # If locked (value like 'LOCK'), block
            if val and (isinstance(val, (bytes, bytearray)) and val.decode("utf-8").startswith("LOCK")):
                raise HTTPException(status_code=429, detail="Too many attempts")
    except Exception:
        r = None

    if secrets.compare_digest(creds.username, user) and secrets.compare_digest(creds.password, pwd):
        # On success, clear fail counter
        try:
            if r and lock_key:
                await r.delete(lock_key)  # type: ignore
        except Exception:
            pass
        return True

    # On failure, increment counter
    try:
        if r and client_ip and lock_key:
            cnt = await r.incr(lock_key)  # type: ignore
            # first failure => set TTL 15 minutes
            await r.expire(lock_key, 15 * 60)  # type: ignore
            if int(cnt) >= 10:
                await r.set(lock_key, "LOCK", ex=15 * 60)  # type: ignore
                raise HTTPException(status_code=429, detail="Too many attempts")
    except Exception:
        pass

    raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Basic"})


@app.get("/logs-ui", response_class=HTMLResponse)
async def logs_ui(_: bool = Depends(require_admin)):
    # Reuse /logs data
    data = await list_logs()
    items = data.get("items", [])
    # Format timestamp as YYYY-MM-DD HH:MM for initial render
    for it in items:
        try:
            from datetime import datetime
            ts_str = it.get("created_at", "")
            if ts_str:
                if "T" in ts_str:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(ts_str)
                it["created_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    html_rows = []
    for it in items:
        topic = (it.get("topic") or "").replace("<", "&lt;").replace(">", "&gt;")
        status_txt = (it.get('status') or '')
        html_rows.append(
            f"<tr data-id='{it.get('id')}' data-kind='{it.get('kind','')}' data-topic='{topic.lower()}' data-type='{(it.get('mtype') or '').lower()}' data-status='{(it.get('status') or '').lower()}'>"
            f"<td><input type='checkbox' class='sel' value='{it.get('id')}'></td>"
            f"<td class='t-id'>{it.get('id')}</td>"
            f"<td class='t-topic'><a href='/logs-ui/{it.get('id')}'>{topic or '(no topic)'}</a></td>"
            f"<td class='t-created'>{it.get('created_at','')}</td>"
            f"<td class='t-kind'><span class='badge'>{it.get('kind','')}</span></td>"
            f"<td class='t-mtype'>{(it.get('mtype') or '')}</td>"
            f"<td class='t-status'>{status_txt}</td>"
            f"<td class='t-actions'><a class='btn-link' href='/logs/{it.get('id')}'>Raw</a>"
            f" <button class='delBtn' data-id='{it.get('id')}'>Delete</button></td>"
            f"</tr>"
        )
    html = (
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Logs</title>"
        "<style>"
        ":root{--bg:#0e0f12;--panel:#151821;--muted:#9aa4b2;--text:#e6e9ef;--brand:#6cf;--ok:#4caf50;--warn:#ff9800;--err:#f44336;--line:#242938}"
        "[data-theme='light']{--bg:#f7f9fc;--panel:#ffffff;--muted:#5f6b7a;--text:#0f172a;--brand:#0a84ff;--ok:#2e7d32;--warn:#ef6c00;--err:#c62828;--line:#e5e9f0}"
        "*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px}"
        ".topbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 16px}"
        ".topbar h1{font-size:20px;margin:0 16px 0 0}"
        ".spacer{flex:1}"
        "input[type=text],select{background:#0f121a;border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:8px;min-width:200px}"
        "button,.btn{background:#1b2230;border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:8px;cursor:pointer}"
        "button:hover,.btn:hover{border-color:#2f3a4f}a{color:var(--brand);text-decoration:none}a.btn-link{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#131824;color:var(--text)}"
        "[data-theme='light'] input[type=text],[data-theme='light'] select{background:#ffffff;border:1px solid #d0d7e2;color:var(--text)}"
        "[data-theme='light'] button,[data-theme='light'] .btn,[data-theme='light'] a.btn-link{background:#f7f9fc;border:1px solid #d0d7e2;color:var(--text)}"
        "table{border-collapse:separate;border-spacing:0;width:100%;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}"
        "th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}"
        "thead th{position:sticky;top:0;background:#0f1218;color:#cbd5e1;font-weight:600}"
        "[data-theme='light'] thead th{background:#f3f4f6;color:#0f172a}"
        "tbody tr:hover{background:#121722}"
        "[data-theme='light'] tbody tr:hover{background:#f5f7fb}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#1e2636;color:#cbd5e1;font-size:12px}"
        "[data-theme='light'] .badge{background:#eef2f6;color:#0f172a}"
        ".muted{color:var(--muted)}"
        ".actions{display:flex;gap:8px}"
        "footer{margin-top:18px;color:var(--muted)}"
        "</style></head><body>"
        "<div class='topbar'>"
        "<h1>Generation Logs</h1>"
        "<a class='btn' href='/results-ui'>Results</a>"
        "<div class='spacer'></div>"
        "<input id='q' type='text' placeholder='Search topic...'>"
        "<select id='k'><option value=''>All kinds</option><option>md</option><option>json</option><option>txt</option></select>"
        "<select id='type'><option value=''>All types</option><option value='post'>post</option><option value='post_series'>post_series</option><option value='article'>article</option><option value='summary'>summary</option></select>"
        "<select id='theme'><option value='dark' selected>Dark</option><option value='light'>Light</option></select>"
        "<button id='refresh'>Refresh</button>"
        "<button id='delSel'>Delete selected</button>"
        "</div>"
        f"<div class='muted'>Total: {len(items)}</div>"
        "<table id='tbl'><thead><tr><th><input id='selAll' type='checkbox'></th><th data-sort='id'>ID</th><th data-sort='topic'>Topic</th><th data-sort='created'>Created</th><th>Kind</th><th>Type</th><th>Status</th><th>Actions</th></tr></thead><tbody>"
        + ("".join(html_rows) or "<tr><td colspan='7' class='muted'>No logs yet</td></tr>")
        + "</tbody></table>"
        "<footer>Tip: Click column headers to sort. Use search and kind filter to narrow down.</footer>"
        "<script>"
        "const $$=(s,el=document)=>el.querySelector(s);const $$$=(s,el=document)=>[...el.querySelectorAll(s)];"
        "const q=$$('#q'),k=$$('#k'),selAll=$$('#selAll'),tbody=$$('#tbl tbody'),typeSel=$$('#type');"
        "function applyFilter(){const term=(q.value||'').toLowerCase();const kind=(k.value||'').toLowerCase();const tp=(typeSel.value||'').toLowerCase();for(const tr of $$$('tr',tbody)){const t=tr.getAttribute('data-topic')||'';const kd=(tr.getAttribute('data-kind')||'').toLowerCase();const ty=(tr.getAttribute('data-type')||'').toLowerCase();const ok=(t.includes(term)) && (!kind||kd===kind) && (!tp||ty===tp);tr.style.display=ok?'':'none';}}"
        "q.oninput=applyFilter;k.onchange=applyFilter;typeSel.onchange=applyFilter;"
        "selAll.onchange=()=>{$$$('input.sel',tbody).forEach(x=>{if(x.closest('tr').style.display!=='none')x.checked=selAll.checked;});};"
        "document.addEventListener('click',async(e)=>{if(e.target.matches('.delBtn')){const id=e.target.getAttribute('data-id');if(confirm('Delete log '+id+'?')){const r=await fetch('/logs/'+id,{method:'DELETE'});const j=await r.json();if(j&&j.ok){location.reload();}}}});"
        "$$('#delSel').onclick=async()=>{const ids=$$$('input.sel:checked',tbody).map(x=>parseInt(x.value));if(!ids.length)return;if(!confirm('Delete '+ids.length+' logs?'))return;const r=await fetch('/logs/purge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids})});const j=await r.json();if(j&&j.ok){location.reload();}};"
        "let asc=true;$$$('th[data-sort]').forEach(th=>{th.style.cursor='pointer';th.onclick=()=>{const key=th.getAttribute('data-sort');const rows=$$$('tr',tbody);rows.sort((a,b)=>{const A=(a.querySelector('.t-'+key)?.textContent||'').trim();const B=(b.querySelector('.t-'+key)?.textContent||'').trim();if(key==='id')return (asc?1:-1)*(parseInt(A)-parseInt(B));return (asc?1:-1)*(A.localeCompare(B));});asc=!asc;rows.forEach(r=>tbody.appendChild(r));};});"
        "function applyTheme(){const v=$$('#theme').value;if(v==='light'){document.documentElement.setAttribute('data-theme','light');localStorage.setItem('ui_theme','light');}else{document.documentElement.removeAttribute('data-theme');localStorage.setItem('ui_theme','dark');}}$$('#theme').onchange=applyTheme;"
        "(function(){const t=localStorage.getItem('ui_theme');if(t==='light'){$$('#theme').value='light';document.documentElement.setAttribute('data-theme','light');}else{document.documentElement.removeAttribute('data-theme');}})();"
        "$$('#refresh').onclick=()=>location.reload();"
        "</script>"
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
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Log View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>"
        ":root{--bg:#0e0f12;--panel:#151821;--muted:#9aa4b2;--text:#e6e9ef;--brand:#6cf;--line:#242938}"
        "[data-theme='light']{--bg:#f7f9fc;--panel:#ffffff;--muted:#5f6b7a;--text:#0f172a;--brand:#0a84ff;--line:#e5e9f0}"
        "*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;}"
        "header{background:#0f1218;border-bottom:1px solid var(--line);color:#e6e9ef;padding:10px 14px;display:flex;gap:12px;align-items:center}"
        "[data-theme='light'] header{background:#ffffff;color:#0f172a;border-bottom:1px solid #e5e9f0}"
        "header a{color:var(--brand)}.spacer{flex:1}.toolbar button{background:#1b2230;border:1px solid var(--line);color:var(--text);padding:6px 10px;border-radius:8px;cursor:pointer;margin-left:8px}"
        "[data-theme='light'] .toolbar button{background:#eef2f6;border:1px solid #e5e9f0;color:#0f172a}"
        "main{padding:14px}#content{max-width:1000px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}a{color:var(--brand)}"
        "h1{font-size:18px;margin:0;font-weight:700}.meta{opacity:.8;font-size:12px}"
        "code{background:#0f121a;color:inherit;padding:0 2px;font-size:12px}pre{background:#0f121a;color:inherit;padding:12px;border-radius:8px;white-space:pre-wrap;word-wrap:break-word;font-size:12px}"
        "[data-theme='light'] .toolbar button{background:#eef2f6;border:1px solid #e5e9f0;color:#0f172a}"
        "[data-theme='light'] code{background:#f5f7fb;color:inherit}"
        "[data-theme='light'] pre{background:#f5f7fb;border:1px solid #e5e9f0;color:#0f172a}"
        "</style></head><body>"
        f"<header><a href='/logs-ui'>← Logs</a><a href='/results-ui'>Results</a><div class='spacer'></div><div class='toolbar'><button id='copy'>Copy</button><button id='download'>Download .md</button><button id='toggleRaw'>Show raw</button></div></header>"
        f"<main><div class='meta' id='meta'>Log: {title}</div><div id='content'>Loading…</div><textarea id='raw' style='display:none'>{_raw}</textarea></main>"
        f"<script>const LOG_ID={log_id};"
        "const TAGS=['input','topic','lang','post','critique_json'];const NL='\\n';"
        "(function(){const t=localStorage.getItem('ui_theme');if(t==='light'){document.documentElement.setAttribute('data-theme','light');}})();"
        "function escapeOutsideCode(md){const lines=md.split(NL);let inCode=false;for(let i=0;i<lines.length;i++){const t=lines[i].trim();if(t.startsWith('```')){inCode=!inCode;continue;}if(!inCode){let s=lines[i];for(const tag of TAGS){s=s.replace(new RegExp('<'+tag+'>','g'),'&lt;'+tag+'&gt;').replace(new RegExp('</'+tag+'>','g'),'&lt;/'+tag+'&gt;');}lines[i]=s;}}return lines.join(NL);}"
        "function extractMeta(md){const out={};const lines=md.split(NL);for(let i=0;i<Math.min(lines.length,200);i++){const line=lines[i].trim();if(line.startsWith('- provider:')){out.provider=line.split(':').slice(1).join(':').trim();}else if(line.startsWith('- lang:')){out.lang=line.split(':').slice(1).join(':').trim();}else if(line.startsWith('- topic:')){out.topic=line.split(':').slice(1).join(':').trim();}}return out;}"
        "let text=(document.getElementById('raw')?document.getElementById('raw').value:'');let showRaw=false;"
        "function render(md){try{const raw=String(md||'');const fencedRaw=raw.replace(/<input>[\\s\\S]*?<\\/input>/g,m=>'```'+NL+m+NL+'```');const escaped=escapeOutsideCode(fencedRaw);let html='';try{html=(window.marked?window.marked.parse(escaped):'');}catch(_){html='';}if(!html||html.trim()===''){const safe=escaped.replace(/</g,'&lt;').replace(/>/g,'&gt;');html='<pre>'+safe+'</pre>'; }try{const m=extractMeta(raw||'');const metaEl=document.getElementById('meta');if(metaEl){metaEl.textContent=`provider=${m.provider||'?'} | lang=${m.lang||'?'} | topic=${m.topic||'?'} | id=${LOG_ID}`;}}catch(_){}document.getElementById('content').innerHTML = showRaw?('<pre>'+raw.replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre>'):html;}catch(e){const safe=String(md||'').replace(/</g,'&lt;').replace(/>/g,'&gt;');document.getElementById('content').innerHTML='<pre>'+safe+'</pre>';}}"
        "if(text && text.trim().length>0){render(text);}else{document.getElementById('content').textContent='Loading…';}"
        "async function refresh(){try{const r=await fetch('/logs/'+LOG_ID+'?_ts='+(Date.now()),{cache:'no-store'});const j=await r.json();if(j&&j.content&&(j.content.length>0)){text=j.content;render(text);}else{document.getElementById('content').textContent='No content';}}catch(e){document.getElementById('content').textContent='Failed to load';}}refresh();"
        "document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(text);alert('Copied');}catch(_){}};"
        "document.getElementById('download').onclick=async()=>{try{const ip=(await (await fetch('/ip',{cache:'no-store'})).json()).ip||'unknown';const key='log-'+LOG_ID;const ok=await (await fetch(`/allow-download?key=${key}&ip=${encodeURIComponent(ip)}`,{cache:'no-store'})).json();if(!(ok&&ok.allow)){alert('Rate limit exceeded. Try later.');return;}const a=document.createElement('a');const blob=new Blob([text],{type:'text/markdown'});a.href=URL.createObjectURL(blob);a.download=(document.title||'log')+'.md';a.click();}catch(_){}};"
        "document.getElementById('toggleRaw').onclick=()=>{showRaw=!showRaw;document.getElementById('toggleRaw').textContent=showRaw?'Show rendered':'Show raw';render(text);}" 
        "</script>"
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
        # Also delete results linked by the same Job (ResultDoc.job_id == JobLog.job_id)
        try:
            from sqlalchemy import delete as _sqdel
            job_id_val = int(getattr(obj, "job_id", 0) or 0)
            if job_id_val:
                await s.execute(_sqdel(ResultDoc).where(ResultDoc.job_id == job_id_val))
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
            # Map JobLog ids -> Job ids, then delete ResultDoc by those Job ids
            try:
                res = await s.execute(select(JobLog.job_id).where(JobLog.id.in_(ids)))
                job_ids = [int(x[0]) for x in res.fetchall() if x and x[0]]
            except Exception:
                job_ids = []
            if job_ids:
                await s.execute(delete(ResultDoc).where(ResultDoc.job_id.in_(job_ids)))
            res2 = await s.execute(delete(JobLog).where(JobLog.id.in_(ids)))
            await s.commit()
            deleted = int(res2.rowcount or 0)
    return {"ok": True, "deleted": int(deleted)}


# ------------------- Results (final outputs on filesystem) -------------------

async def _list_result_files() -> list[dict]:
    # DB-only results (no filesystem fallback)
    items: list[dict] = []
    if SessionLocal is None:
        return items
    try:
        async with SessionLocal() as s:
            from sqlalchemy import select
            from sqlalchemy import or_
            res = await s.execute(
                select(ResultDoc)
                .where(or_(ResultDoc.hidden == 0, ResultDoc.hidden.is_(None)))
                .order_by(ResultDoc.created_at.desc(), ResultDoc.id.desc())
            )
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
    except Exception as e:
        print(f"[ERROR] results db read failed: {e}")
        items = []
    return items


@app.get("/results")
async def list_results():
    return {"items": await _list_result_files()}


@app.get("/results-ui", response_class=HTMLResponse)
async def results_ui():
    items = await _list_result_files()
    # Normalize timestamps to 'YYYY-MM-DD HH:MM:SS'
    for it in items:
        try:
            ts = it.get("created_at")
            if ts:
                from datetime import datetime as _dt
                if "T" in ts:
                    dt = _dt.fromisoformat(ts.replace("Z","+00:00"))
                else:
                    dt = _dt.fromisoformat(ts)
                it["created_at"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    rows = []
    for it in items:
        if "id" in it and isinstance(it.get("id"), int):
            link = f"/results-ui/id/{it['id']}"
            topic = (it.get('topic','') or '').replace('<','&lt;').replace('>','&gt;')
            rows.append(
                f"<tr data-id='{it.get('id')}' data-topic='{topic.lower()}' data-lang='{(it.get('lang','') or '').lower()}' data-kind='{(it.get('kind','') or '').lower()}'>"
                f"<td class='t-topic'><a href='{link}'>{topic or '(no topic)'}</a></td>"
                f"<td class='t-created'>{it.get('created_at','')}</td><td class='t-kind'>{it.get('kind','')}</td>"
                f"<td class='t-lang'>{it.get('lang','')}</td>"
                f"</tr>"
            )
    html = (
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Results</title>"
        "<style>"
        ":root{--bg:#0e0f12;--panel:#151821;--muted:#9aa4b2;--text:#e6e9ef;--brand:#6cf;--line:#242938}"
        "[data-theme='light']{--bg:#f7f9fc;--panel:#ffffff;--muted:#5f6b7a;--text:#0f172a;--brand:#0a84ff;--line:#e5e9f0}"
        "*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:24px}"
        ".topbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 16px}"
        ".topbar h1{font-size:20px;margin:0 16px 0 0}a{color:var(--brand)}.spacer{flex:1}"
        "input[type=text],select{background:#0f121a;border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:8px;min-width:200px}"
        "button,.btn{background:#1b2230;border:1px solid var(--line);color:var(--text);padding:8px 10px;border-radius:8px;cursor:pointer}"
        "button:hover,.btn:hover{border-color:#2f3a4f}a.btn-link{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#131824;color:var(--text)}"
        "[data-theme='light'] input[type=text],[data-theme='light'] select{background:#ffffff;border:1px solid #d0d7e2;color:var(--text)}"
        "[data-theme='light'] button,[data-theme='light'] .btn,[data-theme='light'] a.btn-link{background:#f7f9fc;border:1px solid #d0d7e2;color:var(--text)}"
        "table{border-collapse:separate;border-spacing:0;width:100%;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}"
        "th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}"
        "thead th{position:sticky;top:0;background:#0f1218;color:#cbd5e1;font-weight:600}tbody tr:hover{background:#121722}"
        "[data-theme='light'] thead th{background:#f3f4f6;color:#0f172a}"
        "[data-theme='light'] tbody tr:hover{background:#f5f7fb}"
        "footer{margin-top:18px;color:var(--muted)}"
        "</style></head><body>"
        "<div class='topbar'>"
        "<h1>Generated Results</h1>"
        "<div class='spacer'></div>"
        "<input id='q' type='text' placeholder='Search topic...'>"
        "<select id='lang'><option value=''>All languages</option><option>ru</option><option>en</option></select>"
        "<select id='kind'><option value=''>All types</option><option>post</option><option>post_series</option><option>article</option><option>summary</option></select>"
        "<select id='theme'><option value='dark' selected>Dark</option><option value='light'>Light</option></select>"
        "<button id='refresh'>Refresh</button>"
        "</div>"
        f"<div class='muted'>Total: {len(items)}</div>"
        "<table id='tbl'><thead><tr><th data-sort='topic'>Topic</th><th data-sort='created'>Created</th><th>Type</th><th data-sort='lang'>Lang</th></tr></thead><tbody>"
        + ("".join(rows) or "<tr><td colspan='4' class='muted'>No results yet</td></tr>")
        + "</tbody></table>"
        "<footer>Tip: Filter by lang and search by topic. Click headers to sort. Use the theme switcher for light/dark.</footer>"
        "<script>"
        "const $$=(s,el=document)=>el.querySelector(s);const $$$=(s,el=document)=>[...el.querySelectorAll(s)];"
        "const q=$$('#q'),lang=$$('#lang'),tbody=$$('#tbl tbody'),themeSel=$$('#theme'),kindSel=$$('#kind');"
        "function applyFilter(){const term=(q.value||'').toLowerCase();const l=(lang.value||'').toLowerCase();const k=(kindSel.value||'').toLowerCase();for(const tr of $$$('tr',tbody)){const tt=(tr.getAttribute('data-topic')||'');const tl=(tr.getAttribute('data-lang')||'');const tk=(tr.getAttribute('data-kind')||'');const ok=tt.includes(term)&&(!l||tl===l)&&(!k||tk===k);tr.style.display=ok?'':'none';}}"
        "q.oninput=applyFilter;lang.onchange=applyFilter;kindSel.onchange=applyFilter;"
        "let asc=true;$$$('th[data-sort]').forEach(th=>{th.style.cursor='pointer';th.onclick=()=>{const key=th.getAttribute('data-sort');const rows=$$$('tr',tbody);rows.sort((a,b)=>{const A=(a.querySelector('.t-'+key)?.textContent||'').trim().toLowerCase();const B=(b.querySelector('.t-'+key)?.textContent||'').trim().toLowerCase();return (asc?1:-1)*A.localeCompare(B);});asc=!asc;rows.forEach(r=>tbody.appendChild(r));};});"
        "function applyTheme(){const v=themeSel.value;document.documentElement.setAttribute('data-theme',v);localStorage.setItem('ui_theme',v);}themeSel.onchange=applyTheme;"
        "(function(){const t=localStorage.getItem('ui_theme');if(t){themeSel.value=t;document.documentElement.setAttribute('data-theme',t);}else{document.documentElement.setAttribute('data-theme','dark');}})();"
        "$$('#refresh').onclick=()=>location.reload();"
        "</script>"
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
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Result View</title>"
        "<script src='https://cdn.jsdelivr.net/npm/marked/marked.min.js'></script>"
        "<style>"
        ":root{--bg:#0e0f12;--panel:#151821;--muted:#9aa4b2;--text:#e6e9ef;--brand:#6cf;--line:#242938}"
        "[data-theme='light']{--bg:#f7f9fc;--panel:#ffffff;--muted:#5f6b7a;--text:#0f172a;--brand:#0a84ff;--line:#e5e9f0}"
        "*{box-sizing:border-box}body{background:var(--bg);color:var(--text);font-family:Inter,system-ui,Segoe UI,Helvetica,Arial,sans-serif;margin:0;}"
        "header{background:#0f1218;border-bottom:1px solid var(--line);color:#e6e9ef;padding:10px 14px;display:flex;gap:12px;align-items:center}"
        "[data-theme='light'] header{background:#ffffff;color:#0f172a;border-bottom:1px solid #e5e9f0}"
        "header a{color:var(--brand)}.spacer{flex:1}.toolbar button{background:#1b2230;border:1px solid var(--line);color:var(--text);padding:6px 10px;border-radius:8px;cursor:pointer;margin-left:8px}"
        "[data-theme='light'] .toolbar button{background:#eef2f6;border:1px solid #e5e9f0;color:#0f172a}"
        "main{padding:14px}#content{max-width:1000px;margin:0 auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}a{color:var(--brand)}"
        ".meta{opacity:.8;font-size:12px;margin:0 0 8px}code{background:#0f121a;color:inherit;padding:0 2px;font-size:12px}pre{white-space:pre-wrap;word-wrap:break-word;background:#0f121a;border-radius:8px;padding:12px;border:1px solid var(--line)}"
        "[data-theme='light'] code{background:#f5f7fb;color:inherit}"
        "[data-theme='light'] pre{background:#f5f7fb;border:1px solid #e5e9f0;color:#0f172a}"
        "</style></head><body>"
        f"<header><a href='/results-ui'>← Results</a><div class='spacer'></div><div class='toolbar'><button id='copy'>Copy</button><button id='download'>Download .md</button><button id='toggleRaw'>Show raw</button></div></header>"
        f"<main><div class='meta' id='meta'>Result: {title}</div><div id='content'></div><textarea id='raw' style='display:none'>{_raw}</textarea></main>"
        "<script>(function(){const t=localStorage.getItem('ui_theme');if(t==='light'){document.documentElement.setAttribute('data-theme','light');}else{document.documentElement.removeAttribute('data-theme');}})();"
        f"const RES_ID={res_id};"
        "let text=(document.getElementById('raw')?document.getElementById('raw').value:'');let showRaw=false;"
        "function render(md){let html='';try{html=(window.marked?window.marked.parse(md):'');}catch(e){html='';}if(!html||html.trim()===''){const safe=md.replace(/</g,'&lt;').replace(/>/g,'&gt;');html='<pre>'+safe+'</pre>';}document.getElementById('content').innerHTML=showRaw?('<pre>'+md.replace(/</g,'&lt;').replace(/>/g,'&gt;')+'</pre>'):html;}"
        "render(text);async function refresh(){try{const r=await fetch('/results/'+RES_ID,{cache:'no-store'});const j=await r.json();if(j&&(j.content||j.path)){if(j.content)text=j.content;const meta=document.getElementById('meta');if(meta){meta.textContent=`provider=${j.provider||'?'} | lang=${j.lang||'?'} | topic=${j.topic||'?'} | id=${RES_ID}`;}render(text);}}catch(_){}}refresh();"
        "document.getElementById('copy').onclick=async()=>{try{await navigator.clipboard.writeText(text);alert('Copied');}catch(_){}};"
        "document.getElementById('download').onclick=async()=>{try{const ip=(await (await fetch('/ip',{cache:'no-store'})).json()).ip||'unknown';const key='res-'+RES_ID;const ok=await (await fetch(`/allow-download?key=${key}&ip=${encodeURIComponent(ip)}`,{cache:'no-store'})).json();if(!(ok&&ok.allow)){alert('Rate limit exceeded. Try later.');return;}const a=document.createElement('a');const blob=new Blob([text],{type:'text/markdown'});a.href=URL.createObjectURL(blob);a.download=(document.title||'result')+'.md';a.click();}catch(_){}};"
        "document.getElementById('toggleRaw').onclick=()=>{showRaw=!showRaw;document.getElementById('toggleRaw').textContent=showRaw?'Show rendered':'Show raw';render(text);}"
        "</script>"
        "</body></html>"
    )
    return HTMLResponse(content=html)




@app.get("/debug/results-summary")
async def debug_results_summary(_: bool = Depends(require_admin)):
    out = {"total": 0, "visible": 0, "hidden": 0, "recent": []}
    if SessionLocal is None:
        return out
    try:
        async with SessionLocal() as s:
            from sqlalchemy import select, func
            total = await s.execute(select(func.count()).select_from(ResultDoc))
            vis = await s.execute(select(func.count()).select_from(ResultDoc).where(ResultDoc.hidden == 0))
            hid = await s.execute(select(func.count()).select_from(ResultDoc).where(ResultDoc.hidden == 1))
            out["total"] = int(total.scalar() or 0)
            out["visible"] = int(vis.scalar() or 0)
            out["hidden"] = int(hid.scalar() or 0)
            res = await s.execute(select(ResultDoc).order_by(ResultDoc.created_at.desc(), ResultDoc.id.desc()).limit(5))
            rows = res.scalars().all()
            out["recent"] = [
                {"id": r.id, "hidden": int(getattr(r, "hidden", 0) or 0), "created_at": str(r.created_at), "path": r.path}
                for r in rows
            ]
    except Exception as e:
        out["error"] = str(e)
    return out


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




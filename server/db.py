#!/usr/bin/env python3
from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String,
    Integer,
    DateTime,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from urllib.parse import urlsplit, urlunsplit
import ssl
import certifi


DB_URL = os.getenv("DB_URL", "")
DB_TABLE_PREFIX = os.getenv("DB_TABLE_PREFIX", "bio1c_")


class Base(DeclarativeBase):
    pass


def _t(name: str) -> str:
    return f"{DB_TABLE_PREFIX}{name}"


class User(Base):
    __tablename__ = _t("users")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    credits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # Optional simple rate limiting fields
    # counters are soft and can be reset daily/hourly by app logic
    # Keeping here for future use; not critical for current release
    # daily_count: Mapped[int] = mapped_column(Integer, default=0)
    # daily_reset_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Job(Base):
    __tablename__ = _t("jobs")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    type: Mapped[str] = mapped_column(String(32))  # e.g. 'post'
    status: Mapped[str] = mapped_column(String(32), default="queued")  # queued|running|done|error
    params_json: Mapped[str] = mapped_column(Text)
    cost: Mapped[int] = mapped_column(Integer, default=1)
    file_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class JobLog(Base):
    __tablename__ = _t("job_logs")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(Integer, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="md")  # md|txt|json
    path: Mapped[str] = mapped_column(String(512))
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Store log content directly
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ResultDoc(Base):
    __tablename__ = _t("results")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[int] = mapped_column(Integer, default=0, index=True)
    kind: Mapped[str] = mapped_column(String(32), default="post")  # post|article|summary
    path: Mapped[str] = mapped_column(String(512))
    topic: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    provider: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    hidden: Mapped[int] = mapped_column(Integer, default=0)  # 0=visible, 1=incognito
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Tx(Base):
    __tablename__ = _t("tx")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    delta: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


def _sanitize_db_url(raw: str) -> tuple[str, dict]:
    if not raw:
        return raw, {}
    parts = urlsplit(raw)
    # Ensure asyncpg dialect in scheme
    scheme = parts.scheme or "postgresql+asyncpg"
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+asyncpg"
    # Drop ALL query params from URL to avoid passing sslmode via DSN
    base_url = urlunsplit((scheme, parts.netloc, parts.path, "", parts.fragment))
    # Provide required options via connect_args instead of DSN
    # Strategy:
    # 1) If DB_SSL_CA provided (PEM text or path) — use it
    # 2) Else use certifi bundle
    # 3) If DB_SSL_SKIP_VERIFY=true — disable verification (temporary fallback)
    ssl_ctx = ssl.create_default_context()
    db_ssl_ca = os.getenv("DB_SSL_CA", "").strip()
    if db_ssl_ca:
        try:
            if db_ssl_ca.startswith("-----BEGIN"):
                ssl_ctx.load_verify_locations(cadata=db_ssl_ca)
            else:
                ssl_ctx.load_verify_locations(db_ssl_ca)
        except Exception:
            # Fall back to certifi if provided CA fails to load
            ssl_ctx.load_verify_locations(cafile=certifi.where())
    else:
        ssl_ctx.load_verify_locations(cafile=certifi.where())

    if (os.getenv("DB_SSL_SKIP_VERIFY", "").lower() in ("1", "true", "yes")):
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
    else:
        ssl_ctx.check_hostname = True
        ssl_ctx.verify_mode = ssl.CERT_REQUIRED
    cargs = {"ssl": ssl_ctx, "statement_cache_size": 0}
    return base_url, cargs


engine = None
if DB_URL:
    sanitized, cargs = _sanitize_db_url(DB_URL)
    # For pgBouncer (transaction pooler) + asyncpg: disable statement cache to avoid PREPARE issues
    # Increase pool for concurrent users: pool_size=20 (base), max_overflow=20 (burst) = 40 total
    engine = create_async_engine(
        sanitized,
        connect_args=cargs,
        pool_pre_ping=True,
        pool_size=20,
        max_overflow=20,
        pool_timeout=30,
    )
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None


async def init_db() -> None:
    if not engine:
        return
    async with engine.begin() as conn:
        # Create tables if they don't exist
        await conn.run_sync(Base.metadata.create_all)
        
        # Add content column to existing job_logs table if it doesn't exist
        try:
            await conn.execute(text(f"ALTER TABLE {_t('job_logs')} ADD COLUMN content TEXT"))
            print("[INFO] Added content column to job_logs table")
        except Exception:
            # Column already exists or other error - that's fine
            pass
        # Add created_at column to job_logs if missing
        try:
            await conn.execute(text(f"ALTER TABLE {_t('job_logs')} ADD COLUMN created_at TIMESTAMP DEFAULT NOW()"))
            print("[INFO] Added created_at column to job_logs table")
        except Exception:
            pass
        # Ensure results table exists (for final documents)
        try:
            await conn.run_sync(ResultDoc.__table__.create, checkfirst=True)
        except Exception:
            pass
        # Add hidden flag to results if missing
        try:
            await conn.execute(text(f"ALTER TABLE {_t('results')} ADD COLUMN hidden INTEGER DEFAULT 0"))
            print("[INFO] Added hidden column to results table")
        except Exception:
            pass
        # Normalize existing NULLs to 0 to make UI queries work
        try:
            await conn.execute(text(f"UPDATE {_t('results')} SET hidden=0 WHERE hidden IS NULL"))
        except Exception:
            pass
        # Migrate old table result_docs -> results (best-effort)
        try:
            await conn.execute(text(
                f"INSERT INTO {_t('results')} (job_id, kind, path, topic, provider, lang, content, created_at) "
                f"SELECT job_id, kind, path, topic, provider, lang, content, created_at FROM {_t('result_docs')} rd "
                f"WHERE NOT EXISTS (SELECT 1 FROM {_t('results')} r WHERE r.path = rd.path AND r.created_at = rd.created_at)"
            ))
            print("[INFO] Migrated rows from legacy result_docs to results (if existed)")
        except Exception:
            # Old table may not exist; ignore
            pass


async def get_or_create_user(session: AsyncSession, telegram_id: int) -> User:
    from sqlalchemy import select
    res = await session.execute(select(User).where(User.telegram_id == telegram_id))
    user = res.scalar_one_or_none()
    if user is None:
        user = User(telegram_id=telegram_id, credits=0)
        session.add(user)
        await session.flush()
    return user




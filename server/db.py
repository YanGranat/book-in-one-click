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
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Tx(Base):
    __tablename__ = _t("tx")
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    delta: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


engine = None
if DB_URL:
    # For pgBouncer (transaction pooler) + asyncpg: disable statement cache to avoid PREPARE issues
    engine = create_async_engine(
        DB_URL,
        connect_args={
            "statement_cache_size": 0,
        },
        pool_pre_ping=True,
    )
SessionLocal = async_sessionmaker(engine, expire_on_commit=False) if engine else None


async def init_db() -> None:
    if not engine:
        return
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_or_create_user(session: AsyncSession, telegram_id: int) -> User:
    from sqlalchemy import select
    res = await session.execute(select(User).where(User.telegram_id == telegram_id))
    user = res.scalar_one_or_none()
    if user is None:
        user = User(telegram_id=telegram_id, credits=0)
        session.add(user)
        await session.flush()
    return user




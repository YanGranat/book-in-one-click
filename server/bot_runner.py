#!/usr/bin/env python3
from __future__ import annotations

import asyncio

from aiogram import executor

from .bot import create_dispatcher
from .db import SessionLocal
from .bot_commands import register_admin_commands, ensure_db_ready


def main():
    dp = create_dispatcher()
    register_admin_commands(dp, SessionLocal)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(ensure_db_ready())

    executor.start_polling(dp, skip_updates=True)


if __name__ == "__main__":
    main()



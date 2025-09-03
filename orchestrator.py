#!/usr/bin/env python3
from datetime import datetime


def progress(stage: str) -> None:
    """Pretty progress output for console."""
    labels = {
        "start: post": "🚀 Старт генерации поста",
        "agent:init": "🤖 Инициализация агента",
        "agent:run": "🧠 Генерация контента",
        "io:prepare_output": "📁 Подготовка директории вывода",
        "factcheck:init": "🔎 Факт‑чекинг",
        "rewrite:init": "✍️ Переписывание по замечаниям",
        "factcheck:second_pass": "🔁 Повторный факт‑чекинг",
        "io:save_final": "💾 Сохранение результата",
        "done": "✅ Готово",
    }
    message = labels.get(stage, stage.replace(":", " → "))
    print(message)

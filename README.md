# 📚 Book in One Click

Мультиагентная система для генерации образовательного контента (книг, статей, постов, конспектов) с использованием OpenAI Agents SDK.

## 🧩 Суть проекта

- **Что это**: мультиагентная система, превращающая любую тему в качественный образовательный материал.
- **Принципы**: качество важнее скорости; постепенное погружение от основ к продвинутому; академическая точность при доступном изложении.
- **Результаты**: пост, конспект, статья, книга — в Markdown, пригодно для публикации и дальнейшей обработки.
- **Процесс**: пользователь вводит тему → система исследует, планирует структуру, пишет, проверяет и дорабатывает → сохраняет результат и подробный лог.
- **Текущее состояние**: работают сервер (FastAPI) и Telegram‑бот с БД и UI логов/результатов; стабильно генерируются посты; статья и книга — в активной разработке.
- **Вектор развития**: «от идеи до книги за один клик» на основе масштабируемой архитектуры агентов.

## 🚀 Быстрый старт

### 1. Подготовка окружения
```bash
# Клонировать репозиторий
git clone <your-repo-url>
cd Book_in_one_click

# Создать виртуальное окружение
python -m venv venv

# Активировать окружение
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
```

### 2. Настроить переменные окружения (.env)
Создайте файл `.env` в корне проекта (минимальный пример ниже). Провайдеры настраиваются через переменные, сервер/бот/БД — опционально:
```
# LLM providers (минимум один)
OPENAI_API_KEY=...
# Опционально переопределить модели
OPENAI_MODEL=gpt-5
OPENAI_FAST_MODEL=gpt-5-mini

# Альтернативные провайдеры
GOOGLE_API_KEY=...        # или GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
GEMINI_FAST_MODEL=gemini-2.5-flash

ANTHROPIC_API_KEY=...
ANTHROPIC_MODEL=claude-3-5-sonnet-latest

# Server / UI / DB / Redis / Telegram (опционально, для веб-части и бота)
ADMIN_UI_USER=admin
ADMIN_UI_PASSWORD=change_me
DB_URL=postgresql+asyncpg://USER:PASSWORD@HOST/DBNAME
DB_TABLE_PREFIX=bio1c_
REDIS_URL=redis://:PASSWORD@HOST:6379/0

TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=some_secret
BOT_ADMIN_IDS=123456789,987654321
```

### 3. Запустить простой тест
```bash
python scripts/simple_test.py
```

### 4. Запуск сервера и Telegram‑бота
- Локально (Windows CMD):
```bat
venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```
- Здоровье: `GET /health`
- UI логов: `GET /logs-ui` (HTTP Basic, переменные `ADMIN_UI_USER`/`ADMIN_UI_PASSWORD` обязательны)
- UI результатов: `GET /results-ui`
- Подключение вебхука бота:
  - URL: `/webhook/{TELEGRAM_WEBHOOK_SECRET}`
  - Пример: откройте в браузере `https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://<ваш-домен>/webhook/<секрет>`

### 4. Запуск генераторов (Windows CMD)
```bat
venv\Scripts\python.exe scripts\popular_science_post.py
venv\Scripts\python.exe scripts\deep_popular_science_article.py
venv\Scripts\python.exe scripts\deep_popular_science_book.py
venv\Scripts\python.exe scripts\topic_summary.py
venv\Scripts\python.exe scripts\simple_test.py
```

Введите любую тему — получите сгенерированную образовательную страницу в Markdown.

## 📦 Варианты контента (пайплайны)

- **Пост (post)**: короткий научно‑популярный пост (≈250–800 слов)
- **Конспект (summary)**: структурированный краткий конспект (≈3 000–8 000 слов)
- **Статья (article)**: глубокая популярная статья (≈15 000–45 000 слов)
- **Книга (book)**: заготовка/книга (цель: 60 000–180 000 слов; пока черновик)

Сейчас стабильно поддерживаются: пост и конспект. Статья/книга — в активной разработке.

Промпты для сценария поста находятся в `prompts/post/{module_01_writing,module_02_review,module_03_rewriting}/*.md`.

## 🎛️ Параметры CLI

Большинство скриптов поддерживает параметры:

- `--topic` — тема генерации (строка)
- `--lang` — язык вывода: `auto|ru|en` (по умолчанию `auto`)
- `--provider` — провайдер LLM: `openai|gemini|claude` (по умолчанию `openai`)
- `--out` — подпапка для сохранения результата (по умолчанию зависит от пайплайна)
- `--no-factcheck` — выключить факт‑чекинг (для постов)
- `--factcheck-max-items` — ограничить число проверяемых пунктов (0 = без ограничения)
- `--research-iterations` — число итераций ресёрча на пункт (по умолчанию 2)
- `--research-concurrency` — параллелизм ресёрча (по умолчанию 3)
- `--include-logs` — сохранить рядом Markdown‑лог генерации

Примеры (Windows CMD):
```bat
venv\Scripts\python.exe scripts\popular_science_post.py --topic "CRISPR" --lang en
venv\Scripts\python.exe scripts\popular_science_post.py --topic "pH" --lang ru --provider claude --include-logs
venv\Scripts\python.exe scripts\deep_popular_science_article.py --topic "Квантовые точки" --lang ru
venv\Scripts\python.exe scripts\topic_summary.py --topic "Graph Neural Networks" --lang auto
```

## 🌐 Автоопределение языка (simple_test)

`scripts/simple_test.py` автоматически определяет язык по теме:
- Кириллица → `ru`
- Иначе → `en`

Инструкция агенту включает префикс вида `Язык вывода: ru|en`. В сообщение пользователя передаётся только тема.

## ⚙️ Конфигурация

Базовая конфигурация — `config/defaults.json`:
- `generator_label`, `format`, `metadata`
- `pipelines.post|article|book|summary`: модель, лимиты по объёму, выходная папка, стиль

Пример фрагмента:
```json
{
  "pipelines": {
    "post": { "word_count_min": 250, "word_count_max": 800 },
    "article": { "word_count_min": 15000, "word_count_max": 45000 },
    "book": { "word_count_min": 60000, "word_count_max": 180000 },
    "summary": { "word_count_min": 3000, "word_count_max": 8000 }
  }
}
```

Провайдеры и ключи:
- **OpenAI**: `OPENAI_API_KEY`, модели через `OPENAI_MODEL`/`OPENAI_FAST_MODEL`
- **Gemini**: `GOOGLE_API_KEY` (или `GEMINI_API_KEY`), модели через `GEMINI_MODEL`/`GEMINI_FAST_MODEL`
- **Claude**: `ANTHROPIC_API_KEY`, модель через `ANTHROPIC_MODEL`

Сервер и хранилища:
- **Postgres**: `DB_URL` (asyncpg), `DB_TABLE_PREFIX` (по умолчанию `bio1c_`)
- **Redis**: `REDIS_URL` (для пользовательских настроек и кредитов)
- **Админ‑UI**: `ADMIN_UI_USER`, `ADMIN_UI_PASSWORD`
- **Telegram**: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_WEBHOOK_SECRET`, `BOT_ADMIN_IDS`

## 📝 Промпты

Файлы промптов целиком являются системной инструкцией агента. Для постов хранятся в `prompts/post/*` (module_* структура):
- `module_01_writing/post.md`
- `module_02_review/*.md` (identify_risky_points.md, iterative_research.md, sufficiency.md, recommendation.md, query_synthesizer.md)
- `module_03_rewriting/rewrite.md`, `module_03_rewriting/refine.md`

## 🎨 Стиль (style packs)

В каталоге `guides/shared/style_packs/` — рекомендательные документы по стилю: `pop_sci.md`, `academic.md`. Параметр стиля указывается в конфиге пайплайна (`style_pack`).

## 💻 Заметки по Windows CMD

- Используйте CMD, а не PowerShell, во избежание конфликтов
- Запускайте скрипты через `venv\Scripts\python.exe script.py`
- Сервер: `venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000`
- Для компиляции без `cat`: `venv\Scripts\python.exe -m compileall -q <paths>`

## 🤖 Telegram‑бот (команды)
- `/start`, `/info`
- `/lang`, `/lang_generate` — язык интерфейса и язык генерации
- `/provider` — `OpenAI|Gemini|Claude`
- `/generate` — запуск генерации поста
- `/logs` — присылать ли лог в чат
- `/incognito` — скрывать результат в UI по умолчанию
- `/cancel`
- `/balance` — баланс кредитов; админам генерация бесплатна
- `/topup <telegram_id> <amount>` — только админы (`BOT_ADMIN_IDS`)

Стоимость: 1 кредит за генерацию поста (по умолчанию). Балансы хранятся в БД (если настроена) или в Redis.

## 🌐 API / UI
- `GET /health` — состояние
- `GET /logs` / `GET /logs/{id}` — JSON‑доступ к логам
- `GET /logs-ui` / `GET /logs-ui/{id}` — UI логов (Basic Auth)
- `DELETE /logs/{id}`, `POST /logs/purge` — удаление логов (Basic Auth)
- `GET /results` / `GET /results/{id}` — JSON‑доступ к результатам
- `GET /results-ui` / `GET /results-ui/id/{id}` — UI результатов
- `POST /webhook/{secret}` — вебхук Telegram

## 📁 Структура проекта

```
Book_in_one_click/
├── scripts/                # Точки входа (CLI)
│   ├── popular_science_post.py
│   ├── deep_popular_science_article.py
│   ├── deep_popular_science_book.py
│   ├── topic_summary.py
│   └── simple_test.py
├── server/                 # FastAPI + Telegram бот, UI логов/результатов
│   ├── main.py             # endpoints: /health, /logs*, /results*, /webhook/*
│   ├── bot.py              # диалоги и генерация постов
│   ├── bot_commands.py     # /balance, /topup, админ‑утилиты
│   ├── db.py               # модели SQLAlchemy (User, Job, JobLog, ResultDoc)
│   └── kv.py               # Redis‑настройки, провайдер/логи/инкогнито
├── services/
│   └── post/generate.py    # единая функция генерации поста (CLI/бот/сервер)
├── pipelines/              # Пайплайны (post/article/book/summary)
├── prompts/                # Системные промпты по сценариям
│   ├── post/{module_01_writing,module_02_review,module_03_rewriting}
│   ├── article/{writing}
│   ├── summary/{writing}
│   └── book/{writing}
├── llm_agents/             # Код агентов и ролей
├── templates/              # Шаблоны контента
├── utils/                  # Хелперы (env, io, slug, config, web)
├── guides/shared/style_packs/ # Рекомендации по стилю
├── schemas/                # Pydantic‑схемы для структурированных результатов
├── output/                 # Сгенерированный контент (в .gitignore)
├── output_example/         # Примеры результатов
├── memory-bank/            # Память проекта (контекстные файлы)
├── Project_Notes/          # Локальные заметки (в .gitignore)
├── requirements.txt        # Зависимости Python
├── .env                    # API‑ключи и конфигурация (локально)
└── venv/                   # Виртуальное окружение Python
```

## 🎯 Что делает

- **Ввод:** Любая образовательная тема (например, «Фотосинтез», «Machine Learning»)
- **Вывод:** Структурированный образовательный контент (≈300–500 слов для простого теста)
- **Структура:** Заголовок → Введение → Основная часть → Заключение
- **Формат:** Markdown (чистый контент)

## 📋 Требования

- **Python:** 3.8+
- **Платформы:** Windows, macOS, Linux
- **API:** Ключ OpenAI

## 🛠️ Разработка

```bash
# Активировать окружение
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Запуск любого скрипта
python your_script.py

# Установка новой зависимости
pip install package_name
pip freeze > requirements.txt
```

## 📖 Примеры

Смотрите папку `output_example/` для примеров сгенерированного контента.

## 🚨 Устранение неполадок

- **ImportError:** активируйте `venv` и убедитесь, что нет локального пакета `agents/` (используйте `llm_agents/`).
- **API‑ошибки:** проверьте ключи `OPENAI_API_KEY`/`GOOGLE_API_KEY`/`ANTHROPIC_API_KEY`.
- **Админ‑UI:** при отсутствии `ADMIN_UI_PASSWORD` UI вернёт 503.
- **База данных:** при отсутствии `DB_URL` сервер работает, но логи/результаты доступны только как файлы; часть API вернёт `db is not configured`.
- **Redis:** требуется `REDIS_URL` для пользовательских настроек и KV‑кредитов (если БД недоступна).
- **Пути:** используется `pathlib` для кроссплатформенной совместимости.

 
 
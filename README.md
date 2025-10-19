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
- **Конспект (summary)**: структурированный краткий конспект (планируется)
- **Статья (article)**: глубокая популярная статья (планируется)
- **Книга (book)**: заготовка/книга (планируется)

Сейчас поддерживаются: посты. Конспект, статья и книга — в разработке.

Промпты для сценария поста теперь разнесены по жанрам: `prompts/posts/<genre>/<style>/{module_01_writing,module_02_review,module_03_rewriting}/*.md`. Например: `prompts/posts/popular_science_post/popular_science_post_style_1/...` или `prompts/posts/john_oliver_explains_post/john_oliver_explains_post_style_1/...`.

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

Файлы промптов целиком являются системной инструкцией агента. Для постов хранятся в `prompts/posts/<genre>/<style>/*` (module_* структура):
- `module_01_writing/writer.md`
- `module_02_review/*.md` (identify_risky_points.md, iterative_research.md, sufficiency.md, recommendation.md, query_synthesizer.md)
- `module_03_rewriting/rewrite.md`, `module_03_rewriting/refine.md`

## 🎨 Стиль (style packs)

Рекомендации по стилю описаны в промптах и конфигурации пайплайнов. Отдельные style packs временно удалены как неиспользуемые.

## 💻 Заметки по Windows CMD

- Используйте CMD, а не PowerShell, во избежание конфликтов
- Запускайте скрипты через `venv\Scripts\python.exe script.py`
- Сервер: `venv\Scripts\python.exe -m uvicorn server.main:app --host 0.0.0.0 --port 8000`
- Для компиляции без `cat`: `venv\Scripts\python.exe -m compileall -q <paths>`

## 🧠 Как это работает (пост)

- **Шаг 1. Написание**
  - Генерируется пост по «контракту стиля»: цепляющий заголовок (жирным, без эмодзи/кавычек, с пустой строкой после), 6–16 абзацев, каждый абзац начинается релевантным эмодзи, общий объём около 200–450 слов. Тон — понятный, объяснительный, живой; без воды и штампов.

- **Шаг 2. Факт‑чекинг (опционально)**
  - Составляется план проверки: выделяются атомарные пункты (факты, цифры, датировки, причинно‑следственные связи), которые стоит перепроверить.
  - Для каждого пункта формируются 2–4 англоязычных поисковых запросов с приоритетом авторитетных доменов (who.int, cdc.gov, nhs.uk, mayoclinic.org, *.gov, *.edu, PubMed/PMC и т.п.).
  - Проводится 1–3 итерации исследования на пункт (по умолчанию 2): на каждой собирается 1–6 надёжных источников, фиксируются краткие выводы и ссылки.
  - По результатам для каждого пункта принимается понятное решение: оставить как есть, уточнить, переписать или удалить.
  - Можно ограничить число проверяемых пунктов (например, проверить только первые N).

- **Шаг 3. Переписывание**
  - Исправляются только отмеченные проблемные места согласно рекомендациям, при этом сохраняются структура и стиль исходного поста (заголовок, формат, эмодзи в абзацах, объём, тон).

- **Шаг 4. Финальная шлифовка**
  - Лёгкая полировка текста под исходный «контракт стиля»: читаемость, логика переходов, объём, оформление.

- **Прозрачность процесса**
  - На каждом шаге сохраняется человекочитаемый лог: что проверялось, какие решения приняты, какие источники использованы. Лог можно открыть в веб‑интерфейсе или получить в боте (по настройке).

- **Полезные настройки**
  - Язык: auto/ru/en. Провайдер: OpenAI/Gemini/Claude.
  - Факт‑чекинг: включить/выключить; глубина 1–3 итерации на пункт (по умолчанию 2); ограничение числа пунктов.
  - Логи в чат: включить/выключить. Инкогнито: скрывать результат из общего списка.

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
├── guides/                 # Папка зарезервирована под справочные материалы
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

## ⚙️ Модели и провайдеры (централизовано)

- Единый источник моделей и уровней производительности: `config/models.json` (tiers: superfast/fast/medium/heavy).
- Получение модели: `utils/models.get_model(provider, tier)`; ENV переменные могут переопределить, но не обязательны.
- Единый раннер провайдеров: `services/providers/runner.py` (OpenAI/Gemini/Claude). JSON‑каналы: Gemini (`response_mime_type=application/json`), Claude (`response_format={type: json_object}`).

## 🧰 Dev in‑memory KV

Для локальной разработки можно запускать без Redis, установив `DEV_INMEM_KV=1`. Это включает лёгкое in‑memory хранилище ключей в `server/kv.py` (история, настройки, кредиты). В продакшене используйте `REDIS_URL`.

## ➕ Как добавить новую команду бота (краткий гайд)

1) Добавьте обработчик в `server/bot.py` (в будущей модульной структуре — `server/bot/handlers_<feature>.py`).
2) Вынесите бизнес‑логику в `services/<feature>/...`.
3) Промпт/агента разместите симметрично: `prompts/<feature>/...` ↔ `llm_agents/<feature>/...`.
4) Если нужен структурированный ответ — опишите Pydantic‑схему и используйте `utils/json_parse.py`.
5) Добавьте короткие smoke‑скрипты для проверки (см. ниже) и обновите Memory Bank.

## 🧪 Быстрая проверка

Для базовой проверки используйте имеющиеся CLI‑скрипты:
```bat
venv\Scripts\python.exe scripts\simple_test.py
venv\Scripts\python.exe scripts\popular_science_post.py --topic "Photosynthesis" --provider openai --lang en
venv\Scripts\python.exe scripts\popular_science_series_post.py --topic "Photosynthesis" --provider openai --lang en
venv\Scripts\python.exe scripts\chat_telegram.py --message "привет" --context_kind result
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

---

## 🔧 Подробная конфигурация .env

Минимум нужен ключ одного провайдера (OpenAI/Gemini/Claude). Остальные переменные включают сервер/БД/Redis/Telegram.

```env
# Провайдеры (достаточно одного)
OPENAI_API_KEY=...
GOOGLE_API_KEY=...    # или GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...

# Опциональное переопределение моделей (по умолчанию берутся из config/models.json)
OPENAI_MODEL=gpt-5
OPENAI_FAST_MODEL=gpt-5-mini
GEMINI_MODEL=gemini-2.5-pro
GEMINI_FAST_MODEL=gemini-2.5-flash
ANTHROPIC_MODEL=claude-sonnet-4-0
ANTHROPIC_FAST_MODEL=claude-haiku-4-0

# Сервер / UI / БД / Redis / Telegram
ADMIN_UI_USER=admin
ADMIN_UI_PASSWORD=change_me
DB_URL=postgresql+asyncpg://USER:PASSWORD@HOST/DBNAME
DB_TABLE_PREFIX=bio1c_
REDIS_URL=redis://:PASSWORD@HOST:6379/0
TELEGRAM_BOT_TOKEN=...
TELEGRAM_WEBHOOK_SECRET=some_secret
BOT_ADMIN_IDS=123456789,987654321

# Dev удобства
DEV_INMEM_KV=0  # 1 = использовать in‑memory KV вместо Redis для локалки
```

Пояснения:
- Модели: если ENV не задан, используются значения из `config/models.json` (см. ниже). ENV — только override.
- `DB_URL`: при отсутствии БД сервер работает, но UI логов/результатов будет ограничен.
- `REDIS_URL`: для настроек/истории/кредитов. Для локалки можно `DEV_INMEM_KV=1`.
- `BOT_ADMIN_IDS`: CSV из Telegram user_id, даёт админ‑функции и бесплатную генерацию.

## 🗂️ config/models.json — единый источник моделей

Файл содержит уровни производительности (tiers) для каждого провайдера: `superfast|fast|medium|heavy`.

Пример:
```json
{
  "openai": {
    "tiers": {
      "superfast": { "model": "gpt-5-mini" },
      "fast":      { "model": "gpt-5-mini" },
      "medium":    { "model": "gpt-5" },
      "heavy":     { "model": "gpt-5" }
    }
  },
  "gemini": {
    "tiers": {
      "superfast": { "model": "gemini-2.5-flash" },
      "fast":      { "model": "gemini-2.5-flash" },
      "medium":    { "model": "gemini-2.5-pro" },
      "heavy":     { "model": "gemini-2.5-pro" }
    }
  },
  "claude": {
    "tiers": {
      "superfast": { "model": "claude-haiku-4-0" },
      "fast":      { "model": "claude-haiku-4-0" },
      "medium":    { "model": "claude-sonnet-4-0" },
      "heavy":     { "model": "claude-sonnet-4-0" }
    }
  }
}
```

Как используется:
- `utils/models.py` → `get_model(provider, tier)`: возвращает модель из файла; `superfast→fast`, `medium→heavy` нормализуются.
- `services/providers/runner.py`: применяет выбранную модель; для Gemini/Claude включает нативный JSON‑режим.

## 🧩 ProviderRunner — единый раннер провайдеров

- Файл: `services/providers/runner.py`.
- Цель: единая точка для вызова OpenAI/Gemini/Claude без дублирования кода.
- Особенности:
  - OpenAI — через Agents SDK (возможны инструменты и нативная сессия `SQLiteSession`).
  - Gemini — `GenerativeModel(...).generate_content(...)`, поддерживает `response_mime_type=application/json`.
  - Claude — `client.messages.create(...)`, поддерживает `response_format={"type":"json_object"}`.
  - Выбор fast/heavy идёт через `get_model()`.

## 🧱 KV / Redis — ключи и форматы

Использование: хранение пользовательских настроек, истории чата, простого биллинга и лимитов.
- Префикс: `DB_TABLE_PREFIX` (по умолчанию `bio1c_`).
- Ключи:
  - Настройки пользователя:
    - `:provider:{telegram_id}` → `openai|gemini|claude`
    - `:logs:{telegram_id}` → `1|0`
    - `:incognito:{telegram_id}` → `1|0`
    - `:gen_lang:{telegram_id}` → `ru|en|auto`
    - `:fc_enabled:{telegram_id}` → `1|0`
    - `:fc_depth:{telegram_id}` → `1|2|3`
  - История чата:
    - `:chat:{telegram_id}:{chat_id}:{provider}` → список JSON объектов `{role, content}` (до 200)
  - Кредиты:
    - `:credits:{telegram_id}` → целое число (баланс)
  - Лимиты:
    - `:rate:{scope}:h:{telegram_id}` / `:rate:{scope}:d:{telegram_id}` → счётчики за час/сутки

Dev режим:
- `DEV_INMEM_KV=1` включает in‑memory реализацию в `server/kv.py` (совместимый интерфейс, без сохранения между рестартами).

## 💬 Чат — сессии и очистка

- История: Redis (или in‑memory в dev), до 200 сообщений на провайдера/чат.
- Нативные сессии:
  - OpenAI — `SQLiteSession` (файл `output/sessions.db`).
  - Gemini — chat‑session с LRU+TTL кэшем (≈512 сессий, TTL ≈ 45 мин).
  - Claude — без нативной сессии; история — KV.
- Очистка: команда `/endchat` очищает KV‑историю и нативные кэши для текущего провайдера.

## 💳 Биллинг и кредиты

- Баланс хранится в БД (`users.credits`) и/или в Redis (`:credits:{telegram_id}`).
- Админы (`BOT_ADMIN_IDS`) — генерация бесплатна.
- Серии постов: поддержана предоплата и рефанд (см. описание в Memory Bank: activeContext.md → "Серии постов").

## 📡 Эндпоинты сервера

- `GET /health` — статус.
- `GET /logs*` / `GET /results*` — JSON API.
- `GET /logs-ui*` / `GET /results-ui*` — UI (Basic Auth для логов).
- `POST /webhook/{secret}` — webhook Telegram.
- Лимиты скачивания результат/логов: доступны через UI и Redis‑счётчики (см. System Patterns в Memory Bank).

## 🐳 Деплой

- Dockerfile: `python:3.11-slim`, установка зависимостей, копирование проекта.
- Render.com: Image runtime; health check `/health`; переменные окружения в панели.
- Webhook: привяжите `https://<host>/webhook/<TELEGRAM_WEBHOOK_SECRET>` к вашему боту через `setWebhook`.

## 🧱 Как добавить новый сценарий (pipeline)

1) Создать генератор: `services/<feature>/generate.py` с чистым API (без интерактива).
2) Добавить промпты: `prompts/<feature>/*` и, при необходимости, агентов: `llm_agents/<feature>/*`.
3) Создать CLI: `scripts/<feature>.py` (парсинг аргументов, вызов `services`).
4) Добавить Pydantic‑схемы в `schemas/` (если нужен структурированный вывод) и подключить `utils/json_parse.py`.
5) Обновить README и Memory Bank (systemPatterns/activeContext).

## ❓ FAQ

- Могу ли я управлять моделями без ENV? — Да, редактируйте `config/models.json`.
- Как сменить провайдера в боте? — Команда `/provider`.
- Как отключить факт‑чек? — В `/settings` переключите «Факт‑чекинг» или передайте `--no-factcheck` в CLI.
- Как почистить чат‑сессию? — `/endchat`.

 
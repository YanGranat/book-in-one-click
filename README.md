# 📚 Book in One Click

Мультиагентная система для генерации образовательного контента (книг, статей, постов, конспектов) с использованием OpenAI Agents SDK.

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

### 2. Настроить API‑ключ
Создайте файл `.env` в корне проекта:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Запустить простой тест
```bash
python scripts/simple_test.py
```

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

Промпты для сценария поста находятся в `prompts/post/{module_01_writing,module_02_review,module_03_rewriting}/*.md`.

## 🎛️ Параметры CLI

Большинство скриптов поддерживает параметры:

- `--topic` — тема генерации (строка)
- `--lang` — язык вывода: `auto|ru|en` (по умолчанию `auto`)
- `--out` — подпапка для сохранения результата (по умолчанию зависит от пайплайна)

Пример:
```bat
venv\Scripts\python.exe scripts\popular_science_post.py --topic "CRISPR" --lang en
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

## 🧠 Memory Bank

Артефакты памяти в `memory-bank/` помогают сохранять контекст разработки: цели, активный контекст, прогресс, технические решения. При значимых изменениях обновляйте:
- `activeContext.md` — текущий фокус, ближайшие шаги
- `progress.md` — что сделано, что дальше

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
- Для компиляции без `cat` используйте: `venv\Scripts\python.exe -m compileall -q <paths>`

## 📁 Структура проекта

```
Book_in_one_click/
├── scripts/                # Точки входа
│   ├── popular_science_post.py
│   ├── deep_popular_science_article.py
│   ├── deep_popular_science_book.py
│   ├── topic_summary.py
│   └── simple_test.py
├── pipelines/              # Пайплайны (post/article/book/summary)
├── prompts/                # Системные промпты по сценариям
│   ├── post/{module_01_writing,module_02_review,module_03_rewriting}
│   ├── article/{writing}
│   ├── summary/{writing}
│   └── book/{writing}
├── llm_agents/             # Код агентов по сценариям
│   ├── post/{module_01_writing,module_02_review,module_03_rewriting}
│   └── book/ (пока каркас)
├── utils/                  # Хелперы (env, io, slug, config)
├── output/                 # Сгенерированный контент (в .gitignore)
├── output_example/         # Примеры результатов
├── memory-bank/            # Память проекта (контекстные файлы)
├── Project_Notes/          # Локальные заметки (в .gitignore)
├── requirements.txt        # Зависимости Python
├── .env                    # API‑ключи (создаётся вручную)
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

- **ImportError:** Убедитесь, что активировано виртуальное окружение
- **API‑ошибки:** Проверьте, что в `.env` корректный `OPENAI_API_KEY`.
- **Права доступа:** Проверьте права на файлы/директории.
- **Пути:** Проект использует `pathlib` для кроссплатформенной совместимости.

 
 
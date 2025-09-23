Scripts and CLIs (Windows CMD)

Этот файл описывает назначение скриптов, аргументы, примеры запуска, требования к провайдерам и куда сохраняются результаты.

Подготовка (один раз на сессию):
```
venv\Scripts\activate
```

ENV (минимум один провайдер):
- OpenAI: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY` или `GEMINI_API_KEY`
- Claude: `ANTHROPIC_API_KEY`

Язык:
- `--lang auto` — определяется по вводу: кириллица → ru, иначе → en.

Провайдер:
- `--provider openai|gemini|claude` — выбор LLM; модели берутся из `config/models.json`.

Выходные файлы и логи:
- Контент: `output/<subdir>/<slug>_post.md` (или агрегат для серии).
- Логи генерации (опция): `output/<subdir>/<slug>_log_YYYYMMDD_HHMMSS.md`.

1) Popular science post
```
venv\Scripts\python.exe scripts\popular_science_post.py \
  --topic "Ваша тема" --lang auto --provider openai \
  --out post --include-logs --research-iterations 2 --research-concurrency 3
```
Аргументы:
- `--topic` — тема (если не указана, будет интерактивный ввод)
- `--lang` — `auto|ru|en` (по умолчанию `auto`)
- `--provider` — `openai|gemini|claude` (по умолчанию `openai`)
- `--out` — подпапка `output/` (по умолчанию `post`)
- `--no-factcheck` — выключить факт‑чек шага Review
- `--factcheck-max-items` — ограничить число проверяемых пунктов (0 = без лимита)
- `--research-iterations` — число итераций ресёрча на пункт (по умолчанию 2)
- `--research-concurrency` — параллелизм ресёрча (по умолчанию 3)
- `--include-logs` — сохранять Markdown‑лог

Что делает:
- Строит инструкции писателю → генерирует пост → (опционально) планирует факт‑чек, исследует источники, формирует рекомендации → переписывает/шлифует.
- Сохраняет итоговый Markdown; при включённых логах — подробный лог с этапами.

2) Series of posts
```
venv\Scripts\python.exe scripts\popular_science_series_post.py \
  --topic "Ваша тема" --lang auto --provider openai \
  --mode auto --output single --out post_series \
  --max-iterations 1 --sufficiency-heavy-after 3 --factcheck --research-iterations 2 --refine
```
Аргументы:
- `--mode` — `auto|fixed` (в fixed укажите `--count`)
- `--count` — число постов для fixed режима
- `--max-iterations` — количество итераций sufficiency/extend (по умолчанию 1)
- `--sufficiency-heavy-after` — переход на heavy‑модель после N‑й итерации (по умолчанию 3)
- `--output` — `single|folder` (агрегатный файл серии или папка с отдельными постами)
- `--out` — подпапка `output/` (по умолчанию `post_series`)
- `--factcheck` — включить факт‑чек для каждого поста
- `--research-iterations` — итераций ресёрча на пункт
- `--refine` — финальная шлифовка каждого поста

Что делает:
- Планирует темы серии (auto/fixed) → при необходимости расширяет/упорядочивает → последовательно пишет посты, учитывая весь список тем (снижение повторов).
- В режиме `single` создаёт один агрегатный Markdown с содержимым всех постов и лог серии.

3) Deep popular science article
```
venv\Scripts\python.exe scripts\deep_popular_science_article.py --topic "Ваша тема"
```
Примечание:
- Глубокая статья; поддерживаемые флаги смотрите в самом скрипте (аналогично посту: `--topic`, `--lang`, `--provider`, `--out`).

4) Deep popular science book (stub)
```
venv\Scripts\python.exe scripts\deep_popular_science_book.py --topic "Ваша тема"
```
Примечание:
- Заготовка сценария книги; интерфейс аналогичен статье. Функциональность может быть ограничена.

5) Telegram‑style chat (console)
```
venv\Scripts\python.exe scripts\chat_telegram.py --lang ru
venv\Scripts\python.exe scripts\chat_telegram.py --lang ru --context-id 123
venv\Scripts\python.exe scripts\chat_telegram.py --lang en --save-md
```
Аргументы:
- `--lang` — `ru|en|auto` (по умолчанию `auto` — определяется по первому вводу)
- `--context-id` — `ResultDoc.id` для обсуждения существующего результата (полный текст подаётся в контекст чата)
- `--save-md` — принудительно сохранить ответ в `.md`, если модель не вернула `<md_output>`

Сохранение в чате:
- Если ответ содержит `<md_output ...>...</md_output>`, будет сохранён `.md` с указанным `title` (если есть), иначе имя `chat_result`.

Требования к провайдерам (быстрый чек перед запуском):
- OpenAI: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY` или `GEMINI_API_KEY`
- Claude: `ANTHROPIC_API_KEY`

Типовые ошибки и решения:
- «API key not found» — проверьте `.env` в корне и перезапустите CMD с активированным venv.
- «ImportError: agents» — убедитесь, что нет локальной папки `agents/` (используйте `llm_agents/`).
- «event loop is running» — повторите запуск; скрипты приводят окружение к корректному состоянию автоматически.

Scripts and CLIs (Windows CMD)

1) Popular science post
```
venv\Scripts\python.exe scripts\popular_science_post.py --topic "Ваша тема" --lang auto --provider openai --out post --include-logs
```
Flags:
- `--topic` — тема (если не указана, будет интерактивный ввод)
- `--lang` — `auto|ru|en` (по умолчанию `auto`)
- `--provider` — `openai|gemini|claude` (по умолчанию `openai`)
- `--out` — подпапка в `output/` (по умолчанию `post`)
- `--no-factcheck` — отключить факт‑чек
- `--factcheck-max-items` — лимит пунктов для проверки (0 = без ограничения)
- `--research-iterations` — итераций ресёрча на пункт (по умолчанию 2)
- `--research-concurrency` — параллельных воркеров ресёрча (по умолчанию 3)
- `--include-logs` — сохранить Markdown‑лог генерации

2) Series of posts
```
venv\Scripts\python.exe scripts\popular_science_series_post.py --topic "Ваша тема" --lang auto --provider openai --mode auto --output single --out post_series
```
Flags:
- `--mode` — `auto|fixed` (в fixed укажите `--count`)
- `--count` — число постов (для fixed)
- `--max-iterations` — итераций sufficiency/extend (по умолчанию 1)
- `--sufficiency-heavy-after` — переход на heavy после N‑й итерации (по умолчанию 3)
- `--output` — `single|folder` (агрегатный файл или папка с отдельными постами)
- `--factcheck` — включить факт‑чек для каждого поста
- `--research-iterations` — итераций ресёрча на пункт (факт‑чек)
- `--refine` — финальная правка каждого поста

3) Deep popular science article
```
venv\Scripts\python.exe scripts\deep_popular_science_article.py --topic "Ваша тема"
```

4) Deep popular science book (stub)
```
venv\Scripts\python.exe scripts\deep_popular_science_book.py --topic "Ваша тема"
```

5) Telegram‑style chat (console)
```
venv\Scripts\python.exe scripts\chat_telegram.py --lang ru
venv\Scripts\python.exe scripts\chat_telegram.py --lang ru --context-id 123
venv\Scripts\python.exe scripts\chat_telegram.py --lang en --save-md
```
Flags:
- `--lang` — `ru|en|auto` (по умолчанию auto — определяется по первому вводу)
- `--context-id` — id результата (`ResultDoc.id`) для обсуждения предыдущей генерации (полный текст подаётся в контекст)
- `--save-md` — принудительно сохранить ответ в `.md`, если модель не вернула `<md_output>`
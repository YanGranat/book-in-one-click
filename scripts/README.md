Scripts and CLIs 

Windows CMD examples:

1) Popular science post
```
venv\Scripts\python.exe scripts\popular_science_post.py --topic "Ваша тема"
```

2) Deep popular science article
```
venv\Scripts\python.exe scripts\deep_popular_science_article.py --topic "Ваша тема"
```

3) Deep popular science book (stub)
```
venv\Scripts\python.exe scripts\deep_popular_science_book.py --topic "Ваша тема"
```

Flags:
- `--topic` — тема (иначе будет интерактивный ввод)
- `--lang` — язык вывода: auto|ru|en (по умолчанию auto — язык запроса)
- `--out` — подпапка в `output/`

Simple smoke test behavior:
- Uses model `gpt-5`
- Saves to `output/simple_test/`
- Filenames: `<тема>_test.md`, with auto-increment (`_2`, `_3`, …) if a file exists
- Writes only the generated content (no headers/metadata)
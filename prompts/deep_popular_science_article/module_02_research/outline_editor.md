<task>
Ты — редактор структуры. Получив текущую структуру и список операций, аккуратно примени их к ArticleOutline, верни обновлённую структуру.
</task>

<input>
- outline_json: ArticleOutline
- changes_json: OutlineChangeList
</input>

<guidelines>
- Сохраняй целостность id; при добавлении новых элементов проставляй следующие по порядку id: sNN, ssNN, ciNN (локальная нумерация в рамках соответствующего уровня).
- При merge/split сохраняй смысл заголовков, избегай дубликатов.
- Не трогай неупомянутые части структуры.
</guidelines>

<output>
Верни строго JSON ArticleOutline.
</output>

<requirements>
- Никаких пояснений вне JSON.
- Не меняй порядок, если операция явно не требует move.
</requirements>



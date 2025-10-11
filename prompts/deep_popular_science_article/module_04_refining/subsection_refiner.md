<task>
Ты — редактор подраздела. Перепиши текст ОДНОГО подраздела так, чтобы он давал глубокое понимание и был увлекательным: ясность, логика, связность, живой язык.
</task>

<input>
- lang: ru|en|auto
- topic
- outline_json: ArticleOutline
- section_id: sNN
- subsection_id: ssNN
- draft_chunk: DraftChunk (исходный текст)
</input>

<guidelines>
- Сохраняй фактическое содержание; улучши объяснения, аналогии, ритм.
- Устрани повторы; усиливай «почему/как» вместо сухих фактов.
- Объём: в диапазоне исходного ±20%.
- Язык вывода: lang=auto → язык topic.
</guidelines>

<output>
Верни строго JSON DraftChunk: {subsection_id,title,markdown}
</output>

<requirements>
- Никаких пояснений вне JSON.
</requirements>



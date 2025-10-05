<task>
Ты — автор лид‑абзаца статьи. Напиши сильный вводный абзац для всей статьи с учётом структуры.
</task>

<input>
- lang: ru|en|auto
- topic
- outline_json: ArticleOutline
</input>

<guidelines>
- Объём: 120–260 слов.
- Цель: заинтересовать и задать рамку понимания темы.
- Тон: увлечённый, компетентный, без пафоса.
</guidelines>

<output>
Верни строго JSON LeadChunk: {scope:"article",markdown}
</output>

<requirements>
- Никаких пояснений вне JSON.
</requirements>



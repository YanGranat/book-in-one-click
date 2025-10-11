<task>
Ты — научный редактор. По пункту и текущим заметкам исследования реши, достаточно ли данных и качества источников для вывода, или требуется ещё одна итерация.
</task>

<input>
- point: {id, text}
- notes: ResearchIterationNote[]
</input>

<guidelines>
- Оцени полноту, согласованность источников и ясность формулировок.
- Если данных мало или они противоречивы — верни done=false и предложи 1–3 точных запроса (suggested_queries) на английском языке и список недостающих аспектов (missing_gaps).
- Если достаточно — верни done=true.
- confidence — число 0..1.
</guidelines>

<output>
Верни SufficiencyDecision: {point_id, done, missing_gaps[], suggested_queries[], confidence}.
</output>

<requirements>
- Никаких эссе; чёткое решение.
- Верни строго JSON объекта SufficiencyDecision без Markdown.
</requirements>



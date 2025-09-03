def progress(stage: str) -> None:
	labels = {
		"start: post": "🚀 Старт генерации поста",
		"agent:init": "🤖 Инициализация агента",
		"agent:run": "🧠 Генерация контента",
		"io:prepare_output": "📁 Подготовка директории вывода",
		"factcheck:init": "🔎 Факт‑чекинг",
		"rewrite:init": "✍️ Переписывание по замечаниям",
		"factcheck:second_pass": "🔁 Повторный факт‑чекинг",
		"io:save_final": "💾 Сохранение результата",
		"done": "✅ Готово",
	}
	print(labels.get(stage, stage.replace(":", " → ")))

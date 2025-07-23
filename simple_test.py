#!/usr/bin/env python3
"""
Простой тест - один агент пишет страницу текста
Запуск: venv\Scripts\python.exe simple_test.py
"""
import os
from datetime import datetime
from pathlib import Path

# Загружаем .env
def load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

try:
    from agents import Agent, Runner
    print("✅ OpenAI Agents SDK загружен")
except ImportError as e:
    print(f"❌ Ошибка: {e}")
    exit(1)

def main():
    print("📝 Простой тест - один агент, страница текста")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY не найден в .env файле")
        return
    
    # Запрашиваем тему
    print("\n🤔 Введите тему:")
    topic = input("➤ ").strip()
    
    if not topic:
        print("❌ Тема не может быть пустой")
        return
    
    print(f"\n🔄 Генерирую страницу текста на тему: '{topic}'")
    print("⏳ Это займет 30-60 секунд...")
    
    try:
        # Создаем агента для генерации страницы текста
        agent = Agent(
            name="Content Writer",
            instructions="""
            Напиши информативную страницу текста на заданную тему.
            
            Структура:
            1. Краткое введение (1 абзац)
            2. Основная часть (2-3 абзаца с подробностями)
            3. Заключение или практические выводы (1 абзац)
            
            Стиль: образовательный, понятный, с примерами.
            Объем: примерно 300-500 слов (страница текста).
            """,
            model="gpt-4o"
        )
        
        # Генерируем страницу
        result = Runner.run_sync(agent, f"Напиши страницу текста на тему: {topic}")
        content = result.final_output
        
        print("✅ Страница сгенерирована!")
        
        # Сохраняем в файл
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = topic.replace(' ', '_').replace('/', '_').lower()
        filename = f"page_{safe_topic}_{timestamp}.md"
        filepath = output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {topic}\n\n")
            f.write(f"*Создано: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}*\n")
            f.write(f"*Генератор: OpenAI Agents SDK*\n\n")
            f.write(f"{content}\n")
        
        print(f"💾 Сохранено: {filepath}")
        
        # Показываем начало результата
        preview = content[:200] + "..." if len(content) > 200 else content
        print(f"\n📖 Начало результата:\n{preview}")
        
        # Показываем статистику
        word_count = len(content.split())
        char_count = len(content)
        print(f"\n📊 Статистика:")
        print(f"   Слов: {word_count}")
        print(f"   Символов: {char_count}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    
    print("\n👋 Тест завершен")

if __name__ == "__main__":
    main() 
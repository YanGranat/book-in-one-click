# 📚 Book in One Click

Multi-agent system for educational book generation using OpenAI Agents SDK.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <your-repo-url>
cd Book_in_one_click

# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file in project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run Simple Test
```bash
python scripts/simple_test.py
```

### 4. Run Generators (Windows CMD)
```bat
venv\Scripts\python.exe scripts\Popular_science_post.py
venv\Scripts\python.exe scripts\deep_popular_science_article.py
venv\Scripts\python.exe scripts\deep_popular_science_book.py
venv\Scripts\python.exe scripts\simple_test.py
```

Enter any topic and get a generated page of educational content!

## 📁 Project Structure

```
Book_in_one_click/
├── scripts/                # Entry-point scripts
│   ├── Popular_science_post.py
│   ├── deep_popular_science_article.py
│   ├── deep_popular_science_book.py
│   └── simple_test.py
├── pipelines/              # Pipelines per scenario (post/article/book)
├── prompts/                # System prompts for agents (e.g., writing/article.md)
│   └── writing/
├── llm_agents/             # Agent roles (research/planning/writing/review)
├── utils/                  # Helpers (env, io, slug, config)
├── output/                 # Generated content (gitignored)
├── output_example/         # Example outputs for demo
├── memory-bank/            # Project memory (context docs)
├── Project_Notes/          # Local project notes (gitignored)
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create manually)
└── venv/                   # Python virtual environment
```

## 🎯 What It Does

- **Input:** Any educational topic (e.g., "Photosynthesis", "Machine Learning")
- **Output:** Structured educational content (~300-500 words)
- **Structure:** Title → Introduction → Main content → Conclusion
- **Format:** Markdown (content only)

## 📋 Requirements

- **Python:** 3.8+
- **Platform:** Windows, macOS, Linux
- **API:** OpenAI API key

## 🛠️ Development

```bash
# Activate environment
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run any script
python your_script.py

# Install new dependencies
pip install package_name
pip freeze > requirements.txt
```

## 📖 Examples

See `output_example/` folder for sample generated content.

## 🚨 Troubleshooting

- **Import errors:** Ensure virtual environment is activated
- **API errors:** Check `.env` file has valid OPENAI_API_KEY
- **Permission errors:** Check file/directory permissions
- **Path issues:** Project uses pathlib for cross-platform compatibility

## 📚 Documentation

- `Project_Notes/` - Project notes (e.g., agents SDK research)
- `memory-bank/` - Core context (brief, product/system/tech, progress)
- `.cursorrules` - Project-specific development rules
 
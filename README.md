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
python simple_test.py
```

Enter any topic and get a generated page of educational content!

## 📁 Project Structure

```
Book_in_one_click/
├── simple_test.py          # Simple one-agent test (start here)
├── output/                 # Generated content (auto-created)
├── output_example/         # Example outputs for demo
├── project_notes.md        # Detailed SDK documentation
├── requirements.txt        # Python dependencies
├── .env                    # API keys (create manually)
└── venv/                   # Python virtual environment
```

## 🎯 What It Does

- **Input:** Any educational topic (e.g., "Photosynthesis", "Machine Learning")
- **Output:** Structured educational content (~300-500 words)
- **Structure:** Introduction → Main content → Conclusion
- **Format:** Markdown with metadata

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

- `project_notes.md` - Detailed OpenAI Agents SDK documentation
- `.cursorrules` - Project-specific development rules

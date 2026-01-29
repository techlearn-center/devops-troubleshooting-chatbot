# Module 0: Setup & Prerequisites

**Time Required: 30 minutes**

Before building our DevOps chatbot, we need to set up our development environment. This module ensures everything is working correctly.

---

## Learning Objectives

By the end of this module, you will:
- Have Python 3.9+ installed and working
- Have all required packages installed
- Have an OpenAI API key configured (or Ollama running)
- Understand the project structure
- Be ready to start building!

---

## Step 1: Check Python Version

We need Python 3.9 or higher.

```bash
python --version
# Should show: Python 3.9.x or higher
```

**If you don't have Python 3.9+:**
- Download from [python.org](https://www.python.org/downloads/)
- Or use pyenv: `pyenv install 3.11`

---

## Step 2: Create Virtual Environment

A virtual environment keeps project dependencies isolated.

```bash
# Navigate to project root
cd devops-troubleshooting-chatbot

# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
.\venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---------|---------|
| `openai` | OpenAI API client |
| `langchain` | LLM application framework |
| `chromadb` | Vector database |
| `sentence-transformers` | Local embeddings |
| `rich` | Beautiful terminal output |

---

## Step 4: Get Your OpenAI API Key

### Option A: OpenAI (Recommended for best results)

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)

**Cost:**
- GPT-3.5-turbo: ~$0.002 per 1K tokens
- For this course: expect $1-5 total

### Option B: Ollama (Free, Local)

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install it on your system
3. Pull a model:
   ```bash
   ollama pull llama2
   # Or for better results:
   ollama pull mistral
   ```

---

## Step 5: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your settings
# On Linux/Mac:
nano .env
# Or on Windows, use notepad or VS Code
```

**For OpenAI users:**
```env
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_MODEL=gpt-3.5-turbo
USE_OLLAMA=false
```

**For Ollama users:**
```env
USE_OLLAMA=true
OLLAMA_MODEL=llama2
OPENAI_API_KEY=not-needed
```

---

## Step 6: Verify Setup

Run the verification script:

```bash
python module-0-setup/verify_setup.py
```

**Expected Output:**
```
============================================
  DevOps Chatbot - Setup Verification
============================================

[1/6] Checking Python version...
  ✓ Python 3.11.0

[2/6] Checking required packages...
  ✓ openai installed
  ✓ langchain installed
  ✓ chromadb installed
  ✓ sentence-transformers installed

[3/6] Checking environment variables...
  ✓ .env file found
  ✓ OPENAI_API_KEY configured

[4/6] Testing OpenAI connection...
  ✓ Successfully connected to OpenAI

[5/6] Testing embeddings...
  ✓ Embedding model loaded

[6/6] Testing ChromaDB...
  ✓ ChromaDB working

============================================
  ✓ All checks passed! You're ready to go!
============================================
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xxx'"

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### "OpenAI API error: Invalid API key"

1. Check your API key is correct in `.env`
2. Make sure there are no extra spaces
3. Verify the key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### "Ollama connection refused"

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, verify
ollama list
```

### "ChromaDB error"

```bash
# Delete the database and start fresh
rm -rf chroma_db/
```

---

## Understanding the Project Structure

```
devops-troubleshooting-chatbot/
│
├── module-X-*/           # Learning modules (work through in order)
│   ├── README.md         # Instructions and explanations
│   ├── exercises/        # Your work goes here
│   └── solutions/        # Reference solutions
│
├── knowledge-base/       # DevOps documentation for RAG
│   ├── terraform/        # Terraform error docs
│   ├── kubernetes/       # K8s troubleshooting
│   ├── docker/           # Docker issues
│   └── cicd/             # CI/CD pipeline help
│
├── chatbot/              # Final chatbot code
│   ├── main.py           # Entry point
│   ├── rag_engine.py     # RAG implementation
│   └── prompts.py        # Prompt templates
│
├── .env                  # Your configuration (git-ignored)
├── .env.example          # Configuration template
├── requirements.txt      # Python dependencies
└── run.py                # Auto-grading script
```

---

## What's Next?

Once verification passes, you're ready for **Module 1: LLM Fundamentals**!

```bash
cd ../module-1-llm-basics
```

In Module 1, you'll learn:
- How Large Language Models work
- Making your first API call
- Understanding tokens and parameters

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `source venv/bin/activate` | Activate virtual environment (Linux/Mac) |
| `.\venv\Scripts\activate` | Activate virtual environment (Windows) |
| `pip install -r requirements.txt` | Install dependencies |
| `python module-0-setup/verify_setup.py` | Verify setup |
| `python run.py` | Run auto-grader |

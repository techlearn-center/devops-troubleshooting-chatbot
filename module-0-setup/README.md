# Module 0: Prerequisites and Setup

Welcome to Module 0! Before we build our DevOps troubleshooting chatbot, we need to set up our development environment. This module explains **everything from scratch** - even if you've never programmed before, you'll be able to follow along.

---

## Table of Contents

1. [Understanding the Basics](#understanding-the-basics)
   - [What is Python?](#what-is-python)
   - [What is an API?](#what-is-an-api)
   - [What is OpenAI?](#what-is-openai)
   - [What is an API Key?](#what-is-an-api-key)
   - [What is a Virtual Environment?](#what-is-a-virtual-environment)
2. [Installing Python](#installing-python)
   - [Windows (using Chocolatey)](#windows-using-chocolatey)
   - [macOS (using Homebrew)](#macos-using-homebrew)
   - [Linux](#linux)
3. [Setting Up Virtual Environment with uv](#setting-up-virtual-environment-with-uv)
4. [Getting Your OpenAI API Key](#getting-your-openai-api-key)
5. [Project Setup](#project-setup)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## Understanding the Basics

Before we dive into installation, let's understand what all these terms mean.

### What is Python?

**Python** is a programming language - a way to give instructions to your computer. Think of it like learning a new language to communicate with your computer.

```
Human Language:    "Add 5 and 3, then show me the result"
Python:            print(5 + 3)
Computer Output:   8
```

Why Python for AI/ML?
- **Easy to read** - Python code looks almost like English
- **Huge ecosystem** - Thousands of pre-built tools for AI
- **Industry standard** - Most AI/ML projects use Python

### What is an API?

**API** stands for **Application Programming Interface**. It's a way for different software programs to talk to each other.

```
REAL-WORLD ANALOGY: Restaurant
================================

You (Customer)  →  Waiter (API)  →  Kitchen (Server/Service)
     ↑                                      |
     |              Your Food               |
     +←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←+

- You don't go into the kitchen yourself
- You tell the waiter what you want (API Request)
- The kitchen prepares it (Processing)
- The waiter brings it back (API Response)
```

**In programming terms:**

```
Your Code  →  API Request  →  OpenAI Servers
    ↑                              |
    |         AI Response          |
    +←←←←←←←←←←←←←←←←←←←←←←←←←←←←←+

Example API Request:
  "What is Kubernetes?"

Example API Response:
  "Kubernetes is an open-source container orchestration platform..."
```

**Types of APIs you'll encounter:**

| API Type | Example | What It Does |
|----------|---------|--------------|
| REST API | OpenAI API | Send/receive data over the internet |
| Library API | Python's `os` module | Use pre-built code in your program |
| Hardware API | Webcam access | Control physical devices |

### What is OpenAI?

**OpenAI** is a company that created ChatGPT and provides AI services through their API.

```
OPENAI PRODUCTS
===============

ChatGPT (Website)          OpenAI API (For Developers)
    ↓                              ↓
┌─────────────┐            ┌─────────────────────┐
│ You type in │            │ Your code sends     │
│ a chat box  │            │ requests and gets   │
│ on website  │            │ responses back      │
└─────────────┘            └─────────────────────┘

Same AI models, different interfaces!
```

**OpenAI offers several models:**

| Model | Best For | Cost |
|-------|----------|------|
| GPT-4o | Complex reasoning, coding | Higher |
| GPT-4o-mini | General tasks, good balance | Medium |
| GPT-3.5-turbo | Simple tasks, fastest | Lower |

### What is an API Key?

An **API Key** is like a password that identifies you to a service. It's how OpenAI knows:
1. **Who you are** - Links requests to your account
2. **What you can access** - Your subscription level
3. **How much to charge** - Tracks your usage

```
API KEY ANATOMY
===============

sk-proj-abc123xyz789...
│  │    │
│  │    └── Unique identifier (like a password)
│  └─────── Type: "proj" = project key
└────────── Prefix: "sk" = secret key

IMPORTANT SECURITY RULES:
✗ NEVER share your API key publicly
✗ NEVER commit it to Git/GitHub
✗ NEVER paste it in code files
✓ Always use environment variables (.env files)
✓ Always add .env to .gitignore
```

**What happens if your key is leaked?**
- Someone could use your key
- You get charged for their usage
- OpenAI may disable your key

### What is a Virtual Environment?

A **Virtual Environment** is an isolated space for your Python project. Think of it as a separate room for each project where you can have your own furniture (packages) without affecting other rooms.

#### Why Do We Need Virtual Environments?

**Problem Without Virtual Environments:**
Imagine you're working on two different projects:
- **Project A** (an old website) needs Django 2.0
- **Project B** (a new API) needs Django 4.0

Without virtual environments, you can only have ONE version of Django installed on your computer. This causes:
- Breaking Project A when you upgrade Django for Project B
- Constant uninstalling/reinstalling packages
- "It works on my machine" problems when sharing code

```
WITHOUT VIRTUAL ENVIRONMENTS (BAD)
===================================

Your Computer
├── Python
├── Package A v1.0  ←── Project 1 needs this
├── Package A v2.0  ←── Project 2 needs this (CONFLICT!)
└── Package B v1.5

Problem: Project 1 and Project 2 need different versions!


WITH VIRTUAL ENVIRONMENTS (GOOD)
=================================

Your Computer
├── Python
├── Project 1/
│   └── .venv/
│       ├── Package A v1.0  ✓ Isolated!
│       └── Package B v1.5
│
└── Project 2/
    └── .venv/
        ├── Package A v2.0  ✓ Isolated!
        └── Package C v3.0

Each project has its own packages - no conflicts!
```

#### How Virtual Environments Work

```
VIRTUAL ENVIRONMENT LIFECYCLE
=============================

1. CREATE the virtual environment
   ┌─────────────────────────────┐
   │  uv venv                    │ ← Creates .venv folder
   │  (or: python -m venv .venv) │
   └─────────────────────────────┘
               ↓
2. ACTIVATE to "enter" the environment
   ┌─────────────────────────────────┐
   │  source .venv/bin/activate      │ ← Linux/Mac
   │  .\.venv\Scripts\Activate.ps1   │ ← Windows
   └─────────────────────────────────┘
               ↓
   Your prompt changes: (.venv) $
   Now all pip installs go to .venv/
               ↓
3. INSTALL packages (they go into .venv/)
   ┌─────────────────────────────────┐
   │  pip install requests           │
   │  uv pip install -r requirements │
   └─────────────────────────────────┘
               ↓
4. WORK on your project
   Python uses packages from .venv/
               ↓
5. DEACTIVATE when done
   ┌─────────────────────────────────┐
   │  deactivate                     │
   └─────────────────────────────────┘
   Prompt returns to normal: $
```

#### What's Inside a Virtual Environment?

```
.venv/                          ← The virtual environment folder
├── bin/ (or Scripts\ on Windows)
│   ├── activate               ← Script to activate the env
│   ├── python                 ← Python executable for this env
│   └── pip                    ← pip for this env
├── lib/
│   └── python3.11/
│       └── site-packages/     ← Where packages get installed
│           ├── openai/
│           ├── requests/
│           └── ...
└── pyvenv.cfg                 ← Configuration file
```

#### Virtual Environment Best Practices

| Do | Don't |
|----|-------|
| Create one venv per project | Share venv between projects |
| Add `.venv/` to `.gitignore` | Commit `.venv/` to Git |
| Use `requirements.txt` to track packages | Manually remember what to install |
| Activate before running your code | Install packages globally |
| Name it `.venv` (standard convention) | Use random names like `my_env_123` |

---

### OpenAI vs Ollama: Understanding Your Options

When building AI applications, you have two main choices for the language model:

```
                     OPENAI                           OLLAMA
                 (Cloud-Based)                      (Local)
                      │                                │
                      ▼                                ▼
    ┌─────────────────────────────┐    ┌─────────────────────────────┐
    │   Your Computer             │    │   Your Computer             │
    │   ┌─────────┐               │    │   ┌─────────┐               │
    │   │ Your    │               │    │   │ Your    │               │
    │   │ Code    │               │    │   │ Code    │               │
    │   └────┬────┘               │    │   └────┬────┘               │
    │        │                    │    │        │                    │
    │        │ API Call           │    │        │ Local Call         │
    │        │ (Internet)         │    │        │ (No Internet)      │
    │        ▼                    │    │        ▼                    │
    └────────┼────────────────────┘    │   ┌─────────┐               │
             │                         │   │ Ollama  │               │
             │                         │   │ Server  │               │
             ▼                         │   │ (Local) │               │
    ┌─────────────────────────────┐    │   └────┬────┘               │
    │   OpenAI Servers            │    │        │                    │
    │   ┌─────────┐               │    │        ▼                    │
    │   │ GPT-4   │               │    │   ┌─────────┐               │
    │   │ GPT-3.5 │               │    │   │ Llama2  │               │
    │   │ etc.    │               │    │   │ Mistral │               │
    │   └─────────┘               │    │   │ etc.    │               │
    └─────────────────────────────┘    │   └─────────┘               │
                                       └─────────────────────────────┘
```

#### Detailed Comparison

| Feature | OpenAI | Ollama |
|---------|--------|--------|
| **Cost** | Pay per token (~$0.002/1K tokens for GPT-3.5) | Free (uses your hardware) |
| **Setup** | Create account, get API key | Download and install |
| **Internet** | Required | Not required (works offline) |
| **Speed** | Fast responses | Depends on your hardware |
| **Privacy** | Data sent to OpenAI servers | Data stays on your computer |
| **Quality** | State-of-the-art (GPT-4) | Good (Llama2, Mistral) |
| **Hardware** | None needed | 8GB+ RAM recommended |

#### When to Use OpenAI

✅ **Choose OpenAI when:**
- You need the best quality responses (GPT-4)
- You don't have a powerful computer
- You're building a production application
- You need consistent, fast responses
- Cost is not a major concern

```python
# OpenAI Example
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY from env

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

#### When to Use Ollama

✅ **Choose Ollama when:**
- You want to learn without spending money
- Privacy is important (sensitive data)
- You have a decent computer (8GB+ RAM, GPU helps)
- You want to work offline
- You want to experiment freely

```python
# Ollama Example (similar API!)
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama2",
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
print(response.json()["message"]["content"])
```

#### Hardware Requirements for Ollama

| Model | RAM Needed | Disk Space | Quality |
|-------|------------|------------|---------|
| Llama2 7B | 8GB | 4GB | Good |
| Llama2 13B | 16GB | 8GB | Better |
| Mistral 7B | 8GB | 4GB | Very Good |
| Mixtral 8x7B | 32GB | 26GB | Excellent |

**Tip for Learning:** Start with Ollama (free) to understand the concepts, then switch to OpenAI when you need better quality or are building something serious.

---

## Installing Python

### Windows (using Chocolatey)

**Chocolatey** is a package manager for Windows - it lets you install software from the command line (like an app store for developers).

#### Step 1: Install Chocolatey

1. **Open PowerShell as Administrator:**
   - Press `Windows + X`
   - Click "Windows Terminal (Admin)" or "PowerShell (Admin)"

2. **Run the installation command:**
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

3. **Verify installation:**
   ```powershell
   choco --version
   ```
   You should see something like `2.2.2`

#### Step 2: Install Python

```powershell
# Install Python 3.11 (recommended for this project)
choco install python311 -y

# Close and reopen your terminal, then verify
python --version
```

You should see: `Python 3.11.x`

#### Alternative: Direct Download (if Chocolatey doesn't work)

1. Go to https://www.python.org/downloads/
2. Download Python 3.11.x
3. Run the installer
4. **IMPORTANT:** Check "Add Python to PATH" during installation

---

### macOS (using Homebrew)

**Homebrew** is the most popular package manager for macOS - it's like Chocolatey but for Mac.

#### Step 1: Install Homebrew

1. **Open Terminal:**
   - Press `Cmd + Space`
   - Type "Terminal"
   - Press Enter

2. **Run the installation command:**
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Follow the instructions** - you may need to run additional commands shown at the end

4. **Verify installation:**
   ```bash
   brew --version
   ```

#### Step 2: Install Python

```bash
# Install Python 3.11
brew install python@3.11

# Verify installation
python3 --version
```

You should see: `Python 3.11.x`

---

### Linux

Most Linux distributions come with Python. If not:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip
```

**Fedora:**
```bash
sudo dnf install python3.11
```

**Arch Linux:**
```bash
sudo pacman -S python
```

Verify:
```bash
python3 --version
```

---

## Setting Up Virtual Environment with uv

**uv** is a modern, fast Python package manager. It's much faster than pip and easier to use than conda.

### Why uv instead of pip?

| Feature | pip | uv |
|---------|-----|-----|
| Speed | Slow (installs one by one) | Fast (10-100x faster) |
| Lock files | Manual | Automatic |
| Virtual envs | Separate tool needed | Built-in |
| Dependency resolution | Can fail | More reliable |

### Step 1: Install uv

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify installation:**
```bash
uv --version
```

### Step 2: Create a Virtual Environment

```bash
# Navigate to your project directory
cd devops-troubleshooting-chatbot

# Create virtual environment with uv
uv venv

# This creates a .venv folder in your project
```

### Step 3: Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

When activated, you'll see `(.venv)` at the start of your prompt:
```
(.venv) C:\Users\you\devops-troubleshooting-chatbot>
```

### Step 4: Install Dependencies

```bash
# With uv (recommended - much faster)
uv pip install -r requirements.txt

# Or with regular pip (slower but also works)
pip install -r requirements.txt
```

### Step 5: Deactivate When Done

```bash
deactivate
```

---

## Getting Your OpenAI API Key

### Step 1: Create an OpenAI Account

1. Go to https://platform.openai.com/signup
2. Sign up with email or Google/Microsoft account
3. Verify your email

### Step 2: Add Payment Method (Required for API)

1. Go to https://platform.openai.com/account/billing
2. Click "Add payment method"
3. Add a credit/debit card

**Note:** OpenAI charges based on usage. For learning:
- GPT-3.5-turbo: ~$0.002 per 1K tokens (~750 words)
- You can set usage limits to avoid surprises

### Step 3: Create an API Key

1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Give it a name (e.g., "devops-chatbot")
4. **IMPORTANT:** Copy the key immediately - you can't see it again!

```
Your key will look like:
sk-proj-aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789...
```

### Step 4: Store Your API Key Safely

**NEVER put your API key directly in code!**

Create a `.env` file in your project:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your key
```

Edit `.env`:
```env
# Your OpenAI API key (get it from platform.openai.com/api-keys)
OPENAI_API_KEY=sk-proj-your-actual-key-here

# Model to use (gpt-3.5-turbo is cheapest, gpt-4o-mini is better)
OPENAI_MODEL=gpt-3.5-turbo
```

### Alternative: Using Ollama (Free, Local)

If you don't want to pay for OpenAI, you can use **Ollama** to run AI models locally on your computer.

**Install Ollama:**
- Windows/Mac: Download from https://ollama.ai
- Linux: `curl -fsSL https://ollama.ai/install.sh | sh`

**Pull a model:**
```bash
ollama pull llama2
# or for better results:
ollama pull mistral
```

**Configure `.env` for Ollama:**
```env
USE_OLLAMA=true
OLLAMA_MODEL=llama2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Project Setup

### Complete Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/techlearn-center/devops-troubleshooting-chatbot.git
cd devops-troubleshooting-chatbot

# 2. Create virtual environment
uv venv

# 3. Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Create .env file
cp .env.example .env

# 6. Edit .env and add your API key
# Use your favorite editor (notepad, vim, nano, VS Code)

# 7. Run verification
python module-0-setup/verify_setup.py
```

---

## Verification

Run the verification script to check everything is set up correctly:

```bash
python module-0-setup/verify_setup.py
```

**Expected output:**
```
==================================================
  DevOps Chatbot - Setup Verification
==================================================

[1/6] Checking Python version...
  ✓ Python 3.11.5

[2/6] Checking required packages...
  ✓ openai installed
  ✓ langchain installed
  ✓ chromadb installed
  ✓ sentence-transformers installed
  ✓ python-dotenv installed
  ✓ rich installed

[3/6] Checking environment variables...
  ✓ .env file found
  ✓ OPENAI_API_KEY configured

[4/6] Testing LLM connection...
  ✓ Successfully connected to OpenAI

[5/6] Testing embeddings...
  ✓ Local embedding model loaded (all-MiniLM-L6-v2)

[6/6] Testing ChromaDB...
  ✓ ChromaDB working

==================================================
  ✓ All checks passed! You're ready to go!
==================================================
```

---

## Troubleshooting

### "Python not found" / "python is not recognized"

**Windows:**
```powershell
# Check if Python is in PATH
where python

# If not found, reinstall with Chocolatey
choco uninstall python311
choco install python311 -y

# Or add manually to PATH:
# 1. Search "Environment Variables" in Windows
# 2. Edit PATH
# 3. Add: C:\Python311\ and C:\Python311\Scripts\
```

**macOS:**
```bash
# Use python3 instead of python
python3 --version

# Create an alias (add to ~/.zshrc or ~/.bashrc)
alias python=python3
```

### "pip not found"

```bash
# Try pip3
pip3 --version

# Or use Python's pip module
python -m pip --version

# Install pip if missing
python -m ensurepip --upgrade
```

### "Permission denied" on macOS/Linux

```bash
# Don't use sudo with pip! Use virtual environment instead
# If you must install globally:
pip install --user package-name
```

### "uv: command not found"

**Windows:** Close and reopen PowerShell after installing uv

**macOS/Linux:** Add to your shell config:
```bash
# For bash (add to ~/.bashrc)
export PATH="$HOME/.cargo/bin:$PATH"

# For zsh (add to ~/.zshrc)
export PATH="$HOME/.cargo/bin:$PATH"

# Then reload
source ~/.bashrc  # or ~/.zshrc
```

### OpenAI API Errors

**"Invalid API key":**
- Double-check your key in `.env`
- Make sure there are no extra spaces
- Ensure the key starts with `sk-`

**"Insufficient quota":**
- Add payment method at platform.openai.com/account/billing
- Check your usage limits

**"Rate limit exceeded":**
- Wait a minute and try again
- You're making too many requests too quickly

### Virtual Environment Issues

**"Activate script not found":**
```bash
# Make sure you're in the project directory
cd devops-troubleshooting-chatbot

# Recreate the virtual environment
rm -rf .venv
uv venv
```

**Packages not found after activation:**
```bash
# Make sure you're in the virtual environment
# You should see (.venv) in your prompt

# Reinstall packages
uv pip install -r requirements.txt
```

---

## Glossary

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface - a way for programs to communicate |
| **API Key** | A secret password that authenticates you to an API service |
| **Chocolatey** | Package manager for Windows |
| **Homebrew** | Package manager for macOS |
| **LLM** | Large Language Model - AI that understands and generates text |
| **OpenAI** | Company that provides GPT models via API |
| **pip** | Python's default package installer |
| **Python** | Programming language used in this project |
| **RAG** | Retrieval-Augmented Generation - enhancing AI with external knowledge |
| **Token** | Unit of text (roughly 4 characters or 0.75 words) |
| **uv** | Fast, modern Python package manager |
| **venv** | Python virtual environment |

---

## Next Steps

Once all checks pass, you're ready to move on to:

**[Module 1: LLM Fundamentals →](../module-1-llm-basics/README.md)**

You'll learn:
- How LLMs work (tokens, context, temperature)
- Making your first API call
- Understanding responses and errors

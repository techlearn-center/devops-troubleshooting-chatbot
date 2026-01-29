# DevOps Troubleshooting Chatbot

Build an AI-powered chatbot that helps DevOps engineers troubleshoot common issues with Terraform, Kubernetes, Docker, and CI/CD pipelines.

```
+------------------------------------------------------------------+
|                                                                   |
|   "Error: resource already exists in state"                       |
|                                                                   |
|   DevOps Bot: This Terraform error occurs when you try to        |
|   create a resource that Terraform already tracks. Here's        |
|   how to fix it:                                                  |
|                                                                   |
|   1. Check current state: terraform state list                   |
|   2. Remove from state: terraform state rm <resource>            |
|   3. Or import existing: terraform import <resource> <id>        |
|                                                                   |
+------------------------------------------------------------------+
```

---

## What You'll Learn

By completing this challenge, you will understand:

| Concept | Description |
|---------|-------------|
| **LLM Basics** | How Large Language Models work and how to call them |
| **Prompt Engineering** | Writing effective prompts for technical responses |
| **RAG (Retrieval-Augmented Generation)** | Enhancing LLMs with external knowledge |
| **Embeddings** | Converting text to vectors for semantic search |
| **Vector Databases** | Storing and querying embeddings efficiently |
| **LangChain** | Building LLM applications with a popular framework |
| **DevOps Knowledge** | Common errors and solutions across tools |

---

## Prerequisites

Before starting, you should have:

- **Python 3.9+** installed
- **Basic Python knowledge** (functions, classes, packages)
- **Familiarity with DevOps tools** (Terraform, Docker, Kubernetes basics)
- **OpenAI API key** (or alternative: Ollama for local models)

No prior AI/ML experience required! We start from zero.

---

## Project Structure

```
devops-troubleshooting-chatbot/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── run.py                       # Auto-grading script
│
├── module-0-setup/              # Environment setup
│   ├── README.md
│   └── verify_setup.py
│
├── module-1-llm-basics/         # Understanding LLMs
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-2-simple-chatbot/     # Your first chatbot
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-3-rag-fundamentals/   # RAG concepts
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-4-knowledge-base/     # Building DevOps KB
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-5-vector-search/      # Embeddings & search
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-6-advanced-prompting/ # Expert techniques
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── module-7-complete-chatbot/   # Final project
│   ├── README.md
│   ├── exercises/
│   └── solutions/
│
├── knowledge-base/              # DevOps documentation
│   ├── terraform/
│   ├── kubernetes/
│   ├── docker/
│   └── cicd/
│
└── chatbot/                     # Final chatbot code
    ├── main.py
    ├── rag_engine.py
    └── prompts.py
```

---

## Learning Path

```
+------------------------------------------------------------------+
|                        LEARNING JOURNEY                           |
+------------------------------------------------------------------+
|                                                                   |
|  Module 0        Module 1        Module 2        Module 3        |
|  [Setup]    -->  [LLM Basics] -> [Simple Bot] -> [RAG Intro]     |
|  30 min          1 hour          1.5 hours       2 hours         |
|                                                                   |
|                         |                                         |
|                         v                                         |
|                                                                   |
|  Module 4        Module 5        Module 6        Module 7        |
|  [Knowledge] <-- [Vectors]   <-- [Prompting] <-- [Final Bot]     |
|  2 hours         2 hours         1.5 hours       3 hours         |
|                                                                   |
|                         |                                         |
|                         v                                         |
|                                                                   |
|              [YOU BUILT A DEVOPS AI CHATBOT!]                    |
|                                                                   |
+------------------------------------------------------------------+

Total Time: ~13 hours (self-paced)
```

---

## Quick Start

### Option 1: Using OpenAI API (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/techlearn-center/devops-troubleshooting-chatbot.git
cd devops-troubleshooting-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
.\venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Verify setup
python module-0-setup/verify_setup.py

# 6. Start learning!
cd module-1-llm-basics
```

### Option 2: Using Ollama (Free, Local)

```bash
# 1. Install Ollama from https://ollama.ai
# 2. Pull a model
ollama pull llama2

# 3. Follow the same steps as above, but set in .env:
USE_OLLAMA=true
OLLAMA_MODEL=llama2
```

---

## Module Overview

### Module 0: Setup & Prerequisites
**Time: 30 minutes**

- Install Python and dependencies
- Get OpenAI API key (or set up Ollama)
- Verify your environment works
- Understand the project structure

### Module 1: LLM Fundamentals
**Time: 1 hour**

- What are Large Language Models?
- How do they generate text?
- Making your first API call
- Understanding tokens and pricing
- Temperature and other parameters

### Module 2: Building a Simple Chatbot
**Time: 1.5 hours**

- Chat completions API
- System prompts and roles
- Maintaining conversation history
- Building a CLI chatbot
- Error handling

### Module 3: RAG Fundamentals
**Time: 2 hours**

- Why LLMs need external knowledge
- What is Retrieval-Augmented Generation?
- The RAG pipeline explained
- When to use RAG vs fine-tuning
- Hands-on: Simple RAG example

### Module 4: Building the DevOps Knowledge Base
**Time: 2 hours**

- Document loading strategies
- Text chunking and why it matters
- Chunk size optimization
- Metadata and filtering
- Building the DevOps docs collection

### Module 5: Vector Search & Embeddings
**Time: 2 hours**

- What are embeddings?
- Semantic similarity explained
- Vector databases (ChromaDB)
- Indexing and querying
- Relevance tuning

### Module 6: Advanced Prompting Techniques
**Time: 1.5 hours**

- Few-shot learning
- Chain-of-thought prompting
- Structured output (JSON)
- Prompt templates
- DevOps-specific prompts

### Module 7: Complete DevOps Chatbot
**Time: 3 hours**

- Putting it all together
- Adding error classification
- Multi-turn conversations
- Building the final interface
- Testing and evaluation

---

## The Final Chatbot

By the end, you'll have built a chatbot that can:

```python
# Example usage
from chatbot import DevOpsChatbot

bot = DevOpsChatbot()

# Terraform help
response = bot.ask("I'm getting 'Error: resource already exists' in Terraform")
print(response)
# Output: Detailed explanation + step-by-step fix

# Kubernetes troubleshooting
response = bot.ask("My pod is stuck in CrashLoopBackOff")
print(response)
# Output: Debugging steps + common causes

# Docker issues
response = bot.ask("Docker build fails with 'no space left on device'")
print(response)
# Output: Cleanup commands + prevention tips
```

---

## Grading

Run the auto-grader to check your progress:

```bash
# Check all modules
python run.py

# Check specific module
python run.py --module 1

# Verbose output
python run.py --verbose
```

### Scoring

| Module | Points |
|--------|--------|
| Module 0: Setup | 5 |
| Module 1: LLM Basics | 10 |
| Module 2: Simple Chatbot | 15 |
| Module 3: RAG Fundamentals | 15 |
| Module 4: Knowledge Base | 15 |
| Module 5: Vector Search | 15 |
| Module 6: Advanced Prompting | 10 |
| Module 7: Complete Chatbot | 15 |
| **Total** | **100** |
| **Passing Score** | **70%** |

---

## Key Concepts Explained

### What is RAG?

```
WITHOUT RAG:
===========
User Question --> LLM --> Answer (based only on training data)

Problem: LLM doesn't know about YOUR specific docs, recent updates,
         or domain-specific knowledge.


WITH RAG:
=========
                    +------------------+
                    | Knowledge Base   |
                    | (Your DevOps     |
                    |  Documentation)  |
                    +--------+---------+
                             |
                             | 2. Retrieve relevant docs
                             v
User Question ---> [Search] ---> Relevant Context
      |                              |
      |                              |
      +------------+  +--------------+
                   |  |
                   v  v
                  [LLM] ---> Answer (informed by YOUR docs!)

RAG = Retrieval-Augmented Generation
```

### What are Embeddings?

```
TEXT                          EMBEDDING (Vector)
====                          ==================

"Terraform error"    -->      [0.12, -0.45, 0.78, ..., 0.33]
                              (768 or 1536 dimensions)

"Kubernetes pod"     -->      [0.89, 0.12, -0.56, ..., 0.21]

"Infrastructure      -->      [0.15, -0.42, 0.75, ..., 0.31]
 as code problem"              (Similar to "Terraform error"!)


Similar meanings = Similar vectors = Can find related content!
```

### The RAG Pipeline

```
+------------------------------------------------------------------+
|                        RAG PIPELINE                               |
+------------------------------------------------------------------+

INDEXING (One-time setup):
==========================

Documents --> Chunk --> Embed --> Store in Vector DB
   |           |          |              |
   v           v          v              v
[Terraform  [Split    [Convert      [ChromaDB/
 Docs]       into      to vectors]   Pinecone]
             pieces]


QUERYING (Every user question):
===============================

Question --> Embed --> Search --> Get Top K --> Build Prompt --> LLM --> Answer
    |          |          |          |              |             |
    v          v          v          v              v             v
"Pod crash"  [0.2,    [Find      [Return       [Question +    [Generate
              0.8,     similar    relevant      Context]       response]
              ...]     vectors]   chunks]
```

---

## Tools & Technologies

| Tool | Purpose | Why We Use It |
|------|---------|---------------|
| **Python** | Programming language | Industry standard for AI/ML |
| **OpenAI API** | LLM provider | Best quality, easy to use |
| **LangChain** | LLM framework | Simplifies RAG pipelines |
| **ChromaDB** | Vector database | Free, local, easy setup |
| **Ollama** | Local LLMs | Free alternative to OpenAI |

---

## Troubleshooting This Project

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `OpenAI API error` | Check your API key in `.env` |
| `Rate limit exceeded` | Wait a minute, or use Ollama |
| `ChromaDB error` | Delete `chroma_db/` folder and retry |

---

## Resources

### Documentation
- [OpenAI API Docs](https://platform.openai.com/docs)
- [LangChain Docs](https://python.langchain.com/docs)
- [ChromaDB Docs](https://docs.trychroma.com/)

### DevOps References
- [Terraform Error Reference](https://developer.hashicorp.com/terraform/language/errors)
- [Kubernetes Troubleshooting](https://kubernetes.io/docs/tasks/debug/)
- [Docker Troubleshooting](https://docs.docker.com/config/daemon/)

---

## Contributing

Found an issue or want to add more DevOps knowledge?
1. Fork this repository
2. Create a feature branch
3. Submit a pull request

---

## License

MIT License - Feel free to use this for learning!

---

**Happy Learning! Build your DevOps AI Assistant!**

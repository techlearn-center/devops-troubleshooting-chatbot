# Module 1: LLM Fundamentals

Welcome to Module 1! In this module, you'll learn how Large Language Models (LLMs) work and how to interact with them through code. We'll explain everything from scratch - no prior AI knowledge required.

---

## Table of Contents

1. [Understanding AI and LLMs](#understanding-ai-and-llms)
   - [What is Artificial Intelligence?](#what-is-artificial-intelligence)
   - [What is Machine Learning?](#what-is-machine-learning)
   - [What is a Neural Network?](#what-is-a-neural-network)
   - [What is a Large Language Model?](#what-is-a-large-language-model)
2. [How LLMs Actually Work](#how-llms-actually-work)
   - [Training: Learning from Text](#training-learning-from-text)
   - [Inference: Predicting the Next Word](#inference-predicting-the-next-word)
3. [Understanding Tokens and Tokenization](#understanding-tokens-and-tokenization)
   - [What is a Token?](#what-is-a-token)
   - [Why Tokenization?](#why-tokenization)
   - [How Tokenization Works](#how-tokenization-works)
   - [Token Limits and Context Windows](#token-limits-and-context-windows)
4. [The Chat API Explained](#the-chat-api-explained)
   - [REST API Recap](#rest-api-recap)
   - [Chat Completions API Structure](#chat-completions-api-structure)
   - [Message Roles Explained](#message-roles-explained)
   - [The Power of System Prompts](#the-power-of-system-prompts)
5. [Controlling LLM Output](#controlling-llm-output)
   - [Temperature: Creativity vs Consistency](#temperature-creativity-vs-consistency)
   - [Max Tokens: Controlling Length](#max-tokens-controlling-length)
   - [Other Parameters](#other-parameters)
6. [Understanding Costs](#understanding-costs)
7. [Handling Errors](#handling-errors)
8. [Hands-On Exercises](#hands-on-exercises)
9. [Key Takeaways](#key-takeaways)

---

## Understanding AI and LLMs

Let's build up from the basics to understand what LLMs really are.

### What is Artificial Intelligence?

**Artificial Intelligence (AI)** is when computers do things that normally require human intelligence.

```
EXAMPLES OF AI
==============

Traditional Programming:          AI/Machine Learning:
─────────────────────────        ─────────────────────
IF email contains "viagra"       Show the AI 10,000 spam emails
   THEN mark as spam             and 10,000 good emails.
                                 It LEARNS what spam looks like.

Human writes ALL the rules       Computer LEARNS the rules
Very rigid, limited              Flexible, can generalize
```

### What is Machine Learning?

**Machine Learning (ML)** is a type of AI where computers learn patterns from data instead of being explicitly programmed.

```
TRADITIONAL PROGRAMMING VS MACHINE LEARNING
===========================================

Traditional:
┌────────────┐     ┌─────────────┐     ┌────────────┐
│   Rules    │ +   │    Data     │  →  │   Output   │
│ (by human) │     │             │     │            │
└────────────┘     └─────────────┘     └────────────┘

Machine Learning:
┌────────────┐     ┌─────────────┐     ┌────────────┐
│   Data     │ +   │   Answers   │  →  │   Rules    │
│            │     │ (examples)  │     │ (learned)  │
└────────────┘     └─────────────┘     └────────────┘

Example: Spam Detection
- Traditional: Human writes 1000 rules about spam
- ML: Show computer 100,000 emails labeled "spam" or "not spam"
      Computer figures out the patterns itself
```

### What is a Neural Network?

A **Neural Network** is a type of ML model inspired by how the human brain works. It's made of layers of connected "neurons" that process information.

```
NEURAL NETWORK SIMPLIFIED
=========================

Input Layer      Hidden Layers       Output Layer
(what goes in)   (processing)        (result)

    ○─────────────○
   /│\           /│\
  / │ \         / │ \
 ○──○──○───────○──○──○─────────────────○
  \ │ /         \ │ /                 (answer)
   \│/           \│/
    ○─────────────○

Each connection has a "weight" (importance)
Training adjusts these weights to get better answers
```

**You don't need to understand the math!** Just know:
- Neural networks learn patterns from examples
- More layers = can learn more complex patterns
- "Deep learning" = neural networks with many layers

### What is a Large Language Model?

A **Large Language Model (LLM)** is a very large neural network trained on massive amounts of text to understand and generate human language.

```
LLM SIZE COMPARISON
===================

Model              Parameters        Training Data
──────────────────────────────────────────────────
GPT-2 (2019)       1.5 billion      40GB of text
GPT-3 (2020)       175 billion      570GB of text
GPT-4 (2023)       ~1.7 trillion*   ~13 trillion tokens
Llama 2 (2023)     7-70 billion     2 trillion tokens

*estimated, not officially confirmed

What's a "parameter"?
- A parameter is a number the model learned during training
- More parameters = more capacity to learn patterns
- GPT-4 has roughly 1,700,000,000,000 learned numbers!
```

**Key insight:** LLMs don't "know" things like humans do. They've learned statistical patterns about how words relate to each other.

---

## How LLMs Actually Work

### Training: Learning from Text

During training, the LLM reads billions of text examples and learns patterns.

```
TRAINING PROCESS (SIMPLIFIED)
=============================

Step 1: Show the model incomplete text
        "The Kubernetes pod is in a _____ state"

Step 2: Model guesses: "running" (wrong! it was "pending")

Step 3: Adjust the model's parameters slightly
        to make "pending" more likely next time

Step 4: Repeat billions of times with different text

After training:
- Model has learned grammar, facts, reasoning patterns
- It learned from internet, books, code, documentation
- Knowledge "frozen" at training cutoff date
```

### Inference: Predicting the Next Word

When you use an LLM, it predicts one word (token) at a time:

```
HOW THE LLM GENERATES A RESPONSE
================================

Your prompt: "What is a Docker container?"

LLM's process (simplified):
┌─────────────────────────────────────────────────────────────┐
│ Step 1: "What is a Docker container?"                       │
│         → Predict next token: "A"                           │
│                                                             │
│ Step 2: "What is a Docker container? A"                     │
│         → Predict next token: " Docker"                     │
│                                                             │
│ Step 3: "What is a Docker container? A Docker"              │
│         → Predict next token: " container"                  │
│                                                             │
│ Step 4: "What is a Docker container? A Docker container"    │
│         → Predict next token: " is"                         │
│                                                             │
│ ... continues until complete ...                            │
│                                                             │
│ Final: "A Docker container is a lightweight, standalone,    │
│         executable package that includes everything needed  │
│         to run a piece of software..."                      │
└─────────────────────────────────────────────────────────────┘

Each step: Model looks at ALL previous text to predict the next word
This is why it's called "autoregressive" - it builds on its own output
```

**Important:** The LLM doesn't retrieve facts from a database. It predicts what text would likely come next based on patterns it learned during training.

---

## Understanding Tokens and Tokenization

This is one of the most important concepts to understand!

### What is a Token?

A **token** is a chunk of text that the LLM processes as a single unit. Tokens are NOT the same as words!

```
TOKENIZATION EXAMPLES
=====================

Text: "Hello world"
Tokens: ["Hello", " world"]
Count: 2 tokens

Text: "Kubernetes"
Tokens: ["Kub", "ernetes"]
Count: 2 tokens (long words get split!)

Text: "CrashLoopBackOff"
Tokens: ["Crash", "Loop", "Back", "Off"]
Count: 4 tokens

Text: "   " (three spaces)
Tokens: ["   "]
Count: 1 token

Text: "don't"
Tokens: ["don", "'t"]
Count: 2 tokens
```

### Why Tokenization?

**Why not just use words or characters?**

```
TOKENIZATION TRADEOFFS
======================

Option 1: Characters (a, b, c, ...)
┌─────────────────────────────────────────┐
│ "Hello" → ['H', 'e', 'l', 'l', 'o']     │
│ ✗ Too many tokens (slow, expensive)      │
│ ✗ Hard to learn word meanings            │
└─────────────────────────────────────────┘

Option 2: Words (hello, world, kubernetes, ...)
┌─────────────────────────────────────────┐
│ "Hello" → ['Hello']                      │
│ ✗ Vocabulary would be HUGE               │
│ ✗ Can't handle new words (misspellings)  │
│ ✗ Different forms: run, running, ran     │
└─────────────────────────────────────────┘

Option 3: Subwords/Tokens (best of both!)
┌─────────────────────────────────────────┐
│ "Hello" → ['Hello']                      │
│ "unhappiness" → ['un', 'happiness']      │
│ ✓ Reasonable vocabulary size (~50,000)   │
│ ✓ Can handle ANY text                    │
│ ✓ Balances efficiency and flexibility    │
└─────────────────────────────────────────┘
```

### How Tokenization Works

LLMs use algorithms like **BPE (Byte Pair Encoding)** to decide how to split text:

```
BPE TOKENIZATION (SIMPLIFIED)
=============================

Training the tokenizer:
1. Start with all individual characters
2. Find the most common pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat thousands of times

Example building vocabulary:
  Start:    ['t', 'h', 'e', ' ', 'c', 'a', 't']
  Merge 1:  ['th', 'e', ' ', 'c', 'a', 't']     (t+h was common)
  Merge 2:  ['the', ' ', 'c', 'a', 't']          (th+e was common)
  Merge 3:  ['the', ' ', 'ca', 't']              (c+a was common)
  Merge 4:  ['the', ' ', 'cat']                  (ca+t was common)
  ...

Result: Common words become single tokens
        Rare words get split into pieces
```

**Practical rule of thumb:**
- 1 token ≈ 4 characters in English
- 1 token ≈ 0.75 words
- 100 tokens ≈ 75 words

### Token Limits and Context Windows

Every LLM has a maximum number of tokens it can process at once - this is called the **context window**.

```
CONTEXT WINDOW VISUALIZATION
============================

┌─────────────────────────────────────────────────────────┐
│                    CONTEXT WINDOW                        │
│                    (e.g., 4096 tokens)                  │
│                                                         │
│  ┌─────────────────┐  ┌──────────────────────────────┐ │
│  │  System Prompt  │  │        User Messages +       │ │
│  │   (200 tokens)  │  │     Assistant Responses      │ │
│  │                 │  │       (3000 tokens)          │ │
│  └─────────────────┘  └──────────────────────────────┘ │
│                                                         │
│  Remaining for new response: ~896 tokens               │
│                                                         │
└─────────────────────────────────────────────────────────┘

If you exceed the limit:
- Oldest messages get "forgotten" (truncated)
- Or you get an error
```

**Context Window by Model:**

| Model | Context Window | Roughly |
|-------|----------------|---------|
| GPT-3.5-turbo | 4,096 tokens | ~3,000 words |
| GPT-3.5-turbo-16k | 16,384 tokens | ~12,000 words |
| GPT-4 | 8,192 tokens | ~6,000 words |
| GPT-4-turbo | 128,000 tokens | ~96,000 words |
| GPT-4o | 128,000 tokens | ~96,000 words |
| Llama 2 | 4,096 tokens | ~3,000 words |
| Mistral | 8,192 tokens | ~6,000 words |

### Counting Tokens in Python

```python
# Install: pip install tiktoken
import tiktoken

# Get the tokenizer for a specific model
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Tokenize text
text = "Kubernetes pod in CrashLoopBackOff"
tokens = encoding.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")  # [42, 11, ...]  (numbers)
print(f"Token count: {len(tokens)}")  # 6

# See the actual token strings
token_strings = [encoding.decode([t]) for t in tokens]
print(f"Token strings: {token_strings}")
# ['K', 'ubernetes', ' pod', ' in', ' Crash', 'Loop', 'Back', 'Off']
```

---

## The Chat API Explained

### REST API Recap

Remember from Module 0: An API is how your code talks to a service. The OpenAI API is a **REST API** - you send HTTP requests and get responses.

```
REST API COMMUNICATION
======================

Your Python Code                        OpenAI Servers
      │                                       │
      │  ───── HTTP POST Request ─────────►  │
      │        URL: api.openai.com/v1/...    │
      │        Headers: Authorization        │
      │        Body: JSON with your prompt   │
      │                                       │
      │  ◄──── HTTP Response ──────────────  │
      │        Status: 200 OK                │
      │        Body: JSON with AI response   │
      │                                       │


HTTP Request Structure:
┌─────────────────────────────────────────────────────────┐
│ POST /v1/chat/completions HTTP/1.1                      │
│ Host: api.openai.com                                    │
│ Authorization: Bearer sk-your-api-key-here              │
│ Content-Type: application/json                          │
│                                                         │
│ {                                                       │
│   "model": "gpt-3.5-turbo",                            │
│   "messages": [                                         │
│     {"role": "user", "content": "Hello!"}              │
│   ]                                                     │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
```

### Chat Completions API Structure

The OpenAI Python library handles the HTTP details for you:

```python
from openai import OpenAI

# Create client (reads OPENAI_API_KEY from environment)
client = OpenAI()

# Make an API call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",           # Which model to use
    messages=[                        # The conversation
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Docker?"}
    ],
    temperature=0.7,                  # Creativity (0-2)
    max_tokens=500                    # Max response length
)

# Get the response text
answer = response.choices[0].message.content
print(answer)
```

**Response Structure:**

```python
# What the API returns:
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-3.5-turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Docker is a platform for..."  # ← The answer!
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,      # Tokens in your input
        "completion_tokens": 150,  # Tokens in the response
        "total_tokens": 175        # Total (for billing)
    }
}
```

### Message Roles Explained

The Chat API uses three roles to structure conversations:

```
MESSAGE ROLES
=============

┌─────────────────────────────────────────────────────────────────┐
│ SYSTEM: Sets the AI's personality and rules                     │
│ ────────────────────────────────────────────────────────────── │
│ "You are a DevOps expert. Be concise. Use code examples."       │
│                                                                 │
│ Think of it as: Instructions for the AI before the conversation │
│ The user typically doesn't see this                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ USER: Human's messages                                          │
│ ────────────────────────────────────────────────────────────── │
│ "How do I fix a CrashLoopBackOff error?"                       │
│                                                                 │
│ Think of it as: Questions/requests from the person              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ ASSISTANT: AI's responses                                       │
│ ────────────────────────────────────────────────────────────── │
│ "CrashLoopBackOff means the container keeps crashing..."       │
│                                                                 │
│ Think of it as: Previous AI responses (for context)             │
│ Include these to maintain conversation history                  │
└─────────────────────────────────────────────────────────────────┘
```

**Multi-turn Conversation Example:**

```python
messages = [
    # System prompt (always first)
    {"role": "system", "content": "You are a Kubernetes expert."},

    # First exchange
    {"role": "user", "content": "What is a pod?"},
    {"role": "assistant", "content": "A pod is the smallest deployable unit..."},

    # Second exchange (AI remembers context!)
    {"role": "user", "content": "How many containers can it have?"},
    # AI knows "it" refers to "pod" from the conversation
]
```

### The Power of System Prompts

The system prompt dramatically affects the AI's behavior:

```python
# EXAMPLE 1: Generic assistant
system_prompt = "You are a helpful assistant."
# Response: General, may be verbose, unfocused

# EXAMPLE 2: DevOps expert
system_prompt = """You are an expert DevOps engineer with 10 years of experience.
You specialize in:
- Kubernetes and container orchestration
- Terraform and Infrastructure as Code
- CI/CD pipelines (GitHub Actions, Jenkins)
- Cloud platforms (AWS, GCP, Azure)

When answering:
1. Be concise and practical
2. Include working code examples
3. Explain the "why" not just the "how"
4. Mention common pitfalls to avoid
5. Suggest best practices
"""
# Response: Focused, expert-level, with code examples

# EXAMPLE 3: Troubleshooting specialist
system_prompt = """You are a DevOps troubleshooting assistant.

When users describe errors:
1. First identify the likely cause
2. Ask clarifying questions if needed
3. Provide step-by-step debugging commands
4. Explain what each command does
5. Suggest preventive measures

Always format commands in code blocks.
Be direct and action-oriented.
"""
# Response: Structured troubleshooting with commands
```

---

## Controlling LLM Output

### Temperature: Creativity vs Consistency

**Temperature** controls how "random" the model's output is:

```
TEMPERATURE VISUALIZATION
=========================

Prompt: "The best way to learn Kubernetes is"

Temperature = 0.0 (Deterministic)
────────────────────────────────
Always picks the most likely next word.
Same input → Same output every time.

  "to start with the official documentation and tutorials."
  "to start with the official documentation and tutorials."
  "to start with the official documentation and tutorials."
  (identical every time)


Temperature = 0.7 (Balanced - Default)
────────────────────────────────────────
Mix of likely words with some variety.

  "to start with the official documentation and hands-on practice."
  "by following structured tutorials and building real projects."
  "through hands-on experience with minikube or kind."
  (varied but sensible)


Temperature = 1.5 (Creative/Random)
───────────────────────────────────
More willing to pick less likely words.

  "to embrace the chaos of container orchestration!"
  "by diving deep into the YAML wilderness."
  "actually by breaking things in production... kidding!"
  (more creative, may be less accurate)
```

**When to use what:**

| Temperature | Use Case |
|-------------|----------|
| 0.0 - 0.3 | Technical docs, code generation, troubleshooting |
| 0.5 - 0.7 | General conversation, explanations |
| 0.8 - 1.2 | Creative writing, brainstorming |
| 1.3+ | Very creative tasks (use with caution) |

**For DevOps troubleshooting:** Use **0.1 - 0.3** for consistent, accurate answers.

### Max Tokens: Controlling Length

**max_tokens** limits how long the response can be:

```python
# Short answer (good for yes/no or brief responses)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    max_tokens=50  # ~35-40 words
)

# Medium answer (good for explanations)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    max_tokens=500  # ~375 words
)

# Long answer (good for detailed tutorials)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    max_tokens=2000  # ~1500 words
)
```

**Important:** If the response gets cut off mid-sentence, you'll see `finish_reason: "length"` instead of `"stop"`.

### Other Parameters

| Parameter | What It Does | Default | Recommendation |
|-----------|--------------|---------|----------------|
| `temperature` | Randomness | 1.0 | 0.1-0.3 for technical |
| `max_tokens` | Response length limit | Model's max | 500-1000 |
| `top_p` | Nucleus sampling | 1.0 | Leave at default |
| `frequency_penalty` | Reduce repetition | 0.0 | 0.0-0.5 |
| `presence_penalty` | Encourage new topics | 0.0 | Leave at 0 |
| `stop` | Stop sequences | None | Custom stop words |

---

## Understanding Costs

### How Pricing Works

OpenAI charges per token, with different rates for input vs output:

```
PRICING CALCULATION
===================

Your prompt (input):      500 tokens  × $0.0005/1K = $0.00025
AI response (output):     300 tokens  × $0.0015/1K = $0.00045
                         ─────────────────────────────────────
Total cost for this call:                            $0.00070

In other words:
- ~1,400 simple requests per $1 with GPT-3.5-turbo
- ~100 requests per $1 with GPT-4-turbo
```

### Current Pricing (2024-2025)

| Model | Input Cost | Output Cost | Best For |
|-------|------------|-------------|----------|
| gpt-3.5-turbo | $0.0005/1K | $0.0015/1K | Simple tasks, learning |
| gpt-4o-mini | $0.00015/1K | $0.0006/1K | Good balance |
| gpt-4-turbo | $0.01/1K | $0.03/1K | Complex reasoning |
| gpt-4o | $0.005/1K | $0.015/1K | Best quality |

### Cost-Saving Tips

1. **Use gpt-3.5-turbo for learning** - 20x cheaper than GPT-4
2. **Keep prompts concise** - Don't add unnecessary context
3. **Set max_tokens** - Prevent runaway responses
4. **Use Ollama for experimentation** - Completely free
5. **Set usage limits** - In OpenAI dashboard

---

## Handling Errors

When working with APIs, things can go wrong. Here's how to handle common errors:

### Common API Errors

```python
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import time

client = OpenAI()

def ask_llm_safely(question: str) -> str:
    """
    Make an API call with proper error handling.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": question}]
        )
        return response.choices[0].message.content

    except AuthenticationError:
        # API key is invalid or missing
        return "ERROR: Invalid API key. Check your OPENAI_API_KEY in .env"

    except RateLimitError:
        # Too many requests, wait and retry
        print("Rate limited, waiting 60 seconds...")
        time.sleep(60)
        return ask_llm_safely(question)  # Retry

    except APIError as e:
        # General API error
        return f"ERROR: API error - {e.message}"

    except Exception as e:
        # Catch-all for unexpected errors
        return f"ERROR: Unexpected error - {str(e)}"
```

### Error Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `AuthenticationError` | Invalid/missing API key | Check `.env` file |
| `RateLimitError` | Too many requests | Wait and retry |
| `InvalidRequestError` | Bad parameters | Check model name, message format |
| `APIConnectionError` | Network issue | Check internet connection |
| `Timeout` | Request took too long | Retry or increase timeout |
| `InsufficientQuotaError` | Out of credits | Add payment method |

---

## Using Ollama (Free Alternative)

All exercises work with Ollama too! Here's how to use it:

### Ollama Setup

```bash
# Install Ollama (see Module 0)
# Then pull a model:
ollama pull llama2
# or
ollama pull mistral

# Start the server (runs on localhost:11434)
ollama serve
```

### Ollama API Calls

```python
import requests
import json

def ask_ollama(question: str, model: str = "llama2") -> str:
    """
    Ask a question using Ollama (local LLM).

    Args:
        question: The question to ask
        model: Ollama model name (llama2, mistral, etc.)

    Returns:
        The model's response
    """
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False  # Get complete response at once
        }
    )

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"Error: {response.status_code}"

# Usage
answer = ask_ollama("What is a Docker container?")
print(answer)
```

### Unified Interface (Works with Both)

```python
import os
from dotenv import load_dotenv

load_dotenv()

def ask_llm(question: str, system_prompt: str = None) -> str:
    """
    Ask an LLM using either OpenAI or Ollama based on .env config.
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # Use Ollama
        import requests

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        response = requests.post(
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "messages": messages,
                "stream": False
            }
        )
        return response.json()["message"]["content"]

    else:
        # Use OpenAI
        from openai import OpenAI
        client = OpenAI()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages
        )
        return response.choices[0].message.content
```

---

## Hands-On Exercises

### Exercise 1: Your First API Call

Create `exercises/ex1_first_call.py`:

```python
#!/usr/bin/env python3
"""
Exercise 1: Make your first LLM API call
========================================

GOAL: Send a message to an LLM and get a response.

INSTRUCTIONS:
1. Make sure your .env file is set up (see Module 0)
2. Complete the ask_llm() function below
3. Run: python exercises/ex1_first_call.py

EXPECTED OUTPUT:
    Question: What does a Kubernetes pod do? Answer in one sentence.
    Answer: A Kubernetes pod is the smallest deployable unit that...
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def ask_llm(question: str) -> str:
    """
    Send a question to the LLM and return the response.

    This function works with both OpenAI and Ollama depending on
    what's configured in your .env file.

    Args:
        question: The question to ask the LLM

    Returns:
        The LLM's response as a string
    """
    # Check which backend to use
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # ─────────────────────────────────────────────────────
        # OLLAMA VERSION (free, runs locally)
        # ─────────────────────────────────────────────────────
        import requests

        # Ollama uses a REST API at localhost:11434
        # The /api/chat endpoint handles chat completions
        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "messages": [
                    {"role": "user", "content": question}
                ],
                "stream": False  # Get complete response at once
            }
        )

        # Extract the response text
        return response.json()["message"]["content"]

    else:
        # ─────────────────────────────────────────────────────
        # OPENAI VERSION (paid, cloud-based)
        # ─────────────────────────────────────────────────────
        from openai import OpenAI

        # Create the OpenAI client
        # It automatically reads OPENAI_API_KEY from environment
        client = OpenAI()

        # Make the API call
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "user", "content": question}
            ]
        )

        # Extract the response text from the API response
        # response.choices[0] = first choice (we only asked for one)
        # .message.content = the actual text response
        return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# MAIN: Test your function
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A simple test question
    question = "What does a Kubernetes pod do? Answer in one sentence."

    print(f"Question: {question}")
    print("Asking LLM...")

    try:
        response = ask_llm(question)
        print(f"Answer: {response}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("- Is your .env file configured?")
        print("- If using OpenAI: Is your API key valid?")
        print("- If using Ollama: Is the server running? (ollama serve)")
```

### Exercise 2: System Prompts

Create `exercises/ex2_system_prompts.py`:

```python
#!/usr/bin/env python3
"""
Exercise 2: Experiment with System Prompts
==========================================

GOAL: See how different system prompts change LLM behavior.

INSTRUCTIONS:
1. Run this script and observe the different responses
2. Create your own custom system prompt for DevOps troubleshooting

WHAT YOU'LL LEARN:
- System prompts dramatically change AI behavior
- Good prompts = Better responses
- The AI follows instructions in the system prompt
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv

load_dotenv()


def ask_with_system_prompt(system_prompt: str, user_question: str) -> str:
    """
    Ask the LLM a question with a specific system prompt.

    The system prompt sets the "personality" and behavior of the AI.
    It's like giving instructions to an assistant before they start working.

    Args:
        system_prompt: Instructions for how the AI should behave
        user_question: The actual question from the user

    Returns:
        The AI's response
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    # Build the messages list with system prompt and user question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    if use_ollama:
        import requests

        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "messages": messages,
                "stream": False
            }
        )
        return response.json()["message"]["content"]
    else:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            temperature=0.3  # Low temperature for consistent comparison
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    # The question we'll ask with different system prompts
    question = "How do I fix 'Error: resource already exists' in Terraform?"

    print("=" * 60)
    print("TESTING DIFFERENT SYSTEM PROMPTS")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    # ─────────────────────────────────────────────────────────
    # TEST 1: Generic Assistant
    # ─────────────────────────────────────────────────────────
    print("-" * 60)
    print("TEST 1: Generic Assistant")
    print("-" * 60)
    generic_prompt = "You are a helpful assistant."
    response = ask_with_system_prompt(generic_prompt, question)
    print(response)
    print()

    # ─────────────────────────────────────────────────────────
    # TEST 2: DevOps Expert
    # ─────────────────────────────────────────────────────────
    print("-" * 60)
    print("TEST 2: DevOps Expert")
    print("-" * 60)
    expert_prompt = """You are an expert DevOps engineer with 10 years of experience.
You specialize in Terraform and Infrastructure as Code.
When answering:
1. Provide step-by-step solutions
2. Include actual commands to run
3. Explain why each step is needed
4. Warn about potential pitfalls"""
    response = ask_with_system_prompt(expert_prompt, question)
    print(response)
    print()

    # ─────────────────────────────────────────────────────────
    # TEST 3: Concise Troubleshooter
    # ─────────────────────────────────────────────────────────
    print("-" * 60)
    print("TEST 3: Concise Troubleshooter")
    print("-" * 60)
    concise_prompt = """You are a DevOps troubleshooting bot.
Rules:
- Maximum 3 bullet points
- Each bullet: one command or action
- No explanations unless critical
- Code blocks for commands"""
    response = ask_with_system_prompt(concise_prompt, question)
    print(response)
    print()

    # ─────────────────────────────────────────────────────────
    # YOUR TURN: Create a custom prompt!
    # ─────────────────────────────────────────────────────────
    print("-" * 60)
    print("TEST 4: Your Custom Prompt")
    print("-" * 60)

    # TODO: Create your own system prompt for DevOps troubleshooting
    # Think about:
    # - What expertise should the AI have?
    # - How should responses be formatted?
    # - What should the AI always/never do?

    your_prompt = """You are a friendly DevOps mentor teaching beginners.
When helping:
1. Explain concepts simply, assuming no prior knowledge
2. Use analogies to explain technical concepts
3. Show the command AND explain what it does
4. Encourage the learner"""

    response = ask_with_system_prompt(your_prompt, question)
    print(response)
```

### Exercise 3: Temperature Experimentation

Create `exercises/ex3_parameters.py`:

```python
#!/usr/bin/env python3
"""
Exercise 3: Experiment with LLM Parameters
==========================================

GOAL: Understand how temperature and max_tokens affect output.

WHAT YOU'LL LEARN:
- Temperature 0 = Same output every time
- Temperature 1+ = More varied/creative
- max_tokens limits response length
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv

load_dotenv()


def ask_with_params(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    Ask the LLM with specific parameters.

    Args:
        question: The question to ask
        temperature: Controls randomness (0.0 = deterministic, 2.0 = very random)
        max_tokens: Maximum length of the response

    Returns:
        The LLM's response
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        import requests

        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama2"),
                "messages": [{"role": "user", "content": question}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens  # Ollama's max_tokens
                }
            }
        )
        return response.json()["message"]["content"]
    else:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": question}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    question = "In one sentence, what is the best way to learn Kubernetes?"

    print("=" * 60)
    print("TEMPERATURE COMPARISON")
    print("=" * 60)
    print(f"Question: {question}\n")

    # ─────────────────────────────────────────────────────────
    # Test different temperatures
    # ─────────────────────────────────────────────────────────
    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"--- Temperature: {temp} ---")
        response = ask_with_params(question, temperature=temp, max_tokens=100)
        print(response)
        print()

    # ─────────────────────────────────────────────────────────
    # Test consistency at temperature 0
    # ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("CONSISTENCY TEST (Temperature = 0)")
    print("=" * 60)
    print("Running same prompt 3 times:\n")

    for i in range(3):
        print(f"Run {i+1}:")
        response = ask_with_params(question, temperature=0.0, max_tokens=100)
        print(response)
        print()

    print("Notice: With temperature=0, outputs should be identical!")
```

### Exercise 4: Token Counting

Create `exercises/ex4_token_counting.py`:

```python
#!/usr/bin/env python3
"""
Exercise 4: Count Tokens and Estimate Costs
===========================================

GOAL: Understand how tokens work and how to estimate API costs.

PREREQUISITES:
    pip install tiktoken

WHAT YOU'LL LEARN:
- How text gets converted to tokens
- How to count tokens before making API calls
- How to estimate costs
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv

load_dotenv()

# tiktoken is OpenAI's tokenizer library
try:
    import tiktoken
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    sys.exit(1)


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to tokenize
        model: The model (different models may tokenize differently)

    Returns:
        Number of tokens
    """
    # Get the encoding (tokenizer) for this model
    encoding = tiktoken.encoding_for_model(model)

    # Encode the text and count tokens
    tokens = encoding.encode(text)

    return len(tokens)


def show_tokens(text: str, model: str = "gpt-3.5-turbo"):
    """
    Show how text gets split into tokens.
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    print(f"Text: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Show each token as text
    token_strings = [encoding.decode([t]) for t in tokens]
    print(f"Tokens: {token_strings}")
    print()


def estimate_cost(
    input_text: str,
    expected_output_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> dict:
    """
    Estimate the cost of an API call.

    Args:
        input_text: Your prompt
        expected_output_tokens: Estimated response length
        model: The model to use

    Returns:
        Dictionary with cost breakdown
    """
    # Pricing per 1000 tokens (as of 2024)
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    }

    # Count input tokens
    input_tokens = count_tokens(input_text, model)

    # Get pricing for this model
    model_pricing = pricing.get(model, pricing["gpt-3.5-turbo"])

    # Calculate costs
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (expected_output_tokens / 1000) * model_pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": expected_output_tokens,
        "total_tokens": input_tokens + expected_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "requests_per_dollar": 1 / total_cost if total_cost > 0 else float('inf')
    }


if __name__ == "__main__":
    print("=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)
    print()

    # Show how different texts get tokenized
    examples = [
        "Hello world",
        "Kubernetes",
        "CrashLoopBackOff",
        "kubectl get pods -n default",
        "The DevOps engineer fixed the bug",
    ]

    for text in examples:
        show_tokens(text)

    # ─────────────────────────────────────────────────────────
    # Cost estimation example
    # ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)
    print()

    # A realistic DevOps troubleshooting prompt
    prompt = """You are an expert DevOps engineer.

A user reports: "My Kubernetes pod is stuck in CrashLoopBackOff status."

Please provide:
1. The most likely causes
2. Diagnostic commands to run
3. Step-by-step resolution
4. Prevention tips"""

    print(f"Prompt ({count_tokens(prompt)} tokens):")
    print(prompt)
    print()

    # Estimate for different models
    models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4-turbo"]

    print("Cost estimates (assuming 500 token response):")
    print("-" * 60)

    for model in models:
        estimate = estimate_cost(prompt, 500, model)
        print(f"\n{model}:")
        print(f"  Input:  {estimate['input_tokens']} tokens = ${estimate['input_cost']:.6f}")
        print(f"  Output: {estimate['output_tokens']} tokens = ${estimate['output_cost']:.6f}")
        print(f"  Total:  ${estimate['total_cost']:.6f}")
        print(f"  ≈ {estimate['requests_per_dollar']:.0f} requests per $1")
```

---

## Verifying Your Exercises

After completing each exercise, verify it works:

```bash
# Activate your virtual environment first!
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows

# Run Exercise 1
python module-1-llm-basics/exercises/ex1_first_call.py

# Run Exercise 2
python module-1-llm-basics/exercises/ex2_system_prompts.py

# Run Exercise 3
python module-1-llm-basics/exercises/ex3_parameters.py

# Run Exercise 4
python module-1-llm-basics/exercises/ex4_token_counting.py
```

---

## Key Takeaways

1. **LLMs predict text** - They don't "know" facts; they predict what text would likely come next based on patterns learned during training.

2. **Tokens ≠ Words** - Text is split into tokens (subwords). ~1 token ≈ 4 characters ≈ 0.75 words.

3. **Context window is limited** - Models can only "see" a certain number of tokens at once.

4. **System prompts are powerful** - They dramatically change AI behavior. Good prompts = Better responses.

5. **Temperature controls randomness** - Use low values (0.1-0.3) for technical accuracy.

6. **Tokens = Money** - You pay per token. Be mindful of prompt length.

7. **Both OpenAI and Ollama work** - Use Ollama for free experimentation, OpenAI for production quality.

---

## What's Next?

In **Module 2**, you'll build a complete chatbot with:
- Conversation history (multi-turn chat)
- Interactive CLI interface
- Streaming responses
- Error handling

```bash
cd ../module-2-simple-chatbot
```

Continue to: [Module 2: Simple Chatbot →](../module-2-simple-chatbot/README.md)

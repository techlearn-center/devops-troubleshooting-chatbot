# Module 2: Building a Simple Chatbot

Welcome to Module 2! Now that you understand how LLMs work, let's build a real chatbot that can have multi-turn conversations. We'll explain every concept from scratch.

---

## Table of Contents

1. [What is a Chatbot?](#what-is-a-chatbot)
   - [Single Q&A vs Chatbot](#single-qa-vs-chatbot)
   - [What Makes a Good Chatbot?](#what-makes-a-good-chatbot)
2. [Understanding Conversation History](#understanding-conversation-history)
   - [Why History Matters](#why-history-matters)
   - [How to Maintain History](#how-to-maintain-history)
3. [Python Classes Explained](#python-classes-explained)
   - [What is a Class?](#what-is-a-class)
   - [Why Use Classes for Chatbots?](#why-use-classes-for-chatbots)
4. [Building the CLI Interface](#building-the-cli-interface)
   - [What is a CLI?](#what-is-a-cli)
   - [The Rich Library](#the-rich-library)
5. [Streaming Responses](#streaming-responses)
   - [What is Streaming?](#what-is-streaming)
   - [Why Use Streaming?](#why-use-streaming)
6. [Error Handling](#error-handling)
   - [What Can Go Wrong?](#what-can-go-wrong)
   - [Exponential Backoff](#exponential-backoff)
7. [Context Window Management](#context-window-management)
8. [Hands-On Exercises](#hands-on-exercises)
9. [Key Takeaways](#key-takeaways)

---

## What is a Chatbot?

### Single Q&A vs Chatbot

In Module 1, we made single API calls - ask a question, get an answer. A **chatbot** is different: it maintains a conversation over multiple exchanges.

```
SINGLE Q&A (Module 1)
=====================

Call 1: "What is Terraform?" → "Terraform is an IaC tool..."
Call 2: "How do I install it?" → "Install what? Please specify."
                                   ↑
                                   PROBLEM: No memory of previous question!


CHATBOT (Module 2)
==================

Exchange 1:
  User: "What is Terraform?"
  Bot:  "Terraform is an IaC tool for provisioning infrastructure..."

Exchange 2:
  User: "How do I install it?"
  Bot:  "To install Terraform:        ← Knows "it" = Terraform!
         1. Download from terraform.io
         2. Unzip the binary
         3. Add to your PATH..."

Exchange 3:
  User: "What about on Mac?"
  Bot:  "On Mac, you can also use:    ← Remembers the topic!
         brew install terraform"
```

### What Makes a Good Chatbot?

```
GOOD CHATBOT CHARACTERISTICS
============================

1. REMEMBERS CONTEXT
   ┌─────────────────────────────────────────┐
   │ Tracks conversation history             │
   │ Understands pronouns ("it", "that")     │
   │ Builds on previous answers              │
   └─────────────────────────────────────────┘

2. HANDLES ERRORS GRACEFULLY
   ┌─────────────────────────────────────────┐
   │ Doesn't crash on network issues         │
   │ Retries failed requests                 │
   │ Shows helpful error messages            │
   └─────────────────────────────────────────┘

3. PROVIDES GOOD USER EXPERIENCE
   ┌─────────────────────────────────────────┐
   │ Shows when it's "thinking"              │
   │ Formats responses nicely                │
   │ Streams long responses                  │
   └─────────────────────────────────────────┘

4. MANAGES RESOURCES
   ┌─────────────────────────────────────────┐
   │ Stays within token limits               │
   │ Doesn't waste API credits               │
   │ Handles long conversations              │
   └─────────────────────────────────────────┘
```

---

## Understanding Conversation History

### Why History Matters

The LLM has **no memory** between API calls. Every call is independent. To create a conversation, **you** must send the full history each time.

```
HOW CONVERSATION HISTORY WORKS
==============================

Without History (Each call is isolated):
────────────────────────────────────────

Call 1: messages = [user: "What is Docker?"]
        → "Docker is a containerization platform..."

Call 2: messages = [user: "How do I install it?"]
        → "Please specify what you want to install."
           ↑ LLM doesn't know we were talking about Docker!


With History (Context is preserved):
────────────────────────────────────

Call 1: messages = [
          system: "You are a DevOps assistant.",
          user: "What is Docker?"
        ]
        → "Docker is a containerization platform..."

Call 2: messages = [
          system: "You are a DevOps assistant.",
          user: "What is Docker?",
          assistant: "Docker is a containerization platform...",  ← Previous exchange
          user: "How do I install it?"                            ← New question
        ]
        → "To install Docker on your system..."
           ↑ LLM sees the full conversation!
```

### How to Maintain History

```python
# Initialize with system prompt
messages = [
    {"role": "system", "content": "You are a DevOps assistant."}
]

# User asks first question
user_input = "What is Docker?"
messages.append({"role": "user", "content": user_input})

# Get response from LLM
response = call_llm(messages)  # "Docker is a containerization..."

# Add assistant's response to history
messages.append({"role": "assistant", "content": response})

# User asks follow-up
user_input = "How do I install it?"
messages.append({"role": "user", "content": user_input})

# Call LLM again - it now has full context!
response = call_llm(messages)  # "To install Docker..."
```

**Visual representation of growing history:**

```
MESSAGE HISTORY GROWTH
======================

After Exchange 1:
┌─────────────────────────────────────────────────┐
│ [system] You are a DevOps assistant.            │
│ [user] What is Docker?                          │
│ [assistant] Docker is a containerization...     │
└─────────────────────────────────────────────────┘

After Exchange 2:
┌─────────────────────────────────────────────────┐
│ [system] You are a DevOps assistant.            │
│ [user] What is Docker?                          │
│ [assistant] Docker is a containerization...     │
│ [user] How do I install it?                     │  ← Growing!
│ [assistant] To install Docker...                │
└─────────────────────────────────────────────────┘

After Exchange 5:
┌─────────────────────────────────────────────────┐
│ [system] You are a DevOps assistant.            │
│ [user] What is Docker?                          │
│ [assistant] Docker is a containerization...     │
│ [user] How do I install it?                     │
│ [assistant] To install Docker...                │
│ [user] What about on Mac?                       │
│ [assistant] On Mac, use: brew install docker... │  ← More growth!
│ [user] How do I run my first container?         │
│ [assistant] Try: docker run hello-world...      │
│ [user] What does that command do?               │
│ [assistant] The docker run command...           │
└─────────────────────────────────────────────────┘

⚠️ WARNING: This keeps growing!
   Eventually hits the token limit (context window).
   We'll learn to manage this later in this module.
```

---

## Python Classes Explained

Our chatbot code uses Python **classes**. If you're not familiar with classes, here's a quick explanation.

### What is a Class?

A **class** is a blueprint for creating objects. It bundles data (attributes) and functions (methods) together.

```python
# WITHOUT CLASSES (messy, hard to manage)
# =======================================

messages1 = []  # For chatbot 1
messages2 = []  # For chatbot 2

def chat1(user_input):
    global messages1  # Messy!
    messages1.append({"role": "user", "content": user_input})
    # ... call LLM ...

def chat2(user_input):
    global messages2  # More mess!
    messages2.append({"role": "user", "content": user_input})
    # ... call LLM ...


# WITH CLASSES (clean, organized)
# ===============================

class Chatbot:
    def __init__(self):
        """Called when you create a new Chatbot."""
        self.messages = []  # Each chatbot has its OWN messages

    def chat(self, user_input):
        """Send a message and get response."""
        self.messages.append({"role": "user", "content": user_input})
        # ... call LLM ...

# Create two separate chatbots - each with its own history!
bot1 = Chatbot()
bot2 = Chatbot()

bot1.chat("What is Docker?")   # bot1 has its own messages
bot2.chat("What is Terraform?")  # bot2 has its own messages
```

### Why Use Classes for Chatbots?

```
BENEFITS OF CLASSES
===================

1. ENCAPSULATION
   Each chatbot instance has its own:
   - messages (conversation history)
   - settings (model, temperature)
   - state (is it busy? how many tokens used?)

2. CLEAN CODE
   class Chatbot:
       def chat(self, message):     # Main functionality
       def clear_history(self):     # Helper method
       def count_tokens(self):      # Another helper

3. REUSABILITY
   # Create different chatbots with different personalities
   devops_bot = Chatbot(system_prompt="You are a DevOps expert...")
   python_bot = Chatbot(system_prompt="You are a Python expert...")

4. TESTABILITY
   # Easy to test in isolation
   def test_chatbot():
       bot = Chatbot()
       response = bot.chat("Hello")
       assert response is not None
```

**Class Anatomy:**

```python
class Chatbot:
    """
    A class is defined with the 'class' keyword.
    By convention, class names use CamelCase.
    """

    def __init__(self, system_prompt="You are helpful."):
        """
        __init__ is the CONSTRUCTOR - called when creating a new instance.
        'self' refers to the instance being created.
        """
        # 'self.messages' is an ATTRIBUTE - data stored in the object
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_input):
        """
        This is a METHOD - a function that belongs to the class.
        'self' is always the first parameter (refers to the instance).
        """
        self.messages.append({"role": "user", "content": user_input})
        # ... get response ...
        return response

    def clear_history(self):
        """Another method - clears everything except system prompt."""
        self.messages = [self.messages[0]]


# USING THE CLASS:
# ================

# Create an INSTANCE of the class
bot = Chatbot(system_prompt="You are a DevOps assistant.")

# Call METHODS on the instance
response = bot.chat("What is Kubernetes?")

# Access ATTRIBUTES
print(len(bot.messages))  # How many messages in history?

# Call another method
bot.clear_history()
```

---

## Building the CLI Interface

### What is a CLI?

A **CLI (Command Line Interface)** is a text-based interface where users type commands and see text output. It's the opposite of a GUI (Graphical User Interface) with buttons and windows.

```
CLI vs GUI
==========

CLI (What we're building):
┌────────────────────────────────────────────────────┐
│ $ python chatbot.py                                │
│                                                    │
│ DevOps Chatbot - Type 'quit' to exit               │
│ ─────────────────────────────────────              │
│                                                    │
│ You: What is a Kubernetes pod?                     │
│                                                    │
│ Bot: A pod is the smallest deployable unit in     │
│      Kubernetes. It can contain one or more       │
│      containers that share storage and network.   │
│                                                    │
│ You: _                                             │
└────────────────────────────────────────────────────┘

GUI (Like ChatGPT website):
┌────────────────────────────────────────────────────┐
│  [Logo]  ChatGPT           [New Chat] [Settings]   │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ You: What is a Kubernetes pod?               │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ 🤖 A pod is the smallest deployable unit... │ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Type your message...              [Send ➤]  │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

**Why CLI for learning?**
- Simpler to build (no web framework needed)
- Focuses on the chatbot logic, not UI code
- Easy to run and test
- Foundation for other interfaces later

### The Rich Library

**Rich** is a Python library that makes CLI output beautiful. It adds colors, formatting, progress bars, and more.

```
PLAIN PYTHON vs RICH
====================

Plain Python print():
┌────────────────────────────────────────────────────┐
│ Bot: A Kubernetes pod is the smallest unit...      │
│ It can contain one or more containers.             │
│                                                    │
│ Here is the command:                               │
│ kubectl get pods                                   │
└────────────────────────────────────────────────────┘
Everything is plain text, hard to read.


With Rich library:
┌────────────────────────────────────────────────────┐
│ ┌──────────────── DevOps Bot ────────────────┐    │
│ │                                            │    │
│ │ A Kubernetes **pod** is the smallest unit. │    │
│ │ It can contain one or more containers.     │    │
│ │                                            │    │
│ │ Here is the command:                       │    │
│ │ ┌────────────────────────────────────────┐ │    │
│ │ │ kubectl get pods                       │ │    │
│ │ └────────────────────────────────────────┘ │    │
│ │                                            │    │
│ └────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────┘
Colors, borders, code blocks, markdown!
```

**Key Rich features we'll use:**

```python
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

# Colored text
console.print("[bold green]Success![/bold green]")
console.print("[red]Error occurred[/red]")

# Panels (boxes around content)
console.print(Panel("Hello World", title="Greeting"))

# Markdown rendering (code blocks, bold, lists)
md = Markdown("**Bold** and `code` and\n```python\nprint('hi')\n```")
console.print(md)

# User input with styling
name = Prompt.ask("[bold blue]Your name[/bold blue]")

# Loading spinner
with console.status("[green]Thinking..."):
    # Do something slow
    time.sleep(2)
```

---

## Streaming Responses

### What is Streaming?

**Streaming** means receiving the response word-by-word as it's generated, instead of waiting for the complete response.

```
WITHOUT STREAMING
=================

User: "Explain Docker architecture"

      [Waiting...]     [Waiting...]     [Waiting...]
         2 sec            4 sec            6 sec

      [COMPLETE RESPONSE APPEARS ALL AT ONCE]
      "Docker uses a client-server architecture. The Docker
       daemon runs on the host machine and manages containers.
       The Docker client communicates with the daemon via REST API..."


WITH STREAMING
==============

User: "Explain Docker architecture"

      "Docker"
      "Docker uses"
      "Docker uses a"
      "Docker uses a client-server"
      "Docker uses a client-server architecture."
      "Docker uses a client-server architecture. The"
      "Docker uses a client-server architecture. The Docker"
      ... (continues word by word)

      Text appears as it's generated - feels faster!
```

### Why Use Streaming?

```
STREAMING BENEFITS
==================

1. PERCEIVED SPEED
   ┌──────────────────────────────────────────────────┐
   │ Without streaming: User stares at blank screen   │
   │ With streaming: Text starts appearing instantly  │
   │                                                  │
   │ Same total time, but streaming FEELS faster!     │
   └──────────────────────────────────────────────────┘

2. EARLY FEEDBACK
   ┌──────────────────────────────────────────────────┐
   │ User can start reading while generation happens  │
   │ Can cancel early if response is wrong direction  │
   └──────────────────────────────────────────────────┘

3. BETTER UX
   ┌──────────────────────────────────────────────────┐
   │ Mimics natural conversation (like someone typing)│
   │ User knows the system is working                 │
   └──────────────────────────────────────────────────┘
```

**How streaming works in code:**

```python
# Without streaming - wait for complete response
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=False  # Default
)
print(response.choices[0].message.content)  # All at once


# With streaming - get chunks as they're generated
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    stream=True  # Enable streaming
)

# Response is now an ITERATOR, not a complete response
full_text = ""
for chunk in response:
    # Each chunk contains a small piece of text
    if chunk.choices[0].delta.content:
        text_piece = chunk.choices[0].delta.content
        print(text_piece, end="", flush=True)  # Print without newline
        full_text += text_piece

print()  # Newline at the end
```

---

## Error Handling

### What Can Go Wrong?

When calling APIs, many things can fail:

```
COMMON API ERRORS
=================

1. AUTHENTICATION ERROR
   ┌──────────────────────────────────────────────────┐
   │ Cause: Invalid or missing API key                │
   │ Fix: Check .env file, regenerate key             │
   └──────────────────────────────────────────────────┘

2. RATE LIMIT ERROR
   ┌──────────────────────────────────────────────────┐
   │ Cause: Too many requests in short time           │
   │ Fix: Wait and retry (exponential backoff)        │
   └──────────────────────────────────────────────────┘

3. CONNECTION ERROR
   ┌──────────────────────────────────────────────────┐
   │ Cause: Network issues, API server down           │
   │ Fix: Retry, check internet connection            │
   └──────────────────────────────────────────────────┘

4. TIMEOUT ERROR
   ┌──────────────────────────────────────────────────┐
   │ Cause: Request took too long                     │
   │ Fix: Retry, maybe use smaller prompt             │
   └──────────────────────────────────────────────────┘

5. INVALID REQUEST
   ┌──────────────────────────────────────────────────┐
   │ Cause: Wrong model name, bad parameters          │
   │ Fix: Check model name, validate input            │
   └──────────────────────────────────────────────────┘
```

### Exponential Backoff

**Exponential backoff** is a retry strategy where you wait longer after each failure.

```
EXPONENTIAL BACKOFF EXPLAINED
=============================

Why not just retry immediately?
- If everyone retries immediately, the server gets flooded
- Makes the problem worse (more requests = more overload)

Exponential backoff strategy:
- Attempt 1 fails → Wait 1 second
- Attempt 2 fails → Wait 2 seconds
- Attempt 3 fails → Wait 4 seconds
- Attempt 4 fails → Wait 8 seconds
- ... (doubles each time)

Visual:

Attempt:    1       2       3       4
            │       │       │       │
            ▼       ▼       ▼       ▼
         [FAIL]  [FAIL]  [FAIL]  [SUCCESS]
            │       │       │       │
Wait:      1s      2s      4s     DONE
         ──────>──────>──────>

Formula: wait_time = 2 ^ attempt_number
         2^0=1, 2^1=2, 2^2=4, 2^3=8...
```

**Implementation:**

```python
import time

def call_with_retry(func, max_retries=3):
    """Call a function with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return func()  # Try to call the function
        except RateLimitError:
            if attempt == max_retries - 1:
                raise  # Give up after max retries

            wait_time = 2 ** attempt  # 1, 2, 4, 8...
            print(f"Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
```

---

## Context Window Management

As conversations grow, you'll hit the token limit. Here's how to manage it:

```
CONTEXT WINDOW PROBLEM
======================

Start of conversation:
┌───────────────────────────────────────────────────────┐
│ Context Window (4096 tokens)                          │
│ ┌─────────────────────────────────────────────────┐  │
│ │ [system] 50 tokens                              │  │
│ │ [user] 20 tokens                                │  │
│ │ [assistant] 100 tokens                          │  │
│ │                                                 │  │
│ │           ... lots of room left ...             │  │
│ │                                                 │  │
│ └─────────────────────────────────────────────────┘  │
│ Used: 170 tokens | Remaining: 3926 tokens             │
└───────────────────────────────────────────────────────┘

After many exchanges:
┌───────────────────────────────────────────────────────┐
│ Context Window (4096 tokens)                          │
│ ┌─────────────────────────────────────────────────┐  │
│ │ [system] 50 tokens                              │  │
│ │ [user] message 1                                │  │
│ │ [assistant] response 1                          │  │
│ │ [user] message 2                                │  │
│ │ [assistant] response 2                          │  │
│ │ [user] message 3                                │  │
│ │ [assistant] response 3                          │  │
│ │ ... many more messages ...                      │  │
│ │ [user] message 20                               │  │
│ │ [assistant] response 20                         │  │
│ └─────────────────────────────────────────────────┘  │
│ Used: 3800 tokens | Remaining: 296 tokens ⚠️          │
└───────────────────────────────────────────────────────┘

Problem: Not enough room for new response!


SOLUTION: Remove old messages
┌───────────────────────────────────────────────────────┐
│ Context Window (4096 tokens)                          │
│ ┌─────────────────────────────────────────────────┐  │
│ │ [system] 50 tokens         ← ALWAYS KEEP        │  │
│ │ [user] message 15          ← Removed 1-14       │  │
│ │ [assistant] response 15                         │  │
│ │ [user] message 16                               │  │
│ │ [assistant] response 16                         │  │
│ │ ... recent messages ...                         │  │
│ │ [user] message 20                               │  │
│ │ [assistant] response 20                         │  │
│ └─────────────────────────────────────────────────┘  │
│ Used: 1500 tokens | Remaining: 2596 tokens ✓          │
└───────────────────────────────────────────────────────┘
```

**Context management strategies:**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Sliding Window** | Keep last N messages | Simple, predictable | Loses old context |
| **Token Budget** | Keep messages until token limit | Efficient | More complex |
| **Summarization** | Summarize old messages | Preserves context | Extra API call |

---

## Hands-On Exercises

### Exercise 1: Chatbot with History

Create `exercises/ex1_chatbot_history.py` - a basic chatbot that maintains conversation context.

**What you'll learn:**
- How to store conversation history
- How to add messages to history
- How to build multi-turn conversations

### Exercise 2: Rich CLI Interface

Create `exercises/ex2_rich_interface.py` - add beautiful formatting to your chatbot.

**What you'll learn:**
- Using the Rich library
- Creating panels and styled output
- Markdown rendering

### Exercise 3: Streaming Responses

Create `exercises/ex3_streaming.py` - show responses as they're generated.

**What you'll learn:**
- How streaming works
- Processing response chunks
- Better user experience

### Exercise 4: Error Handling

Create `exercises/ex4_error_handling.py` - handle failures gracefully.

**What you'll learn:**
- Catching specific exceptions
- Implementing retry logic
- Exponential backoff

### Exercise 5: Complete Chatbot

Create `exercises/ex5_complete_chatbot.py` - combine everything into a production-ready chatbot.

**What you'll learn:**
- Putting it all together
- Context window management
- Building robust applications

---

## Running the Exercises

```bash
# Activate your virtual environment first!
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows

# Run each exercise
python module-2-simple-chatbot/exercises/ex1_chatbot_history.py
python module-2-simple-chatbot/exercises/ex2_rich_interface.py
python module-2-simple-chatbot/exercises/ex3_streaming.py
python module-2-simple-chatbot/exercises/ex4_error_handling.py
python module-2-simple-chatbot/exercises/ex5_complete_chatbot.py

# Or run the complete solution
python module-2-simple-chatbot/solutions/complete_chatbot.py
```

---

## Expected Output

When you run the complete chatbot, you should see:

```
┌─────────────────────────────────────────────────────────┐
│              DevOps Troubleshooting Chatbot              │
│                                                          │
│  I can help with Terraform, Kubernetes, Docker, and     │
│  CI/CD issues. Paste your error message or describe     │
│  your problem.                                           │
│                                                          │
│  Commands: 'quit' to exit, 'clear' to reset history     │
└─────────────────────────────────────────────────────────┘

You: What is a Kubernetes pod?

┌──────────────────── DevOps Bot ─────────────────────────┐
│                                                          │
│ A **Kubernetes pod** is the smallest deployable unit    │
│ in Kubernetes. Key points:                               │
│                                                          │
│ - Can contain one or more containers                    │
│ - Containers in a pod share:                            │
│   - Network namespace (same IP)                         │
│   - Storage volumes                                     │
│                                                          │
│ Example command to see pods:                            │
│ ```bash                                                 │
│ kubectl get pods                                        │
│ ```                                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘

You: How do I see more details?

┌──────────────────── DevOps Bot ─────────────────────────┐
│                                                          │
│ To see more details about your pods:                    │
│                                                          │
│ ```bash                                                 │
│ # Describe a specific pod                               │
│ kubectl describe pod <pod-name>                         │
│                                                          │
│ # See all pods with more columns                        │
│ kubectl get pods -o wide                                │
│ ```                                                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Key Takeaways

1. **History is YOUR responsibility** - LLMs don't remember; you must send full conversation history with each API call.

2. **Classes organize code** - Use classes to bundle data (history) with behavior (chat method).

3. **Rich makes CLI beautiful** - The Rich library adds colors, panels, and markdown rendering.

4. **Streaming improves UX** - Show responses word-by-word for better perceived speed.

5. **Errors will happen** - Always implement retry logic with exponential backoff.

6. **Manage context size** - Remove old messages to stay within token limits.

---

## What's Next?

Our chatbot works great, but it only knows what the LLM learned during training. What if we want it to answer questions about:
- Our company's specific Terraform modules?
- Internal Kubernetes configurations?
- Custom CI/CD pipelines?

In **Module 3**, we'll learn about **RAG (Retrieval-Augmented Generation)** - a technique to give our chatbot access to external knowledge!

Continue to: [Module 3: RAG Fundamentals →](../module-3-rag-fundamentals/README.md)

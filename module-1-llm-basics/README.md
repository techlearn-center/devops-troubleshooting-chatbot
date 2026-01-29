# Module 1: LLM Fundamentals

**Time Required: 1 hour**

Before building our chatbot, let's understand how Large Language Models (LLMs) work and how to interact with them.

---

## Learning Objectives

By the end of this module, you will:
- Understand what LLMs are and how they work
- Make your first API call to an LLM
- Understand tokens, pricing, and parameters
- Control LLM output with temperature and other settings

---

## What is a Large Language Model?

### The Simple Explanation

An LLM is a computer program that predicts the next word in a sequence. It was trained on massive amounts of text from the internet, books, and other sources.

```
Input:  "The DevOps engineer fixed the..."

LLM thinks: Based on patterns I've seen:
  - "bug" - 30% likely
  - "error" - 25% likely
  - "pipeline" - 20% likely
  - "deployment" - 15% likely
  - ...

Output: "bug"

Full sequence: "The DevOps engineer fixed the bug"
```

### Why This Matters for Our Chatbot

LLMs are good at:
- Understanding natural language questions
- Generating human-like explanations
- Following instructions
- Summarizing information

But LLMs have limitations:
- They don't "know" facts - they predict likely text
- They can hallucinate (make up information)
- Their knowledge has a cutoff date
- They don't have access to your specific docs

**This is why we'll add RAG later** - to give the LLM access to real DevOps documentation!

---

## Understanding Tokens

LLMs don't see text as words - they see **tokens**.

### What is a Token?

```
Text: "Kubernetes pod failed"

Tokenized: ["Kub", "ernetes", " pod", " failed"]
           Token 1   Token 2   Token 3   Token 4

Roughly: 1 token ≈ 4 characters ≈ 0.75 words
```

### Why Tokens Matter

1. **Pricing** - You pay per token (input + output)
2. **Context limits** - Models have max token limits
3. **Performance** - More tokens = slower + more expensive

### Token Limits by Model

| Model | Max Tokens | Context Window |
|-------|------------|----------------|
| gpt-3.5-turbo | 4,096 | ~3,000 words |
| gpt-3.5-turbo-16k | 16,384 | ~12,000 words |
| gpt-4 | 8,192 | ~6,000 words |
| gpt-4-turbo | 128,000 | ~96,000 words |

---

## Exercise 1: Your First API Call

Create `exercises/ex1_first_call.py`:

```python
"""
Exercise 1: Make your first LLM API call
========================================
Goal: Send a message to GPT and get a response
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Create the OpenAI client
client = OpenAI()

# TODO: Make your first API call
# Use client.chat.completions.create()

def ask_llm(question: str) -> str:
    """
    Send a question to the LLM and return the response.

    Args:
        question: The question to ask

    Returns:
        The LLM's response text
    """
    # YOUR CODE HERE
    # Hint: Use the chat completions API
    # response = client.chat.completions.create(...)
    pass


# Test your function
if __name__ == "__main__":
    question = "What does a Kubernetes pod do? Answer in one sentence."
    response = ask_llm(question)
    print(f"Question: {question}")
    print(f"Answer: {response}")
```

---

## Understanding the Chat API

The OpenAI Chat API uses **messages** with different **roles**:

```python
messages = [
    {"role": "system", "content": "You are a helpful DevOps assistant."},
    {"role": "user", "content": "What is Terraform?"},
    {"role": "assistant", "content": "Terraform is an IaC tool..."},
    {"role": "user", "content": "How do I install it?"}
]
```

### Message Roles

| Role | Purpose | Example |
|------|---------|---------|
| `system` | Sets behavior and context | "You are a DevOps expert" |
| `user` | Human's input | "What is a pod?" |
| `assistant` | LLM's previous responses | "A pod is..." |

### The System Prompt

The system prompt is **crucial** for controlling LLM behavior:

```python
# Generic assistant
system_prompt = "You are a helpful assistant."

# DevOps specialist
system_prompt = """You are an expert DevOps engineer specializing in:
- Terraform and Infrastructure as Code
- Kubernetes and container orchestration
- CI/CD pipelines
- Cloud platforms (AWS, GCP, Azure)

When answering:
1. Be concise and practical
2. Include code examples when helpful
3. Suggest best practices
4. Warn about common pitfalls
"""
```

---

## Exercise 2: System Prompts

Create `exercises/ex2_system_prompts.py`:

```python
"""
Exercise 2: Experiment with system prompts
==========================================
Goal: See how different system prompts change LLM behavior
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def ask_with_system_prompt(system_prompt: str, user_question: str) -> str:
    """
    Ask the LLM a question with a specific system prompt.

    Args:
        system_prompt: The system message to set context
        user_question: The user's question

    Returns:
        The LLM's response
    """
    # YOUR CODE HERE
    pass


# Test different system prompts
if __name__ == "__main__":
    question = "How do I fix 'Error: resource already exists' in Terraform?"

    # Test 1: Generic assistant
    print("=== Generic Assistant ===")
    generic_prompt = "You are a helpful assistant."
    print(ask_with_system_prompt(generic_prompt, question))

    # Test 2: DevOps expert
    print("\n=== DevOps Expert ===")
    expert_prompt = """You are an expert DevOps engineer with 10 years of experience.
    You specialize in Terraform and Infrastructure as Code.
    Always provide step-by-step solutions with code examples."""
    print(ask_with_system_prompt(expert_prompt, question))

    # Test 3: YOUR custom prompt
    print("\n=== Your Custom Prompt ===")
    # TODO: Create your own system prompt for DevOps troubleshooting
    your_prompt = ""  # YOUR CODE HERE
    print(ask_with_system_prompt(your_prompt, question))
```

---

## Controlling LLM Output

### Temperature

Controls randomness in output:

```
Temperature 0.0: Deterministic, always same output
Temperature 0.7: Balanced (default)
Temperature 1.0: Creative, more varied
Temperature 2.0: Very random, potentially nonsensical
```

**For DevOps troubleshooting:** Use low temperature (0.1-0.3) for consistent, accurate answers.

### Max Tokens

Limits response length:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[...],
    max_tokens=500  # Limit response to ~375 words
)
```

### Other Parameters

| Parameter | Purpose | Recommended Value |
|-----------|---------|-------------------|
| `temperature` | Randomness | 0.1-0.3 for technical |
| `max_tokens` | Response length | 500-1000 |
| `top_p` | Nucleus sampling | 1.0 (default) |
| `frequency_penalty` | Reduce repetition | 0.0-0.5 |
| `presence_penalty` | Encourage new topics | 0.0 |

---

## Exercise 3: Parameter Experimentation

Create `exercises/ex3_parameters.py`:

```python
"""
Exercise 3: Experiment with LLM parameters
==========================================
Goal: Understand how parameters affect output
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


def ask_with_params(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 500
) -> str:
    """
    Ask the LLM with specific parameters.

    Args:
        question: The question to ask
        temperature: Randomness (0.0 to 2.0)
        max_tokens: Maximum response length

    Returns:
        The LLM's response
    """
    # YOUR CODE HERE
    pass


if __name__ == "__main__":
    question = "Explain what happens when a Kubernetes deployment fails."

    # Test different temperatures
    print("=== Temperature Comparison ===\n")

    for temp in [0.0, 0.5, 1.0]:
        print(f"--- Temperature: {temp} ---")
        response = ask_with_params(question, temperature=temp, max_tokens=150)
        print(response)
        print()

    # Run the same prompt 3 times with temp=0 vs temp=1
    # Notice: temp=0 gives same output, temp=1 varies
```

---

## Understanding Costs

### Pricing (as of 2024)

| Model | Input Cost | Output Cost |
|-------|------------|-------------|
| gpt-3.5-turbo | $0.0005/1K tokens | $0.0015/1K tokens |
| gpt-4-turbo | $0.01/1K tokens | $0.03/1K tokens |
| gpt-4 | $0.03/1K tokens | $0.06/1K tokens |

### Estimating Costs

```python
# Rough estimation
input_words = 500  # Your prompt
output_words = 200  # Expected response

input_tokens = input_words * 1.3  # ~650 tokens
output_tokens = output_words * 1.3  # ~260 tokens

# GPT-3.5-turbo cost
cost = (650 * 0.0005 + 260 * 0.0015) / 1000
# = $0.000325 + $0.00039 = $0.0007 per request
# = ~1,400 requests per $1
```

---

## Exercise 4: Token Counting

Create `exercises/ex4_token_counting.py`:

```python
"""
Exercise 4: Count tokens and estimate costs
===========================================
Goal: Understand token usage and costs
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

load_dotenv()
client = OpenAI()


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.

    Args:
        text: The text to count
        model: The model to use for tokenization

    Returns:
        Number of tokens
    """
    # Use tiktoken to count tokens
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def estimate_cost(input_text: str, output_tokens: int, model: str = "gpt-3.5-turbo") -> float:
    """
    Estimate the cost of an API call.

    Args:
        input_text: The input prompt
        output_tokens: Expected output tokens
        model: The model to use

    Returns:
        Estimated cost in dollars
    """
    # YOUR CODE HERE
    # Hint: Use the pricing table above
    pass


if __name__ == "__main__":
    # Test with a DevOps prompt
    prompt = """You are an expert DevOps engineer.
    Explain in detail how to troubleshoot a Kubernetes pod
    that is stuck in CrashLoopBackOff status. Include:
    1. Common causes
    2. Diagnostic commands
    3. Step-by-step resolution"""

    input_tokens = count_tokens(prompt)
    print(f"Input tokens: {input_tokens}")

    # Estimate for ~500 output tokens
    cost = estimate_cost(prompt, 500)
    print(f"Estimated cost: ${cost:.6f}")
```

---

## Key Takeaways

1. **LLMs predict text** - They don't "know" facts, they predict likely continuations
2. **System prompts matter** - They set the behavior and expertise of the LLM
3. **Temperature controls randomness** - Use low values for technical accuracy
4. **Tokens = money** - Be aware of input/output token costs
5. **Limitations exist** - LLMs can hallucinate; we'll fix this with RAG

---

## Solutions

Check `solutions/` directory for complete implementations.

---

## What's Next?

In **Module 2**, you'll build a complete chatbot with:
- Conversation history
- CLI interface
- Error handling
- Streaming responses

```bash
cd ../module-2-simple-chatbot
```

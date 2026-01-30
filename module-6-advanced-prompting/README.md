# Module 6: Advanced Prompting Techniques

**Time Required: 1.5 hours**

The quality of your chatbot's responses depends heavily on **how you ask** the LLM. This module teaches you advanced prompting techniques that dramatically improve response quality.

---

## What You'll Learn

By the end of this module, you will:
- Understand why prompts matter so much
- Master few-shot learning for consistent, structured responses
- Use chain-of-thought prompting for complex troubleshooting
- Generate structured output (JSON) for programmatic use
- Build a reusable prompt library
- Compare prompting strategies side by side

---

## Why Prompts Matter (The Restaurant Analogy)

Think of an LLM like a brilliant chef in a restaurant:

```
BAD ORDER (vague prompt):
  Customer: "Make me something good"
  Chef: *makes a random dish* (maybe great, maybe not what you wanted)

GOOD ORDER (specific prompt):
  Customer: "I'd like a medium-rare steak with mashed potatoes,
             no onions, with a side of garlic butter"
  Chef: *makes exactly what you want*
```

The same applies to LLMs:
```
BAD PROMPT:
  "Help me with my error"
  → LLM: "Can you provide more details?" (Unhelpful!)

GOOD PROMPT:
  "You are a DevOps expert. My Terraform apply shows 'Error: resource
   already exists'. Explain why this happens and give me step-by-step
   commands to fix it. Format your answer with headers and code blocks."
  → LLM: *Detailed, structured, actionable response*
```

**Key Insight:** The same LLM can give a 2/10 or 10/10 response. The difference is the prompt.

---

## The Four Prompting Strategies

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROMPTING STRATEGIES                             │
│                                                                     │
│  1. ZERO-SHOT         "Just answer my question"                     │
│     (no examples)      Good for: Simple, clear questions            │
│                                                                     │
│  2. FEW-SHOT          "Here are examples, now do the same"          │
│     (with examples)    Good for: Consistent formatting/style        │
│                                                                     │
│  3. CHAIN-OF-THOUGHT  "Think step by step"                          │
│     (reasoning)        Good for: Complex debugging/analysis         │
│                                                                     │
│  4. STRUCTURED        "Respond in this exact JSON format"           │
│     (formatted)        Good for: Integration with other tools       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Strategy 1: Zero-Shot (Baseline)

The simplest approach - just ask your question:

```python
# Zero-shot: no examples, just the question
messages = [
    {"role": "user", "content": "How do I fix a Kubernetes pod in CrashLoopBackOff?"}
]
```

**When to use:** Simple, well-known topics where the LLM already has strong knowledge.
**Limitation:** Inconsistent formatting, may miss important details.

---

## Strategy 2: Few-Shot Learning

### What is Few-Shot Learning?

"Few-shot" means giving the LLM a **few examples** before asking your question. The LLM learns the pattern from examples and follows it:

```
WITHOUT EXAMPLES (zero-shot):
  User: "Pod stuck in Pending"
  LLM:  "Your pod is pending. Check the status." (Vague, unhelpful!)

WITH 2 EXAMPLES (few-shot):
  You show the LLM:
    Example 1: Error X → here's how I want you to respond
    Example 2: Error Y → here's how I want you to respond
    Now answer: "Pod stuck in Pending"
  LLM:  "When a pod is stuck in Pending, follow these steps:
         1. Check events: kubectl describe pod <name>
         2. Look for scheduling issues: kubectl get events
         3. Common causes: insufficient resources, node selector..."
         (Structured, actionable, follows the pattern!)
```

### How It Works

The LLM sees the **pattern** in your examples and mirrors it:

```python
FEW_SHOT_PROMPT = """You are a DevOps troubleshooting expert.

Here are examples of how to respond:

---
USER: Terraform shows "Error: resource already exists"

ASSISTANT: This error occurs when Terraform tries to create a resource
that already exists but isn't tracked in state.

**Diagnosis:**
```bash
terraform state list | grep <resource_name>
```

**Solutions:**
1. **Import the existing resource:**
   ```bash
   terraform import <resource_type>.<name> <resource_id>
   ```
2. **Remove from state (if duplicate):**
   ```bash
   terraform state rm <resource_type>.<name>
   ```

**Prevention:** Always use remote state with locking.

---
USER: Kubernetes pod in CrashLoopBackOff

ASSISTANT: CrashLoopBackOff means your container is crashing repeatedly.

**Diagnosis:**
```bash
kubectl logs <pod-name> --previous
kubectl describe pod <pod-name>
```

**Common Causes:**
1. Application error (check logs)
2. Missing configuration/secrets
3. Resource limits too low
4. Liveness probe failing

**Quick Fix:** Check the logs first - 80% of the time the error is there.

---
Now help with this issue:
USER: {user_question}

ASSISTANT:"""
```

**Why this works:** The LLM sees that every response has:
- A one-line explanation
- A **Diagnosis** section with bash commands
- A **Solutions** or **Common Causes** list
- A **Prevention** or **Quick Fix** tip

It follows this exact pattern for the new question!

---

## Strategy 3: Chain-of-Thought (CoT) Prompting

### What is Chain-of-Thought?

Instead of asking for an immediate answer, you ask the LLM to **think step by step**. This dramatically improves accuracy for complex problems:

```
WITHOUT CoT:
  "Why is my deployment failing?" → *Random guess, often wrong*

WITH CoT:
  "Think through this step by step:
   1. What could cause this?
   2. How do we diagnose each cause?
   3. What's the most likely fix?"
  → *Systematic, thorough, accurate analysis*
```

### The Science Behind It

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Regular Prompt:     Question ──────────────────→ Answer         │
│                      (jumps to conclusion)                       │
│                                                                  │
│  Chain-of-Thought:   Question → Step 1 → Step 2 → Step 3 →     │
│                      Analysis → Root Cause → Answer              │
│                      (reasons through the problem)               │
│                                                                  │
│  Result: CoT finds the right answer more often because           │
│  it "shows its work" like a math student!                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
COT_PROMPT = """You are a DevOps expert. When troubleshooting,
think through the problem step by step.

User Issue: {user_question}

Let's analyze this systematically:

1. **Understanding the Error:**
   What exactly is happening? What does this error message mean?

2. **Possible Causes:**
   What are all the things that could cause this?

3. **Diagnostic Steps:**
   What commands or checks should we run to identify the root cause?

4. **Solutions:**
   Based on the likely causes, what are the solutions?

5. **Prevention:**
   How can we prevent this in the future?

Now, let me help you:
"""
```

### When to Use CoT

| Scenario | Use CoT? | Why |
|----------|----------|-----|
| "How do I restart a pod?" | No | Simple question, CoT is overkill |
| "My app is slow after deploying" | Yes | Multiple possible causes |
| "Pipeline fails intermittently" | Yes | Complex, needs systematic analysis |
| "What's the kubectl command for..." | No | Factual lookup, no reasoning needed |

---

## Strategy 4: Structured Output (JSON)

### Why JSON Output?

Sometimes you don't just want text - you want **data** you can use in code:

```
TEXT RESPONSE (hard to parse programmatically):
  "The error is a Terraform state issue. It's high severity.
   You should run terraform import..."

JSON RESPONSE (easy to use in code):
  {
    "error_type": "terraform",
    "severity": "high",
    "summary": "Resource exists but not in state",
    "solutions": [
      {"title": "Import resource", "command": "terraform import..."}
    ]
  }
```

### Implementation

```python
import json

STRUCTURED_PROMPT = """Analyze this DevOps issue and respond with ONLY valid JSON.

User Issue: {user_question}

Respond in this exact format:
{{
    "error_type": "terraform|kubernetes|docker|cicd|unknown",
    "severity": "critical|high|medium|low",
    "summary": "One-line summary of the issue",
    "diagnosis": ["Step 1", "Step 2", "Step 3"],
    "solutions": [
        {{
            "title": "Solution title",
            "command": "Command to run (if applicable)",
            "explanation": "Why this works"
        }}
    ],
    "prevention": "How to prevent this in the future"
}}

JSON Response:"""


def get_structured_response(client, question: str) -> dict:
    """Get a structured JSON response from the LLM."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": STRUCTURED_PROMPT.format(
                user_question=question
            )}
        ],
        temperature=0.1  # Low temperature = more consistent JSON
    )

    content = response.choices[0].message.content
    return json.loads(content)


# Usage:
# result = get_structured_response(client, "Terraform plan shows drift")
# print(result["severity"])      # "high"
# print(result["solutions"][0])  # {"title": "...", "command": "..."}
```

**Pro tip:** Use `temperature=0.1` for JSON output. Higher temperatures cause formatting errors.

---

## Temperature: The Creativity Dial

Temperature controls how "creative" vs "precise" the LLM is:

```
Temperature = 0.0    Temperature = 0.5    Temperature = 1.0
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PRECISE     │    │  BALANCED    │    │  CREATIVE    │
│              │    │              │    │              │
│ Same answer  │    │ Mostly same  │    │ Different    │
│ every time   │    │ with variety │    │ every time   │
│              │    │              │    │              │
│ Good for:    │    │ Good for:    │    │ Good for:    │
│ - JSON       │    │ - General    │    │ - Brainstorm │
│ - Facts      │    │   help       │    │ - Creative   │
│ - Commands   │    │ - Explaining │    │   writing    │
│ - Code       │    │ - Teaching   │    │ - Ideas      │
└──────────────┘    └──────────────┘    └──────────────┘
```

For DevOps troubleshooting, **use 0.1-0.3** (we want accurate, consistent answers, not creative ones).

---

## Building a Prompt Library

### What is a Prompt Library?

Instead of writing prompts from scratch every time, create a **reusable library** of tested prompts:

```python
from string import Template


class DevOpsPromptLibrary:
    """
    A library of tested, optimized prompts for different scenarios.

    Think of it like a recipe book - each prompt is a recipe that's been
    tested and refined to produce consistent, high-quality results.
    """

    # Base system prompt - sets the LLM's "personality"
    SYSTEM = """You are an expert DevOps engineer with 10+ years of experience.
You specialize in troubleshooting infrastructure and deployment issues.
Always be specific, include commands, and explain your reasoning."""

    # Few-shot prompt for error troubleshooting
    ERROR_TROUBLESHOOT = Template("""
I'll help you troubleshoot this error. Let me analyze it:

$context

User's Error: $error

Based on similar issues I've seen, here's my analysis:
""")

    # Chain-of-thought for complex issues
    COMPLEX_ISSUE = Template("""
This is a complex issue. Let me think through it systematically.

Issue: $issue

**Step 1 - Understanding:** What's actually happening here?
**Step 2 - Root Cause Analysis:** What could cause this?
**Step 3 - Diagnostic Commands:** Let's gather more information.
**Step 4 - Solution:** Based on my analysis...
**Step 5 - Prevention:** To avoid this in the future...
""")

    # Structured JSON prompt
    STRUCTURED = Template("""
Analyze this DevOps issue and respond with JSON only.

Issue: $issue

{
    "category": "terraform|kubernetes|docker|cicd",
    "severity": "critical|high|medium|low",
    "summary": "",
    "steps": [],
    "commands": [],
    "prevention": ""
}
""")

    # Quick answer prompt
    QUICK_HELP = Template("""
Give a brief, actionable answer to: $question

Keep it under 3 sentences. Include one command if relevant.
""")


# Usage:
# prompts = DevOpsPromptLibrary()
# prompt = prompts.COMPLEX_ISSUE.substitute(issue="Ingress returns 502")
```

---

## Prompt Engineering Best Practices

### DO:
- **Be specific** about the format you want (headers, bullet points, code blocks)
- **Provide examples** when you need consistent output (few-shot)
- **Ask for reasoning** on complex problems (chain-of-thought)
- **Use low temperature** (0.1-0.3) for factual/technical responses
- **Include context** - the more relevant info, the better the answer
- **Set the role** - "You are a DevOps expert" beats "Help me"

### DON'T:
- Use vague instructions ("be helpful", "do your best")
- Assume the model knows your preferences without telling it
- Skip system prompts for specialized tasks
- Use high temperature for technical content
- Put multiple unrelated questions in one prompt

---

## Exercises

The `exercises/` folder contains 4 hands-on exercises:

| Exercise | What You'll Build |
|----------|-------------------|
| `ex1_prompt_library.py` | A reusable prompt library class with template methods |
| `ex2_few_shot_builder.py` | Dynamic few-shot prompt builder with example management |
| `ex3_chain_of_thought.py` | CoT troubleshooting engine with step-by-step analysis |
| `ex4_strategy_comparison.py` | Side-by-side comparison of all 4 prompting strategies |

Check `solutions/` for complete implementations.

---

## Key Takeaways

1. **Few-shot learning** - Examples dramatically improve consistency
2. **Chain-of-thought** - Step-by-step reasoning catches issues zero-shot misses
3. **Structured output** - JSON enables programmatic integration
4. **Temperature matters** - Low (0.1-0.3) for facts, higher for creativity
5. **Prompt libraries** - Reusable, tested prompts save time and improve quality
6. **Iterate and test** - Good prompts take refinement, not just one attempt

---

## What's Next?

In **Module 7**, we'll put everything together into a complete DevOps troubleshooting chatbot that combines:
- RAG-powered knowledge retrieval (Modules 3-5)
- Advanced prompting techniques (this module)
- Beautiful CLI interface (Module 2)
- Error handling and conversation management

```bash
cd ../module-7-complete-chatbot
```

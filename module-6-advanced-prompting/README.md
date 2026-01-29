# Module 6: Advanced Prompting Techniques

**Time Required: 1.5 hours**

The quality of your chatbot's responses depends heavily on how you prompt the LLM. Learn advanced techniques to get better, more actionable answers.

---

## Learning Objectives

By the end of this module, you will:
- Master few-shot learning for consistent responses
- Use chain-of-thought for complex troubleshooting
- Generate structured output (JSON) for integration
- Create reusable prompt templates

---

## Few-Shot Learning

### What is Few-Shot Learning?

Give the LLM examples of how to respond:

```
WITHOUT EXAMPLES:
User: "Pod stuck in Pending"
LLM: "Your pod is pending. Check the status." (Unhelpful!)

WITH EXAMPLES (Few-shot):
User: "Pod stuck in Pending"
LLM: "When a pod is stuck in Pending, follow these steps:
      1. Check events: kubectl describe pod <name>
      2. Look for scheduling issues: kubectl get events
      3. Common causes: insufficient resources, node selector..."
      (Much better!)
```

### Implementing Few-Shot

```python
FEW_SHOT_PROMPT = """You are a DevOps troubleshooting expert.

Here are examples of how to respond to common issues:

---
USER: Terraform shows "Error: resource already exists"

ASSISTANT: This error occurs when Terraform tries to create a resource that already exists in your cloud provider but isn't tracked in state.

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
# Check logs
kubectl logs <pod-name> --previous

# Check events
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

---

## Chain-of-Thought Prompting

For complex issues, have the LLM reason step-by-step:

```python
COT_PROMPT = """You are a DevOps expert. When troubleshooting, think through the problem step by step.

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

### Example Output with CoT

```
User: "My GitHub Actions workflow keeps failing with 'resource not accessible'"

LLM Response:

1. **Understanding the Error:**
   "Resource not accessible by integration" means the GitHub token doesn't
   have permission to access something the workflow needs.

2. **Possible Causes:**
   - Workflow trying to access another repository
   - Missing permissions in workflow file
   - Fork PR restrictions
   - GITHUB_TOKEN scope limitations

3. **Diagnostic Steps:**
   ```yaml
   # Add to workflow for debugging
   - name: Debug Token
     run: |
       echo "Token scopes:"
       curl -H "Authorization: token ${{ github.token }}" \
         https://api.github.com/repos/${{ github.repository }}
   ```

4. **Solutions:**
   - Add explicit permissions:
     ```yaml
     permissions:
       contents: read
       pull-requests: write
     ```

5. **Prevention:**
   Always define explicit permissions at the job level.
```

---

## Structured Output (JSON)

For integration with other systems, get JSON responses:

```python
STRUCTURED_PROMPT = """You are a DevOps assistant that returns structured JSON responses.

User Issue: {user_question}

Respond with ONLY valid JSON in this format:
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


import json

def get_structured_response(question: str) -> dict:
    """Get a structured JSON response from the LLM."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": STRUCTURED_PROMPT.format(
                user_question=question
            )}
        ],
        temperature=0.1  # Low temperature for consistent JSON
    )

    # Parse JSON from response
    content = response.choices[0].message.content
    return json.loads(content)
```

---

## Exercise 1: Create a Prompt Library

```python
"""
Exercise 1: DevOps Prompt Library
=================================
Goal: Create reusable prompt templates for different scenarios
"""

from string import Template


class DevOpsPromptLibrary:
    """Library of optimized prompts for DevOps troubleshooting."""

    # Base system prompt
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

    # Chain-of-thought prompt
    COMPLEX_ISSUE = Template("""
This is a complex issue. Let me think through it systematically.

Issue: $issue

**Step 1 - Understanding:**
What's actually happening here?

**Step 2 - Root Cause Analysis:**
What could cause this?

**Step 3 - Diagnostic Commands:**
Let's gather more information:

**Step 4 - Solution:**
Based on my analysis:

**Step 5 - Prevention:**
To avoid this in the future:
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


# Usage
prompts = DevOpsPromptLibrary()

# For complex debugging
prompt = prompts.COMPLEX_ISSUE.substitute(
    issue="Kubernetes ingress returns 502 bad gateway intermittently"
)

# For quick questions
prompt = prompts.QUICK_HELP.substitute(
    question="How do I restart a deployment?"
)
```

---

## Exercise 2: Response Quality Comparison

```python
"""
Exercise 2: Compare Prompting Strategies
========================================
Goal: See how different prompts affect response quality
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()


TEST_QUESTION = "My Terraform apply is stuck and not progressing"


def test_basic_prompt():
    """Minimal prompt - baseline."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": TEST_QUESTION}]
    )
    return response.choices[0].message.content


def test_system_prompt():
    """With system prompt."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a DevOps expert."},
            {"role": "user", "content": TEST_QUESTION}
        ]
    )
    return response.choices[0].message.content


def test_few_shot():
    """With examples."""
    # TODO: Implement few-shot prompt
    pass


def test_cot():
    """With chain-of-thought."""
    # TODO: Implement CoT prompt
    pass


def compare_all():
    """Compare all strategies."""
    strategies = [
        ("Basic", test_basic_prompt),
        ("System Prompt", test_system_prompt),
        ("Few-Shot", test_few_shot),
        ("Chain-of-Thought", test_cot),
    ]

    for name, func in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {name}")
        print('='*60)
        try:
            response = func()
            print(response[:500] + "..." if len(response) > 500 else response)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    compare_all()
```

---

## Prompt Engineering Best Practices

### DO:
- Be specific about the format you want
- Provide examples when possible
- Ask the model to explain its reasoning
- Use low temperature (0.1-0.3) for factual responses
- Include relevant context

### DON'T:
- Use vague instructions ("be helpful")
- Assume the model knows your preferences
- Skip system prompts for specialized tasks
- Use high temperature for technical content

---

## Key Takeaways

1. **Few-shot learning** - Examples dramatically improve consistency
2. **Chain-of-thought** - Step-by-step reasoning for complex issues
3. **Structured output** - JSON for programmatic integration
4. **Temperature matters** - Low for facts, higher for creativity
5. **Iterate and test** - Good prompts take refinement

---

## What's Next?

In **Module 7**, we'll put everything together into a complete DevOps troubleshooting chatbot with:
- RAG-powered knowledge
- Advanced prompting
- Beautiful CLI interface
- Error handling

```bash
cd ../module-7-complete-chatbot
```

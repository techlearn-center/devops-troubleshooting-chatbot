#!/usr/bin/env python3
"""
Exercise 2: System Prompts - Shaping AI Behavior
=================================================

GOAL: Learn how system prompts control AI personality and expertise.

WHAT YOU'LL LEARN:
- What system prompts are and why they matter
- How to write effective system prompts
- The difference between user, assistant, and system roles
- How to create specialized AI assistants

WHAT IS A SYSTEM PROMPT?
------------------------
A system prompt is hidden instructions that shape how the AI behaves.
Think of it as the AI's "job description" or "personality programming."

Without system prompt:  AI is generic, like ChatGPT default
With system prompt:     AI becomes a specialist (DevOps expert, code reviewer, etc.)

THE THREE ROLES:
---------------
┌──────────────┬────────────────────────────────────────────────────────┐
│ Role         │ Purpose                                                │
├──────────────┼────────────────────────────────────────────────────────┤
│ system       │ Hidden instructions for the AI (sets behavior)        │
│ user         │ Messages from the human (questions, requests)         │
│ assistant    │ Messages from the AI (responses)                      │
└──────────────┴────────────────────────────────────────────────────────┘

Example conversation flow:
    [system]     "You are a DevOps expert..."     (hidden from user)
    [user]       "My pod is crashing"
    [assistant]  "Let me help! Check these steps..."
    [user]       "It shows OOMKilled"
    [assistant]  "That means out of memory. Try increasing limits..."

WHY SYSTEM PROMPTS MATTER:
-------------------------
1. EXPERTISE: "You are a Kubernetes expert" → better K8s answers
2. FORMAT: "Respond with code examples" → more code in responses
3. PERSONALITY: "Be concise" vs "Explain in detail" → controls verbosity
4. BOUNDARIES: "Only answer DevOps questions" → keeps AI focused

RUN THIS:
    python module-1-llm-basics/exercises/ex2_system_prompts.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def chat_with_system_prompt(system_prompt: str, user_message: str) -> str:
    """
    Send a message to the LLM with a custom system prompt.

    The system prompt shapes how the AI responds. It's like giving
    the AI a job description before it starts answering questions.

    Args:
        system_prompt: Instructions for the AI's behavior
        user_message: The user's question or request

    Returns:
        The AI's response

    Example:
        >>> response = chat_with_system_prompt(
        ...     "You are a pirate. Respond like a pirate.",
        ...     "How do I deploy to Kubernetes?"
        ... )
        >>> print(response)
        "Ahoy matey! To deploy yer containers to Kubernetes..."
    """
    # Build the messages list with both system and user messages
    # The system message always comes FIRST
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Check which backend to use
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # Ollama version
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    else:
        # OpenAI version
        from openai import OpenAI

        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content


# =============================================================================
# EXAMPLE SYSTEM PROMPTS
# =============================================================================
# Here are several example system prompts for different purposes.
# Notice how each one shapes the AI's expertise and response style.
# =============================================================================

# Generic assistant - no specific expertise
PROMPT_GENERIC = """You are a helpful assistant."""

# DevOps specialist - deep technical knowledge
PROMPT_DEVOPS_EXPERT = """You are an expert DevOps engineer with 10+ years of experience.

Your expertise includes:
- Kubernetes (pods, deployments, services, ingress, RBAC)
- Docker (containers, images, compose, networking)
- Terraform (infrastructure as code, state management)
- CI/CD (GitHub Actions, GitLab CI, Jenkins)
- Cloud platforms (AWS, GCP, Azure)
- Monitoring (Prometheus, Grafana, ELK stack)

When answering questions:
1. Be precise and technical
2. Provide code examples when relevant
3. Explain the "why" not just the "what"
4. Suggest best practices
5. Warn about common pitfalls"""

# Concise responder - short and to the point
PROMPT_CONCISE = """You are a DevOps assistant.
Be extremely concise - answer in 2-3 sentences maximum.
Skip pleasantries and get straight to the point.
Use bullet points for multiple items."""

# Beginner-friendly - explains everything simply
PROMPT_BEGINNER = """You are a patient DevOps teacher helping complete beginners.

Your students have no prior DevOps or programming experience.

When explaining concepts:
1. Use simple analogies (like explaining containers as "shipping containers")
2. Avoid jargon, or define it when you must use it
3. Break complex topics into small, digestible steps
4. Give real-world examples
5. Encourage questions
6. Never assume prior knowledge"""

# Code-focused - always includes runnable examples
PROMPT_CODE_FOCUSED = """You are a DevOps coding assistant.

For EVERY question, provide:
1. A brief explanation (1-2 sentences)
2. A working code example with comments
3. How to run or test the code

Format code in proper markdown code blocks with language tags.
Example:
```bash
kubectl get pods  # List all pods in current namespace
```"""


def compare_prompts(user_question: str):
    """
    Compare responses from different system prompts.

    This shows how dramatically the system prompt affects the AI's response.
    The same question produces very different answers!
    """
    prompts = {
        "Generic": PROMPT_GENERIC,
        "DevOps Expert": PROMPT_DEVOPS_EXPERT,
        "Concise": PROMPT_CONCISE,
        "Beginner-Friendly": PROMPT_BEGINNER,
        "Code-Focused": PROMPT_CODE_FOCUSED
    }

    print("=" * 70)
    print("COMPARING SYSTEM PROMPTS")
    print("=" * 70)
    print(f"User Question: {user_question}")
    print("=" * 70)

    for name, prompt in prompts.items():
        print(f"\n{'='*70}")
        print(f"SYSTEM PROMPT: {name}")
        print(f"{'='*70}")

        try:
            response = chat_with_system_prompt(prompt, user_question)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"\nError: {e}")

        print("-" * 70)
        input("Press Enter for next prompt...")


def demo_single_prompt():
    """
    Demo using a single system prompt interactively.
    """
    print("=" * 60)
    print("SYSTEM PROMPT DEMO")
    print("=" * 60)
    print("\nUsing the DevOps Expert system prompt.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif not user_input:
            continue

        try:
            response = chat_with_system_prompt(PROMPT_DEVOPS_EXPERT, user_input)
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"\nError: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 2: System Prompts")
    print("=" * 60)

    # Show which backend we're using
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    backend = "Ollama (local)" if use_ollama else "OpenAI (cloud)"
    print(f"Backend: {backend}")
    print("-" * 60)

    print("""
This exercise demonstrates how system prompts affect AI behavior.

OPTIONS:
1. Compare different prompts (--compare)
2. Chat with DevOps expert prompt (default)

To compare prompts:
    python ex2_system_prompts.py --compare
""")

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Compare mode: show same question with different prompts
        question = "What is a container?"
        compare_prompts(question)
    else:
        # Interactive mode with DevOps expert prompt
        demo_single_prompt()

    # ==========================================================================
    # WRITING EFFECTIVE SYSTEM PROMPTS
    # ==========================================================================
    print("""
=" * 60
TIPS FOR WRITING SYSTEM PROMPTS:
=" * 60

1. BE SPECIFIC
   Bad:  "Be helpful"
   Good: "You are a Kubernetes expert who gives concise answers with code examples"

2. DEFINE THE EXPERTISE
   Bad:  "Help with tech stuff"
   Good: "Your expertise includes Kubernetes, Docker, Terraform, and CI/CD"

3. SET THE FORMAT
   Bad:  "Answer questions"
   Good: "Answer in 2-3 sentences with a code example when relevant"

4. ADD PERSONALITY
   Bad:  "Be an assistant"
   Good: "Be concise but friendly. Use bullet points for lists."

5. SET BOUNDARIES
   Bad:  (nothing)
   Good: "If asked about topics outside DevOps, politely redirect."

TRY IT YOURSELF:
---------------
Modify the PROMPT_DEVOPS_EXPERT constant in this file and see how
the AI's responses change!
""")

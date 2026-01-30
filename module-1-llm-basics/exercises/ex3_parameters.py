#!/usr/bin/env python3
"""
Exercise 3: API Parameters - Controlling LLM Output
====================================================

GOAL: Learn how to control LLM behavior using API parameters.

WHAT YOU'LL LEARN:
- Temperature: Controls randomness/creativity
- Max tokens: Limits response length
- Top P: Alternative to temperature
- Other parameters and their effects

KEY PARAMETERS EXPLAINED:
------------------------

┌─────────────────┬───────────────────────────────────────────────────────────┐
│ Parameter       │ What It Does                                              │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ temperature     │ Controls randomness (0.0 = deterministic, 2.0 = random)  │
│ max_tokens      │ Maximum length of response (1 token ≈ 4 characters)      │
│ top_p           │ Alternative to temperature (nucleus sampling)            │
│ frequency_penalty│ Reduces repetition of words                             │
│ presence_penalty │ Encourages talking about new topics                     │
└─────────────────┴───────────────────────────────────────────────────────────┘

TEMPERATURE IN DETAIL:
---------------------
Temperature controls how "random" the AI's word choices are.

    Temperature = 0.0 (DETERMINISTIC)
    ┌────────────────────────────────────┐
    │ Next word probabilities:           │
    │   "container" → 80% ████████░░     │ ← Always picks this (highest)
    │   "image"     → 15% ██░░░░░░░░     │
    │   "pod"       → 5%  █░░░░░░░░░     │
    └────────────────────────────────────┘

    Temperature = 1.0 (BALANCED)
    ┌────────────────────────────────────┐
    │ Next word probabilities:           │
    │   "container" → 50% █████░░░░░     │ ← Usually picks this
    │   "image"     → 35% ████░░░░░░     │ ← Sometimes picks this
    │   "pod"       → 15% ██░░░░░░░░     │ ← Occasionally this
    └────────────────────────────────────┘

    Temperature = 2.0 (CREATIVE/RANDOM)
    ┌────────────────────────────────────┐
    │ Next word probabilities:           │
    │   "container" → 35% ████░░░░░░     │
    │   "image"     → 30% ███░░░░░░░     │ ← More likely to pick
    │   "pod"       → 35% ████░░░░░░     │ ← unexpected words
    └────────────────────────────────────┘

WHEN TO USE WHAT TEMPERATURE:
----------------------------
- 0.0 - 0.3: Technical docs, code, factual answers (consistent, accurate)
- 0.5 - 0.7: General conversation, balanced responses
- 0.8 - 1.2: Creative writing, brainstorming (varied, exploratory)
- 1.5 - 2.0: Very creative, experimental (can be incoherent!)

MAX_TOKENS EXPLAINED:
--------------------
This limits how long the response can be.

    Token ≈ ~4 characters or ~3/4 of a word

    Examples:
    - "Kubernetes" = 3 tokens (Kub-ern-etes)
    - "pod" = 1 token
    - "Hello, world!" = 3 tokens (Hello, | world | !)

    max_tokens = 50  → Short response (~35 words)
    max_tokens = 200 → Medium response (~150 words)
    max_tokens = 500 → Long response (~375 words)

RUN THIS:
    python module-1-llm-basics/exercises/ex3_parameters.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def ask_with_parameters(
    question: str,
    temperature: float = 0.7,
    max_tokens: int = 200,
    top_p: float = 1.0
) -> str:
    """
    Ask the LLM with specific parameters.

    Args:
        question: The question to ask
        temperature: Randomness (0.0 to 2.0, default 0.7)
        max_tokens: Maximum response length (default 200)
        top_p: Nucleus sampling (0.0 to 1.0, default 1.0)

    Returns:
        The LLM's response

    Example:
        >>> # Get a creative response
        >>> response = ask_with_parameters("What is Docker?", temperature=1.5)

        >>> # Get a precise, consistent response
        >>> response = ask_with_parameters("What is Docker?", temperature=0.0)
    """
    messages = [
        {"role": "user", "content": question}
    ]

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # Ollama version
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")

        # Ollama uses 'options' for parameters
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,  # Ollama calls it num_predict
                    "top_p": top_p
                }
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
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.choices[0].message.content


def demo_temperature():
    """
    Demonstrate how temperature affects responses.

    We ask the same question multiple times with different temperatures
    to see how the responses vary.
    """
    print("\n" + "=" * 60)
    print("TEMPERATURE DEMONSTRATION")
    print("=" * 60)

    question = "What is a container? Give a one-sentence definition."
    print(f"\nQuestion: {question}\n")

    temperatures = [0.0, 0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n{'─' * 60}")
        print(f"Temperature: {temp}")
        print(f"{'─' * 60}")

        # Ask the same question twice to show consistency/variance
        for i in range(2):
            try:
                response = ask_with_parameters(question, temperature=temp, max_tokens=100)
                print(f"\nAttempt {i + 1}: {response}")
            except Exception as e:
                print(f"\nError: {e}")

    print(f"\n{'=' * 60}")
    print("""
OBSERVATIONS:
- At temperature 0.0: Responses should be identical
- At temperature 0.5: Responses are similar but with slight variations
- At temperature 1.0: Responses have noticeable differences
- At temperature 1.5: Responses may be quite different (and potentially less coherent)
""")


def demo_max_tokens():
    """
    Demonstrate how max_tokens affects response length.
    """
    print("\n" + "=" * 60)
    print("MAX TOKENS DEMONSTRATION")
    print("=" * 60)

    question = "Explain what Kubernetes does."
    print(f"\nQuestion: {question}")

    token_limits = [30, 100, 300]

    for max_tokens in token_limits:
        print(f"\n{'─' * 60}")
        print(f"Max Tokens: {max_tokens}")
        print(f"{'─' * 60}")

        try:
            response = ask_with_parameters(
                question,
                temperature=0.3,
                max_tokens=max_tokens
            )
            # Count approximate tokens (rough estimate)
            approx_tokens = len(response) // 4
            print(f"\nResponse (~{approx_tokens} tokens):\n{response}")
        except Exception as e:
            print(f"\nError: {e}")

    print(f"\n{'=' * 60}")
    print("""
OBSERVATIONS:
- Lower max_tokens: Response is cut short (may be incomplete)
- Higher max_tokens: Response can be complete and detailed
- The AI doesn't always use all tokens - it stops when done
""")


def demo_for_devops():
    """
    Show recommended parameters for DevOps troubleshooting.
    """
    print("\n" + "=" * 60)
    print("RECOMMENDED DEVOPS PARAMETERS")
    print("=" * 60)

    devops_questions = [
        "My Kubernetes pod shows 'CrashLoopBackOff'. What should I check?",
        "Write a Dockerfile for a Python Flask application.",
        "Explain the difference between a Deployment and a StatefulSet."
    ]

    # Recommended parameters for DevOps troubleshooting
    # - Low temperature: Consistent, accurate answers
    # - Moderate max_tokens: Room for explanation + code
    params = {
        "temperature": 0.2,
        "max_tokens": 500,
        "top_p": 0.95
    }

    print(f"\nRecommended Parameters:")
    print(f"  temperature = {params['temperature']} (low for accuracy)")
    print(f"  max_tokens = {params['max_tokens']} (room for code examples)")
    print(f"  top_p = {params['top_p']} (slight variation allowed)")

    for question in devops_questions:
        print(f"\n{'─' * 60}")
        print(f"Question: {question}")
        print(f"{'─' * 60}")

        try:
            response = ask_with_parameters(
                question,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                top_p=params["top_p"]
            )
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"\nError: {e}")

        input("\nPress Enter for next question...")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 3: API Parameters")
    print("=" * 60)

    # Show which backend we're using
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    backend = "Ollama (local)" if use_ollama else "OpenAI (cloud)"
    print(f"Backend: {backend}")

    print("""
This exercise demonstrates how API parameters affect LLM responses.

OPTIONS:
1. Temperature demo (--temp)
2. Max tokens demo (--tokens)
3. DevOps recommended settings (--devops)
4. Interactive mode (default)

Examples:
    python ex3_parameters.py --temp
    python ex3_parameters.py --tokens
    python ex3_parameters.py --devops
""")

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--temp":
            demo_temperature()
        elif sys.argv[1] == "--tokens":
            demo_max_tokens()
        elif sys.argv[1] == "--devops":
            demo_for_devops()
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("INTERACTIVE MODE")
        print("=" * 60)
        print("Ask questions with custom parameters.")
        print("Type 'quit' to exit.\n")

        while True:
            user_input = input("\nYour question: ").strip()

            if user_input.lower() == 'quit':
                break
            elif not user_input:
                continue

            # Get parameters
            try:
                temp = float(input("Temperature (0.0-2.0, default 0.5): ") or "0.5")
                tokens = int(input("Max tokens (50-1000, default 200): ") or "200")
            except ValueError:
                print("Invalid input, using defaults")
                temp = 0.5
                tokens = 200

            try:
                response = ask_with_parameters(user_input, temperature=temp, max_tokens=tokens)
                print(f"\nResponse:\n{response}")
            except Exception as e:
                print(f"\nError: {e}")

    print("""
=" * 60
PARAMETER CHEAT SHEET:
=" * 60

┌─────────────────────────────────────────────────────────────────────┐
│ USE CASE                      │ temperature │ max_tokens │ top_p   │
├─────────────────────────────────────────────────────────────────────┤
│ Code generation               │ 0.0 - 0.2   │ 500 - 1000 │ 0.95    │
│ Troubleshooting               │ 0.2 - 0.3   │ 300 - 500  │ 0.95    │
│ Technical documentation       │ 0.3 - 0.5   │ 500 - 1000 │ 0.95    │
│ General Q&A                   │ 0.5 - 0.7   │ 200 - 400  │ 1.0     │
│ Brainstorming                 │ 0.8 - 1.0   │ 300 - 500  │ 1.0     │
│ Creative writing              │ 1.0 - 1.5   │ 500+       │ 1.0     │
└─────────────────────────────────────────────────────────────────────┘
""")

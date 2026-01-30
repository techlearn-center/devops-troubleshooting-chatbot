#!/usr/bin/env python3
"""
SOLUTION: Exercise 3 - API Parameters
======================================

This is the complete solution for Exercise 3.

WHAT THIS CODE DEMONSTRATES:
---------------------------
1. How temperature affects response randomness
2. How max_tokens limits response length
3. Recommended parameter settings for DevOps use cases
4. Interactive parameter experimentation

KEY CONCEPTS:
------------
- Temperature: 0.0 (deterministic) to 2.0 (random)
- Max Tokens: Limits response length
- Top P: Alternative to temperature (nucleus sampling)

RUNNING THIS CODE:
-----------------
    # Temperature demo
    python module-1-llm-basics/solutions/ex3_parameters.py --temp

    # Max tokens demo
    python module-1-llm-basics/solutions/ex3_parameters.py --tokens

    # DevOps recommended settings
    python module-1-llm-basics/solutions/ex3_parameters.py --devops

    # Interactive mode
    python module-1-llm-basics/solutions/ex3_parameters.py
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
    top_p: float = 1.0,
    system_prompt: str = None
) -> str:
    """
    Ask the LLM with specific parameters.

    Args:
        question: The question to ask
        temperature: Randomness (0.0-2.0)
        max_tokens: Maximum response length
        top_p: Nucleus sampling (0.0-1.0)
        system_prompt: Optional system prompt

    Returns:
        The LLM's response
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")

        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p
                }
            }
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    else:
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
    """Demonstrate temperature effects."""
    print("\n" + "=" * 60)
    print("TEMPERATURE DEMONSTRATION")
    print("=" * 60)

    question = "Define what a container is in one sentence."

    for temp in [0.0, 0.5, 1.0, 1.5]:
        print(f"\n{'─' * 60}")
        print(f"Temperature: {temp}")
        print(f"{'─' * 60}")

        for i in range(3):
            try:
                response = ask_with_parameters(question, temperature=temp, max_tokens=80)
                print(f"  {i+1}. {response[:150]}")
            except Exception as e:
                print(f"  Error: {e}")


def demo_max_tokens():
    """Demonstrate max_tokens effects."""
    print("\n" + "=" * 60)
    print("MAX TOKENS DEMONSTRATION")
    print("=" * 60)

    question = "Explain what Kubernetes does and why it's useful."

    for tokens in [30, 100, 300]:
        print(f"\n{'─' * 60}")
        print(f"Max Tokens: {tokens}")
        print(f"{'─' * 60}")

        try:
            response = ask_with_parameters(question, temperature=0.3, max_tokens=tokens)
            word_count = len(response.split())
            print(f"Response ({word_count} words):\n{response}")
        except Exception as e:
            print(f"Error: {e}")


def demo_devops_settings():
    """Show recommended settings for DevOps."""
    print("\n" + "=" * 60)
    print("RECOMMENDED DEVOPS SETTINGS")
    print("=" * 60)

    system_prompt = "You are a DevOps expert. Be precise and include code examples."

    settings = {
        "Troubleshooting": {"temperature": 0.1, "max_tokens": 500},
        "Code Generation": {"temperature": 0.0, "max_tokens": 800},
        "Explanation": {"temperature": 0.4, "max_tokens": 400},
        "Brainstorming": {"temperature": 0.9, "max_tokens": 600},
    }

    questions = {
        "Troubleshooting": "My pod shows ImagePullBackOff. How do I debug this?",
        "Code Generation": "Write a Dockerfile for a Python Flask app.",
        "Explanation": "What's the difference between a Deployment and StatefulSet?",
        "Brainstorming": "What are creative ways to implement blue-green deployments?",
    }

    for use_case, params in settings.items():
        print(f"\n{'='*60}")
        print(f"USE CASE: {use_case}")
        print(f"Settings: temperature={params['temperature']}, max_tokens={params['max_tokens']}")
        print(f"{'='*60}")

        try:
            response = ask_with_parameters(
                questions[use_case],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                system_prompt=system_prompt
            )
            print(f"\nQuestion: {questions[use_case]}")
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter to continue...")


def interactive_mode():
    """Interactive parameter testing."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PARAMETER TESTING")
    print("=" * 60)
    print("Experiment with different parameters!")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() == 'quit':
            break
        if not question:
            continue

        try:
            temp = float(input("Temperature (0.0-2.0) [0.5]: ") or "0.5")
            tokens = int(input("Max tokens (50-1000) [200]: ") or "200")
        except ValueError:
            temp, tokens = 0.5, 200

        try:
            response = ask_with_parameters(question, temperature=temp, max_tokens=tokens)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    print(f"Backend: {'Ollama' if use_ollama else 'OpenAI'}")

    if len(sys.argv) > 1:
        if sys.argv[1] == "--temp":
            demo_temperature()
        elif sys.argv[1] == "--tokens":
            demo_max_tokens()
        elif sys.argv[1] == "--devops":
            demo_devops_settings()
    else:
        interactive_mode()

    print("\n" + "=" * 60)
    print("PARAMETER QUICK REFERENCE")
    print("=" * 60)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Task                     │ Temperature │ Max Tokens │ Notes        │
├─────────────────────────────────────────────────────────────────────┤
│ Code generation          │ 0.0 - 0.2   │ 500-1000   │ Deterministic│
│ Troubleshooting          │ 0.2 - 0.3   │ 300-500    │ Consistent   │
│ Documentation            │ 0.3 - 0.5   │ 500-1000   │ Balanced     │
│ General Q&A              │ 0.5 - 0.7   │ 200-400    │ Natural      │
│ Brainstorming            │ 0.8 - 1.2   │ 300-500    │ Creative     │
└─────────────────────────────────────────────────────────────────────┘
""")

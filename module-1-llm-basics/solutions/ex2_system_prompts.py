#!/usr/bin/env python3
"""
SOLUTION: Exercise 2 - System Prompts
======================================

This is the complete solution for Exercise 2.

WHAT THIS CODE DEMONSTRATES:
---------------------------
1. How system prompts shape AI behavior
2. Multiple system prompt examples for different use cases
3. Interactive comparison of different prompts
4. Best practices for writing effective prompts

HOW IT WORKS:
------------
The system prompt is sent as the first message in the conversation.
It's marked with role="system" and tells the AI how to behave.

The AI doesn't show this message to users - it's "hidden instructions."

RUNNING THIS CODE:
-----------------
    # Interactive mode with DevOps expert prompt
    python module-1-llm-basics/solutions/ex2_system_prompts.py

    # Compare different system prompts
    python module-1-llm-basics/solutions/ex2_system_prompts.py --compare
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# SYSTEM PROMPT COLLECTION
# =============================================================================
# These are production-ready system prompts for different use cases.
# Study how each one shapes the AI's behavior.
# =============================================================================

SYSTEM_PROMPTS = {
    "generic": """You are a helpful assistant.""",

    "devops_expert": """You are a senior DevOps engineer with 15+ years of experience.

Your deep expertise includes:
- Kubernetes: Architecture, troubleshooting, RBAC, networking, operators
- Docker: Multi-stage builds, security, optimization, compose
- Terraform: Modules, state management, providers, workspaces
- CI/CD: GitHub Actions, GitLab CI, Jenkins, ArgoCD
- Cloud: AWS, GCP, Azure (networking, IAM, compute, storage)
- Monitoring: Prometheus, Grafana, ELK, Datadog
- Security: OWASP, container scanning, secrets management

Response Guidelines:
1. Start with understanding the problem (ask if unclear)
2. Explain the root cause, not just the fix
3. Provide working code examples with comments
4. Suggest prevention strategies
5. Link concepts to industry best practices

Format with markdown: **bold** for key terms, `code` for commands, code blocks for scripts.""",

    "concise": """You are a DevOps assistant.
Rules:
- Maximum 3 sentences per response
- Use bullet points for lists
- Include one code example if relevant
- No pleasantries or filler words
- Get straight to the solution""",

    "beginner_teacher": """You are a patient DevOps instructor teaching complete beginners.

Your students:
- Have never used command line before
- Don't know what containers or servers are
- Need everything explained from first principles

Your teaching approach:
1. Use real-world analogies (containers = shipping containers, etc.)
2. Define EVERY technical term before using it
3. Break complex tasks into tiny, numbered steps
4. Show exactly what to type and what to expect
5. Explain common mistakes and how to avoid them
6. Celebrate small wins and encourage questions
7. Never assume prior knowledge""",

    "code_only": """You are a code-focused DevOps assistant.

For EVERY response:
1. Start with 1-2 sentence explanation
2. Show complete, runnable code example
3. Add comments explaining each line
4. Show expected output
5. Include error handling

Format all code with proper markdown:
```bash
# Comment explaining what this does
command --flag value
```

No lengthy explanations - let the code speak.""",

    "security_focused": """You are a DevSecOps security specialist.

Your priority: Security-first solutions.

For every recommendation:
1. Identify potential security risks
2. Suggest secure alternatives
3. Reference OWASP, CIS, or NIST when relevant
4. Check for: exposed secrets, excessive permissions, unvalidated input
5. Recommend least-privilege approaches

Always ask: "What could go wrong if this is misconfigured?"

Format warnings clearly:
⚠️ SECURITY: [issue description]
✅ SECURE: [recommended approach]"""
}


def chat_with_system_prompt(system_prompt: str, user_message: str) -> str:
    """
    Send a message to the LLM with a specific system prompt.

    Args:
        system_prompt: Instructions for the AI's behavior
        user_message: The user's question or request

    Returns:
        The AI's response
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        import requests
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")

        response = requests.post(
            f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": False}
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
            temperature=0.3
        )
        return response.choices[0].message.content


def compare_prompts():
    """Compare responses from different system prompts."""
    question = "How do I check why my Kubernetes pod is failing?"

    print("=" * 70)
    print("SYSTEM PROMPT COMPARISON")
    print("=" * 70)
    print(f"\nQuestion: {question}\n")

    for name, prompt in SYSTEM_PROMPTS.items():
        print(f"\n{'='*70}")
        print(f"PROMPT: {name.upper().replace('_', ' ')}")
        print(f"{'='*70}")

        try:
            response = chat_with_system_prompt(prompt, question)
            print(f"\nResponse:\n{response[:500]}...")
        except Exception as e:
            print(f"Error: {e}")

        input("\nPress Enter for next prompt...")


def interactive_chat():
    """Interactive chat with selectable system prompts."""
    print("=" * 60)
    print("INTERACTIVE SYSTEM PROMPT CHAT")
    print("=" * 60)
    print("\nAvailable prompts:")
    for i, name in enumerate(SYSTEM_PROMPTS.keys(), 1):
        print(f"  {i}. {name}")

    print("\nCommands: 'switch' to change prompt, 'quit' to exit\n")

    current_prompt = "devops_expert"
    print(f"Using: {current_prompt}")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'switch':
            print("\nSelect prompt:")
            for i, name in enumerate(SYSTEM_PROMPTS.keys(), 1):
                print(f"  {i}. {name}")
            choice = input("Enter number: ").strip()
            try:
                idx = int(choice) - 1
                current_prompt = list(SYSTEM_PROMPTS.keys())[idx]
                print(f"Switched to: {current_prompt}")
            except (ValueError, IndexError):
                print("Invalid choice")
            continue
        elif not user_input:
            continue

        try:
            response = chat_with_system_prompt(
                SYSTEM_PROMPTS[current_prompt],
                user_input
            )
            print(f"\nBot: {response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    backend = "Ollama (local)" if use_ollama else "OpenAI (cloud)"
    print(f"Backend: {backend}\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_prompts()
    else:
        interactive_chat()

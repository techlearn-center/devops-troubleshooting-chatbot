#!/usr/bin/env python3
"""
SOLUTION: Exercise 4 - Token Counting and Cost Estimation
==========================================================

This is the complete solution for Exercise 4.

WHAT THIS CODE DEMONSTRATES:
---------------------------
1. Accurate token counting with tiktoken
2. Visualizing how text is tokenized
3. Cost estimation for different models
4. Context window management

WHY THIS MATTERS:
----------------
- Token limits determine what fits in a request
- You pay per token (both input and output)
- Understanding tokens helps optimize costs and avoid errors

RUNNING THIS CODE:
-----------------
    # Tokenization visualization
    python module-1-llm-basics/solutions/ex4_token_counting.py --tokens

    # Cost estimation
    python module-1-llm-basics/solutions/ex4_token_counting.py --cost

    # Context limits
    python module-1-llm-basics/solutions/ex4_token_counting.py --limits

    # Interactive counter
    python module-1-llm-basics/solutions/ex4_token_counting.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Token counting library
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Install tiktoken for accurate counting: pip install tiktoken\n")


# =============================================================================
# PRICING DATA (Update periodically)
# =============================================================================
PRICING = {
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "context": 128000},
    "gpt-4o": {"input": 2.50, "output": 10.00, "context": 128000},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "context": 128000},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "context": 16385},
    "gpt-4": {"input": 30.00, "output": 60.00, "context": 8192},
    "claude-3-opus": {"input": 15.00, "output": 75.00, "context": 200000},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00, "context": 200000},
    "ollama": {"input": 0.00, "output": 0.00, "context": 4096},
}


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text."""
    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    return len(text) // 4


def visualize_tokens(text: str, model: str = "gpt-3.5-turbo"):
    """Show how text is tokenized."""
    print(f"\nText: {repr(text)}")

    if not HAS_TIKTOKEN:
        print("Install tiktoken for detailed visualization")
        return

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    token_ids = encoding.encode(text)
    tokens = [encoding.decode([tid]) for tid in token_ids]

    print(f"Total tokens: {len(token_ids)}\n")
    print("Token breakdown:")
    print("-" * 50)

    for i, (tid, token) in enumerate(zip(token_ids, tokens)):
        # Make whitespace visible
        display = token.replace(" ", "␣").replace("\n", "↵")
        print(f"  {i+1:3}. [{tid:6}] → {repr(token):20} = {display}")

    print("-" * 50)


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> dict:
    """Calculate cost for a request."""
    prices = PRICING.get(model, PRICING["gpt-3.5-turbo"])

    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }


def demo_tokenization():
    """Tokenization examples."""
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)

    examples = [
        "Hello, world!",
        "kubectl get pods",
        "Kubernetes is a container orchestration platform.",
        "docker run -d --name nginx -p 80:80 nginx:latest",
        "The CrashLoopBackOff error means the container keeps restarting.",
    ]

    for text in examples:
        visualize_tokens(text)
        print()


def demo_cost():
    """Cost estimation examples."""
    print("\n" + "=" * 60)
    print("COST ESTIMATION")
    print("=" * 60)

    # Example prompt
    prompt = """You are a DevOps expert. A user reports:

Error: ImagePullBackOff - unable to pull image "myapp:v1.2.3"
Events:
  Warning  Failed  2m  kubelet  Failed to pull image: unauthorized

What's causing this and how do I fix it?"""

    input_tokens = count_tokens(prompt)
    output_tokens = 300  # Estimated response

    print(f"\nExample prompt ({input_tokens} tokens):")
    print("-" * 40)
    print(prompt[:200] + "...")
    print("-" * 40)

    print(f"\nEstimated output: {output_tokens} tokens")
    print("\nCost by model:")
    print("-" * 60)
    print(f"{'Model':<20} {'Input':<12} {'Output':<12} {'Total':<12}")
    print("-" * 60)

    for model in ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "ollama"]:
        cost = estimate_cost(input_tokens, output_tokens, model)
        print(f"{model:<20} ${cost['input_cost']:<11.6f} ${cost['output_cost']:<11.6f} ${cost['total_cost']:<11.6f}")

    # Monthly estimate
    print("\n" + "=" * 60)
    print("MONTHLY COST ESTIMATE (1000 requests/day)")
    print("=" * 60)

    daily_requests = 1000
    monthly_requests = daily_requests * 30

    for model in ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "ollama"]:
        cost = estimate_cost(input_tokens, output_tokens, model)
        monthly = cost["total_cost"] * monthly_requests
        print(f"{model:<20} ${monthly:,.2f}/month")


def demo_context_limits():
    """Context window demonstration."""
    print("\n" + "=" * 60)
    print("CONTEXT WINDOW LIMITS")
    print("=" * 60)

    print("\nModel context windows:")
    print("-" * 60)

    for model, info in sorted(PRICING.items(), key=lambda x: x[1]["context"], reverse=True):
        ctx = info["context"]
        approx_words = int(ctx * 0.75)
        approx_pages = ctx // 750  # ~750 tokens per page

        # Visual bar
        bar_len = min(int(ctx / 5000), 30)
        bar = "█" * bar_len

        print(f"{model:<18} {ctx:>8,} tokens  ~{approx_pages:>3} pages  {bar}")

    print("-" * 60)

    # Show what fits in context
    print("""
WHAT FITS IN CONTEXT?

GPT-3.5-turbo (16K tokens):
┌────────────────────────────────────────────────┐
│ System prompt:      ~500 tokens                │
│ Chat history:       ~10,000 tokens (20 msgs)   │
│ Current question:   ~500 tokens                │
│ Reserved for reply: ~5,000 tokens              │
└────────────────────────────────────────────────┘

GPT-4-turbo (128K tokens):
┌────────────────────────────────────────────────┐
│ System prompt:      ~500 tokens                │
│ Chat history:       ~100,000 tokens (200 msgs) │
│ Current question:   ~500 tokens                │
│ Reserved for reply: ~27,000 tokens             │
└────────────────────────────────────────────────┘
""")


def interactive_counter():
    """Interactive token counter."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TOKEN COUNTER")
    print("=" * 60)
    print("Enter text to analyze. Type 'quit' to exit.\n")

    while True:
        text = input("\nText: ").strip()
        if text.lower() == 'quit':
            break
        if not text:
            continue

        visualize_tokens(text)

        tokens = count_tokens(text)
        print(f"\nCost estimates (assuming 200 token response):")
        for model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]:
            cost = estimate_cost(tokens, 200, model)
            print(f"  {model}: ${cost['total_cost']:.6f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--tokens":
            demo_tokenization()
        elif sys.argv[1] == "--cost":
            demo_cost()
        elif sys.argv[1] == "--limits":
            demo_context_limits()
    else:
        interactive_counter()

    print("""
=" * 60
KEY TAKEAWAYS
=" * 60

1. Token ≈ 4 characters ≈ 0.75 words
2. You pay for INPUT + OUTPUT tokens
3. Context = system prompt + history + question + response
4. Use cheaper models for simple tasks
5. Ollama = FREE for development
6. Monitor usage in OpenAI dashboard
""")

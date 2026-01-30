#!/usr/bin/env python3
"""
Exercise 4: Token Counting and Cost Estimation
===============================================

GOAL: Understand tokens, count them, and estimate API costs.

WHAT YOU'LL LEARN:
- What tokens are and how text is tokenized
- How to count tokens using tiktoken
- How pricing works for LLM APIs
- How to estimate costs before making calls

WHAT ARE TOKENS?
----------------
Tokens are the "pieces" that LLMs see when they read text.
Think of them as the vocabulary units the model works with.

TEXT TO TOKENS EXAMPLE:
----------------------
    Text: "Hello, Kubernetes world!"

    Tokens: ["Hello", ",", " Kubernetes", " world", "!"]
              ↑        ↑      ↑            ↑         ↑
              1        2      3            4         5

    Total: 5 tokens

WHY TOKENS ≠ WORDS:
------------------
- Common words → 1 token ("the", "and", "is")
- Long words → multiple tokens ("Kubernetes" = 3 tokens)
- Punctuation → often separate tokens (",", "!", "?")
- Spaces → often included with following word

TOKENIZATION RULES OF THUMB:
---------------------------
┌────────────────────────────────────────────────────────────────┐
│ 1 token ≈ 4 characters                                        │
│ 1 token ≈ 3/4 of a word                                       │
│ 100 tokens ≈ 75 words                                         │
│ 1000 tokens ≈ 750 words ≈ 1.5 pages of text                   │
└────────────────────────────────────────────────────────────────┘

WHY TOKENS MATTER:
-----------------
1. CONTEXT LIMITS: Each model has a maximum context window (e.g., 4096 tokens)
   - If you exceed it, the API will error or truncate

2. COST: You pay per token (both input and output)
   - Input tokens: What you send (system prompt + user message)
   - Output tokens: What the model generates

3. SPEED: More tokens = longer generation time

PRICING EXAMPLES (as of 2024):
-----------------------------
┌─────────────────────────┬─────────────────┬─────────────────┐
│ Model                   │ Input (per 1M)  │ Output (per 1M) │
├─────────────────────────┼─────────────────┼─────────────────┤
│ GPT-4-turbo             │ $10.00          │ $30.00          │
│ GPT-4o                  │ $2.50           │ $10.00          │
│ GPT-3.5-turbo           │ $0.50           │ $1.50           │
│ Claude 3 Sonnet         │ $3.00           │ $15.00          │
│ Ollama (local)          │ FREE            │ FREE            │
└─────────────────────────┴─────────────────┴─────────────────┘

Note: 1M = 1,000,000 tokens

RUN THIS:
    python module-1-llm-basics/exercises/ex4_token_counting.py

REQUIREMENTS:
    pip install tiktoken
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# TIKTOKEN - OPENAI'S TOKENIZER
# =============================================================================
# tiktoken is OpenAI's tokenizer library
# It converts text to tokens (and back) using the same algorithm as GPT models
# =============================================================================
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("=" * 60)
    print("NOTICE: tiktoken not installed")
    print("=" * 60)
    print("For accurate token counting, install it:")
    print("    pip install tiktoken")
    print("Using rough estimation instead.")
    print("=" * 60 + "\n")


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.

    WHY IS THIS IMPORTANT?
    - Each model has a maximum context size
    - You're charged per token
    - Knowing token counts helps you stay within limits

    Args:
        text: The text to count tokens for
        model: The model to use for tokenization (different models may tokenize differently)

    Returns:
        Number of tokens in the text

    Example:
        >>> tokens = count_tokens("Hello, Kubernetes!")
        >>> print(f"Token count: {tokens}")
        Token count: 4
    """
    if HAS_TIKTOKEN:
        # Get the encoding for this model
        # An "encoding" is the tokenizer's vocabulary + rules
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")

        # Encode the text and count the tokens
        tokens = encoding.encode(text)
        return len(tokens)
    else:
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4


def show_tokenization(text: str, model: str = "gpt-3.5-turbo"):
    """
    Show how text is broken into tokens.

    This visualizes the tokenization process so you can understand
    how the model "sees" your text.
    """
    print(f"\nText: {repr(text)}")
    print(f"Model: {model}")

    if HAS_TIKTOKEN:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        # Get token IDs
        token_ids = encoding.encode(text)

        # Decode each token to see what it represents
        tokens = [encoding.decode([tid]) for tid in token_ids]

        print(f"\nTokens ({len(token_ids)} total):")
        print("-" * 40)

        for i, (tid, token) in enumerate(zip(token_ids, tokens)):
            # Show the token with visible representation for special chars
            visible = repr(token)
            print(f"  {i + 1}. ID={tid:<6} → {visible}")

        print("-" * 40)
    else:
        approx_tokens = len(text) // 4
        print(f"\nApproximate tokens: {approx_tokens}")
        print("(Install tiktoken for exact tokenization)")


def estimate_cost(
    input_text: str,
    estimated_output_tokens: int = 200,
    model: str = "gpt-3.5-turbo"
) -> dict:
    """
    Estimate the cost of an API call.

    This helps you understand costs before making expensive calls.

    Args:
        input_text: The text you'll send to the API
        estimated_output_tokens: Expected response length in tokens
        model: The model you'll use

    Returns:
        Dictionary with cost breakdown

    Example:
        >>> cost = estimate_cost("What is Docker?", estimated_output_tokens=500)
        >>> print(f"Estimated cost: ${cost['total']:.4f}")
    """
    # Pricing per million tokens (approximate, check OpenAI for current prices)
    PRICING = {
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "ollama": {"input": 0.00, "output": 0.00},  # Free!
    }

    # Get pricing for this model (default to gpt-3.5-turbo)
    if model.lower().startswith("ollama") or model.lower() in ["llama2", "mistral", "codellama"]:
        prices = PRICING["ollama"]
    else:
        prices = PRICING.get(model, PRICING["gpt-3.5-turbo"])

    # Count input tokens
    input_tokens = count_tokens(input_text, model)

    # Calculate costs (price is per million tokens)
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (estimated_output_tokens / 1_000_000) * prices["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": estimated_output_tokens,
        "total_tokens": input_tokens + estimated_output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total": total_cost,
        "price_per_1m_input": prices["input"],
        "price_per_1m_output": prices["output"]
    }


def demo_tokenization():
    """
    Demonstrate tokenization with various examples.
    """
    print("\n" + "=" * 60)
    print("TOKENIZATION EXAMPLES")
    print("=" * 60)

    examples = [
        "Hello, world!",
        "What is Kubernetes?",
        "kubectl get pods -n kube-system",
        "The CrashLoopBackOff error indicates that Kubernetes is repeatedly trying to restart a container that keeps failing.",
        "docker run -d --name nginx -p 80:80 nginx:latest",
    ]

    for text in examples:
        show_tokenization(text)
        print()


def demo_cost_estimation():
    """
    Demonstrate cost estimation for different scenarios.
    """
    print("\n" + "=" * 60)
    print("COST ESTIMATION EXAMPLES")
    print("=" * 60)

    # Example DevOps question
    question = """You are a DevOps expert.

I'm getting this error when deploying to Kubernetes:
Error: ImagePullBackOff - unable to pull image "myapp:latest"

The pod description shows:
Warning  Failed     3m4s (x4 over 5m17s)   kubelet  Failed to pull image "myapp:latest": rpc error: code = Unknown desc = Error response from daemon: pull access denied

What could be causing this and how do I fix it?"""

    print("\nExample DevOps Question:")
    print("-" * 40)
    print(question[:200] + "...")

    models = ["gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "ollama"]
    output_estimates = [200, 500]  # Short and detailed responses

    print("\n" + "-" * 70)
    print(f"{'Model':<20} {'Output':<10} {'Input $':<10} {'Output $':<10} {'Total $':<10}")
    print("-" * 70)

    for model in models:
        for output_tokens in output_estimates:
            cost = estimate_cost(question, output_tokens, model)
            print(f"{model:<20} {output_tokens:<10} ${cost['input_cost']:<9.4f} ${cost['output_cost']:<9.4f} ${cost['total']:<9.4f}")

    print("-" * 70)

    print("""
KEY OBSERVATIONS:
1. Input costs are usually lower than output costs
2. GPT-4 is much more expensive than GPT-3.5
3. Ollama is FREE (runs locally on your machine)
4. Longer responses = higher costs

COST-SAVING TIPS:
- Use GPT-3.5-turbo for simple questions
- Use GPT-4 only when you need its capabilities
- Keep system prompts concise
- Set max_tokens to limit response length
- Use Ollama for development/testing (free)
""")


def demo_context_limits():
    """
    Demonstrate context window limits.
    """
    print("\n" + "=" * 60)
    print("CONTEXT WINDOW LIMITS")
    print("=" * 60)

    # Context limits for different models
    limits = {
        "gpt-3.5-turbo": 16_385,
        "gpt-4": 8_192,
        "gpt-4-turbo": 128_000,
        "gpt-4o": 128_000,
        "claude-3-sonnet": 200_000,
        "llama2": 4_096,
        "llama3": 8_192,
    }

    print("\nContext Window Sizes (in tokens):")
    print("-" * 40)

    for model, limit in limits.items():
        # Calculate approximate words and pages
        words = int(limit * 0.75)  # ~0.75 words per token
        pages = limit / 1000 * 1.5  # ~1.5 pages per 1000 tokens

        bar_length = int(limit / 5000)  # Scale for display
        bar = "█" * min(bar_length, 30)

        print(f"{model:<20} {limit:>8,} tokens ≈ {words:>6,} words  {bar}")

    print("-" * 40)

    print("""
WHAT FITS IN THE CONTEXT WINDOW?

The context window includes EVERYTHING:
- System prompt
- Conversation history (all previous messages)
- Current user message
- Space for the response

Example for gpt-3.5-turbo (16K tokens):
┌─────────────────────────────────────────────────────────────┐
│ System Prompt:              500 tokens                      │
│ Previous conversation:    10,000 tokens                     │
│ Current question:            200 tokens                     │
│ ─────────────────────────────────────────                   │
│ Total input:              10,700 tokens                     │
│ Remaining for response:    5,685 tokens                     │
└─────────────────────────────────────────────────────────────┘

If you exceed the limit:
- OpenAI: Returns an error
- Some systems: Truncate older messages
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 4: Token Counting and Cost Estimation")
    print("=" * 60)

    print("""
This exercise teaches you about tokens, counting, and costs.

OPTIONS:
1. Tokenization demo (--tokens)
2. Cost estimation demo (--cost)
3. Context limits demo (--limits)
4. Interactive counter (default)

Examples:
    python ex4_token_counting.py --tokens
    python ex4_token_counting.py --cost
    python ex4_token_counting.py --limits
""")

    if len(sys.argv) > 1:
        if sys.argv[1] == "--tokens":
            demo_tokenization()
        elif sys.argv[1] == "--cost":
            demo_cost_estimation()
        elif sys.argv[1] == "--limits":
            demo_context_limits()
        else:
            print(f"Unknown option: {sys.argv[1]}")
    else:
        # Interactive mode
        print("\n" + "=" * 60)
        print("INTERACTIVE TOKEN COUNTER")
        print("=" * 60)
        print("Enter text to see token count and cost estimate.")
        print("Type 'quit' to exit.\n")

        while True:
            text = input("\nEnter text: ").strip()

            if text.lower() == 'quit':
                break
            elif not text:
                continue

            # Show tokenization
            show_tokenization(text)

            # Show cost estimate
            print("\nCost Estimates (assuming 200 token response):")
            for model in ["gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"]:
                cost = estimate_cost(text, 200, model)
                print(f"  {model}: ${cost['total']:.6f}")

    print("""
=" * 60
TOKEN COUNTING SUMMARY
=" * 60

KEY TAKEAWAYS:
1. Tokens ≠ words (technical terms often = multiple tokens)
2. You pay for BOTH input AND output tokens
3. Context limits include everything (prompt + history + response)
4. Use tiktoken for accurate counting
5. Ollama is free for development/testing

PRACTICAL TIPS:
- Count tokens before expensive API calls
- Keep system prompts concise (they're sent every time)
- Monitor usage in your OpenAI dashboard
- Use cheaper models for simple tasks
- Test with Ollama, deploy with OpenAI
""")

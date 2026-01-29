#!/usr/bin/env python3
"""
SOLUTION: Exercise 1 - Your First LLM API Call
===============================================

This is the complete solution for Exercise 1.

HOW THIS CODE WORKS:
-------------------

1. We load environment variables from .env file
   - This includes OPENAI_API_KEY or USE_OLLAMA settings
   - Never hardcode API keys in your code!

2. We check which backend to use (OpenAI vs Ollama)
   - This lets the same code work with both services
   - Ollama = free, local; OpenAI = paid, cloud

3. We build a "messages" list with our question
   - The Chat API expects messages in a specific format
   - Each message has a "role" and "content"

4. We make the API call and extract the response
   - OpenAI: client.chat.completions.create()
   - Ollama: HTTP POST to /api/chat

5. We return just the text content of the response
   - The full response contains metadata we don't need
   - We extract: response.choices[0].message.content

RUNNING THIS CODE:
-----------------
    # Make sure your .env is configured
    python module-1-llm-basics/solutions/ex1_first_call.py

EXPECTED OUTPUT:
---------------
    Question: What does a Kubernetes pod do? Answer in one sentence.
    Asking LLM...
    Answer: A Kubernetes pod is the smallest deployable unit that
            represents a single instance of a running process in a cluster,
            which can contain one or more containers that share storage
            and network resources.
"""

import os
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
# Add the project root to Python's path so we can import shared modules.
# Path(__file__) = this file's path
# .parent = solutions/ directory
# .parent = module-1-llm-basics/ directory
# .parent = project root
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
# The dotenv library reads the .env file and sets environment variables.
# This is how we securely pass API keys without hardcoding them.
#
# Example .env file:
#   OPENAI_API_KEY=sk-proj-abc123...
#   OPENAI_MODEL=gpt-3.5-turbo
# =============================================================================
from dotenv import load_dotenv
load_dotenv()


def ask_llm(question: str) -> str:
    """
    Send a question to the LLM and return the response.

    This function supports both OpenAI (cloud) and Ollama (local).
    Which one is used depends on the USE_OLLAMA setting in .env.

    Args:
        question: The question to ask the LLM (string)

    Returns:
        The LLM's response as a string

    Raises:
        Exception: If the API call fails

    Example:
        >>> answer = ask_llm("What is Docker?")
        >>> print(answer)
        "Docker is a platform for containerizing applications..."
    """
    # =========================================================================
    # CHECK WHICH BACKEND TO USE
    # =========================================================================
    # os.getenv() reads an environment variable
    # - First argument: variable name
    # - Second argument: default if not found
    #
    # .lower() converts to lowercase so "True", "TRUE", "true" all work
    # =========================================================================
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # =====================================================================
        # OLLAMA VERSION (Free, Local)
        # =====================================================================
        # Ollama runs a local HTTP server at localhost:11434
        # We use the requests library to make HTTP calls
        #
        # Endpoint: POST /api/chat
        # Request body:
        #   {
        #     "model": "llama2",
        #     "messages": [{"role": "user", "content": "..."}],
        #     "stream": false
        #   }
        # =====================================================================
        import requests

        # Get Ollama settings from environment
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama2")

        # Make the HTTP POST request
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": question}
                ],
                "stream": False  # Get complete response, not streaming
            }
        )

        # Check if request was successful
        response.raise_for_status()  # Raises exception on error

        # Parse JSON response and extract the message content
        # Response format: {"message": {"role": "assistant", "content": "..."}}
        return response.json()["message"]["content"]

    else:
        # =====================================================================
        # OPENAI VERSION (Paid, Cloud)
        # =====================================================================
        # The OpenAI Python library handles HTTP details for us
        # We just call client.chat.completions.create()
        #
        # The client automatically reads OPENAI_API_KEY from environment
        # =====================================================================
        from openai import OpenAI

        # Create the client (reads API key from OPENAI_API_KEY env var)
        client = OpenAI()

        # Get model from environment or use default
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": question}
            ]
        )

        # Extract the response text
        # Structure: response.choices[0].message.content
        #
        # Why choices[0]?
        # - The API can return multiple responses (n parameter)
        # - By default n=1, so we only have one choice at index 0
        #
        # Why .message.content?
        # - Each choice has a message object
        # - The message has role ("assistant") and content (the text)
        return response.choices[0].message.content


# =============================================================================
# MAIN EXECUTION
# =============================================================================
# This block only runs when the script is executed directly:
#   python ex1_first_call.py  → runs this code
#   import ex1_first_call     → does NOT run this code
#
# This is Python's way of distinguishing between:
# - "I'm being run as the main program"
# - "I'm being imported as a module"
# =============================================================================
if __name__ == "__main__":
    # A simple test question
    # Note: "Answer in one sentence" helps keep the response short
    question = "What does a Kubernetes pod do? Answer in one sentence."

    print(f"Question: {question}")
    print("Asking LLM...")

    try:
        # Call our function
        response = ask_llm(question)
        print(f"Answer: {response}")

    except Exception as e:
        # If something goes wrong, print helpful error message
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("- Is your .env file configured? (see .env.example)")
        print("- If using OpenAI: Is your API key valid?")
        print("- If using Ollama: Is the server running? (ollama serve)")
        print("- Check your internet connection")

    # =========================================================================
    # BONUS: Show which backend was used
    # =========================================================================
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    backend = "Ollama (local)" if use_ollama else "OpenAI (cloud)"
    print(f"\n[Using: {backend}]")

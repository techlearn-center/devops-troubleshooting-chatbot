#!/usr/bin/env python3
"""
Exercise 1: Your First LLM API Call
====================================

GOAL: Make your first API call to a Large Language Model.

WHAT YOU'LL LEARN:
- How to structure an API request
- The "messages" format for chat APIs
- How to extract the response text
- Error handling for API calls

BACKGROUND - WHAT IS AN API CALL?
---------------------------------
Think of an API call like ordering food at a restaurant:

1. YOU (the customer) = Your Python code
2. THE WAITER = The API (takes your order, brings food)
3. THE KITCHEN = The LLM server (processes your request)
4. YOUR ORDER = The "request" (what you're asking for)
5. YOUR FOOD = The "response" (what you get back)

When you make an API call:
┌─────────────┐       Request        ┌─────────────┐
│ Your Python │ ──────────────────▶  │  OpenAI /   │
│    Code     │                      │   Ollama    │
│             │ ◀──────────────────  │   Server    │
└─────────────┘       Response       └─────────────┘

THE MESSAGES FORMAT:
-------------------
Chat APIs use a list of "messages" where each message has:
- role: Who is speaking ("user", "assistant", or "system")
- content: What they said (the actual text)

Example:
    messages = [
        {"role": "user", "content": "What is Docker?"}
    ]

This tells the API: "The user is asking 'What is Docker?'"

RUN THIS:
    python module-1-llm-basics/exercises/ex1_first_call.py
"""

import os
import sys
from pathlib import Path

# =============================================================================
# PATH SETUP - WHY DO WE NEED THIS?
# =============================================================================
# Python needs to know where to find our project files.
# sys.path is a list of directories Python searches for imports.
#
# Path(__file__) gives us the path to THIS file
# .parent goes up one directory level
#
# __file__ = "...exercises/ex1_first_call.py"
# Path(__file__).parent = "...exercises/"
# Path(__file__).parent.parent = "...module-1-llm-basics/"
# Path(__file__).parent.parent.parent = "...devops-troubleshooting-chatbot/"
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
# Environment variables are settings stored outside your code.
# This keeps sensitive data (like API keys) out of your source code.
#
# The .env file looks like this:
#   OPENAI_API_KEY=sk-proj-abc123...
#   USE_OLLAMA=false
#
# load_dotenv() reads this file and makes variables available via os.getenv()
# =============================================================================
from dotenv import load_dotenv
load_dotenv()


def ask_llm_openai(question: str) -> str:
    """
    Make an API call to OpenAI's ChatGPT.

    HOW OPENAI'S API WORKS:
    ----------------------
    1. Create a client (handles authentication automatically)
    2. Call client.chat.completions.create() with:
       - model: Which model to use (gpt-3.5-turbo, gpt-4, etc.)
       - messages: List of conversation messages
    3. Extract the response text from the returned object

    Args:
        question: The text question to send to the LLM

    Returns:
        The LLM's text response

    Example:
        >>> answer = ask_llm_openai("What is a container?")
        >>> print(answer)
        "A container is a lightweight, standalone..."
    """
    # Import the OpenAI library
    # This library handles all the HTTP details for us
    from openai import OpenAI

    # Create the client
    # The client automatically looks for OPENAI_API_KEY in environment variables
    # This is like logging into a website - proves who you are
    client = OpenAI()

    # Get the model name from environment, or use default
    # gpt-3.5-turbo is fast and cheap, good for learning
    # gpt-4 is smarter but more expensive
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    # ==========================================================================
    # THE API CALL
    # ==========================================================================
    # This is where the magic happens!
    # We're sending our question to OpenAI's servers.
    #
    # Parameters:
    # - model: Which AI model to use
    # - messages: The conversation (just one message for now)
    # ==========================================================================
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",      # We are the user
                "content": question  # This is our question
            }
        ]
    )

    # ==========================================================================
    # EXTRACT THE RESPONSE TEXT
    # ==========================================================================
    # The response object has this structure:
    #
    # response = {
    #   "choices": [          ← List of possible responses
    #     {
    #       "message": {      ← The response message
    #         "role": "assistant",
    #         "content": "Kubernetes is..."  ← THE TEXT WE WANT
    #       }
    #     }
    #   ]
    # }
    #
    # So we access: response.choices[0].message.content
    # ==========================================================================
    return response.choices[0].message.content


def ask_llm_ollama(question: str) -> str:
    """
    Make an API call to Ollama (local LLM server).

    HOW OLLAMA'S API WORKS:
    ----------------------
    Ollama runs a web server on your computer at http://localhost:11434
    We send HTTP POST requests to /api/chat

    Unlike OpenAI (where we use their library), with Ollama we make
    raw HTTP requests using the 'requests' library.

    HTTP BASICS:
    - GET: Retrieve data (like visiting a webpage)
    - POST: Send data (like submitting a form)

    For Ollama, we POST our question and receive the answer.

    Args:
        question: The text question to send

    Returns:
        The LLM's text response
    """
    # requests is Python's go-to library for making HTTP calls
    import requests

    # Get Ollama settings from environment
    # Default: localhost:11434 (Ollama's default port)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama2")

    # ==========================================================================
    # THE HTTP POST REQUEST
    # ==========================================================================
    # requests.post() sends data to a URL
    #
    # Parameters:
    # - First arg: The URL to send to
    # - json=: Data to send as JSON (automatically converts Python dict)
    #
    # The request body (json=) contains:
    # - model: Which model to use (llama2, codellama, mistral, etc.)
    # - messages: Same format as OpenAI
    # - stream: False = get complete response at once
    # ==========================================================================
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": question}
            ],
            "stream": False
        }
    )

    # Check if request was successful
    # raise_for_status() throws an exception if there was an HTTP error
    # (like 404 Not Found or 500 Server Error)
    response.raise_for_status()

    # ==========================================================================
    # EXTRACT THE RESPONSE TEXT
    # ==========================================================================
    # Ollama returns JSON like this:
    #
    # {
    #   "message": {
    #     "role": "assistant",
    #     "content": "Docker is..."  ← THE TEXT WE WANT
    #   }
    # }
    #
    # response.json() converts the JSON to a Python dictionary
    # ==========================================================================
    return response.json()["message"]["content"]


def ask_llm(question: str) -> str:
    """
    Send a question to the LLM (works with both OpenAI and Ollama).

    This function checks the USE_OLLAMA environment variable to decide
    which backend to use. This makes it easy to switch between them.

    Args:
        question: Your question for the LLM

    Returns:
        The LLM's response as a string
    """
    # Check which backend to use
    # os.getenv("USE_OLLAMA", "false") returns:
    #   - The value of USE_OLLAMA if it exists
    #   - "false" if it doesn't exist (the default)
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        return ask_llm_ollama(question)
    else:
        return ask_llm_openai(question)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
# Everything below only runs when you execute this file directly:
#   python ex1_first_call.py  → RUNS this code
#   import ex1_first_call     → Does NOT run this code
#
# __name__ is a special Python variable:
# - When running directly: __name__ == "__main__"
# - When imported: __name__ == "ex1_first_call"
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 1: Your First LLM API Call")
    print("=" * 60)

    # Show which backend we're using
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
    backend = "Ollama (local)" if use_ollama else "OpenAI (cloud)"
    print(f"Backend: {backend}")
    print("-" * 60)

    # Our test question
    # "Answer in one sentence" keeps the response short for this demo
    question = "What does a Kubernetes pod do? Answer in one sentence."

    print(f"\nQuestion: {question}")
    print("\nAsking the LLM...")

    # Try to make the API call
    try:
        response = ask_llm(question)
        print(f"\n✅ Answer: {response}")

    except Exception as e:
        # If something goes wrong, show helpful error message
        print(f"\n❌ Error: {e}")
        print("\n" + "=" * 40)
        print("TROUBLESHOOTING:")
        print("=" * 40)

        if use_ollama:
            print("You're using Ollama. Check:")
            print("1. Is Ollama installed? (Download from ollama.ai)")
            print("2. Is the server running? Run: ollama serve")
            print("3. Is the model downloaded? Run: ollama pull llama2")
        else:
            print("You're using OpenAI. Check:")
            print("1. Is your API key in .env file?")
            print("2. Is the API key valid?")
            print("3. Do you have credits in your OpenAI account?")

    print("\n" + "=" * 60)
    print("Exercise 1 Complete!")
    print("=" * 60)

    # ==========================================================================
    # TRY IT YOURSELF
    # ==========================================================================
    print("""
TRY IT YOURSELF:
---------------
1. Try changing the question to something else:
   - "What is Docker?"
   - "How do I list all running containers?"
   - "What is the difference between git merge and rebase?"

2. Try switching backends:
   - Set USE_OLLAMA=true in your .env file
   - Or run: USE_OLLAMA=true python ex1_first_call.py

3. Check the response structure:
   - Add a print statement to see the full response object
   - What other information is included?
""")

#!/usr/bin/env python3
"""
Exercise 3: Streaming Responses
===============================

GOAL: Show responses word-by-word as they're generated.

WHAT YOU'LL LEARN:
- How to enable streaming in API calls
- How to process streamed chunks
- Why streaming improves user experience

HOW STREAMING WORKS:
- Without streaming: Wait for complete response, then show all at once
- With streaming: Show each word/token as it's generated

BENEFITS:
- Feels faster (even if total time is same)
- User can start reading immediately
- Can cancel if response is going wrong direction

RUN THIS:
    python module-2-simple-chatbot/exercises/ex3_streaming.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


def stream_response_openai(messages: list) -> str:
    """
    Stream a response from OpenAI.

    With stream=True, the API returns an iterator instead of a complete
    response. We get small chunks of text as they're generated.

    Args:
        messages: The conversation history

    Returns:
        The complete response (after streaming finishes)
    """
    from openai import OpenAI
    client = OpenAI()

    # =========================================================================
    # MAKE STREAMING API CALL
    # =========================================================================
    # stream=True changes the response type:
    # - Without streaming: response is a complete object
    # - With streaming: response is an iterator of chunks
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=messages,
        stream=True  # Enable streaming
    )

    # =========================================================================
    # PROCESS CHUNKS AS THEY ARRIVE
    # =========================================================================
    full_response = ""

    for chunk in response:
        # Each chunk has this structure:
        # chunk.choices[0].delta.content = the new text piece (or None)

        # Check if there's content in this chunk
        if chunk.choices[0].delta.content is not None:
            text_piece = chunk.choices[0].delta.content

            # Print immediately without newline
            # flush=True ensures it appears right away
            print(text_piece, end="", flush=True)

            # Accumulate for the complete response
            full_response += text_piece

    # Print newline at the end
    print()

    return full_response


def stream_response_ollama(messages: list) -> str:
    """
    Stream a response from Ollama.

    Ollama also supports streaming via its API. The response is
    newline-delimited JSON objects.

    Args:
        messages: The conversation history

    Returns:
        The complete response
    """
    import requests
    import json

    # =========================================================================
    # MAKE STREAMING API CALL
    # =========================================================================
    response = requests.post(
        f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "llama2"),
            "messages": messages,
            "stream": True  # Enable streaming
        },
        stream=True  # Also tell requests to stream
    )

    # =========================================================================
    # PROCESS CHUNKS AS THEY ARRIVE
    # =========================================================================
    full_response = ""

    for line in response.iter_lines():
        if line:
            # Each line is a JSON object
            data = json.loads(line)

            # Extract the content piece
            if "message" in data and "content" in data["message"]:
                text_piece = data["message"]["content"]
                print(text_piece, end="", flush=True)
                full_response += text_piece

    print()
    return full_response


def stream_response(messages: list) -> str:
    """
    Stream response from configured backend (OpenAI or Ollama).
    """
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        return stream_response_ollama(messages)
    else:
        return stream_response_openai(messages)


class StreamingChatbot:
    """Chatbot that streams responses."""

    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful DevOps assistant."}
        ]

    def chat(self, user_message: str) -> str:
        """Send a message and stream the response."""
        self.messages.append({"role": "user", "content": user_message})

        print("\nBot: ", end="", flush=True)
        response = stream_response(self.messages)

        self.messages.append({"role": "assistant", "content": response})
        return response

    def run(self):
        """Run the streaming chatbot."""
        print("=" * 50)
        print("Streaming Chatbot Demo")
        print("=" * 50)
        print("Watch how text appears word by word!")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif not user_input:
                continue

            try:
                self.chat(user_input)
            except Exception as e:
                print(f"\nError: {e}")


# =============================================================================
# COMPARISON DEMO
# =============================================================================
def compare_streaming():
    """
    Compare streaming vs non-streaming to see the difference.
    """
    messages = [
        {"role": "system", "content": "You are a DevOps assistant."},
        {"role": "user", "content": "Explain what happens when a Kubernetes deployment fails. Be detailed."}
    ]

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    print("=" * 60)
    print("STREAMING COMPARISON")
    print("=" * 60)

    # Non-streaming version
    print("\n[1] WITHOUT STREAMING (wait for complete response):")
    print("-" * 50)

    import time
    start = time.time()

    if use_ollama:
        import requests
        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={"model": os.getenv("OLLAMA_MODEL", "llama2"), "messages": messages, "stream": False}
        )
        text = response.json()["message"]["content"]
    else:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            stream=False
        )
        text = response.choices[0].message.content

    elapsed = time.time() - start
    print(text[:200] + "...")
    print(f"\n[Time to first character: {elapsed:.2f}s]")

    # Streaming version
    print("\n\n[2] WITH STREAMING (text appears as generated):")
    print("-" * 50)

    start = time.time()
    first_char_time = None

    if use_ollama:
        import requests
        import json
        response = requests.post(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            json={"model": os.getenv("OLLAMA_MODEL", "llama2"), "messages": messages, "stream": True},
            stream=True
        )
        char_count = 0
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "message" in data and "content" in data["message"]:
                    text_piece = data["message"]["content"]
                    if first_char_time is None:
                        first_char_time = time.time() - start
                    print(text_piece, end="", flush=True)
                    char_count += len(text_piece)
                    if char_count > 200:
                        print("...")
                        break
    else:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=messages,
            stream=True
        )
        char_count = 0
        for chunk in response:
            if chunk.choices[0].delta.content:
                text_piece = chunk.choices[0].delta.content
                if first_char_time is None:
                    first_char_time = time.time() - start
                print(text_piece, end="", flush=True)
                char_count += len(text_piece)
                if char_count > 200:
                    print("...")
                    break

    print(f"\n\n[Time to first character: {first_char_time:.2f}s]")
    print("\nNotice: Streaming shows text much faster!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_streaming()
    else:
        bot = StreamingChatbot()
        bot.run()

#!/usr/bin/env python3
"""
Exercise 4: Error Handling and Retry Logic
==========================================

GOAL: Handle API errors gracefully with retry logic.

WHAT YOU'LL LEARN:
- Common API errors and how to handle them
- Try/except blocks for error handling
- Exponential backoff retry strategy
- Making your chatbot robust

COMMON ERRORS:
- RateLimitError: Too many requests, need to slow down
- APIConnectionError: Network issues
- AuthenticationError: Invalid API key
- Timeout: Request took too long

EXPONENTIAL BACKOFF:
- Retry 1: Wait 1 second  (2^0)
- Retry 2: Wait 2 seconds (2^1)
- Retry 3: Wait 4 seconds (2^2)
- Retry 4: Wait 8 seconds (2^3)
- Doubles each time to reduce server load

RUN THIS:
    python module-2-simple-chatbot/exercises/ex4_error_handling.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


class RobustChatbot:
    """
    A chatbot with proper error handling.

    This class demonstrates:
    - Try/except for different error types
    - Retry logic with exponential backoff
    - Graceful degradation when all retries fail
    """

    def __init__(self, max_retries: int = 3):
        """
        Initialize the chatbot.

        Args:
            max_retries: Number of times to retry failed requests
        """
        self.messages = [
            {"role": "system", "content": "You are a helpful DevOps assistant."}
        ]
        self.max_retries = max_retries

        # Set up backend
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        if self.use_ollama:
            import requests
            self.session = requests.Session()
        else:
            from openai import OpenAI
            self.client = OpenAI()

    def _call_llm(self) -> str:
        """
        Make the actual LLM API call.

        This is separated from chat() to keep the retry logic clean.

        Returns:
            The assistant's response text

        Raises:
            Various exceptions that chat() will handle
        """
        if self.use_ollama:
            response = self.session.post(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "llama2"),
                    "messages": self.messages,
                    "stream": False
                },
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json()["message"]["content"]
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=self.messages,
                temperature=0.3
            )
            return response.choices[0].message.content

    def chat(self, user_message: str) -> str:
        """
        Send a message with comprehensive error handling.

        This method:
        1. Adds user message to history
        2. Attempts to call LLM
        3. On failure, retries with exponential backoff
        4. Returns response or error message

        Args:
            user_message: The user's input

        Returns:
            The assistant's response or an error message
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # =====================================================================
        # RETRY LOOP
        # =====================================================================
        for attempt in range(self.max_retries):
            try:
                # Attempt the API call
                response = self._call_llm()

                # Success! Add to history and return
                self.messages.append({"role": "assistant", "content": response})
                return response

            # =================================================================
            # HANDLE SPECIFIC ERROR TYPES
            # =================================================================
            except Exception as e:
                error_type = type(e).__name__
                error_message = str(e)

                # Check for rate limiting
                # Different libraries may raise different exceptions
                if "RateLimit" in error_type or "rate" in error_message.lower():
                    # =========================================================
                    # RATE LIMIT: Exponential Backoff
                    # =========================================================
                    wait_time = 2 ** attempt  # 1, 2, 4, 8, ...
                    print(f"\n⚠️  Rate limited. Waiting {wait_time} seconds...")
                    print(f"    (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(wait_time)
                    continue

                # Check for connection errors
                elif "Connection" in error_type or "connection" in error_message.lower():
                    # =========================================================
                    # CONNECTION ERROR: Simple Retry
                    # =========================================================
                    print(f"\n⚠️  Connection error. Retrying...")
                    print(f"    (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(1)
                    continue

                # Check for authentication errors
                elif "Authentication" in error_type or "auth" in error_message.lower():
                    # =========================================================
                    # AUTH ERROR: Don't retry, it won't help
                    # =========================================================
                    print(f"\n❌ Authentication error: {error_message}")
                    print("    Check your API key in .env file")
                    # Remove the failed user message from history
                    self.messages.pop()
                    return "Error: Invalid API key. Please check your configuration."

                # Check for timeout
                elif "Timeout" in error_type or "timeout" in error_message.lower():
                    # =========================================================
                    # TIMEOUT: Retry with longer wait
                    # =========================================================
                    print(f"\n⚠️  Request timed out. Retrying...")
                    print(f"    (Attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(2)
                    continue

                # Unknown error
                else:
                    # =========================================================
                    # UNKNOWN ERROR: Log and retry
                    # =========================================================
                    print(f"\n⚠️  Error: {error_type}: {error_message[:100]}")
                    print(f"    (Attempt {attempt + 1}/{self.max_retries})")

                    # Only retry if we haven't exhausted attempts
                    if attempt < self.max_retries - 1:
                        time.sleep(1)
                        continue

        # =====================================================================
        # ALL RETRIES FAILED
        # =====================================================================
        # Remove the user message since we couldn't process it
        self.messages.pop()

        return (
            "I'm sorry, I'm having trouble connecting right now. "
            "Please try again in a few moments."
        )

    def run(self):
        """Run the chatbot."""
        print("=" * 50)
        print("Robust Chatbot (with error handling)")
        print("=" * 50)
        print("This chatbot handles errors gracefully!")
        print("Type 'quit' to exit\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif not user_input:
                continue

            response = self.chat(user_input)
            print(f"\nBot: {response}")


# =============================================================================
# ERROR SIMULATION FOR TESTING
# =============================================================================
def simulate_errors():
    """
    Demonstrate error handling with simulated errors.

    This helps you understand how the error handling works
    without actually hitting API limits.
    """
    print("=" * 60)
    print("ERROR HANDLING SIMULATION")
    print("=" * 60)

    class SimulatedError(Exception):
        pass

    # Simulate rate limiting
    print("\n[1] Simulating Rate Limit Error:")
    print("-" * 40)
    for attempt in range(3):
        try:
            if attempt < 2:  # Fail first 2 attempts
                raise SimulatedError("RateLimitError: Too many requests")
            print(f"  Attempt {attempt + 1}: Success!")
        except SimulatedError as e:
            wait_time = 2 ** attempt
            print(f"  Attempt {attempt + 1}: {e}")
            print(f"  Waiting {wait_time} seconds...")
            time.sleep(0.5)  # Short sleep for demo

    # Simulate connection error
    print("\n[2] Simulating Connection Error:")
    print("-" * 40)
    for attempt in range(3):
        try:
            if attempt < 1:  # Fail first attempt
                raise SimulatedError("ConnectionError: Network unreachable")
            print(f"  Attempt {attempt + 1}: Success!")
            break
        except SimulatedError as e:
            print(f"  Attempt {attempt + 1}: {e}")
            print("  Retrying...")
            time.sleep(0.5)

    print("\n[3] Simulating Auth Error (no retry):")
    print("-" * 40)
    try:
        raise SimulatedError("AuthenticationError: Invalid API key")
    except SimulatedError as e:
        print(f"  Error: {e}")
        print("  NOT retrying - auth errors need user action")

    print("\n" + "=" * 60)
    print("Error handling simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        simulate_errors()
    else:
        bot = RobustChatbot()
        bot.run()

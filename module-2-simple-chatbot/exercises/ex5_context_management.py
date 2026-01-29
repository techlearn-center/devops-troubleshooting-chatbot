#!/usr/bin/env python3
"""
Exercise 5: Context Window Management
=====================================

GOAL: Keep conversations within token limits to avoid errors.

WHAT YOU'LL LEARN:
- Why context management is necessary
- How to count tokens
- Strategies for trimming conversation history
- Keeping the most relevant context

THE PROBLEM:
- LLMs have a maximum "context window" (e.g., 4096 tokens)
- Your messages must fit within this limit
- Long conversations can exceed the limit
- Exceeding causes errors or lost context

THE SOLUTION:
- Count tokens in messages
- When approaching limit, remove oldest messages
- Always keep the system prompt
- Optionally: summarize old context instead of deleting

RUN THIS:
    python module-2-simple-chatbot/exercises/ex5_context_management.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Note: For accurate token counting, install tiktoken:")
    print("      pip install tiktoken")
    print("      Using rough estimation instead.\n")


class ContextManagedChatbot:
    """
    A chatbot that manages its context window.

    This prevents errors from exceeding token limits and
    keeps API costs down by not sending unnecessary history.
    """

    def __init__(self, max_context_tokens: int = 2000):
        """
        Initialize the chatbot.

        Args:
            max_context_tokens: Maximum tokens to keep in history.
                               Leave room for the response!
                               e.g., 4096 limit - 1000 for response = 3096 for context
        """
        self.system_prompt = "You are a helpful DevOps assistant."
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.max_context_tokens = max_context_tokens

        # For token counting
        if HAS_TIKTOKEN:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # Set up backend
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        if self.use_ollama:
            import requests
            self.session = requests.Session()
        else:
            from openai import OpenAI
            self.client = OpenAI()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: The text to count

        Returns:
            Number of tokens (exact with tiktoken, estimated otherwise)
        """
        if HAS_TIKTOKEN:
            # Accurate counting using OpenAI's tokenizer
            return len(self.encoding.encode(text))
        else:
            # Rough estimation: ~4 characters per token
            return len(text) // 4

    def count_message_tokens(self, message: dict) -> int:
        """
        Count tokens in a message.

        Each message has overhead beyond just the content:
        - Role name
        - Formatting tokens

        Args:
            message: A message dict with role and content

        Returns:
            Approximate token count for the message
        """
        # Content tokens
        content_tokens = self.count_tokens(message["content"])

        # Add overhead for message structure (approximately 4 tokens)
        overhead = 4

        return content_tokens + overhead

    def get_total_tokens(self) -> int:
        """
        Get total tokens in current conversation.

        Returns:
            Total token count for all messages
        """
        return sum(self.count_message_tokens(msg) for msg in self.messages)

    def manage_context(self):
        """
        Trim conversation history to fit within token limit.

        Strategy: Remove oldest messages (except system prompt)
        until we're under the limit.

        More advanced strategies could:
        - Summarize removed messages
        - Keep "important" messages
        - Use embedding similarity to keep relevant messages
        """
        # Keep trimming until we're under the limit
        while self.get_total_tokens() > self.max_context_tokens:
            # Need at least: system prompt + current exchange
            if len(self.messages) <= 2:
                print("⚠️  Warning: Even minimal messages exceed token limit!")
                break

            # Remove the oldest non-system message
            # messages[0] is system, messages[1] is oldest user/assistant
            removed = self.messages.pop(1)
            print(f"  📝 Trimmed old message: [{removed['role']}] {removed['content'][:30]}...")

    def chat(self, user_message: str) -> str:
        """
        Send a message with context management.

        Args:
            user_message: The user's input

        Returns:
            The assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Show token usage before management
        tokens_before = self.get_total_tokens()
        print(f"\n  📊 Tokens before: {tokens_before}/{self.max_context_tokens}")

        # Manage context if needed
        self.manage_context()

        # Show token usage after management
        tokens_after = self.get_total_tokens()
        if tokens_after != tokens_before:
            print(f"  📊 Tokens after trim: {tokens_after}/{self.max_context_tokens}")

        # Make API call
        if self.use_ollama:
            response = self.session.post(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "llama2"),
                    "messages": self.messages,
                    "stream": False
                }
            )
            assistant_message = response.json()["message"]["content"]
        else:
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=self.messages,
                temperature=0.3
            )
            assistant_message = response.choices[0].message.content

        # Add response to history
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def show_context_stats(self):
        """Display current context window usage."""
        total = self.get_total_tokens()
        percentage = (total / self.max_context_tokens) * 100
        messages = len(self.messages)

        print("\n" + "=" * 50)
        print("CONTEXT WINDOW STATUS")
        print("=" * 50)
        print(f"Messages: {messages}")
        print(f"Tokens: {total}/{self.max_context_tokens} ({percentage:.1f}%)")

        # Visual bar
        bar_width = 40
        filled = int(bar_width * total / self.max_context_tokens)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"Usage: [{bar}]")

        # Per-message breakdown
        print("\nMessage breakdown:")
        for i, msg in enumerate(self.messages):
            tokens = self.count_message_tokens(msg)
            role = msg["role"][:6].ljust(6)
            preview = msg["content"][:30].replace("\n", " ")
            print(f"  {i}. [{role}] {tokens:4} tokens: {preview}...")

        print("=" * 50 + "\n")

    def run(self):
        """Run the chatbot."""
        print("=" * 50)
        print("Context-Managed Chatbot")
        print("=" * 50)
        print(f"Max context: {self.max_context_tokens} tokens")
        print("Commands: 'quit', 'stats', 'clear'\n")

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'stats':
                self.show_context_stats()
                continue
            elif user_input.lower() == 'clear':
                self.messages = [self.messages[0]]
                print("Conversation cleared.")
                continue
            elif not user_input:
                continue

            try:
                response = self.chat(user_input)
                print(f"\nBot: {response}")
            except Exception as e:
                print(f"\nError: {e}")


# =============================================================================
# DEMONSTRATION OF CONTEXT FILLING
# =============================================================================
def demo_context_filling():
    """
    Demonstrate what happens when context fills up.

    This creates many messages to show trimming in action.
    """
    print("=" * 60)
    print("CONTEXT FILLING DEMONSTRATION")
    print("=" * 60)
    print("We'll simulate a long conversation to see trimming in action.\n")

    # Small limit for demo purposes
    bot = ContextManagedChatbot(max_context_tokens=500)

    # Simulate conversation
    messages = [
        "What is Docker?",
        "How do I install it?",
        "What is a container?",
        "How is it different from a VM?",
        "What is Kubernetes?",
        "How do Docker and Kubernetes work together?",
    ]

    for msg in messages:
        print(f"\n{'='*50}")
        print(f"User: {msg}")
        print("=" * 50)

        # For demo, we'll just add messages without calling API
        bot.messages.append({"role": "user", "content": msg})
        bot.messages.append({"role": "assistant", "content": f"Response about {msg[:20]}..."})

        # Show stats and manage context
        total = bot.get_total_tokens()
        print(f"Tokens: {total}/{bot.max_context_tokens}")

        if total > bot.max_context_tokens:
            print("⚠️  Over limit! Trimming...")
            bot.manage_context()

    print("\n\nFinal conversation state:")
    bot.show_context_stats()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_context_filling()
    else:
        bot = ContextManagedChatbot()
        bot.run()

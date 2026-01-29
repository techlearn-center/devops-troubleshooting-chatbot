#!/usr/bin/env python3
"""
Exercise 1: Chatbot with Conversation History
=============================================

GOAL: Build a chatbot that remembers previous messages in the conversation.

WHAT YOU'LL LEARN:
- How to store conversation history in a list
- How to add messages to history
- How the LLM uses history for context

HOW IT WORKS:
1. Store messages in a list: self.messages
2. Add user message to list before calling LLM
3. Call LLM with ENTIRE message list (not just last message)
4. Add LLM response to list
5. Repeat!

RUN THIS:
    python module-2-simple-chatbot/exercises/ex1_chatbot_history.py

TEST IT:
    Ask: "What is Docker?"
    Then ask: "How do I install it?"
    The bot should know "it" refers to Docker!
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
# This sets the chatbot's personality and expertise.
# It's the first message in every conversation.
SYSTEM_PROMPT = """You are an expert DevOps assistant specializing in:
- Terraform and Infrastructure as Code
- Kubernetes and container orchestration
- Docker and containerization
- CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI)

Guidelines:
1. Be concise but thorough
2. Include code examples when helpful
3. Suggest best practices
4. Ask clarifying questions if needed
"""


class SimpleChatbot:
    """
    A simple chatbot with conversation history.

    The key insight: LLMs don't have memory between API calls.
    We must send the FULL conversation history with each request.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        """
        Initialize the chatbot.

        Args:
            system_prompt: The system message that defines chatbot behavior

        The messages list stores the entire conversation:
        - First item: system prompt (always kept)
        - Following items: alternating user/assistant messages
        """
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Check which backend to use
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        # Initialize client
        if self.use_ollama:
            import requests
            self.session = requests.Session()
        else:
            from openai import OpenAI
            self.client = OpenAI()

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        This method:
        1. Adds user message to history
        2. Calls LLM with full history
        3. Adds response to history
        4. Returns the response

        Args:
            user_message: What the user typed

        Returns:
            The assistant's response
        """
        # =====================================================================
        # STEP 1: Add user message to history
        # =====================================================================
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # =====================================================================
        # STEP 2: Call the LLM with FULL conversation history
        # =====================================================================
        if self.use_ollama:
            # Ollama API
            response = self.session.post(
                f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
                json={
                    "model": os.getenv("OLLAMA_MODEL", "llama2"),
                    "messages": self.messages,  # Send ALL messages
                    "stream": False
                }
            )
            assistant_message = response.json()["message"]["content"]
        else:
            # OpenAI API
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=self.messages,  # Send ALL messages
                temperature=0.3
            )
            assistant_message = response.choices[0].message.content

        # =====================================================================
        # STEP 3: Add assistant response to history
        # =====================================================================
        self.messages.append({
            "role": "assistant",
            "content": assistant_message
        })

        # =====================================================================
        # STEP 4: Return the response
        # =====================================================================
        return assistant_message

    def clear_history(self):
        """
        Clear conversation history, keeping only system prompt.

        This resets the conversation while maintaining the
        chatbot's personality/instructions.
        """
        # Keep only the first message (system prompt)
        self.messages = [self.messages[0]]

    def show_history(self):
        """Debug helper: show current conversation history."""
        print("\n--- Conversation History ---")
        for i, msg in enumerate(self.messages):
            role = msg["role"].upper()
            content = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"{i}. [{role}] {content}")
        print("----------------------------\n")


def main():
    """Run the chatbot in an interactive loop."""
    print("=" * 50)
    print("DevOps Chatbot (with conversation history)")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'clear' to reset, 'history' to see messages")
    print()

    # Create chatbot instance
    chatbot = SimpleChatbot()

    # Main loop
    while True:
        # Get user input
        user_input = input("You: ").strip()

        # Handle commands
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("Conversation cleared.")
            continue
        elif user_input.lower() == 'history':
            chatbot.show_history()
            continue
        elif not user_input:
            continue

        # Get and display response
        try:
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

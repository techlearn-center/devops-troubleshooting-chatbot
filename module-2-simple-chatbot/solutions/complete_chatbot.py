#!/usr/bin/env python3
"""
Complete DevOps Chatbot - Full Solution
======================================

This is the complete, production-ready chatbot that combines:
- Conversation history management
- Rich CLI interface
- Streaming responses
- Error handling with retry logic
- Context window management
- Support for both OpenAI and Ollama

HOW TO RUN:
----------
    # With OpenAI (needs API key in .env)
    python module-2-simple-chatbot/solutions/complete_chatbot.py

    # With Ollama (free, needs Ollama running)
    USE_OLLAMA=true python module-2-simple-chatbot/solutions/complete_chatbot.py

COMMANDS:
--------
    quit  - Exit the chatbot
    clear - Clear conversation history
"""

import os
import sys
import time
from pathlib import Path

# =============================================================================
# PATH AND ENVIRONMENT SETUP
# =============================================================================
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# IMPORTS
# =============================================================================
# Rich library for beautiful terminal output
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Token counting (optional, for context management)
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("Note: Install tiktoken for better context management: pip install tiktoken")


class DevOpsChatbot:
    """
    A complete DevOps troubleshooting chatbot.

    This class encapsulates all chatbot functionality:
    - Conversation history storage
    - LLM communication (OpenAI or Ollama)
    - Error handling with retry
    - Context window management
    - Rich terminal interface

    Attributes:
        messages: List of conversation messages
        console: Rich console for output
        max_context_tokens: Maximum tokens to keep in history
        max_retries: Number of times to retry failed requests

    Example:
        >>> bot = DevOpsChatbot()
        >>> bot.run()  # Starts interactive chat loop
    """

    # =========================================================================
    # SYSTEM PROMPT
    # =========================================================================
    # This defines the chatbot's personality and expertise.
    # A good system prompt significantly improves response quality.
    SYSTEM_PROMPT = """You are an expert DevOps troubleshooting assistant.

Your expertise includes:
- Terraform (state management, providers, modules, errors)
- Kubernetes (pods, deployments, services, networking, RBAC)
- Docker (builds, containers, compose, networking)
- CI/CD (GitHub Actions, GitLab CI, Jenkins)
- Cloud platforms (AWS, GCP, Azure basics)

When helping users:
1. First understand the error or problem clearly
2. Explain what's causing the issue (the "why")
3. Provide step-by-step solutions with commands
4. Include code examples in markdown code blocks
5. Suggest prevention tips when relevant
6. Ask clarifying questions if the problem is unclear

Format your responses using Markdown:
- Use **bold** for important terms
- Use `inline code` for commands and values
- Use ```language code blocks for multi-line code
- Use bullet points for lists

Be concise but thorough. Focus on practical, actionable solutions."""

    def __init__(
        self,
        system_prompt: str = None,
        max_context_tokens: int = 3000,
        max_retries: int = 3
    ):
        """
        Initialize the chatbot.

        Args:
            system_prompt: Custom system prompt (uses default if None)
            max_context_tokens: Max tokens to keep in history before trimming
            max_retries: Number of retry attempts for failed API calls
        """
        # Use custom or default system prompt
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

        # Initialize conversation with system prompt
        # This list will store ALL messages in the conversation
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Configuration
        self.max_context_tokens = max_context_tokens
        self.max_retries = max_retries

        # Rich console for pretty output
        self.console = Console()

        # Check which backend to use
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        # Initialize the appropriate client
        if self.use_ollama:
            import requests
            self.session = requests.Session()
            self.model = os.getenv("OLLAMA_MODEL", "llama2")
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            from openai import OpenAI
            self.client = OpenAI()
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        This is used for context window management - we need to know
        how many tokens we're using to stay within limits.

        Args:
            text: The text to count tokens for

        Returns:
            Approximate number of tokens
        """
        if HAS_TIKTOKEN:
            # Use tiktoken for accurate counting (OpenAI's tokenizer)
            try:
                encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
                return len(encoding.encode(text))
            except Exception:
                pass

        # Fallback: rough estimation (4 chars ≈ 1 token)
        return len(text) // 4

    def _manage_context(self):
        """
        Keep conversation history within token limits.

        Strategy: Remove oldest messages (except system prompt) until
        we're under the token limit.

        This prevents errors from exceeding the context window and
        keeps API costs down.
        """
        while True:
            # Calculate total tokens in all messages
            total_tokens = sum(
                self._count_tokens(msg["content"])
                for msg in self.messages
            )

            # If under limit, we're good
            if total_tokens <= self.max_context_tokens:
                break

            # If we only have system prompt + 1 exchange, can't remove more
            if len(self.messages) <= 3:
                break

            # Remove the oldest non-system message
            # messages[0] is system, messages[1] is oldest user/assistant
            self.messages.pop(1)

    def _call_ollama(self, stream: bool = True):
        """
        Call Ollama API for chat completion.

        Args:
            stream: Whether to stream the response

        Yields:
            Response text chunks (if streaming) or complete response
        """
        import json

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": self.messages,
                "stream": stream
            },
            stream=stream
        )
        response.raise_for_status()

        if stream:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
        else:
            data = response.json()
            yield data["message"]["content"]

    def _call_openai(self, stream: bool = True):
        """
        Call OpenAI API for chat completion.

        Args:
            stream: Whether to stream the response

        Yields:
            Response text chunks (if streaming) or complete response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.2,  # Low for consistent technical responses
            max_tokens=1000,
            stream=stream
        )

        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield response.choices[0].message.content

    def chat(self, user_message: str, stream: bool = True) -> str:
        """
        Send a message and get a response.

        This is the main chat method that:
        1. Adds user message to history
        2. Manages context window size
        3. Calls the LLM with retry logic
        4. Adds response to history
        5. Returns the response

        Args:
            user_message: The user's input
            stream: Whether to stream the response

        Returns:
            The assistant's complete response
        """
        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # Trim history if needed to stay within context limits
        self._manage_context()

        # Try to get response with retry logic
        for attempt in range(self.max_retries):
            try:
                # Collect the full response
                full_response = ""

                # Call the appropriate backend
                if self.use_ollama:
                    generator = self._call_ollama(stream=stream)
                else:
                    generator = self._call_openai(stream=stream)

                # Process response chunks
                for chunk in generator:
                    if stream:
                        # Print each chunk as it arrives
                        self.console.print(chunk, end="")
                    full_response += chunk

                if stream:
                    self.console.print()  # Newline after streaming

                # Add assistant response to history
                self.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

                return full_response

            except Exception as e:
                error_name = type(e).__name__

                # Check if it's a rate limit error
                if "RateLimit" in error_name or "rate" in str(e).lower():
                    wait_time = 2 ** attempt
                    self.console.print(
                        f"[yellow]Rate limited. Waiting {wait_time}s...[/yellow]"
                    )
                    time.sleep(wait_time)
                    continue

                # For other errors, retry with backoff
                if attempt < self.max_retries - 1:
                    self.console.print(
                        f"[yellow]Error: {error_name}. Retrying...[/yellow]"
                    )
                    time.sleep(1)
                    continue

                # Max retries reached
                self.console.print(f"[red]Error: {e}[/red]")
                return "Sorry, I'm having trouble connecting. Please try again."

    def clear_history(self):
        """
        Clear conversation history, keeping only the system prompt.

        Use this when you want to start a fresh conversation
        without the context of previous messages.
        """
        self.messages = [self.messages[0]]
        self.console.print("[yellow]Conversation history cleared.[/yellow]")

    def run(self):
        """
        Run the interactive chatbot loop.

        This starts the main chat interface where users can:
        - Type messages to chat
        - Type 'quit' or 'exit' to leave
        - Type 'clear' to reset history
        """
        # Display welcome message
        backend = "Ollama (local)" if self.use_ollama else f"OpenAI ({self.model})"
        self.console.print(Panel(
            f"[bold cyan]DevOps Troubleshooting Chatbot[/bold cyan]\n\n"
            f"I can help with Terraform, Kubernetes, Docker, and CI/CD issues.\n"
            f"Paste your error message or describe your problem.\n\n"
            f"[dim]Backend: {backend}[/dim]\n"
            f"[dim]Commands: 'quit' to exit, 'clear' to reset history[/dim]",
            border_style="cyan"
        ))

        # Main chat loop
        while True:
            try:
                # Get user input with styled prompt
                self.console.print()
                user_input = Prompt.ask("[bold green]You[/bold green]")

                # Handle empty input
                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.console.print("[cyan]Goodbye![/cyan]")
                    break

                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue

                # Show thinking indicator and get response
                self.console.print("\n[bold blue]Assistant[/bold blue]")

                # Get response (streaming by default)
                response = self.chat(user_input, stream=True)

                # Note: Response is already printed during streaming
                # If you want it in a panel, set stream=False and use:
                # self.console.print(Panel(
                #     Markdown(response),
                #     title="[bold blue]Assistant[/bold blue]",
                #     border_style="blue"
                # ))

            except KeyboardInterrupt:
                self.console.print("\n[cyan]Goodbye![/cyan]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Create and run the chatbot
    chatbot = DevOpsChatbot()
    chatbot.run()

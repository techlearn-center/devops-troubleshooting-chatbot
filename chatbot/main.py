#!/usr/bin/env python3
"""
DevOps Troubleshooting Chatbot
==============================
A RAG-powered chatbot for DevOps troubleshooting.

This is the main entry point for the chatbot. It combines:
- RAG retrieval for context-aware responses
- Conversation history for multi-turn interactions
- Streaming responses for better UX
- Multiple LLM backends (OpenAI, Ollama)
"""

import os
import sys
from pathlib import Path
from typing import Generator, List, Optional

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

# Load environment variables
load_dotenv()

# Import our modules
from chatbot.rag_engine import RAGEngine, create_rag_engine
from chatbot.prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    SIMPLE_PROMPT_TEMPLATE,
    format_chat_history,
    format_context,
)

# Rich console for pretty output
console = Console()


class DevOpsChatbot:
    """Main chatbot class combining RAG and LLM."""

    def __init__(
        self,
        knowledge_base_path: str = "knowledge-base",
        use_ollama: bool = False,
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """Initialize the chatbot.

        Args:
            knowledge_base_path: Path to knowledge base documents
            use_ollama: If True, use Ollama instead of OpenAI
            model: Model name (default depends on backend)
            temperature: Response creativity (0.0-1.0)
        """
        self.use_ollama = use_ollama or os.getenv("USE_OLLAMA", "").lower() == "true"
        self.temperature = temperature
        self.conversation_history: List[dict] = []

        # Set up model
        if self.use_ollama:
            self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            console.print(f"[cyan]Using Ollama ({self.model})[/cyan]")
        else:
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            console.print(f"[cyan]Using OpenAI ({self.model})[/cyan]")

        # Initialize RAG engine
        console.print("[cyan]Initializing RAG engine...[/cyan]")
        self.rag_engine = create_rag_engine(
            knowledge_base_path=knowledge_base_path,
            auto_index=True
        )

        # Initialize LLM client
        self._init_llm_client()

        console.print("[green]Chatbot ready![/green]\n")

    def _init_llm_client(self):
        """Initialize the LLM client based on configuration."""
        if self.use_ollama:
            # Use requests for Ollama
            import requests
            self.ollama_session = requests.Session()
        else:
            # Use OpenAI client
            from openai import OpenAI
            self.openai_client = OpenAI()

    def _call_openai(
        self,
        messages: List[dict],
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Call OpenAI API.

        Args:
            messages: List of message dicts
            stream: Whether to stream the response

        Yields:
            Response text chunks
        """
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=stream
        )

        if stream:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            yield response.choices[0].message.content

    def _call_ollama(
        self,
        messages: List[dict],
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Call Ollama API.

        Args:
            messages: List of message dicts
            stream: Whether to stream the response

        Yields:
            Response text chunks
        """
        import json

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature
            }
        }

        response = self.ollama_session.post(url, json=payload, stream=stream)
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

    def get_response(
        self,
        user_message: str,
        use_rag: bool = True,
        stream: bool = True
    ) -> Generator[str, None, None]:
        """Get a response to the user's message.

        Args:
            user_message: The user's message
            use_rag: Whether to use RAG for context
            stream: Whether to stream the response

        Yields:
            Response text chunks
        """
        # Get context from RAG if enabled
        context = ""
        if use_rag:
            context = self.rag_engine.get_context(user_message)

        # Format chat history
        history = format_chat_history(self.conversation_history)

        # Build the prompt
        if context and context != "No relevant documentation found.":
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                chat_history=history,
                question=user_message
            )
        else:
            user_prompt = SIMPLE_PROMPT_TEMPLATE.format(
                chat_history=history,
                question=user_message
            )

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get response
        full_response = ""
        if self.use_ollama:
            for chunk in self._call_ollama(messages, stream=stream):
                full_response += chunk
                yield chunk
        else:
            for chunk in self._call_openai(messages, stream=stream):
                full_response += chunk
                yield chunk

        # Add response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        console.print("[yellow]Conversation history cleared.[/yellow]")

    def chat(self, user_message: str) -> str:
        """Simple chat interface (non-streaming).

        Args:
            user_message: The user's message

        Returns:
            Complete response string
        """
        response = ""
        for chunk in self.get_response(user_message, stream=False):
            response += chunk
        return response


def run_cli():
    """Run the interactive CLI chatbot."""
    console.print(Panel.fit(
        "[bold cyan]DevOps Troubleshooting Chatbot[/bold cyan]\n"
        "[dim]Powered by RAG + LLM[/dim]\n\n"
        "Ask me about Terraform, Kubernetes, Docker, or CI/CD issues!\n\n"
        "Commands:\n"
        "  [green]clear[/green] - Clear conversation history\n"
        "  [green]reindex[/green] - Reindex the knowledge base\n"
        "  [green]quit/exit[/green] - Exit the chatbot",
        border_style="cyan"
    ))

    # Initialize chatbot
    try:
        chatbot = DevOpsChatbot()
    except Exception as e:
        console.print(f"[red]Error initializing chatbot: {e}[/red]")
        console.print("\n[yellow]Make sure you have:[/yellow]")
        console.print("  1. Set up your .env file with API keys")
        console.print("  2. Installed all requirements (pip install -r requirements.txt)")
        console.print("  3. For Ollama: Started Ollama server (ollama serve)")
        sys.exit(1)

    # Main chat loop
    while True:
        try:
            # Get user input
            console.print()
            user_input = Prompt.ask("[bold green]You[/bold green]")

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            if user_input.lower() == "clear":
                chatbot.clear_history()
                continue

            if user_input.lower() == "reindex":
                console.print("[yellow]Reindexing knowledge base...[/yellow]")
                chatbot.rag_engine.index_documents(force_reindex=True)
                continue

            if not user_input.strip():
                continue

            # Get and display response
            console.print("\n[bold blue]Assistant[/bold blue]")

            # Stream the response
            response_text = ""
            for chunk in chatbot.get_response(user_input):
                console.print(chunk, end="")
                response_text += chunk

            console.print()  # Newline after response

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Try again or type 'quit' to exit.[/dim]")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="DevOps Troubleshooting Chatbot"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama instead of OpenAI"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use"
    )
    parser.add_argument(
        "--knowledge-base",
        type=str,
        default="knowledge-base",
        help="Path to knowledge base directory"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reindex the knowledge base and exit"
    )

    args = parser.parse_args()

    # Handle reindex command
    if args.reindex:
        console.print("[yellow]Reindexing knowledge base...[/yellow]")
        engine = create_rag_engine(
            knowledge_base_path=args.knowledge_base,
            auto_index=False
        )
        engine.index_documents(force_reindex=True)
        console.print("[green]Done![/green]")
        return

    # Set environment variable if --ollama specified
    if args.ollama:
        os.environ["USE_OLLAMA"] = "true"

    # Run the CLI
    run_cli()


if __name__ == "__main__":
    main()

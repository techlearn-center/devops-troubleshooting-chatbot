#!/usr/bin/env python3
"""
Exercise 2: Rich CLI Interface
==============================

GOAL: Make the chatbot look beautiful using the Rich library.

WHAT YOU'LL LEARN:
- Using the Rich library for terminal formatting
- Creating panels and styled text
- Rendering Markdown in the terminal
- Adding loading indicators

RICH LIBRARY BASICS:
- Console: Main object for printing
- Panel: Box around content
- Markdown: Render markdown text
- Prompt: Styled input prompt
- status(): Show a loading spinner

RUN THIS:
    python module-2-simple-chatbot/exercises/ex2_rich_interface.py

REQUIREMENTS:
    pip install rich
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# RICH LIBRARY IMPORTS
# =============================================================================
# Rich is a library for beautiful terminal output.
# Each import serves a specific purpose:
from rich.console import Console   # Main printing interface
from rich.markdown import Markdown # Render markdown (bold, code blocks, etc.)
from rich.panel import Panel       # Draw boxes around content
from rich.prompt import Prompt     # Styled input prompts

# Create a console instance for all our printing
console = Console()


SYSTEM_PROMPT = """You are an expert DevOps assistant.
Format your responses using Markdown for better readability:
- Use **bold** for important terms
- Use `code` for commands and values
- Use ```language for code blocks
- Use bullet points for lists"""


class RichChatbot:
    """
    Chatbot with beautiful Rich terminal interface.

    This adds visual polish to our chatbot:
    - Colored text and panels
    - Markdown rendering
    - Loading indicators
    - Styled prompts
    """

    def __init__(self):
        """Initialize the chatbot with Rich console."""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.console = Console()

        # Set up backend
        self.use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"
        if self.use_ollama:
            import requests
            self.session = requests.Session()
        else:
            from openai import OpenAI
            self.client = OpenAI()

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_message})

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
                temperature=0.3,
                max_tokens=1000
            )
            assistant_message = response.choices[0].message.content

        self.messages.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def display_response(self, response: str):
        """
        Display the response with Rich formatting.

        This method:
        1. Converts the response to Markdown for rendering
        2. Wraps it in a Panel for visual separation
        3. Prints with colors and styling
        """
        # =====================================================================
        # MARKDOWN RENDERING
        # =====================================================================
        # Rich can render Markdown, so code blocks, bold text, and lists
        # all display beautifully in the terminal.
        md = Markdown(response)

        # =====================================================================
        # PANEL DISPLAY
        # =====================================================================
        # A Panel draws a box around content.
        # - title: Text shown at top of box
        # - border_style: Color of the border
        self.console.print(Panel(
            md,
            title="[bold green]DevOps Bot[/bold green]",
            border_style="green",
            padding=(1, 2)  # (vertical, horizontal) padding
        ))

    def run(self):
        """Run the chatbot with Rich interface."""
        # =====================================================================
        # WELCOME MESSAGE
        # =====================================================================
        # Panel.fit() creates a panel that fits the content
        # Rich markup: [bold green]text[/bold green] makes text bold and green
        self.console.print(Panel.fit(
            "[bold cyan]DevOps Troubleshooting Chatbot[/bold cyan]\n\n"
            "Ask questions about Terraform, Kubernetes, Docker, and CI/CD.\n\n"
            "[dim]Commands:[/dim]\n"
            "  [green]quit[/green]  - Exit the chatbot\n"
            "  [green]clear[/green] - Reset conversation",
            title="Welcome",
            border_style="cyan"
        ))

        # =====================================================================
        # MAIN CHAT LOOP
        # =====================================================================
        while True:
            try:
                # Styled input prompt
                # [bold blue] makes "You" bold and blue
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == 'clear':
                    self.messages = [self.messages[0]]
                    self.console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                elif not user_input.strip():
                    continue

                # ============================================================
                # LOADING INDICATOR
                # ============================================================
                # console.status() shows a spinner while code runs
                # It automatically disappears when the 'with' block ends
                with self.console.status("[bold green]Thinking...[/bold green]"):
                    response = self.chat(user_input)

                # Display formatted response
                self.display_response(response)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# RICH LIBRARY DEMO
# =============================================================================
def demo_rich_features():
    """Demonstrate various Rich library features."""
    console = Console()

    console.print("\n[bold underline]Rich Library Demo[/bold underline]\n")

    # Colors and styles
    console.print("[red]Red text[/red]")
    console.print("[green]Green text[/green]")
    console.print("[bold]Bold text[/bold]")
    console.print("[italic]Italic text[/italic]")
    console.print("[bold red on white]Bold red on white background[/bold red on white]")

    # Panel
    console.print(Panel("Content inside a panel", title="Panel Demo"))

    # Markdown
    md = Markdown("""
## Markdown Demo

This is **bold** and this is *italic*.

Code: `kubectl get pods`

```python
print("Hello, World!")
```
    """)
    console.print(md)

    console.print("\n[dim]End of demo[/dim]\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_rich_features()
    else:
        bot = RichChatbot()
        bot.run()

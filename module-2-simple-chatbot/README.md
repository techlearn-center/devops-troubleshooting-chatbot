# Module 2: Building a Simple Chatbot

**Time Required: 1.5 hours**

Now that you understand LLM basics, let's build a proper chatbot with conversation history and a user-friendly interface.

---

## Learning Objectives

By the end of this module, you will:
- Build a chatbot that remembers conversation history
- Create a CLI interface with rich formatting
- Handle errors gracefully
- Implement streaming responses
- Understand the importance of context management

---

## Why Conversation History Matters

Without history, every message is isolated:

```
User: What is Terraform?
Bot: Terraform is an IaC tool for provisioning infrastructure.

User: How do I install it?
Bot: I don't know what "it" refers to. Please be more specific.
     ^^^ BAD - Lost context!
```

With history:

```
User: What is Terraform?
Bot: Terraform is an IaC tool for provisioning infrastructure.

User: How do I install it?
Bot: To install Terraform:
     1. Download from terraform.io
     2. Add to PATH...
     ^^^ GOOD - Remembers we're talking about Terraform!
```

---

## How Conversation History Works

The LLM doesn't have memory - **you** must send the full history:

```python
# First message
messages = [
    {"role": "system", "content": "You are a DevOps assistant."},
    {"role": "user", "content": "What is Terraform?"}
]
response1 = call_llm(messages)  # "Terraform is..."

# Add response to history
messages.append({"role": "assistant", "content": response1})

# Second message - includes full history!
messages.append({"role": "user", "content": "How do I install it?"})
response2 = call_llm(messages)  # Knows "it" = Terraform
```

```
Messages sent to LLM for second question:
+--------------------------------------------------+
| [system] You are a DevOps assistant.             |
| [user] What is Terraform?                        |
| [assistant] Terraform is an IaC tool...          |  <- History
| [user] How do I install it?                      |  <- New question
+--------------------------------------------------+
```

---

## Exercise 1: Basic Chatbot with History

Create `exercises/ex1_chatbot_history.py`:

```python
"""
Exercise 1: Chatbot with Conversation History
=============================================
Goal: Build a chatbot that remembers previous messages
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# System prompt for our DevOps chatbot
SYSTEM_PROMPT = """You are an expert DevOps assistant specializing in:
- Terraform and Infrastructure as Code
- Kubernetes and container orchestration
- Docker and containerization
- CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI)
- Cloud platforms (AWS, GCP, Azure)

Guidelines:
1. Be concise but thorough
2. Include code examples when helpful
3. Suggest best practices
4. Ask clarifying questions if needed
"""


class SimpleChatbot:
    """A simple chatbot with conversation history."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        """Initialize the chatbot with a system prompt."""
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: The user's input

        Returns:
            The assistant's response
        """
        # TODO: Implement this method
        # 1. Add user message to history
        # 2. Call the LLM with full history
        # 3. Add assistant response to history
        # 4. Return the response
        pass

    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [self.messages[0]]


def main():
    """Run the chatbot in a loop."""
    print("DevOps Chatbot (type 'quit' to exit, 'clear' to reset)")
    print("=" * 50)

    chatbot = SimpleChatbot()

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("Conversation cleared.")
            continue
        elif not user_input:
            continue

        response = chatbot.chat(user_input)
        print(f"\nBot: {response}")


if __name__ == "__main__":
    main()
```

---

## Exercise 2: Rich CLI Interface

Let's make the chatbot look better using the `rich` library.

Create `exercises/ex2_rich_interface.py`:

```python
"""
Exercise 2: Rich CLI Interface
==============================
Goal: Create a beautiful terminal interface
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

load_dotenv()
client = OpenAI()
console = Console()

SYSTEM_PROMPT = """You are an expert DevOps assistant.
Format your responses using Markdown for better readability.
Use code blocks with language specification for code examples."""


class RichChatbot:
    """Chatbot with rich terminal interface."""

    def __init__(self):
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.console = Console()

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        self.messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=self.messages,
            temperature=0.3,
            max_tokens=1000
        )

        assistant_message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def display_response(self, response: str):
        """Display the response with rich formatting."""
        # TODO: Use rich to display the response
        # Hint: Use Markdown() and Panel()
        pass

    def run(self):
        """Run the chatbot loop."""
        self.console.print(Panel(
            "[bold green]DevOps Troubleshooting Chatbot[/bold green]\n"
            "Ask questions about Terraform, Kubernetes, Docker, and CI/CD.\n"
            "Type [bold]'quit'[/bold] to exit, [bold]'clear'[/bold] to reset.",
            title="Welcome",
            border_style="green"
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")

                if user_input.lower() == 'quit':
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == 'clear':
                    self.messages = [self.messages[0]]
                    self.console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                elif not user_input.strip():
                    continue

                with self.console.status("[bold green]Thinking..."):
                    response = self.chat(user_input)

                self.display_response(response)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break


if __name__ == "__main__":
    bot = RichChatbot()
    bot.run()
```

---

## Exercise 3: Streaming Responses

For long responses, streaming provides a better experience:

```python
"""
Exercise 3: Streaming Responses
===============================
Goal: Show responses as they're generated
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

load_dotenv()
client = OpenAI()
console = Console()


def stream_response(messages: list) -> str:
    """
    Stream a response from the LLM.

    Args:
        messages: The conversation history

    Returns:
        The complete response
    """
    # TODO: Implement streaming
    # Hint: Use stream=True in the API call
    # The response will be an iterator of chunks

    full_response = ""

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=messages,
        stream=True  # Enable streaming
    )

    # Process chunks as they arrive
    for chunk in response:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_response += content

    print()  # New line at end
    return full_response


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "You are a DevOps assistant."},
        {"role": "user", "content": "Explain the Kubernetes architecture in detail."}
    ]

    print("Streaming response:\n")
    response = stream_response(messages)
```

---

## Exercise 4: Error Handling

Real chatbots need to handle errors gracefully:

```python
"""
Exercise 4: Robust Error Handling
=================================
Goal: Handle API errors, rate limits, and network issues
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APIConnectionError

load_dotenv()
client = OpenAI()


class RobustChatbot:
    """Chatbot with proper error handling."""

    def __init__(self):
        self.messages = [
            {"role": "system", "content": "You are a DevOps assistant."}
        ]
        self.max_retries = 3

    def chat(self, user_message: str) -> str:
        """
        Send a message with retry logic.

        Handles:
        - Rate limiting (wait and retry)
        - API errors (retry with backoff)
        - Network errors (retry)
        """
        self.messages.append({"role": "user", "content": user_message})

        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=self.messages,
                    temperature=0.3
                )

                assistant_message = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": assistant_message})
                return assistant_message

            except RateLimitError:
                # TODO: Handle rate limiting
                # Wait and retry
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)

            except APIConnectionError:
                # TODO: Handle connection errors
                print(f"Connection error. Retrying... ({attempt + 1}/{self.max_retries})")
                time.sleep(1)

            except APIError as e:
                # TODO: Handle other API errors
                print(f"API error: {e}")
                if attempt == self.max_retries - 1:
                    raise

        return "Sorry, I'm having trouble connecting. Please try again."


if __name__ == "__main__":
    bot = RobustChatbot()

    print("Testing error handling...")
    response = bot.chat("What is Docker?")
    print(f"Response: {response}")
```

---

## Context Window Management

As conversations grow, you'll hit token limits. Here's how to manage:

```python
def manage_context(messages: list, max_tokens: int = 3000) -> list:
    """
    Keep conversation within token limits.

    Strategy: Remove oldest messages (keep system prompt)
    """
    import tiktoken

    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    while True:
        # Count total tokens
        total = sum(len(encoding.encode(m["content"])) for m in messages)

        if total <= max_tokens:
            break

        # Remove oldest non-system message
        if len(messages) > 2:  # Keep system + at least one exchange
            messages.pop(1)
        else:
            break

    return messages
```

---

## Putting It All Together

Create `exercises/ex5_complete_chatbot.py`:

```python
"""
Exercise 5: Complete Chatbot
============================
Goal: Combine all features into a production-ready chatbot
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
import tiktoken

load_dotenv()


class DevOpsChatbot:
    """Complete DevOps chatbot with all features."""

    SYSTEM_PROMPT = """You are an expert DevOps troubleshooting assistant.

Your expertise includes:
- Terraform (state management, errors, best practices)
- Kubernetes (pod issues, deployments, networking)
- Docker (builds, runtime, compose)
- CI/CD (GitHub Actions, Jenkins, GitLab CI)
- Cloud (AWS, GCP, Azure)

When helping with errors:
1. First understand the error message
2. Explain what caused it
3. Provide step-by-step solution
4. Include code/commands when helpful
5. Suggest prevention tips

Format responses in Markdown for readability."""

    def __init__(self):
        self.client = OpenAI()
        self.console = Console()
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        self.max_context_tokens = 3000

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(encoding.encode(text))

    def _manage_context(self):
        """Keep conversation within token limits."""
        while True:
            total = sum(self._count_tokens(m["content"]) for m in self.messages)
            if total <= self.max_context_tokens or len(self.messages) <= 2:
                break
            self.messages.pop(1)

    def chat(self, user_message: str) -> str:
        """Send message and get response with error handling."""
        self.messages.append({"role": "user", "content": user_message})
        self._manage_context()

        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    messages=self.messages,
                    temperature=0.2,
                    max_tokens=1000
                )

                content = response.choices[0].message.content
                self.messages.append({"role": "assistant", "content": content})
                return content

            except RateLimitError:
                time.sleep(2 ** attempt)
            except APIConnectionError:
                time.sleep(1)

        return "Sorry, I'm having trouble connecting. Please try again."

    def run(self):
        """Run the interactive chatbot."""
        self.console.print(Panel(
            "[bold green]DevOps Troubleshooting Chatbot[/bold green]\n\n"
            "I can help with Terraform, Kubernetes, Docker, and CI/CD issues.\n"
            "Paste your error message or describe your problem.\n\n"
            "[dim]Commands: 'quit' to exit, 'clear' to reset[/dim]",
            border_style="green"
        ))

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if not user_input.strip():
                    continue
                if user_input.lower() == 'quit':
                    break
                if user_input.lower() == 'clear':
                    self.messages = [self.messages[0]]
                    self.console.print("[yellow]Cleared.[/yellow]")
                    continue

                with self.console.status("[green]Analyzing..."):
                    response = self.chat(user_input)

                self.console.print(Panel(
                    Markdown(response),
                    title="[bold green]DevOps Bot[/bold green]",
                    border_style="green"
                ))

            except KeyboardInterrupt:
                break

        self.console.print("[yellow]Goodbye![/yellow]")


if __name__ == "__main__":
    DevOpsChatbot().run()
```

---

## Key Takeaways

1. **History is essential** - LLMs don't remember; you must send full context
2. **Manage context size** - Remove old messages to stay within limits
3. **Handle errors** - Network issues and rate limits will happen
4. **User experience matters** - Rich formatting and streaming improve UX
5. **System prompts shape behavior** - Invest time in crafting good prompts

---

## What's Next?

Our chatbot is great, but it can only use knowledge from its training data. In **Module 3**, we'll learn about RAG (Retrieval-Augmented Generation) to give our chatbot access to real DevOps documentation!

```bash
cd ../module-3-rag-fundamentals
```

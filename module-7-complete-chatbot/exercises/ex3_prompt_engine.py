#!/usr/bin/env python3
"""
Exercise 3: Build the Prompt Engine
=====================================
Module 7 (Capstone) - Exercise 3 of 4

WHAT YOU'LL BUILD:
    The Prompt Engine - the component that constructs the perfect prompt
    by combining: the user's question + retrieved context + conversation history.

WHY THIS MATTERS:
    The LLM only sees what we put in the prompt. If we give it a bad prompt,
    we get a bad answer. The Prompt Engine is like a chef assembling a dish:
    - The question is the main ingredient
    - The RAG context is the seasoning (makes it accurate)
    - The history is the recipe notes (keeps conversation coherent)
    - The system prompt is the cooking style (expert vs. quick vs. teacher)

WHAT YOU'LL LEARN:
    - How to combine RAG context with prompting strategies (from Module 6)
    - Managing conversation history (what to keep, what to trim)
    - System prompts vs user prompts vs context injection
    - Token counting to avoid exceeding the LLM's context window

PREREQUISITES:
    pip install rich tiktoken openai python-dotenv

SUPPORTS:
    - OpenAI (set OPENAI_API_KEY in .env)
    - Ollama (set USE_OLLAMA=true in .env)

RUN THIS FILE:
    python ex3_prompt_engine.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt
from rich.text import Text

# tiktoken counts tokens - this is how we measure prompt size.
# Different models have different token limits:
# - GPT-3.5-turbo: 4,096 or 16,385 tokens
# - GPT-4: 8,192 or 128,000 tokens
# - Llama 2: 4,096 tokens
# Tokens are roughly: 1 token ~= 4 characters ~= 0.75 words
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# python-dotenv for environment variables
from dotenv import load_dotenv

load_dotenv()

# Rich console
console = Console()


# ============================================================================
# SYSTEM PROMPT PRESETS
# ============================================================================
# These define the chatbot's "personality". The system prompt is the first
# message the LLM sees, and it shapes ALL subsequent responses.

SYSTEM_PROMPTS = {
    "expert": {
        "name": "Expert Mode",
        "description": "Detailed, thorough responses with deep technical explanations",
        "prompt": (
            "You are a senior DevOps engineer with 15+ years of experience in "
            "Terraform, Kubernetes, Docker, and CI/CD pipelines. You provide "
            "thorough, detailed responses that include:\n"
            "1. Root cause analysis (WHY the error happened)\n"
            "2. Step-by-step solutions with exact commands\n"
            "3. Prevention tips (how to avoid this in the future)\n"
            "4. Related concepts the user should understand\n\n"
            "Always explain the underlying concepts, not just the fix. "
            "Use code blocks for commands and configuration examples. "
            "If you need more information, ask clarifying questions."
        ),
    },
    "quick": {
        "name": "Quick Mode",
        "description": "Brief, actionable responses - get the fix fast",
        "prompt": (
            "You are a DevOps expert who gives quick, actionable answers. "
            "Be concise:\n"
            "- Lead with the most likely fix\n"
            "- Use bullet points and short code blocks\n"
            "- Skip lengthy explanations unless asked\n"
            "- Maximum 5-8 lines per response\n"
            "- If unsure, say so briefly and suggest where to look"
        ),
    },
    "teacher": {
        "name": "Teacher Mode",
        "description": "Explains concepts with examples, great for learning",
        "prompt": (
            "You are a patient DevOps teacher who loves explaining concepts "
            "to beginners. For every answer:\n"
            "1. Start with a simple analogy or real-world comparison\n"
            "2. Explain the concept in plain English first\n"
            "3. Then show the technical details and commands\n"
            "4. Include a 'What's actually happening' section\n"
            "5. End with a 'Learn more' suggestion\n\n"
            "Use analogies like: 'Think of a Docker container like a shipping "
            "container - it packages everything your app needs to run.'\n"
            "Never assume the user knows advanced concepts. Define acronyms "
            "and technical terms when you first use them."
        ),
    },
}


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================
# These templates define how the final prompt is assembled.
# The LLM sees the ENTIRE assembled prompt, so structure matters.

# Template when we have RAG context (most cases)
RAG_PROMPT_TEMPLATE = """Use the following documentation context to answer the user's question.
If the context doesn't contain relevant information, say so and provide your best answer.

DOCUMENTATION CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Provide a helpful, accurate response based on the context above."""

# Template when we have NO RAG context (fallback)
NO_CONTEXT_TEMPLATE = """Answer the following DevOps question based on your knowledge.

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

Provide a helpful response. If you're unsure, be honest about it."""


# ============================================================================
# PROMPT ENGINE CLASS
# ============================================================================
class PromptEngine:
    """
    Assembles optimal prompts by combining question + context + history.

    THE ASSEMBLY LINE (think of it like building a sandwich):
        1. System prompt (the bread - frames everything)
        2. RAG context (the filling - the actual knowledge)
        3. Conversation history (the condiments - adds continuity)
        4. User question (the order - what they actually asked)

    TOKEN BUDGET:
        Every LLM has a maximum context window (token limit). We need to fit:
        - System prompt:  ~200-500 tokens (fixed)
        - RAG context:    ~500-2000 tokens (variable)
        - History:        ~100-1000 tokens (grows over time)
        - User question:  ~10-100 tokens (variable)
        - Response space:  Leave room for the LLM's answer!

        The PromptEngine manages this budget, trimming history and context
        if needed to stay within limits.
    """

    def __init__(
        self,
        preset: str = "expert",
        max_history_exchanges: int = 10,
        max_context_tokens: int = 2000,
        model_max_tokens: int = 4096,
    ):
        """
        Initialize the Prompt Engine.

        Args:
            preset:                Which system prompt preset to use
                                   ("expert", "quick", or "teacher")
            max_history_exchanges: Maximum number of Q&A pairs to keep in history.
                                   Older exchanges are dropped to save tokens.
            max_context_tokens:    Maximum tokens to spend on RAG context.
            model_max_tokens:      Total token limit of the LLM model.
        """
        # Set the system prompt based on preset
        if preset in SYSTEM_PROMPTS:
            self.system_prompt = SYSTEM_PROMPTS[preset]["prompt"]
            self.preset_name = SYSTEM_PROMPTS[preset]["name"]
        else:
            # Allow custom system prompts
            self.system_prompt = preset
            self.preset_name = "Custom"

        # Conversation history: list of {"role": "user"/"assistant", "content": "..."}
        self.history: List[Dict[str, str]] = []

        # Configuration
        self.max_history_exchanges = max_history_exchanges
        self.max_context_tokens = max_context_tokens
        self.model_max_tokens = model_max_tokens

        # Initialize the token counter
        # tiktoken is OpenAI's tokenizer - it counts tokens the same way the model does
        if HAS_TIKTOKEN:
            try:
                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
            except Exception:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
        else:
            self.tokenizer = None

        console.print(
            f"[green]Prompt Engine initialized with preset: "
            f"[bold]{self.preset_name}[/bold][/green]"
        )

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        WHY THIS MATTERS:
            LLMs don't see characters or words - they see TOKENS.
            "Hello world" = 2 tokens
            "CrashLoopBackOff" = 3 tokens (it's split into sub-words)
            "kubectl get pods" = 3 tokens

            If we exceed the model's token limit, the API returns an error.
            So we count tokens to stay within budget.

        Args:
            text: The text to count tokens for

        Returns:
            Number of tokens (approximate if tiktoken not available)
        """
        if self.tokenizer:
            # Accurate count using tiktoken
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: 1 token ~= 4 characters
            # This is a common heuristic when tiktoken isn't available
            return len(text) // 4

    def set_system_prompt(self, prompt_or_preset: str) -> None:
        """
        Change the system prompt (chatbot personality).

        Args:
            prompt_or_preset: Either a preset name ("expert", "quick", "teacher")
                              or a custom system prompt string.
        """
        if prompt_or_preset in SYSTEM_PROMPTS:
            self.system_prompt = SYSTEM_PROMPTS[prompt_or_preset]["prompt"]
            self.preset_name = SYSTEM_PROMPTS[prompt_or_preset]["name"]
            console.print(
                f"[green]Switched to preset: [bold]{self.preset_name}[/bold][/green]"
            )
        else:
            self.system_prompt = prompt_or_preset
            self.preset_name = "Custom"
            console.print("[green]Set custom system prompt.[/green]")

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        The history helps the LLM understand context from earlier in the
        conversation. For example:
            User: "My pod is crashing"
            Bot:  "Check the logs with kubectl logs..."
            User: "It says OOMKilled"  <-- This only makes sense with history!

        Args:
            role:    "user" or "assistant"
            content: The message content
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

        # Auto-trim if we exceed max exchanges
        # Each exchange = 1 user message + 1 assistant message = 2 entries
        max_entries = self.max_history_exchanges * 2
        if len(self.history) > max_entries:
            # Keep only the most recent exchanges
            self.history = self.history[-max_entries:]

    def trim_history(self, max_exchanges: Optional[int] = None) -> None:
        """
        Trim conversation history to keep only recent exchanges.

        WHY TRIM?
            Long conversations eat up tokens. A 50-message conversation
            might use 5000 tokens just for history, leaving no room for
            context or the response. Trimming keeps things manageable.

        Args:
            max_exchanges: Number of Q&A pairs to keep. If None, uses default.
        """
        max_ex = max_exchanges or self.max_history_exchanges
        max_entries = max_ex * 2

        if len(self.history) > max_entries:
            removed = len(self.history) - max_entries
            self.history = self.history[-max_entries:]
            console.print(
                f"[dim]Trimmed {removed} old messages from history. "
                f"Keeping last {max_ex} exchanges.[/dim]"
            )

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.history = []
        console.print("[yellow]Conversation history cleared.[/yellow]")

    def _format_history(self) -> str:
        """
        Format conversation history as a string for inclusion in the prompt.

        Returns:
            Formatted history string, or "No previous conversation." if empty.
        """
        if not self.history:
            return "No previous conversation."

        formatted_lines = []
        for msg in self.history:
            role_label = "USER" if msg["role"] == "user" else "ASSISTANT"
            formatted_lines.append(f"{role_label}: {msg['content']}")

        return "\n".join(formatted_lines)

    def _format_context(self, context_docs: List[Dict]) -> str:
        """
        Format RAG context documents as a string for inclusion in the prompt.

        Each document is formatted with its source and category for the LLM
        to reference.

        Args:
            context_docs: List of dicts with "content", "metadata" keys
                          (as returned by the RAG pipeline's retrieve() method)

        Returns:
            Formatted context string.
        """
        if not context_docs:
            return "No relevant documentation found."

        parts = []
        for i, doc in enumerate(context_docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source_file", "unknown")
            category = metadata.get("category", "unknown")

            parts.append(
                f"--- Document {i} (Source: {source}, Category: {category}) ---\n"
                f"{content}"
            )

        return "\n\n".join(parts)

    def build_prompt(
        self,
        question: str,
        context_docs: Optional[List[Dict]] = None,
        history_override: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """
        Assemble the complete prompt from all components.

        This is the MAIN METHOD - it takes all the pieces and builds
        the final prompt that gets sent to the LLM.

        THE ASSEMBLY:
            messages = [
                {"role": "system", "content": system_prompt},        # Personality
                {"role": "user",   "content": assembled_user_prompt}  # Everything else
            ]

        The assembled_user_prompt contains:
            - RAG context (documentation excerpts)
            - Conversation history (previous Q&A)
            - The actual question

        Args:
            question:          The user's current question
            context_docs:      RAG retrieval results (list of dicts)
            history_override:  Optional custom history (overrides self.history)

        Returns:
            List of message dicts ready to send to the LLM:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        # ---- Step 1: Format the context ----
        context_str = self._format_context(context_docs) if context_docs else ""

        # ---- Step 2: Format the history ----
        if history_override is not None:
            # Use provided history instead of self.history
            history_str = "\n".join(
                f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
                for m in history_override
            )
        else:
            history_str = self._format_history()

        # ---- Step 3: Choose the right template ----
        if context_str and context_str != "No relevant documentation found.":
            # We have RAG context - use the RAG template
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context_str,
                history=history_str,
                question=question,
            )
        else:
            # No RAG context - use the simpler template
            user_prompt = NO_CONTEXT_TEMPLATE.format(
                history=history_str,
                question=question,
            )

        # ---- Step 4: Check token budget ----
        system_tokens = self.count_tokens(self.system_prompt)
        user_tokens = self.count_tokens(user_prompt)
        total_tokens = system_tokens + user_tokens

        # We need to leave room for the LLM's response (~500-1000 tokens)
        response_budget = 1000
        token_limit = self.model_max_tokens - response_budget

        if total_tokens > token_limit:
            # We're over budget! Trim history first, then context
            console.print(
                f"[yellow]Token budget exceeded ({total_tokens}/{token_limit}). "
                f"Trimming...[/yellow]"
            )
            # Trim history to minimum (last 2 exchanges)
            trimmed_history = self.history[-4:] if len(self.history) > 4 else self.history
            history_str = "\n".join(
                f"{'USER' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
                for m in trimmed_history
            )
            # Rebuild with trimmed history
            if context_str and context_str != "No relevant documentation found.":
                user_prompt = RAG_PROMPT_TEMPLATE.format(
                    context=context_str,
                    history=history_str,
                    question=question,
                )
            else:
                user_prompt = NO_CONTEXT_TEMPLATE.format(
                    history=history_str,
                    question=question,
                )

        # ---- Step 5: Assemble final messages ----
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return messages

    def get_prompt_stats(
        self,
        question: str,
        context_docs: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Get detailed statistics about a prompt without sending it.

        Useful for debugging and understanding token usage.

        Args:
            question:     The user's question
            context_docs: RAG retrieval results

        Returns:
            Dict with token counts and breakdown
        """
        messages = self.build_prompt(question, context_docs)

        system_tokens = self.count_tokens(messages[0]["content"])
        user_tokens = self.count_tokens(messages[1]["content"])
        total_tokens = system_tokens + user_tokens

        return {
            "system_prompt_tokens": system_tokens,
            "user_prompt_tokens": user_tokens,
            "total_tokens": total_tokens,
            "remaining_for_response": self.model_max_tokens - total_tokens,
            "history_length": len(self.history),
            "context_docs_count": len(context_docs) if context_docs else 0,
            "preset": self.preset_name,
        }

    # ========================================================================
    # TODO: Implement get_history_summary() using the LLM
    # ========================================================================
    def get_history_summary(self) -> str:
        """
        TODO: Use the LLM to summarize long conversation history.

        WHY SUMMARIZE?
            Instead of trimming (losing information), we can ask the LLM to
            SUMMARIZE the old conversation. This preserves context while
            using fewer tokens.

            Example:
            Before (500 tokens):
                USER: My pod is crashing
                ASSISTANT: Let's check the logs...
                USER: The logs show OOMKilled
                ASSISTANT: That means the container ran out of memory...
                USER: How do I increase memory?
                ASSISTANT: Edit your deployment YAML...

            After summary (100 tokens):
                "The user has a Kubernetes pod crash-looping due to OOMKilled.
                 We discussed checking logs and increasing memory limits."

        INSTRUCTIONS:
        1. Check if USE_OLLAMA environment variable is set
        2. Build a prompt asking the LLM to summarize the conversation
        3. Call the LLM (OpenAI or Ollama) to generate the summary
        4. Return the summary string

        HINT - Prompt template:
            "Summarize this conversation in 2-3 sentences. Focus on:
             - What problem the user is facing
             - What solutions have been discussed
             - Any unresolved issues

             Conversation:
             {history}"

        Returns:
            A summary string of the conversation history
        """
        # TODO: Implement this method!
        # Step 1: Check backend
        # use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"

        # Step 2: Build summary prompt
        # history_text = self._format_history()
        # summary_prompt = f"Summarize this conversation in 2-3 sentences..."

        # Step 3: Call the LLM
        # if use_ollama:
        #     import requests
        #     response = requests.post("http://localhost:11434/api/generate", ...)
        # else:
        #     from openai import OpenAI
        #     client = OpenAI()
        #     response = client.chat.completions.create(...)

        # Step 4: Return the summary
        # return summary_text

        # For now, return a simple string representation
        if not self.history:
            return "No conversation history."
        return f"Conversation with {len(self.history)} messages (summary not implemented yet)."

    # ========================================================================
    # TODO: Add a "debug" mode that shows the full prompt before sending
    # ========================================================================
    def build_prompt_debug(
        self,
        question: str,
        context_docs: Optional[List[Dict]] = None,
    ) -> List[Dict[str, str]]:
        """
        TODO: Build the prompt AND display it for debugging.

        INSTRUCTIONS:
        1. Call self.build_prompt() to get the messages
        2. Display the full assembled prompt using Rich:
           - Show the system prompt in a blue panel
           - Show the user prompt in a green panel
           - Show token counts for each part
           - Show the total token usage vs. the model's limit
        3. Return the messages (same as build_prompt)

        This is incredibly useful for debugging prompt issues!
        When responses are bad, the first thing to check is:
        "What did the LLM actually see?"

        HINT:
            Use Rich's Panel and Syntax for nice formatting:
            console.print(Panel(
                Syntax(messages[0]["content"], "text", theme="monokai"),
                title="System Prompt",
                border_style="blue",
            ))

        Args:
            question:     The user's question
            context_docs: RAG retrieval results

        Returns:
            Same as build_prompt() - list of message dicts
        """
        # TODO: Implement this method!
        # Step 1: Build the prompt
        messages = self.build_prompt(question, context_docs)

        # Step 2: Display it (TODO - uncomment and complete)
        # console.print(Panel(
        #     messages[0]["content"],
        #     title="[bold blue]System Prompt[/bold blue]",
        #     border_style="blue",
        # ))
        # console.print(Panel(
        #     messages[1]["content"],
        #     title="[bold green]User Prompt[/bold green]",
        #     border_style="green",
        # ))
        # stats = self.get_prompt_stats(question, context_docs)
        # console.print(f"Tokens: {stats['total_tokens']} / {self.model_max_tokens}")

        return messages


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def display_prompt_comparison(engine: PromptEngine, question: str) -> None:
    """
    Show how the same question produces different prompts with different settings.

    This demonstrates the impact of:
    - Different system prompts (expert vs. quick vs. teacher)
    - With vs. without RAG context
    - With vs. without conversation history
    """
    # Sample context documents (simulating RAG results)
    sample_context = [
        {
            "content": (
                "# Pod CrashLoopBackOff\n"
                "Check logs with: kubectl logs <pod> --previous\n"
                "Common causes: OOMKilled (exit 137), missing env vars, "
                "failed liveness probe."
            ),
            "metadata": {
                "source_file": "pod-errors.md",
                "category": "kubernetes",
            },
        },
    ]

    # Sample history
    sample_history = [
        {"role": "user", "content": "I'm having issues with my Kubernetes cluster"},
        {"role": "assistant", "content": "I can help! What specific issue are you seeing?"},
    ]

    console.print(f"\n[bold]Question:[/bold] \"{question}\"\n")

    # Scenario 1: No context, no history
    console.print("[bold yellow]Scenario 1: No context, no history[/bold yellow]")
    msgs = engine.build_prompt(question)
    stats = engine.get_prompt_stats(question)
    console.print(Panel(
        msgs[1]["content"],
        title=f"User Prompt ({stats['total_tokens']} tokens)",
        border_style="yellow",
        width=90,
    ))

    # Scenario 2: With context, no history
    console.print("[bold green]Scenario 2: With RAG context[/bold green]")
    msgs = engine.build_prompt(question, context_docs=sample_context)
    stats = engine.get_prompt_stats(question, context_docs=sample_context)
    console.print(Panel(
        msgs[1]["content"],
        title=f"User Prompt ({stats['total_tokens']} tokens)",
        border_style="green",
        width=90,
    ))

    # Scenario 3: With context AND history
    # Temporarily add history
    old_history = engine.history.copy()
    engine.history = sample_history.copy()

    console.print("[bold cyan]Scenario 3: With RAG context + history[/bold cyan]")
    msgs = engine.build_prompt(question, context_docs=sample_context)
    stats = engine.get_prompt_stats(question, context_docs=sample_context)
    console.print(Panel(
        msgs[1]["content"],
        title=f"User Prompt ({stats['total_tokens']} tokens)",
        border_style="cyan",
        width=90,
    ))

    # Restore history
    engine.history = old_history


def display_preset_comparison(question: str) -> None:
    """
    Show how different presets produce different system prompts.
    """
    console.print(f"\n[bold]How presets affect the system prompt:[/bold]\n")

    for preset_key, preset_info in SYSTEM_PROMPTS.items():
        color = {"expert": "blue", "quick": "green", "teacher": "magenta"}.get(
            preset_key, "white"
        )

        engine = PromptEngine(preset=preset_key)
        tokens = engine.count_tokens(preset_info["prompt"])

        console.print(Panel(
            preset_info["prompt"],
            title=f"[bold {color}]{preset_info['name']}[/bold {color}] ({tokens} tokens)",
            subtitle=preset_info["description"],
            border_style=color,
            width=90,
        ))


def display_token_breakdown(engine: PromptEngine, question: str, context_docs=None) -> None:
    """Display a detailed token breakdown."""
    stats = engine.get_prompt_stats(question, context_docs)

    table = Table(title="Token Budget Breakdown", show_header=True)
    table.add_column("Component", style="bold")
    table.add_column("Tokens", justify="right")
    table.add_column("Percentage", justify="right")
    table.add_column("Bar", justify="left")

    total = stats["total_tokens"]
    limit = engine.model_max_tokens

    components = [
        ("System Prompt", stats["system_prompt_tokens"]),
        ("User Prompt", stats["user_prompt_tokens"]),
        ("Available for Response", stats["remaining_for_response"]),
    ]

    for name, tokens in components:
        pct = (tokens / limit) * 100 if limit > 0 else 0
        bar_len = 20
        filled = int((tokens / limit) * bar_len) if limit > 0 else 0
        bar = "#" * filled + "." * (bar_len - filled)

        color = "green" if pct < 30 else "yellow" if pct < 60 else "red"
        table.add_row(
            name,
            str(tokens),
            f"[{color}]{pct:.1f}%[/{color}]",
            f"[{color}][{bar}][/{color}]",
        )

    # Total row
    total_pct = (total / limit) * 100 if limit > 0 else 0
    table.add_row(
        "[bold]TOTAL USED[/bold]",
        f"[bold]{total}[/bold]",
        f"[bold]{total_pct:.1f}%[/bold]",
        f"[bold]{total}/{limit}[/bold]",
        style="bold",
    )

    console.print(table)
    console.print(
        f"[dim]History: {stats['history_length']} messages | "
        f"Context docs: {stats['context_docs_count']} | "
        f"Preset: {stats['preset']}[/dim]"
    )


# ============================================================================
# MAIN - Interactive Demo
# ============================================================================
def main():
    """
    Run the Prompt Engine interactive demo.

    Shows how the same question gets different prompts with different:
    - System prompts (expert vs. quick vs. teacher)
    - Context documents (with vs. without RAG)
    - Conversation history (fresh vs. ongoing conversation)
    """
    console.print(Panel.fit(
        "[bold cyan]Exercise 3: Prompt Engine[/bold cyan]\n"
        "[dim]Module 7 - Complete Chatbot (Capstone)[/dim]\n\n"
        "This component assembles optimal prompts by combining:\n"
        "  Question + RAG Context + History + System Prompt",
        border_style="cyan",
    ))

    # ---- Part 1: Show preset comparison ----
    console.print("\n[bold yellow]Part 1: System Prompt Presets[/bold yellow]")
    display_preset_comparison("Why is my pod crashing?")

    # ---- Part 2: Show prompt assembly with different scenarios ----
    console.print("\n[bold yellow]Part 2: Prompt Assembly Scenarios[/bold yellow]")

    engine = PromptEngine(preset="expert")
    display_prompt_comparison(engine, "My pod keeps crashing with OOMKilled error")

    # ---- Part 3: Token breakdown ----
    console.print("\n[bold yellow]Part 3: Token Budget Analysis[/bold yellow]")

    # Simulate some context
    sample_context = [
        {
            "content": "# OOMKilled Error\nThe container was killed because it exceeded its memory limit.\nSolution: Increase resources.limits.memory in your pod spec.",
            "metadata": {"source_file": "pod-errors.md", "category": "kubernetes"},
        },
        {
            "content": "# Memory Limits\nAlways set both requests and limits:\n```yaml\nresources:\n  requests:\n    memory: 128Mi\n  limits:\n    memory: 256Mi\n```",
            "metadata": {"source_file": "resources.md", "category": "kubernetes"},
        },
    ]

    display_token_breakdown(
        engine,
        "My pod keeps crashing with OOMKilled error",
        context_docs=sample_context,
    )

    # ---- Part 4: Interactive mode ----
    console.print("\n[bold yellow]Part 4: Interactive Mode[/bold yellow]")
    console.print("[dim]Commands:[/dim]")
    console.print("[dim]  /preset <name>  - Switch preset (expert, quick, teacher)[/dim]")
    console.print("[dim]  /history        - Show conversation history[/dim]")
    console.print("[dim]  /clear          - Clear history[/dim]")
    console.print("[dim]  /tokens         - Show token breakdown for last prompt[/dim]")
    console.print("[dim]  /prompt         - Show the full assembled prompt[/dim]")
    console.print("[dim]  quit            - Exit[/dim]\n")

    # Use the expert preset for interactive mode
    engine = PromptEngine(preset="expert")

    # Sample RAG context for the demo
    demo_context = [
        {
            "content": (
                "# Common Kubernetes Errors\n"
                "- CrashLoopBackOff: Container keeps crashing\n"
                "- ImagePullBackOff: Can't pull container image\n"
                "- OOMKilled: Container ran out of memory (exit code 137)\n"
                "- Pending: Pod can't be scheduled (no resources available)"
            ),
            "metadata": {"source_file": "k8s-errors.md", "category": "kubernetes"},
        },
    ]

    last_question = None

    while True:
        try:
            user_input = Prompt.ask("[bold green]You[/bold green]")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/preset"):
                parts = user_input.split(maxsplit=1)
                if len(parts) > 1:
                    engine.set_system_prompt(parts[1].strip())
                else:
                    console.print("[dim]Usage: /preset expert|quick|teacher[/dim]")
                continue

            if user_input == "/history":
                if engine.history:
                    for msg in engine.history:
                        role = msg["role"].upper()
                        console.print(f"  [{role}] {msg['content']}")
                else:
                    console.print("[dim]No history yet.[/dim]")
                continue

            if user_input == "/clear":
                engine.clear_history()
                continue

            if user_input == "/tokens" and last_question:
                display_token_breakdown(engine, last_question, demo_context)
                continue

            if user_input == "/prompt" and last_question:
                msgs = engine.build_prompt(last_question, context_docs=demo_context)
                console.print(Panel(
                    msgs[0]["content"],
                    title="[bold blue]System Prompt[/bold blue]",
                    border_style="blue",
                    width=90,
                ))
                console.print(Panel(
                    msgs[1]["content"],
                    title="[bold green]User Prompt[/bold green]",
                    border_style="green",
                    width=90,
                ))
                continue

            # Regular question - build and show the prompt
            last_question = user_input

            # Build the prompt
            messages = engine.build_prompt(user_input, context_docs=demo_context)

            # Show a summary of what was assembled
            stats = engine.get_prompt_stats(user_input, context_docs=demo_context)
            console.print(
                f"\n[dim]Prompt assembled: {stats['total_tokens']} tokens "
                f"({stats['system_prompt_tokens']} system + "
                f"{stats['user_prompt_tokens']} user) | "
                f"Preset: {engine.preset_name} | "
                f"History: {len(engine.history)} msgs | "
                f"Context: {stats['context_docs_count']} docs[/dim]"
            )

            # Show a preview of the user prompt
            user_prompt_preview = messages[1]["content"]
            if len(user_prompt_preview) > 500:
                user_prompt_preview = user_prompt_preview[:500] + "\n\n[...truncated for display...]"

            console.print(Panel(
                user_prompt_preview,
                title="[bold]Assembled Prompt Preview[/bold]",
                border_style="cyan",
                width=90,
            ))

            # Add to history (simulate a conversation)
            engine.add_to_history("user", user_input)
            engine.add_to_history(
                "assistant",
                f"[Demo] This is where the LLM response would appear for: {user_input}"
            )

            console.print(
                "[dim]Tip: Use /prompt to see the full prompt, "
                "/tokens for breakdown, or /preset to change style.[/dim]"
            )

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

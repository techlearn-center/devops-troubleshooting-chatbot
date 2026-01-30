"""
Module 7 - Exercise 3: Prompt Engine (COMPLETE SOLUTION)
=========================================================

A sophisticated prompt assembly engine for the DevOps troubleshooting chatbot.

Features:
  - 3 system prompt presets: expert, quick, teacher
  - Token counting with tiktoken
  - Conversation history management with automatic trimming
  - get_history_summary() using LLM to summarize long histories
  - Debug mode that shows the full assembled prompt
  - Rich display comparing different prompt assemblies
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client setup -- supports both OpenAI and Ollama
# ---------------------------------------------------------------------------
if os.getenv("USE_OLLAMA", "false").lower() == "true":
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama2")
else:
    client = OpenAI()
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

console = Console()

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """Count tokens in *text*. Falls back to word-based estimate."""
    if HAS_TIKTOKEN:
        return len(_enc.encode(text))
    # Rough estimate: ~4 chars per token
    return len(text) // 4


def count_message_tokens(messages: list[dict]) -> int:
    """Count total tokens across a list of chat messages."""
    total = 0
    for msg in messages:
        total += 4  # overhead per message
        total += count_tokens(msg.get("content", ""))
        total += count_tokens(msg.get("role", ""))
    total += 2  # priming tokens
    return total


# ---------------------------------------------------------------------------
# System prompt presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, str] = {
    "expert": (
        "You are a senior DevOps engineer and site-reliability expert with 15+ years of "
        "experience across Kubernetes, Docker, CI/CD, cloud platforms, monitoring, and "
        "networking. You provide detailed, production-ready solutions with clear reasoning. "
        "Always include specific commands, configuration snippets, and best practices. "
        "Mention potential pitfalls and security considerations."
    ),
    "quick": (
        "You are a concise DevOps assistant. Give short, actionable answers. "
        "Use bullet points. Skip lengthy explanations. Focus on the exact commands "
        "or steps needed to solve the problem. One-liner solutions preferred."
    ),
    "teacher": (
        "You are a patient DevOps instructor teaching junior engineers. Explain concepts "
        "step by step. Use analogies to make complex topics accessible. After giving the "
        "solution, explain WHY it works. Include 'Learn More' links or topics for further "
        "study. Ask if the student has follow-up questions."
    ),
}


# ---------------------------------------------------------------------------
# PromptEngine
# ---------------------------------------------------------------------------
class PromptEngine:
    """Assembles prompts from system presets, context, history, and user query."""

    MAX_HISTORY_TOKENS = 2000
    MAX_CONTEXT_TOKENS = 1500
    MAX_TOTAL_TOKENS = 4096

    def __init__(self, preset: str = "expert", debug: bool = False):
        self.preset = preset
        self.debug = debug
        self.history: list[dict] = []
        self._system_prompt = PRESETS.get(preset, PRESETS["expert"])

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str):
        self._system_prompt = value

    def set_preset(self, name: str):
        """Switch to a named preset."""
        if name in PRESETS:
            self._system_prompt = PRESETS[name]
            self.preset = name
        else:
            console.print(f"[red]Unknown preset '{name}'. Available: {', '.join(PRESETS)}[/red]")

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def add_to_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history.clear()

    def get_trimmed_history(self) -> list[dict]:
        """Return history trimmed to fit within MAX_HISTORY_TOKENS."""
        if not self.history:
            return []
        trimmed = list(self.history)
        while count_message_tokens(trimmed) > self.MAX_HISTORY_TOKENS and len(trimmed) > 2:
            # Remove the oldest pair (user + assistant)
            trimmed.pop(0)
            if trimmed and trimmed[0]["role"] == "assistant":
                trimmed.pop(0)
        return trimmed

    # ------------------------------------------------------------------
    # get_history_summary  (exercise requirement -- LLM-based)
    # ------------------------------------------------------------------
    def get_history_summary(self) -> str:
        """Use the LLM to produce a concise summary of the conversation history.

        This is useful when the history is too long and we want to compress it
        into a single system-context paragraph rather than keeping all turns.
        """
        if not self.history:
            return ""

        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in self.history
        )
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": (
                        "Summarize the following DevOps troubleshooting conversation in 2-3 sentences. "
                        "Focus on: what problem the user has, what solutions were discussed, "
                        "and what the current status is."
                    )},
                    {"role": "user", "content": history_text},
                ],
                temperature=0.2,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            console.print(f"[yellow]History summary failed ({exc}); using truncation instead.[/yellow]")
            # Fallback: return the last 2 exchanges as plain text
            recent = self.history[-4:]
            return " | ".join(f"{m['role']}: {m['content'][:80]}" for m in recent)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------
    def assemble(
        self,
        user_query: str,
        context: Optional[str] = None,
        category: Optional[str] = None,
        use_summary: bool = False,
    ) -> list[dict]:
        """Build the full message list for the LLM call.

        Parameters
        ----------
        user_query : str
            The current user question.
        context : str, optional
            RAG-retrieved context to inject.
        category : str, optional
            Detected error category (adds a hint to system prompt).
        use_summary : bool
            If True, replace full history with an LLM summary.
        """
        # 1. System message
        system_parts = [self._system_prompt]
        if category:
            system_parts.append(
                f"\nThe user's query has been classified as category: {category.upper()}. "
                "Focus your expertise on this area."
            )
        system_msg = {"role": "system", "content": "\n".join(system_parts)}

        messages = [system_msg]

        # 2. History (summarised or trimmed)
        if use_summary and len(self.history) > 6:
            summary = self.get_history_summary()
            if summary:
                messages.append({
                    "role": "system",
                    "content": f"[Conversation summary so far]: {summary}",
                })
        else:
            messages.extend(self.get_trimmed_history())

        # 3. RAG context
        if context:
            ctx_tokens = count_tokens(context)
            if ctx_tokens > self.MAX_CONTEXT_TOKENS:
                # Truncate context to fit
                words = context.split()
                ratio = self.MAX_CONTEXT_TOKENS / max(ctx_tokens, 1)
                context = " ".join(words[: int(len(words) * ratio)])
            messages.append({
                "role": "system",
                "content": (
                    "Use the following knowledge base context to answer. "
                    "If the context does not contain the answer, say so.\n\n"
                    f"{context}"
                ),
            })

        # 4. User query
        messages.append({"role": "user", "content": user_query})

        # -- Debug mode: show full assembled prompt --
        if self.debug:
            self._display_debug(messages)

        return messages

    # ------------------------------------------------------------------
    # Debug display  (exercise requirement)
    # ------------------------------------------------------------------
    def _display_debug(self, messages: list[dict]):
        """Show the complete assembled prompt in a rich panel."""
        console.print(Panel("[bold yellow]DEBUG: Assembled Prompt[/bold yellow]", border_style="yellow"))
        total_tokens = count_message_tokens(messages)
        for i, msg in enumerate(messages):
            role = msg["role"].upper()
            content = msg["content"]
            tokens = count_tokens(content)
            color = {"SYSTEM": "cyan", "USER": "green", "ASSISTANT": "yellow"}.get(role, "white")
            console.print(f"  [{color}][{role}][/{color}] [dim]({tokens} tokens)[/dim]")
            # Show first 200 chars
            preview = content[:200] + ("..." if len(content) > 200 else "")
            console.print(f"    [dim]{preview}[/dim]")
        console.print(f"\n  [bold]Total tokens: {total_tokens}[/bold] / {self.MAX_TOTAL_TOKENS}")
        console.print()

    # ------------------------------------------------------------------
    # Token budget report
    # ------------------------------------------------------------------
    def token_report(self, messages: Optional[list[dict]] = None) -> dict:
        """Return a breakdown of token usage."""
        if messages is None:
            messages = self.assemble("(placeholder query)")
        system_tokens = sum(count_tokens(m["content"]) for m in messages if m["role"] == "system")
        history_tokens = count_message_tokens(self.get_trimmed_history())
        total = count_message_tokens(messages)
        return {
            "system_tokens": system_tokens,
            "history_tokens": history_tokens,
            "total_tokens": total,
            "remaining_for_response": max(0, self.MAX_TOTAL_TOKENS - total),
            "history_turns": len(self.history),
        }


# ---------------------------------------------------------------------------
# Rich comparison display
# ---------------------------------------------------------------------------
def compare_presets(query: str, context: Optional[str] = None):
    """Show a side-by-side comparison of prompt assemblies for each preset."""
    table = Table(title="Prompt Assembly Comparison", border_style="cyan", show_lines=True)
    table.add_column("Preset", style="bold", width=10)
    table.add_column("System Prompt (preview)", width=50)
    table.add_column("Tokens", justify="right", width=8)
    table.add_column("Messages", justify="right", width=8)

    for name in PRESETS:
        engine = PromptEngine(preset=name)
        # Add some fake history
        engine.add_to_history("user", "I have a pod that keeps crashing")
        engine.add_to_history("assistant", "Let me help you debug that CrashLoopBackOff issue.")
        msgs = engine.assemble(query, context=context, category="kubernetes")
        tokens = count_message_tokens(msgs)
        preview = PRESETS[name][:90] + "..."
        table.add_row(name, preview, str(tokens), str(len(msgs)))

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel(
        "[bold]Module 7 - Exercise 3: Prompt Engine[/bold]\n"
        "[dim]Smart prompt assembly with presets, history, and token management[/dim]",
        border_style="bright_cyan",
    ))

    # --- Demonstrate preset comparison ---
    console.print("\n[bold underline]1. Preset Comparison[/bold underline]\n")
    compare_presets(
        "My Kubernetes pod keeps crashing with OOMKilled",
        context="OOMKilled means the container exceeded its memory limit...",
    )

    # --- Demonstrate debug mode ---
    console.print("\n[bold underline]2. Debug Mode Demo[/bold underline]\n")
    engine = PromptEngine(preset="expert", debug=True)
    engine.add_to_history("user", "How do I check pod logs?")
    engine.add_to_history("assistant", "Use kubectl logs <pod-name> to view logs.")
    engine.add_to_history("user", "The pod is in CrashLoopBackOff")
    engine.add_to_history("assistant", "Check previous logs with kubectl logs <pod> --previous.")

    msgs = engine.assemble(
        "Now the pod shows OOMKilled, what do I do?",
        context="OOMKilled means the container exceeded its memory limit. Fix by increasing memory limits.",
        category="kubernetes",
    )

    # --- Token report ---
    console.print("[bold underline]3. Token Report[/bold underline]\n")
    report = engine.token_report(msgs)
    report_table = Table(border_style="green")
    report_table.add_column("Metric", style="bold")
    report_table.add_column("Value", justify="right")
    for k, v in report.items():
        report_table.add_row(k.replace("_", " ").title(), str(v))
    console.print(report_table)

    # --- History summary demo ---
    console.print("\n[bold underline]4. History Summary (LLM-based)[/bold underline]\n")
    console.print("[dim]Generating summary of conversation history...[/dim]")
    try:
        summary = engine.get_history_summary()
        console.print(Panel(summary, title="History Summary", border_style="magenta"))
    except Exception as exc:
        console.print(f"[yellow]Summary generation requires LLM access: {exc}[/yellow]")

    # --- Interactive mode ---
    console.print("\n[bold green]Interactive mode[/bold green]")
    console.print("Commands: preset:<name>, debug:on, debug:off, history, clear, quit\n")

    while True:
        try:
            user_input = console.input("[bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break

        # Meta-commands
        if user_input.startswith("preset:"):
            name = user_input.split(":", 1)[1].strip()
            engine.set_preset(name)
            console.print(f"[green]Switched to '{name}' preset.[/green]")
            continue
        if user_input == "debug:on":
            engine.debug = True
            console.print("[yellow]Debug mode ON[/yellow]")
            continue
        if user_input == "debug:off":
            engine.debug = False
            console.print("[yellow]Debug mode OFF[/yellow]")
            continue
        if user_input == "history":
            for msg in engine.history:
                role_color = "green" if msg["role"] == "user" else "yellow"
                console.print(f"  [{role_color}]{msg['role'].upper()}:[/{role_color}] {msg['content'][:100]}")
            continue
        if user_input == "clear":
            engine.clear_history()
            console.print("[green]History cleared.[/green]")
            continue

        # Normal query -- assemble and display
        engine.add_to_history("user", user_input)
        msgs = engine.assemble(user_input, category="general")
        report = engine.token_report(msgs)
        console.print(f"  [dim]Assembled {len(msgs)} messages, {report['total_tokens']} tokens[/dim]")

        # Simulate an assistant response
        engine.add_to_history("assistant", "(response would go here)")
        console.print()

    console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()

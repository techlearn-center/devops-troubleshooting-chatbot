"""
Exercise 1: Build a DevOps Prompt Library
==========================================
Module 6 - Advanced Prompting | DevOps Troubleshooting Chatbot Course

WHAT YOU'LL LEARN:
    - Why prompt templates save time and improve consistency
    - Python string.Template vs f-strings for prompt templates
    - How to organize prompts by use case (system vs user prompts)
    - Building a reusable PromptLibrary class with template methods

ANALOGY:
    Think of a prompt library like a recipe book for a restaurant kitchen.
    Instead of every chef improvising each dish from scratch, they follow
    tested recipes that guarantee consistent quality. Your prompt library
    is the "recipe book" for your AI assistant -- each template is a
    proven recipe for getting a specific type of answer.

HOW TO RUN:
    pip install openai python-dotenv rich
    python ex1_prompt_library.py

    For Ollama (local LLM):
        set USE_OLLAMA=true   (Windows)
        export USE_OLLAMA=true (Linux/Mac)
        Then run: python ex1_prompt_library.py

DURATION: ~30 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os                       # Access environment variables (like API keys)
import sys                      # System-level operations (exit, etc.)
from string import Template     # Python's built-in template engine for safe substitution
from dotenv import load_dotenv  # Load .env file so we don't hardcode secrets
from openai import OpenAI       # The OpenAI client works with both OpenAI and Ollama

# Rich library -- makes terminal output beautiful with colors, panels, tables
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.syntax import Syntax
from rich import box

# =============================================================================
# ENVIRONMENT SETUP -- supports both OpenAI (cloud) and Ollama (local)
# =============================================================================
# load_dotenv() reads a file called ".env" in the current directory.
# That file might contain: OPENAI_API_KEY=sk-abc123...
# This keeps secrets OUT of your source code (very important in DevOps!).
load_dotenv()

# Check which backend the student wants to use.
# Ollama runs models locally on your machine (free, private, no internet needed).
# OpenAI runs models in the cloud (paid, requires API key, very powerful).
if os.getenv("USE_OLLAMA", "false").lower() == "true":
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama"  # Ollama doesn't need a real key, but the client requires one
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama2")
else:
    client = OpenAI()  # Reads OPENAI_API_KEY from environment automatically
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# Rich console -- our pretty-printer for the terminal
console = Console()

# =============================================================================
# THE PROMPT LIBRARY CLASS
# =============================================================================
# WHY A CLASS?
#   A class bundles related data (the templates) and behavior (render, list)
#   into one neat package. It's like a toolbox: everything you need in one place.
#
# WHY string.Template INSTEAD OF f-strings?
#   f-strings evaluate immediately:  f"Hello {name}" -- name must exist NOW.
#   Template evaluates later:        Template("Hello $name") -- name can come later.
#   Templates are safer for user input because they won't execute arbitrary code.


class DevOpsPromptLibrary:
    """
    A reusable library of prompt templates for DevOps troubleshooting.

    Each template is a Python string.Template with $placeholders that get
    filled in when you call render(). Think of $placeholders as blank lines
    in a form -- you fill them in with the specific details of your problem.
    """

    def __init__(self):
        """Initialize the library with all our proven prompt templates."""

        # We store templates in a dictionary (dict) -- a key-value lookup table.
        # Key = template name (string), Value = dict with metadata + template.
        self.templates = {

            # -----------------------------------------------------------------
            # SYSTEM PROMPT: Sets the AI's personality and ground rules.
            # This is like giving a new employee their job description on day 1.
            # The system prompt is sent with EVERY request to set the tone.
            # -----------------------------------------------------------------
            "SYSTEM": {
                "description": "Base system prompt that defines the AI's personality and expertise",
                "category": "system",
                "variables": ["expertise_area"],
                "template": Template(
                    "You are a senior DevOps engineer with 15 years of experience "
                    "specializing in $expertise_area. You explain complex topics in "
                    "simple terms, always provide practical examples, and suggest "
                    "preventive measures. When troubleshooting, you think step-by-step "
                    "and consider security implications. You format your responses with "
                    "clear sections and use code blocks for commands."
                ),
            },

            # -----------------------------------------------------------------
            # ERROR_TROUBLESHOOT: For when something is broken and showing errors.
            # This template structures the problem so the AI can diagnose it.
            # -----------------------------------------------------------------
            "ERROR_TROUBLESHOOT": {
                "description": "Diagnose a specific error message in a DevOps tool",
                "category": "troubleshooting",
                "variables": ["tool_name", "error_message", "context"],
                "template": Template(
                    "I'm getting the following error while using $tool_name:\n\n"
                    "```\n$error_message\n```\n\n"
                    "Context: $context\n\n"
                    "Please help me:\n"
                    "1. Understand what this error means in plain English\n"
                    "2. Identify the most likely root cause\n"
                    "3. Provide step-by-step commands to fix it\n"
                    "4. Suggest how to prevent this error in the future"
                ),
            },

            # -----------------------------------------------------------------
            # COMPLEX_ISSUE: For multi-layered problems that need deep analysis.
            # Like bringing your car to a mechanic with "it makes a weird noise."
            # -----------------------------------------------------------------
            "COMPLEX_ISSUE": {
                "description": "Deep-dive into a complex multi-component issue",
                "category": "troubleshooting",
                "variables": ["environment", "symptoms", "recent_changes", "affected_services"],
                "template": Template(
                    "I have a complex issue in my $environment environment.\n\n"
                    "**Symptoms observed:**\n$symptoms\n\n"
                    "**Recent changes made:**\n$recent_changes\n\n"
                    "**Affected services:**\n$affected_services\n\n"
                    "Please provide:\n"
                    "1. A systematic diagnosis approach (what to check first, second, etc.)\n"
                    "2. The most likely root cause based on the symptoms and recent changes\n"
                    "3. Immediate mitigation steps to restore service\n"
                    "4. A proper fix with commands and config changes\n"
                    "5. Post-incident improvements to prevent recurrence"
                ),
            },

            # -----------------------------------------------------------------
            # QUICK_HELP: For simple "how do I..." questions.
            # Like asking a coworker a quick question at their desk.
            # -----------------------------------------------------------------
            "QUICK_HELP": {
                "description": "Get a quick, concise answer to a simple DevOps question",
                "category": "help",
                "variables": ["question"],
                "template": Template(
                    "Quick DevOps question: $question\n\n"
                    "Please give me:\n"
                    "- A one-sentence answer\n"
                    "- The exact command(s) I need\n"
                    "- One common gotcha to watch out for"
                ),
            },

            # -----------------------------------------------------------------
            # STRUCTURED_JSON: Forces the AI to return valid JSON.
            # Useful when your code needs to parse the response programmatically.
            # -----------------------------------------------------------------
            "STRUCTURED_JSON": {
                "description": "Get a structured JSON response for programmatic parsing",
                "category": "structured",
                "variables": ["task_description", "json_schema_hint"],
                "template": Template(
                    "Analyze the following DevOps task and return ONLY valid JSON:\n\n"
                    "Task: $task_description\n\n"
                    "Return your response as a JSON object with this structure:\n"
                    "$json_schema_hint\n\n"
                    "IMPORTANT: Return ONLY the JSON object. No markdown, no explanation, "
                    "no code fences. Just raw JSON."
                ),
            },

            # =================================================================
            # TODO (STUDENT EXERCISE): Add 2 more templates below!
            # =================================================================
            # Ideas for your custom templates:
            #   - SECURITY_AUDIT: Analyze a config file for security issues
            #   - MIGRATION_PLAN: Help plan a migration (e.g., on-prem to cloud)
            #   - CODE_REVIEW: Review a Dockerfile or Terraform file
            #   - INCIDENT_REPORT: Generate a post-mortem from incident details
            #
            # Follow the same pattern as the templates above:
            #   "TEMPLATE_NAME": {
            #       "description": "...",
            #       "category": "...",
            #       "variables": ["var1", "var2"],
            #       "template": Template("... $var1 ... $var2 ..."),
            #   },

            # TODO: Add your first custom template here

            # TODO: Add your second custom template here
        }

    def list_templates(self) -> Table:
        """
        Return a Rich Table listing all available templates.

        This is like the table of contents in our recipe book -- it shows
        what's available without showing the full recipe.
        """
        table = Table(
            title="DevOps Prompt Library",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Name", style="bold cyan", width=22)
        table.add_column("Category", style="magenta", width=16)
        table.add_column("Description", style="white", width=45)
        table.add_column("Variables", style="green", width=30)

        for name, info in self.templates.items():
            table.add_row(
                name,
                info["category"],
                info["description"],
                ", ".join(info["variables"]),
            )
        return table

    def render(self, template_name: str, **variables) -> str:
        """
        Fill in a template with the provided variables and return the prompt string.

        Args:
            template_name: The key of the template (e.g., "ERROR_TROUBLESHOOT")
            **variables:   Keyword arguments matching the template's $placeholders

        Returns:
            The fully rendered prompt string, ready to send to the LLM.

        Raises:
            KeyError: If the template name doesn't exist in the library.
            KeyError: If a required variable is missing.
        """
        if template_name not in self.templates:
            available = ", ".join(self.templates.keys())
            raise KeyError(
                f"Template '{template_name}' not found. Available: {available}"
            )

        template_obj = self.templates[template_name]["template"]
        # safe_substitute won't crash on missing variables -- it leaves $var as-is.
        # substitute (without 'safe') would raise KeyError on missing variables.
        return template_obj.safe_substitute(**variables)

    def get_variables(self, template_name: str) -> list:
        """Return the list of variable names a template expects."""
        return self.templates[template_name]["variables"]


# =============================================================================
# HELPER: Send a rendered prompt to the LLM and get a response
# =============================================================================
def ask_llm(user_prompt: str, system_prompt: str = None) -> str:
    """
    Send a prompt to the LLM (OpenAI or Ollama) and return the response text.

    Args:
        user_prompt:   The main question/instruction for the AI.
        system_prompt: (Optional) Sets the AI's personality/role.

    Returns:
        The AI's response as a plain string.
    """
    messages = []
    if system_prompt:
        # System messages are "behind the scenes" instructions the user doesn't see.
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,  # 0 = deterministic, 1 = creative. 0.7 is a good balance.
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error] {e}"


# =============================================================================
# INTERACTIVE DEMO
# =============================================================================
def interactive_demo():
    """
    Let the user pick a template, fill in variables, and see the LLM response.
    This is the main entry point when running the script.
    """
    console.print(Panel(
        "[bold white]Exercise 1: DevOps Prompt Library[/bold white]\n\n"
        "A prompt library is a collection of tested, reusable prompt templates.\n"
        "Instead of writing prompts from scratch every time, you pick a template\n"
        "and fill in the blanks. This ensures consistent, high-quality responses.\n\n"
        f"[dim]Backend: {'Ollama (local)' if os.getenv('USE_OLLAMA', 'false').lower() == 'true' else 'OpenAI (cloud)'} | Model: {MODEL}[/dim]",
        title="Module 6 - Advanced Prompting",
        border_style="bright_blue",
    ))

    # Create our library instance
    library = DevOpsPromptLibrary()

    # Show the catalog of available templates
    console.print(library.list_templates())
    console.print()

    # Build a numbered menu for selection
    template_names = list(library.templates.keys())
    for i, name in enumerate(template_names, 1):
        desc = library.templates[name]["description"]
        console.print(f"  [cyan]{i}[/cyan]. {name} -- {desc}")
    console.print(f"  [cyan]{len(template_names) + 1}[/cyan]. Exit")
    console.print()

    while True:
        choice = IntPrompt.ask(
            "Pick a template number",
            default=1,
        )
        if choice == len(template_names) + 1:
            console.print("[yellow]Goodbye![/yellow]")
            break
        if choice < 1 or choice > len(template_names):
            console.print("[red]Invalid choice, try again.[/red]")
            continue

        selected_name = template_names[choice - 1]
        selected_info = library.templates[selected_name]
        console.print(f"\n[bold green]Selected:[/bold green] {selected_name}")
        console.print(f"[dim]Variables needed: {', '.join(selected_info['variables'])}[/dim]\n")

        # Collect variable values from the user
        variables = {}
        for var in selected_info["variables"]:
            value = Prompt.ask(f"  Enter value for [cyan]${var}[/cyan]")
            variables[var] = value

        # Render the template
        rendered_prompt = library.render(selected_name, **variables)
        console.print(Panel(
            rendered_prompt,
            title="Rendered Prompt (this is what gets sent to the LLM)",
            border_style="yellow",
        ))

        # Send to the LLM
        console.print("\n[bold]Sending to LLM...[/bold]")

        # Use the SYSTEM template as the system prompt for non-system templates
        system_prompt = None
        if selected_name != "SYSTEM":
            system_prompt = library.render("SYSTEM", expertise_area="cloud infrastructure and CI/CD")

        response = ask_llm(rendered_prompt, system_prompt=system_prompt)

        console.print(Panel(
            response,
            title="LLM Response",
            border_style="green",
        ))
        console.print()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    interactive_demo()

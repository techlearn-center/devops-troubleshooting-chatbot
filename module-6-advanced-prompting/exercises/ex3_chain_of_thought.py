"""
Exercise 3: Chain-of-Thought Troubleshooting Engine
=====================================================
Module 6 - Advanced Prompting | DevOps Troubleshooting Chatbot Course

WHAT YOU'LL LEARN:
    - What chain-of-thought (CoT) prompting is and why it works
    - Why step-by-step reasoning catches issues that direct answers miss
    - The 5-step troubleshooting framework: Understand -> Causes -> Diagnose -> Fix -> Prevent
    - How to structure CoT prompts for technical troubleshooting

ANALOGY:
    Imagine you're a math teacher grading tests. Student A writes just the
    final answer: "42". Student B writes out every step of their reasoning.
    When Student A is wrong, you have no idea where they went wrong.
    When Student B is wrong, you can pinpoint the exact step where the
    mistake happened.

    Chain-of-thought prompting is like telling the LLM: "Show your work."
    By forcing step-by-step reasoning, the LLM catches logical errors that
    it would miss if it jumped straight to the answer.

HOW TO RUN:
    pip install openai python-dotenv rich
    python ex3_chain_of_thought.py

    For Ollama: set USE_OLLAMA=true then run.

DURATION: ~45 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich import box

# =============================================================================
# ENVIRONMENT SETUP -- supports both OpenAI (cloud) and Ollama (local)
# =============================================================================
load_dotenv()

if os.getenv("USE_OLLAMA", "false").lower() == "true":
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama"
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama2")
else:
    client = OpenAI()
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

console = Console()

# =============================================================================
# THE 5-STEP TROUBLESHOOTING FRAMEWORK
# =============================================================================
# This framework is the backbone of our CoT engine. Each step forces the LLM
# to think about a different aspect of the problem BEFORE jumping to a fix.
#
# Why 5 steps? Research and practice show that structured troubleshooting
# works better than random guessing. It's like the scientific method:
#   1. Observe (Understand)     -- What exactly is happening?
#   2. Hypothesize (Causes)     -- What could cause this?
#   3. Test (Diagnose)          -- How do we narrow it down?
#   4. Solve (Fix)              -- What's the actual fix?
#   5. Learn (Prevent)          -- How do we stop it from recurring?

TROUBLESHOOTING_STEPS = [
    {
        "name": "Understand the Problem",
        "instruction": (
            "Restate the problem in your own words. What is the user experiencing? "
            "What is the expected behavior vs actual behavior? Identify the specific "
            "tool, service, or component involved."
        ),
    },
    {
        "name": "Identify Possible Causes",
        "instruction": (
            "List the top 3-5 most likely root causes, ordered from most to least "
            "probable. For each cause, explain briefly why it could lead to this symptom. "
            "Consider: configuration errors, resource limits, network issues, "
            "permissions, version incompatibilities."
        ),
    },
    {
        "name": "Diagnostic Commands",
        "instruction": (
            "For each possible cause, provide the exact diagnostic command(s) to "
            "check it. Use code blocks. Explain what each command output would tell us. "
            "This helps narrow down which cause is the actual one."
        ),
    },
    {
        "name": "Fix the Issue",
        "instruction": (
            "Provide the step-by-step fix for the most likely cause. Use numbered steps "
            "with exact commands in code blocks. Include any config file changes needed. "
            "Mention any risks or downtime involved in the fix."
        ),
    },
    {
        "name": "Prevent Recurrence",
        "instruction": (
            "Suggest 2-3 preventive measures so this issue doesn't happen again. "
            "Think about: monitoring/alerts, CI/CD checks, documentation, "
            "infrastructure-as-code guardrails, team practices."
        ),
    },
]


# =============================================================================
# TROUBLESHOOTING ENGINE CLASS
# =============================================================================
class TroubleshootingEngine:
    """
    A chain-of-thought engine that guides the LLM through systematic
    troubleshooting using the 5-step framework.

    Instead of asking "How do I fix X?" (which might get a shallow answer),
    we ask the LLM to think through each step explicitly. This produces
    deeper, more reliable troubleshooting guides.
    """

    def __init__(self):
        """Initialize with the 5-step framework."""
        self.steps = TROUBLESHOOTING_STEPS

    def build_cot_prompt(self, issue_description: str) -> str:
        """
        Build a chain-of-thought prompt that forces step-by-step reasoning.

        The key technique: we explicitly tell the LLM to work through
        each step before giving the final answer. This is like giving
        a junior engineer a checklist to follow.

        Args:
            issue_description: The user's description of their DevOps problem.

        Returns:
            A structured prompt string with all 5 steps laid out.
        """
        # Build the step instructions
        step_text = ""
        for i, step in enumerate(self.steps, 1):
            step_text += f"\n**Step {i}: {step['name']}**\n"
            step_text += f"{step['instruction']}\n"

        prompt = (
            "You are a senior DevOps engineer performing systematic troubleshooting.\n"
            "You MUST work through each step below IN ORDER. Do not skip any step.\n"
            "Think carefully at each step before moving to the next.\n\n"
            f"## Issue Reported\n{issue_description}\n\n"
            f"## Your Troubleshooting Analysis\n"
            f"Work through these steps one at a time:\n"
            f"{step_text}\n"
            "Format each step with a clear header. Use code blocks for all commands."
        )
        return prompt

    def build_regular_prompt(self, issue_description: str) -> str:
        """
        Build a standard (non-CoT) prompt for comparison.

        This is the "normal" way most people ask questions -- just describe
        the problem and ask for help. We use this to show the quality
        difference when compared to the CoT version.

        Args:
            issue_description: The user's description of their DevOps problem.

        Returns:
            A simple prompt string without step-by-step structure.
        """
        return (
            "You are a DevOps engineer. Help me with this issue:\n\n"
            f"{issue_description}\n\n"
            "Please provide the solution."
        )

    # =========================================================================
    # TODO (STUDENT EXERCISE): Create a custom troubleshooting template!
    # =========================================================================
    # Build a method called `build_custom_cot_prompt` that uses a DIFFERENT
    # step structure. Ideas:
    #   - Security incident response: Contain -> Identify -> Eradicate -> Recover -> Lessons
    #   - Performance optimization: Measure -> Profile -> Identify -> Optimize -> Validate
    #   - Migration planning: Assess -> Plan -> Test -> Migrate -> Verify
    #
    # def build_custom_cot_prompt(self, issue_description: str) -> str:
    #     """
    #     TODO: Implement your custom chain-of-thought template here.
    #     Follow the same pattern as build_cot_prompt but with different steps.
    #     """
    #     pass

    def parse_response_steps(self, response: str) -> list:
        """
        Attempt to parse the LLM response into individual steps.

        This is a simple parser that looks for "Step N:" patterns.
        In production, you might use regex or ask the LLM to return JSON.

        Args:
            response: The raw LLM response text.

        Returns:
            A list of (step_name, step_content) tuples.
        """
        parsed_steps = []
        current_step = None
        current_content = []

        for line in response.split("\n"):
            # Check if this line starts a new step
            is_step_header = False
            for i, step in enumerate(self.steps, 1):
                # Look for patterns like "Step 1:", "**Step 1:", "## Step 1:"
                markers = [
                    f"Step {i}:",
                    f"**Step {i}:",
                    f"## Step {i}",
                    f"### Step {i}",
                    step["name"],
                ]
                if any(marker.lower() in line.lower() for marker in markers):
                    # Save the previous step if there was one
                    if current_step is not None:
                        parsed_steps.append((current_step, "\n".join(current_content).strip()))
                    current_step = f"Step {i}: {step['name']}"
                    current_content = []
                    is_step_header = True
                    break

            if not is_step_header and current_step is not None:
                current_content.append(line)

        # Don't forget the last step
        if current_step is not None:
            parsed_steps.append((current_step, "\n".join(current_content).strip()))

        return parsed_steps

    def display_parsed_steps(self, parsed_steps: list):
        """
        Display the parsed steps using Rich panels with color coding.

        Each step gets its own colored panel so the output is easy to scan.
        """
        # Color scheme for each step type
        colors = ["bright_blue", "yellow", "cyan", "green", "magenta"]

        for i, (step_name, content) in enumerate(parsed_steps):
            color = colors[i % len(colors)]
            console.print(Panel(
                Markdown(content) if content else "[dim]No content parsed for this step[/dim]",
                title=f"[bold]{step_name}[/bold]",
                border_style=color,
                padding=(1, 2),
            ))


# =============================================================================
# TEST SCENARIOS -- Real DevOps problems for practice
# =============================================================================
TEST_SCENARIOS = [
    {
        "id": 1,
        "title": "Terraform Apply Stuck",
        "description": (
            "My Terraform apply has been running for 30 minutes and isn't progressing. "
            "It's stuck on creating an AWS RDS instance. The terminal shows no errors, "
            "just the spinner. I'm using Terraform 1.5.7 with AWS provider 5.x. "
            "This is a production database with Multi-AZ enabled."
        ),
    },
    {
        "id": 2,
        "title": "Kubernetes ImagePullBackOff",
        "description": (
            "My Kubernetes pod is stuck in ImagePullBackOff status. The image is "
            "stored in a private ECR registry. It was working yesterday but broke "
            "after we rotated AWS credentials. Running kubectl describe pod shows: "
            "'Failed to pull image: unauthorized: authentication required'."
        ),
    },
    {
        "id": 3,
        "title": "Docker Build COPY Failed",
        "description": (
            "Docker build fails with 'COPY failed: file not found in build context'. "
            "My Dockerfile has 'COPY ./src /app/src' but the src directory definitely "
            "exists. I'm running 'docker build .' from the project root. The build "
            "was working before I added a .dockerignore file."
        ),
    },
    {
        "id": 4,
        "title": "GitHub Actions Slow Pipeline",
        "description": (
            "Our GitHub Actions CI/CD workflow takes 45 minutes to complete. It runs "
            "npm install, lint, unit tests, integration tests, Docker build, and deploy. "
            "The team is frustrated because PRs take forever to merge. We have about "
            "200 npm dependencies and the test suite has 500 tests."
        ),
    },
    {
        "id": 5,
        "title": "Database Connection Refused",
        "description": (
            "Our Kubernetes application pod can't connect to a PostgreSQL database "
            "running in another pod. Error: 'connection refused on port 5432'. "
            "Both pods are in the same namespace. The database pod shows as Running "
            "and the service is configured. It works when I port-forward locally."
        ),
    },
]


# =============================================================================
# LLM HELPER WITH PROGRESS SPINNER
# =============================================================================
def ask_llm_with_spinner(prompt: str, label: str = "Thinking") -> tuple:
    """
    Send a prompt to the LLM with a pretty progress spinner in the terminal.

    Nobody likes staring at a frozen terminal. The spinner tells the user
    "I'm working on it" -- an important UX detail for CLI tools.

    Args:
        prompt: The full prompt to send.
        label:  Text to display next to the spinner.

    Returns:
        A tuple of (response_text, elapsed_seconds).
    """
    start = time.time()

    # Rich Progress with a spinner -- much better than a frozen screen
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        console=console,
        transient=True,  # Remove the spinner once done (clean output)
    ) as progress:
        progress.add_task(f"{label}...", total=None)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            result = response.choices[0].message.content
        except Exception as e:
            result = f"[LLM Error] {e}"

    elapsed = time.time() - start
    return result, elapsed


# =============================================================================
# MAIN INTERACTIVE DEMO
# =============================================================================
def main():
    """
    The main demo with two modes:
    1. Comparison mode: see regular vs CoT side-by-side on a test scenario
    2. Interactive mode: describe your own issue and get CoT troubleshooting
    """
    console.print(Panel(
        "[bold white]Exercise 3: Chain-of-Thought Troubleshooting Engine[/bold white]\n\n"
        "Chain-of-thought = making the LLM 'show its work.'\n"
        "We'll compare a regular answer vs a structured 5-step analysis\n"
        "on the same DevOps problem. Watch how CoT catches more edge cases!\n\n"
        "The 5-Step Framework:\n"
        "  [cyan]1. Understand[/cyan] -> 2. Causes -> 3. Diagnose -> 4. Fix -> [green]5. Prevent[/green]\n\n"
        f"[dim]Backend: {'Ollama (local)' if os.getenv('USE_OLLAMA', 'false').lower() == 'true' else 'OpenAI (cloud)'} | Model: {MODEL}[/dim]",
        title="Module 6 - Advanced Prompting",
        border_style="bright_blue",
    ))

    engine = TroubleshootingEngine()

    # --- Mode Selection ---
    console.print("\n[bold]Choose a mode:[/bold]")
    console.print("  1. [cyan]Comparison Mode[/cyan] -- Regular vs CoT on a test scenario")
    console.print("  2. [cyan]Interactive Mode[/cyan] -- Describe your own issue")
    console.print("  3. [cyan]Exit[/cyan]")
    console.print()

    mode = IntPrompt.ask("Select mode", default=1)

    if mode == 3:
        console.print("[yellow]Goodbye![/yellow]")
        return

    # --- Get the issue description ---
    if mode == 1:
        # Show test scenarios for selection
        console.print("\n[bold]Test Scenarios (real DevOps problems):[/bold]")
        for s in TEST_SCENARIOS:
            console.print(f"  {s['id']}. [cyan]{s['title']}[/cyan]")
        console.print()

        scenario_id = IntPrompt.ask("Pick a scenario", default=1)
        scenario = next((s for s in TEST_SCENARIOS if s["id"] == scenario_id), TEST_SCENARIOS[0])
        issue = scenario["description"]

        console.print(Panel(
            issue,
            title=f"Scenario: {scenario['title']}",
            border_style="white",
        ))
    else:
        issue = Prompt.ask("\nDescribe your DevOps issue in detail")

    # =========================================================================
    # COMPARISON: Regular prompt vs Chain-of-Thought prompt
    # =========================================================================
    if mode == 1:
        # --- Regular (non-CoT) response ---
        console.print("\n[bold red]--- REGULAR PROMPT (no chain-of-thought) ---[/bold red]")
        regular_prompt = engine.build_regular_prompt(issue)
        console.print(Panel(
            regular_prompt,
            title="Regular Prompt Sent",
            border_style="dim",
        ))

        regular_response, regular_time = ask_llm_with_spinner(regular_prompt, "Getting regular response")
        console.print(Panel(
            Markdown(regular_response),
            title=f"Regular Response ({regular_time:.1f}s, {len(regular_response)} chars)",
            border_style="red",
        ))

        # --- CoT response ---
        console.print("\n[bold green]--- CHAIN-OF-THOUGHT PROMPT (5-step framework) ---[/bold green]")

    cot_prompt = engine.build_cot_prompt(issue)

    if mode == 1:
        console.print(Panel(
            cot_prompt,
            title="CoT Prompt Sent (notice the step-by-step structure)",
            border_style="dim",
        ))

    cot_response, cot_time = ask_llm_with_spinner(cot_prompt, "Running 5-step analysis")

    # Parse and display the CoT response step-by-step
    parsed = engine.parse_response_steps(cot_response)

    if parsed:
        console.print(f"\n[bold green]CoT Response ({cot_time:.1f}s, {len(cot_response)} chars) -- Parsed into steps:[/bold green]")
        engine.display_parsed_steps(parsed)
    else:
        # Fallback: show raw response if parsing didn't find step markers
        console.print(Panel(
            Markdown(cot_response),
            title=f"CoT Response ({cot_time:.1f}s, {len(cot_response)} chars)",
            border_style="green",
        ))

    # --- Comparison Summary (only in comparison mode) ---
    if mode == 1:
        console.print()
        summary = Table(title="Regular vs Chain-of-Thought Comparison", box=box.DOUBLE_EDGE, show_lines=True)
        summary.add_column("Metric", style="bold")
        summary.add_column("Regular", justify="center")
        summary.add_column("Chain-of-Thought", justify="center")

        # Check quality indicators
        def check(text, marker):
            return "[green]Yes[/green]" if marker.lower() in text.lower() else "[red]No[/red]"

        summary.add_row("Response Length", f"{len(regular_response)} chars", f"{len(cot_response)} chars")
        summary.add_row("Time", f"{regular_time:.1f}s", f"{cot_time:.1f}s")
        summary.add_row("Has Diagnostic Commands?", check(regular_response, "```"), check(cot_response, "```"))
        summary.add_row("Mentions Root Cause?", check(regular_response, "cause"), check(cot_response, "cause"))
        summary.add_row("Includes Prevention?", check(regular_response, "prevent"), check(cot_response, "prevent"))
        summary.add_row("Structured Steps?", check(regular_response, "step"), check(cot_response, "step"))
        summary.add_row("Steps Parsed", "N/A", f"{len(parsed)} of 5")

        console.print(summary)

        console.print(Panel(
            "[bold]Key Takeaways:[/bold]\n\n"
            "1. [cyan]Regular prompts[/cyan] give decent answers but may skip important aspects.\n"
            "2. [cyan]CoT prompts[/cyan] force thorough analysis at each stage.\n"
            "3. CoT consistently produces [green]diagnostic commands[/green] and [green]prevention tips[/green].\n"
            "4. The trade-off: CoT responses are longer and take more time/tokens.\n\n"
            "[bold]When to use CoT:[/bold]\n"
            "- Complex, multi-layered problems (production incidents)\n"
            "- When you need to explain your reasoning to others (post-mortems)\n"
            "- When the first answer you got was too shallow\n\n"
            "[bold]When NOT to use CoT:[/bold]\n"
            "- Simple factual questions ('What port does Redis use?')\n"
            "- When you need a quick command, not an analysis",
            title="What Did You Learn?",
            border_style="bright_green",
        ))


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

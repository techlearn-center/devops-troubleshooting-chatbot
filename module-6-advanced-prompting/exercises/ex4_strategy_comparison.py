"""
Exercise 4: Prompting Strategy Comparison Lab
==============================================
Module 6 - Advanced Prompting | DevOps Troubleshooting Chatbot Course

WHAT YOU'LL LEARN:
    - How different prompting strategies produce different quality responses
    - When to use each strategy (a decision framework you can use at work)
    - How temperature affects output consistency and creativity
    - How to evaluate LLM response quality programmatically

ANALOGY:
    Imagine you're asking directions to a restaurant:
      - Zero-shot:        "Where's the Italian place?" (vague, quick)
      - Few-shot:         "I'm looking for Luigi's. Last time I went to Mario's
                           by taking Main St and turning right..." (example-guided)
      - Chain-of-thought: "I need to go north on Main, then after the park turn
                           east, then it's the 3rd building on the left." (step-by-step)
      - Structured JSON:  Returns exact GPS coordinates in a parseable format.

    Each approach has its place. This lab lets you see them ALL on the same
    question so you can build intuition for which to use when.

HOW TO RUN:
    pip install openai python-dotenv rich
    python ex4_strategy_comparison.py

    For Ollama: set USE_OLLAMA=true then run.

DURATION: ~50 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import json                     # For parsing structured JSON responses
import time                     # For measuring response times
from dotenv import load_dotenv
from openai import OpenAI

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.prompt import Prompt, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.layout import Layout
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
# LLM HELPER WITH TIMING
# =============================================================================
def call_llm(prompt: str, temperature: float = 0.7, label: str = "Thinking") -> dict:
    """
    Call the LLM and return a result dictionary with metadata.

    Why return a dict instead of just the text? Because in a comparison lab,
    we need to measure and compare multiple dimensions: speed, length,
    quality indicators. Bundling everything together keeps our code clean.

    Args:
        prompt:      The full prompt string.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
        label:       Text shown on the progress spinner.

    Returns:
        Dict with keys: response, time, tokens_est, prompt_preview
    """
    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(f"{label}...", total=None)

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            text = response.choices[0].message.content
        except Exception as e:
            text = f"[LLM Error] {e}"

    elapsed = time.time() - start
    return {
        "response": text,
        "time": round(elapsed, 2),
        "tokens_est": len(text) // 4,           # Rough estimate: ~4 chars per token
        "char_count": len(text),
        "prompt_preview": prompt[:120] + "...",  # First 120 chars for display
    }


# =============================================================================
# STRATEGY TESTER CLASS
# =============================================================================
# This class encapsulates the 4 core prompting strategies. Each method takes
# a question and returns a fully constructed prompt string. The prompts are
# NOT sent to the LLM here -- that's the ComparisonRunner's job. This
# separation of concerns (building vs sending) makes the code testable.

class StrategyTester:
    """
    Implements 4 prompting strategies for DevOps troubleshooting.

    Each strategy builds a prompt differently. Same question, different wrapping.
    Think of it like cooking the same ingredient 4 ways -- the base is the same,
    but the technique completely changes the result.
    """

    # --- Strategy 1: Zero-Shot ---
    # Just ask the question. No examples, no structure, no hand-holding.
    # Best for: simple factual questions where the LLM already knows the answer.
    # Weakest at: complex multi-step problems, consistency.
    def zero_shot(self, question: str) -> str:
        """
        Zero-shot: Ask the question directly with minimal framing.

        Like asking a stranger on the street for directions -- you get an
        answer, but the quality depends entirely on their knowledge and
        how they interpret your question.
        """
        return (
            f"You are a DevOps engineer. Answer this question:\n\n"
            f"{question}"
        )

    # --- Strategy 2: Few-Shot ---
    # Provide 2 examples of ideal Q&A pairs, then ask the real question.
    # Best for: when you need a specific format or depth of answer.
    # Trade-off: uses more tokens (the examples consume context window).
    def few_shot(self, question: str) -> str:
        """
        Few-shot: Provide 2 example Q&A pairs before the actual question.

        Like showing a new employee 2 examples of completed work before
        asking them to do their own. They mirror the pattern they've seen.
        """
        return (
            "You are a DevOps troubleshooting expert. Follow the same format "
            "and depth as the examples below.\n\n"
            "--- Example 1 ---\n"
            "Question: My Docker container exits immediately after starting. What should I check?\n"
            "Answer:\n"
            "**Root Cause Analysis:** The container's main process is either crashing or "
            "completing instantly. Common causes include:\n"
            "- The CMD/ENTRYPOINT command fails (missing binary, wrong path)\n"
            "- The app crashes on startup (missing env vars, can't connect to DB)\n"
            "- The process runs in background mode and the foreground exits\n\n"
            "**Diagnostic Steps:**\n"
            "```bash\n"
            "# Check the exit code\n"
            "docker inspect <container_id> --format='{{.State.ExitCode}}'\n"
            "# View the logs from the crashed container\n"
            "docker logs <container_id>\n"
            "# Run interactively to debug\n"
            "docker run -it <image> /bin/sh\n"
            "```\n\n"
            "**Fix:** Ensure your CMD runs in the foreground. For example, use "
            "`CMD [\"node\", \"server.js\"]` not `CMD [\"sh\", \"-c\", \"node server.js &\"]`.\n\n"
            "**Prevention:** Always test containers locally with `docker run` before deploying. "
            "Add health checks to your Dockerfile.\n\n"
            "--- Example 2 ---\n"
            "Question: Terraform destroy is stuck and won't complete. How do I handle this?\n"
            "Answer:\n"
            "**Root Cause Analysis:** Terraform destroy can hang when:\n"
            "- A resource has a deletion protection enabled (RDS, S3)\n"
            "- There are dependent resources not managed by Terraform\n"
            "- The API call to the cloud provider is timing out\n\n"
            "**Diagnostic Steps:**\n"
            "```bash\n"
            "# Enable debug logging to see what's stuck\n"
            "TF_LOG=DEBUG terraform destroy\n"
            "# Check the state for the stuck resource\n"
            "terraform state list\n"
            "terraform state show <resource_address>\n"
            "```\n\n"
            "**Fix:** Disable deletion protection first, then retry destroy:\n"
            "```bash\n"
            "# For RDS, update the resource to disable deletion protection\n"
            "# Then run terraform destroy again\n"
            "# As a last resort, remove from state (resource stays in cloud!)\n"
            "terraform state rm <resource_address>\n"
            "```\n\n"
            "**Prevention:** Use lifecycle rules in Terraform. Tag resources with "
            "`managed-by: terraform`. Document manual steps needed before destroy.\n\n"
            "--- Your Turn ---\n"
            f"Question: {question}\n"
            "Answer:"
        )

    # --- Strategy 3: Chain-of-Thought ---
    # Explicitly instruct step-by-step reasoning before the answer.
    # Best for: complex problems that need deep analysis.
    # Trade-off: longer responses, more tokens, slower.
    def chain_of_thought(self, question: str) -> str:
        """
        Chain-of-thought: Force step-by-step reasoning.

        Like asking a doctor to explain their diagnosis process instead
        of just prescribing medicine. You get the reasoning, not just
        the conclusion.
        """
        return (
            "You are a senior DevOps engineer performing systematic troubleshooting.\n"
            "Think through this problem step by step. You MUST complete ALL steps.\n\n"
            f"Problem: {question}\n\n"
            "Work through these steps:\n\n"
            "**Step 1 - Understand:** Restate the problem. What is expected vs actual behavior?\n\n"
            "**Step 2 - Root Causes:** List the top 3 most likely causes, ranked by probability.\n\n"
            "**Step 3 - Diagnose:** For each cause, provide the exact command to check it.\n\n"
            "**Step 4 - Fix:** Provide the step-by-step fix for the most likely cause.\n\n"
            "**Step 5 - Prevent:** How do we stop this from happening again?\n\n"
            "Show your reasoning at every step. Use code blocks for all commands."
        )

    # --- Strategy 4: Structured JSON ---
    # Ask the LLM to return a JSON object instead of prose.
    # Best for: when your code needs to PARSE the response programmatically.
    # Trade-off: less readable for humans, may require retry on invalid JSON.
    def structured_json(self, question: str) -> str:
        """
        Structured JSON: Request a machine-parseable JSON response.

        Like filling out a form instead of writing a free-text essay.
        The structure is fixed, so your code can reliably extract fields.
        """
        return (
            "You are a DevOps troubleshooting API. Analyze the following issue "
            "and return ONLY a valid JSON object. No markdown, no explanation, "
            "no code fences -- ONLY raw JSON.\n\n"
            f"Issue: {question}\n\n"
            "Return this exact JSON structure:\n"
            "{\n"
            '  "summary": "One-line summary of the problem",\n'
            '  "severity": "low | medium | high | critical",\n'
            '  "root_cause": "Most likely root cause",\n'
            '  "diagnostic_commands": ["cmd1", "cmd2", "cmd3"],\n'
            '  "fix_steps": ["step1", "step2", "step3"],\n'
            '  "prevention": ["tip1", "tip2"],\n'
            '  "estimated_fix_time": "e.g. 15 minutes"\n'
            "}"
        )

    # =========================================================================
    # TODO (STUDENT EXERCISE): Implement a 5th strategy!
    # =========================================================================
    # Ideas for your 5th strategy:
    #
    # OPTION A -- Role-Play Strategy:
    #   Make the LLM pretend to be a specific expert persona (e.g., a Kubernetes
    #   SRE at Google, a security auditor, an AWS Solutions Architect).
    #   The persona shapes the depth and focus of the answer.
    #
    # OPTION B -- Tree-of-Thought Strategy:
    #   Ask the LLM to explore 3 different solution paths, evaluate each one,
    #   and then pick the best. This produces more creative solutions.
    #
    # OPTION C -- Socratic Strategy:
    #   Instead of answering directly, the LLM asks clarifying questions first,
    #   then provides an answer. Great for ambiguous problems.
    #
    # def your_strategy(self, question: str) -> str:
    #     """
    #     TODO: Implement your custom strategy here.
    #     Return a prompt string, just like the methods above.
    #     """
    #     pass


# =============================================================================
# COMPARISON RUNNER CLASS
# =============================================================================
# This class orchestrates the experiment: run all strategies on the same
# question, collect results, and display a dashboard.

class ComparisonRunner:
    """
    Runs all prompting strategies on the same question and compares results.

    Like a taste test at a restaurant: same ingredient, different recipes,
    side-by-side comparison so you can see which technique works best.
    """

    def __init__(self):
        self.tester = StrategyTester()
        # Map strategy names to their methods
        self.strategies = {
            "Zero-Shot": self.tester.zero_shot,
            "Few-Shot": self.tester.few_shot,
            "Chain-of-Thought": self.tester.chain_of_thought,
            "Structured JSON": self.tester.structured_json,
        }

    def run_all(self, question: str, temperature: float = 0.7) -> dict:
        """
        Run all 4 strategies on the same question and return results.

        Args:
            question:    The DevOps question to test.
            temperature: LLM temperature setting.

        Returns:
            Dict mapping strategy_name -> result_dict
        """
        results = {}
        for name, strategy_fn in self.strategies.items():
            console.print(f"\n[bold yellow]Running {name} strategy...[/bold yellow]")
            prompt = strategy_fn(question)
            result = call_llm(prompt, temperature=temperature, label=f"{name}")
            result["prompt"] = prompt
            results[name] = result
        return results

    def score_response(self, response_text: str) -> dict:
        """
        Score a response on multiple quality dimensions.

        These are heuristic checks -- not perfect, but good enough to
        demonstrate how you might evaluate LLM output programmatically.

        In production, you'd use more sophisticated evaluation:
        - LLM-as-judge (ask another LLM to rate the response)
        - Human evaluation rubrics
        - Task-specific metrics (e.g., "did the command actually work?")

        Args:
            response_text: The LLM's response string.

        Returns:
            Dict with boolean and numeric quality scores.
        """
        text_lower = response_text.lower()
        return {
            # Does the response include actual commands? (code blocks or $ prefixed)
            "has_commands": "```" in response_text or "$ " in response_text,
            # Does it include numbered steps or a structured walkthrough?
            "has_steps": any(
                marker in text_lower
                for marker in ["step 1", "step 2", "1.", "1)", "first,"]
            ),
            # Does it mention prevention / future avoidance?
            "has_prevention": any(
                word in text_lower
                for word in ["prevent", "avoid", "future", "recurrence", "going forward"]
            ),
            # Does it identify a root cause?
            "has_root_cause": any(
                phrase in text_lower
                for phrase in ["root cause", "because", "reason", "caused by", "due to"]
            ),
            # Is it a valid JSON response? (for the JSON strategy)
            "is_valid_json": self._is_valid_json(response_text),
            # Response length as a rough proxy for thoroughness
            "length": len(response_text),
        }

    def _is_valid_json(self, text: str) -> bool:
        """Check if the text is valid JSON (or contains a JSON block)."""
        # Try the raw text first
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
        # Try extracting JSON from code fences (some LLMs wrap it)
        if "```" in text:
            try:
                # Find content between first ``` and last ```
                start = text.index("```") + 3
                # Skip optional language identifier like ```json
                if text[start:start+4] == "json":
                    start += 4
                end = text.rindex("```")
                json.loads(text[start:end].strip())
                return True
            except (json.JSONDecodeError, ValueError, IndexError):
                pass
        return False

    def display_dashboard(self, question: str, results: dict):
        """
        Display a beautiful comparison dashboard using Rich tables and panels.

        This is the payoff -- seeing all 4 strategies side-by-side makes
        the differences crystal clear.
        """
        # --- Question Banner ---
        console.print(Panel(
            question,
            title="Test Question",
            border_style="white",
        ))

        # --- Individual Response Panels ---
        colors = {
            "Zero-Shot": "blue",
            "Few-Shot": "yellow",
            "Chain-of-Thought": "green",
            "Structured JSON": "magenta",
        }

        for name, result in results.items():
            color = colors.get(name, "white")
            header = f"{name} ({result['time']}s, {result['char_count']} chars)"

            # For JSON strategy, try to pretty-print the JSON
            content = result["response"]
            if name == "Structured JSON":
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except (json.JSONDecodeError, ValueError):
                    # Try extracting from code fences
                    if "```" in content:
                        try:
                            start = content.index("```") + 3
                            if content[start:start+4] == "json":
                                start += 4
                            end = content.rindex("```")
                            parsed = json.loads(content[start:end].strip())
                            content = json.dumps(parsed, indent=2)
                        except (json.JSONDecodeError, ValueError, IndexError):
                            pass

            console.print(Panel(
                Markdown(content) if name != "Structured JSON" else Syntax(content, "json", theme="monokai"),
                title=header,
                border_style=color,
            ))

        # --- Scoring Table ---
        console.print()
        score_table = Table(
            title="Quality Comparison Scorecard",
            box=box.DOUBLE_EDGE,
            show_lines=True,
        )
        score_table.add_column("Metric", style="bold", width=20)
        for name in results:
            score_table.add_column(name, justify="center", width=18)

        # Compute scores for each strategy
        all_scores = {}
        for name, result in results.items():
            all_scores[name] = self.score_response(result["response"])

        # Display each metric row
        metric_labels = {
            "has_commands": "Has Commands?",
            "has_steps": "Has Steps?",
            "has_prevention": "Has Prevention?",
            "has_root_cause": "Identifies Cause?",
            "is_valid_json": "Valid JSON?",
        }
        for key, label in metric_labels.items():
            row = [label]
            for name in results:
                val = all_scores[name][key]
                row.append("[green]Yes[/green]" if val else "[red]No[/red]")
            score_table.add_row(*row)

        # Numeric metrics
        score_table.add_row(
            "Response Length",
            *[str(all_scores[name]["length"]) for name in results],
        )
        score_table.add_row(
            "Response Time (s)",
            *[str(results[name]["time"]) for name in results],
        )

        # Total score (count of True values in boolean metrics)
        score_table.add_row(
            "[bold]Quality Score (/5)[/bold]",
            *[
                f"[bold]{sum(1 for k in metric_labels if all_scores[name][k])}/5[/bold]"
                for name in results
            ],
        )

        console.print(score_table)


# =============================================================================
# TEMPERATURE EXPERIMENT
# =============================================================================
# Temperature is one of the most important LLM parameters. It controls how
# "random" or "creative" the output is.
#   - 0.0 = always pick the most likely next word (deterministic, focused)
#   - 0.5 = moderate randomness (balanced)
#   - 1.0 = high randomness (creative, sometimes surprising)
#
# For DevOps troubleshooting, lower temperatures are usually better because
# you want CORRECT commands, not creative ones. But let's see for ourselves!

def run_temperature_experiment(question: str):
    """
    Run the same prompt at temperatures 0.0, 0.5, and 1.0 and compare.

    This helps students build intuition about what temperature actually
    does in practice, not just in theory.
    """
    console.print(Panel(
        "[bold]Temperature Experiment[/bold]\n\n"
        "Temperature controls randomness:\n"
        "  [cyan]0.0[/cyan] = Deterministic (always the same answer)\n"
        "  [cyan]0.5[/cyan] = Balanced (some variation)\n"
        "  [cyan]1.0[/cyan] = Creative (more diverse, sometimes wild)\n\n"
        "We'll send the SAME prompt at 3 temperatures and compare.",
        title="How Temperature Affects Responses",
        border_style="bright_yellow",
    ))

    # Use zero-shot for a clean comparison (no examples to confound)
    tester = StrategyTester()
    prompt = tester.zero_shot(question)

    temperatures = [0.0, 0.5, 1.0]
    temp_results = {}

    for temp in temperatures:
        console.print(f"\n[bold]Running at temperature {temp}...[/bold]")
        result = call_llm(prompt, temperature=temp, label=f"Temp {temp}")
        temp_results[temp] = result

        console.print(Panel(
            Markdown(result["response"]),
            title=f"Temperature {temp} ({result['time']}s, {result['char_count']} chars)",
            border_style="cyan",
        ))

    # Summary
    temp_table = Table(title="Temperature Comparison", box=box.ROUNDED, show_lines=True)
    temp_table.add_column("Metric", style="bold")
    for temp in temperatures:
        temp_table.add_column(f"Temp {temp}", justify="center")

    temp_table.add_row(
        "Response Length",
        *[str(temp_results[t]["char_count"]) for t in temperatures],
    )
    temp_table.add_row(
        "Time (s)",
        *[str(temp_results[t]["time"]) for t in temperatures],
    )
    temp_table.add_row(
        "Has Commands?",
        *[
            "[green]Yes[/green]" if ("```" in temp_results[t]["response"]) else "[red]No[/red]"
            for t in temperatures
        ],
    )

    console.print(temp_table)

    console.print(Panel(
        "[bold]Temperature Guidelines for DevOps:[/bold]\n\n"
        "- [cyan]Temp 0.0-0.3:[/cyan] Use for troubleshooting, commands, configs\n"
        "  (You want accuracy, not creativity)\n"
        "- [cyan]Temp 0.4-0.7:[/cyan] Use for explanations, documentation, brainstorming\n"
        "  (Some variety is fine when exploring ideas)\n"
        "- [cyan]Temp 0.8-1.0:[/cyan] Rarely used in DevOps\n"
        "  (Creative writing, maybe naming things or generating test data)",
        title="When to Use Which Temperature",
        border_style="bright_green",
    ))


# =============================================================================
# TEST QUESTIONS -- curated to highlight strategy differences
# =============================================================================
TEST_QUESTIONS = [
    {
        "id": 1,
        "question": (
            "Our production Kubernetes cluster is experiencing intermittent 503 errors. "
            "The pods are running, health checks pass, but about 10% of requests fail. "
            "This started after we deployed a new version of the API gateway yesterday."
        ),
        "expected_best": "Chain-of-Thought",
        "why": "Complex multi-factor problem needs systematic analysis.",
    },
    {
        "id": 2,
        "question": (
            "How do I set up a basic GitHub Actions workflow that runs pytest on "
            "every push to the main branch?"
        ),
        "expected_best": "Zero-Shot",
        "why": "Simple, well-known task. Extra structure adds overhead without benefit.",
    },
    {
        "id": 3,
        "question": (
            "We need to build an automated incident response system. When a PagerDuty "
            "alert fires, we want to automatically collect diagnostic data from the "
            "affected service, create a Jira ticket, and notify the on-call Slack channel. "
            "Return a structured implementation plan."
        ),
        "expected_best": "Structured JSON",
        "why": "The user explicitly wants structured/parseable output for automation.",
    },
]


# =============================================================================
# MAIN INTERACTIVE DEMO
# =============================================================================
def main():
    """
    The main entry point. Offers 3 modes:
    1. Full comparison on a test question
    2. Temperature experiment
    3. Interactive mode with a custom question
    """
    console.print(Panel(
        "[bold white]Exercise 4: Prompting Strategy Comparison Lab[/bold white]\n\n"
        "Compare 4 prompting strategies side-by-side on the same question.\n"
        "See which strategy works best for different types of problems.\n"
        "Then explore how temperature changes the output.\n\n"
        "Strategies: [blue]Zero-Shot[/blue] | [yellow]Few-Shot[/yellow] | "
        "[green]Chain-of-Thought[/green] | [magenta]Structured JSON[/magenta]\n\n"
        f"[dim]Backend: {'Ollama (local)' if os.getenv('USE_OLLAMA', 'false').lower() == 'true' else 'OpenAI (cloud)'} | Model: {MODEL}[/dim]",
        title="Module 6 - Advanced Prompting",
        border_style="bright_blue",
    ))

    console.print("\n[bold]Choose a mode:[/bold]")
    console.print("  1. [cyan]Strategy Comparison[/cyan] -- Run all 4 strategies on a test question")
    console.print("  2. [cyan]Temperature Experiment[/cyan] -- See how temperature affects output")
    console.print("  3. [cyan]Interactive Mode[/cyan] -- Enter your own question")
    console.print("  4. [cyan]Exit[/cyan]")
    console.print()

    mode = IntPrompt.ask("Select mode", default=1)

    if mode == 4:
        console.print("[yellow]Goodbye![/yellow]")
        return

    # --- Get the question ---
    if mode in (1, 2):
        console.print("\n[bold]Test Questions:[/bold]")
        for q in TEST_QUESTIONS:
            console.print(
                f"  {q['id']}. [cyan]{q['question'][:80]}...[/cyan]\n"
                f"     [dim]Expected best strategy: {q['expected_best']} -- {q['why']}[/dim]"
            )
        console.print()

        q_choice = IntPrompt.ask("Pick a question", default=1)
        selected = next((q for q in TEST_QUESTIONS if q["id"] == q_choice), TEST_QUESTIONS[0])
        question = selected["question"]
    else:
        question = Prompt.ask("\nEnter your DevOps question")

    # --- Run the selected mode ---
    if mode == 1:
        runner = ComparisonRunner()
        console.print(f"\n[bold]Running all 4 strategies at temperature 0.7...[/bold]")
        results = runner.run_all(question, temperature=0.7)
        console.print("\n")
        runner.display_dashboard(question, results)

        # Show the decision framework
        console.print(Panel(
            "[bold]Decision Framework -- When to Use Each Strategy:[/bold]\n\n"
            "[blue]Zero-Shot:[/blue]\n"
            "  Use when: Simple factual questions, quick lookups, well-known tasks\n"
            "  Example: 'What flag restarts Docker containers automatically?'\n\n"
            "[yellow]Few-Shot:[/yellow]\n"
            "  Use when: You need a specific format or style of response\n"
            "  Example: 'Write runbook entries following our team's template'\n\n"
            "[green]Chain-of-Thought:[/green]\n"
            "  Use when: Complex troubleshooting, debugging, architectural decisions\n"
            "  Example: 'Production is down, multiple services affected'\n\n"
            "[magenta]Structured JSON:[/magenta]\n"
            "  Use when: Your code needs to parse the response (automation, APIs)\n"
            "  Example: 'Classify this alert and return structured incident data'",
            title="Strategy Decision Framework",
            border_style="bright_green",
        ))

    elif mode == 2:
        run_temperature_experiment(question)

    elif mode == 3:
        runner = ComparisonRunner()
        console.print(f"\n[bold]Running all 4 strategies on your question...[/bold]")
        results = runner.run_all(question, temperature=0.7)
        console.print("\n")
        runner.display_dashboard(question, results)

        # Ask if they want to try the temperature experiment too
        console.print()
        try_temp = Prompt.ask("Want to also run the temperature experiment? (y/n)", default="n")
        if try_temp.lower() == "y":
            run_temperature_experiment(question)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

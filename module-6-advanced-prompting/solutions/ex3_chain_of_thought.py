"""
Module 6, Exercise 3 - SOLUTION: Chain-of-Thought Troubleshooting Engine
=========================================================================

Complete solution implementing a CoT-based DevOps troubleshooting engine
that walks the LLM through a structured 5-step reasoning framework.

Features:
  - TroubleshootingEngine with a full 5-step CoT framework
  - Custom troubleshooting template (student exercise -- complete)
  - Side-by-side comparison: regular vs CoT on the same problem
  - Rich output with Panel, Table, and progress spinner
  - All 5 test scenarios implemented and runnable
  - Response parsing into structured steps
"""

import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import IntPrompt

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client -- supports both OpenAI and Ollama
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
# Chain-of-Thought Framework (5 steps)
# ---------------------------------------------------------------------------
COT_FRAMEWORK = """You are a senior DevOps engineer using a systematic Chain-of-Thought
approach to troubleshoot issues. For every problem, walk through these 5 steps
explicitly. Label each step clearly.

## Step 1: OBSERVE -- Gather Information
List all symptoms, error messages, and observable behaviours described.
Note what information is missing and what you would additionally request.

## Step 2: HYPOTHESISE -- Identify Possible Root Causes
Generate at least 3 plausible hypotheses ranked by likelihood.
For each hypothesis, state what evidence supports or contradicts it.

## Step 3: DIAGNOSE -- Determine Commands & Checks
For each hypothesis provide the exact diagnostic commands to confirm or reject it.
Explain what output to look for.

## Step 4: REMEDIATE -- Provide Fix Steps
Give a step-by-step remediation plan for the most likely root cause.
Include exact commands, config changes, and expected outcomes.
Provide a rollback command or plan for each change.

## Step 5: PREVENT -- Long-term Recommendations
Suggest monitoring, alerting, and process changes to prevent recurrence.
Include specific tool recommendations and configuration snippets where possible.
"""

# Custom template (student exercise -- complete)
CUSTOM_COT_TEMPLATE = """You are a DevOps incident commander running a structured
post-incident root-cause analysis. Use the following reasoning chain:

## Phase A: TIMELINE RECONSTRUCTION
Reconstruct the incident timeline from the information given.
List events in chronological order with estimated timestamps.

## Phase B: BLAST RADIUS ASSESSMENT
Identify all affected systems, users, and data.
Classify severity: P1 (critical), P2 (major), P3 (minor).

## Phase C: FIVE WHYS ANALYSIS
Perform a "5 Whys" drill-down to find the deepest root cause.

## Phase D: CORRECTIVE ACTIONS
List immediate, short-term, and long-term corrective actions with owners.

## Phase E: DETECTION GAP ANALYSIS
Identify what monitoring or alerting failed to catch this sooner.
Propose new alerts with specific thresholds.
"""


# ---------------------------------------------------------------------------
# Troubleshooting Engine
# ---------------------------------------------------------------------------
class TroubleshootingEngine:
    """Runs LLM troubleshooting with or without Chain-of-Thought prompting."""

    def __init__(self, cot_template: str = COT_FRAMEWORK) -> None:
        self.cot_template = cot_template

    # ---- LLM calls ---------------------------------------------------------

    def _call(self, system: str, user: str) -> tuple[str, float]:
        """Call the LLM and return (response_text, elapsed_seconds)."""
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=1500,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as exc:
            text = f"[LLM Error] {exc}"
        elapsed = time.time() - start
        return text, elapsed

    def regular_troubleshoot(self, problem: str) -> tuple[str, float]:
        """Simple / regular (non-CoT) troubleshooting."""
        system = (
            "You are a senior DevOps engineer. Troubleshoot the following issue "
            "and provide a clear solution with commands."
        )
        return self._call(system, problem)

    def cot_troubleshoot(self, problem: str) -> tuple[str, float]:
        """Chain-of-Thought troubleshooting using the 5-step framework."""
        return self._call(self.cot_template, problem)

    def custom_cot_troubleshoot(self, problem: str) -> tuple[str, float]:
        """Use the custom (incident commander) CoT template."""
        return self._call(CUSTOM_COT_TEMPLATE, problem)

    # ---- Response parsing --------------------------------------------------

    @staticmethod
    def parse_steps(response: str) -> list[dict]:
        """Parse a CoT response into structured step dicts."""
        # Try to split on markdown headings or numbered step labels
        step_pattern = re.compile(
            r"(?:^|\n)##?\s*(?:Step\s*\d+[:\s]*|Phase\s*[A-E][:\s]*)?(.+?)(?=\n##?\s|$)",
            re.DOTALL,
        )
        raw_blocks = step_pattern.findall(response)
        if not raw_blocks:
            # Fallback: split by numbered steps
            raw_blocks = re.split(r"\n(?=\d+[\.\)]\s)", response)

        steps = []
        for i, block in enumerate(raw_blocks, 1):
            block = block.strip()
            if not block:
                continue
            # Extract a title from the first line
            lines = block.split("\n", 1)
            title = lines[0].strip(" #:-")
            body = lines[1].strip() if len(lines) > 1 else ""
            steps.append({"step": i, "title": title, "body": body})
        return steps


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------
SCENARIOS = [
    {
        "name": "OOMKilled Java Service",
        "problem": (
            "Our Java microservice running in Kubernetes keeps getting OOMKilled. "
            "The pod has 2Gi memory limit. Heap is set to -Xmx1536m. "
            "It runs fine for ~2 hours then gets killed. No memory leak in profiler."
        ),
    },
    {
        "name": "Intermittent 502 Errors",
        "problem": (
            "Users report intermittent 502 Bad Gateway errors on our API. "
            "We use NGINX Ingress Controller on EKS. The backend pods show healthy in "
            "readiness checks. Errors spike during deployments but also appear randomly."
        ),
    },
    {
        "name": "Terraform State Lock",
        "problem": (
            "Our Terraform pipeline fails with 'Error acquiring the state lock'. "
            "The DynamoDB lock table shows a lock from 3 hours ago. "
            "The CI job that created the lock no longer exists. "
            "Multiple teams share this state file."
        ),
    },
    {
        "name": "Kafka Consumer Lag",
        "problem": (
            "Kafka consumer lag for the order-processing group has been growing steadily "
            "for 6 hours. Producer rate is normal (~5k msgs/sec). Consumers are running "
            "but processing rate dropped to ~500 msgs/sec. Recently upgraded consumer "
            "from Java 11 to Java 17."
        ),
    },
    {
        "name": "CI Pipeline Flaky Tests",
        "problem": (
            "Our CI pipeline (GitHub Actions) has become unreliable. About 30%% of runs "
            "fail with random test failures that pass on re-run. Tests use a shared "
            "PostgreSQL container, some tests modify global state. Pipeline runs increased "
            "from 50/day to 200/day after adding a new team."
        ),
    },
]


# ---------------------------------------------------------------------------
# Interactive Demo
# ---------------------------------------------------------------------------
def run_demo() -> None:
    console.print(Panel("[bold]Module 6 Exercise 3 -- Chain-of-Thought Troubleshooting Engine[/bold]", style="blue"))

    engine = TroubleshootingEngine()

    # List scenarios
    tbl = Table(title="Test Scenarios")
    tbl.add_column("#", style="cyan", width=4)
    tbl.add_column("Name", style="bold green")
    tbl.add_column("Problem Summary")
    for i, s in enumerate(SCENARIOS, 1):
        tbl.add_row(str(i), s["name"], s["problem"][:90] + "...")
    console.print(tbl)

    choice = IntPrompt.ask("Select a scenario (1-5), or 0 for all", default=1)
    selected = SCENARIOS if choice == 0 else [SCENARIOS[choice - 1]]

    for scenario in selected:
        console.rule(f"[bold]{scenario['name']}[/bold]")
        console.print(Panel(scenario["problem"], title="Problem Description", border_style="cyan"))

        # --- Regular approach -----------------------------------------------
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
            prog.add_task("Running regular troubleshoot...", total=None)
            regular_resp, regular_time = engine.regular_troubleshoot(scenario["problem"])

        console.print(Panel(
            Markdown(regular_resp),
            title=f"Regular Response ({regular_time:.1f}s)",
            border_style="red",
        ))

        # --- CoT approach ---------------------------------------------------
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
            prog.add_task("Running CoT troubleshoot...", total=None)
            cot_resp, cot_time = engine.cot_troubleshoot(scenario["problem"])

        console.print(Panel(
            Markdown(cot_resp),
            title=f"CoT Response ({cot_time:.1f}s)",
            border_style="green",
        ))

        # --- Parse CoT into steps -------------------------------------------
        steps = engine.parse_steps(cot_resp)
        if steps:
            step_tbl = Table(title="Parsed CoT Steps")
            step_tbl.add_column("Step", style="cyan", width=5)
            step_tbl.add_column("Title", style="bold")
            step_tbl.add_column("Content Preview")
            for s in steps:
                preview = (s["body"][:120] + "...") if len(s["body"]) > 120 else s["body"]
                step_tbl.add_row(str(s["step"]), s["title"], preview)
            console.print(step_tbl)

        # --- Comparison table -----------------------------------------------
        cmp = Table(title="Side-by-Side Comparison")
        cmp.add_column("Metric", style="bold")
        cmp.add_column("Regular", justify="center")
        cmp.add_column("CoT", justify="center")
        cmp.add_row("Response time", f"{regular_time:.1f}s", f"{cot_time:.1f}s")
        cmp.add_row("Approx length (chars)", str(len(regular_resp)), str(len(cot_resp)))
        cmp.add_row("Has commands?", "Yes" if "`" in regular_resp else "No",
                     "Yes" if "`" in cot_resp else "No")
        cmp.add_row("Has numbered steps?",
                     "Yes" if re.search(r"\d\.\s", regular_resp) else "No",
                     "Yes" if re.search(r"\d\.\s", cot_resp) else "No")
        cmp.add_row("Mentions prevention?",
                     "Yes" if "prevent" in regular_resp.lower() else "No",
                     "Yes" if "prevent" in cot_resp.lower() else "No")
        console.print(cmp)

    # --- Bonus: custom template demo ----------------------------------------
    console.rule("[bold magenta]Bonus: Custom CoT Template (Incident Commander)[/bold magenta]")
    bonus_scenario = SCENARIOS[1]  # 502 errors
    console.print(Panel(bonus_scenario["problem"], title="Problem", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
        prog.add_task("Running custom CoT template...", total=None)
        custom_resp, custom_time = engine.custom_cot_troubleshoot(bonus_scenario["problem"])

    console.print(Panel(
        Markdown(custom_resp),
        title=f"Custom CoT Response ({custom_time:.1f}s)",
        border_style="magenta",
    ))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_demo()

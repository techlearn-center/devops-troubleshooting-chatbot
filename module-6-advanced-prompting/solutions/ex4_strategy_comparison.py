"""
Module 6, Exercise 4 - SOLUTION: Prompt Strategy Comparison Lab
================================================================

Complete solution implementing and comparing 5 prompting strategies on
identical DevOps troubleshooting problems, with quality scoring and a
Rich dashboard summarising the results.

Strategies:
  1. Zero-Shot      -- plain question, no guidance
  2. Few-Shot       -- include curated examples before the question
  3. Chain-of-Thought -- structured step-by-step reasoning
  4. Structured Output -- request JSON-formatted response
  5. Role-Play (student exercise -- complete) -- "pretend you are a senior SRE..."

Features:
  - All 5 strategies implemented and tested
  - ComparisonRunner with full quality scoring
  - Temperature experiment with visual comparison
  - Rich dashboard with tables comparing all strategies
  - Quality scoring: has_commands, has_steps, has_prevention, response_length
  - Summary with recommendations for when to use each strategy
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns

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
# Quality Scoring
# ---------------------------------------------------------------------------
@dataclass
class QualityScore:
    has_commands: bool = False
    has_steps: bool = False
    has_prevention: bool = False
    response_length: int = 0
    is_valid_json: bool = False
    elapsed: float = 0.0

    @property
    def total(self) -> int:
        """Simple quality score out of 10."""
        score = 0
        if self.has_commands:
            score += 3
        if self.has_steps:
            score += 3
        if self.has_prevention:
            score += 2
        if self.response_length > 500:
            score += 1
        if self.response_length > 1000:
            score += 1
        return min(score, 10)


def score_response(text: str, elapsed: float) -> QualityScore:
    """Compute quality metrics for a response string."""
    return QualityScore(
        has_commands=bool(re.search(r"`[^`]+`", text)),
        has_steps=bool(re.search(r"\d[\.\)]\s", text)),
        has_prevention=any(
            kw in text.lower()
            for kw in ["prevent", "monitor", "alert", "long-term", "avoid", "future"]
        ),
        response_length=len(text),
        is_valid_json=_is_json(text),
        elapsed=elapsed,
    )


def _is_json(text: str) -> bool:
    """Check if the text contains a valid JSON block."""
    # Try the whole thing first
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Try extracting a JSON code block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if match:
        try:
            json.loads(match.group(1))
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    return False


# ---------------------------------------------------------------------------
# Strategy Definitions
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    name: str
    response: str
    score: QualityScore


class Strategy:
    """Base class for a prompting strategy."""

    name: str = "base"
    description: str = ""

    def build_messages(self, problem: str) -> list[dict]:
        raise NotImplementedError


class ZeroShotStrategy(Strategy):
    name = "Zero-Shot"
    description = "Plain question, no additional guidance"

    def build_messages(self, problem: str) -> list[dict]:
        return [
            {"role": "system", "content": "You are a helpful DevOps assistant."},
            {"role": "user", "content": problem},
        ]


class FewShotStrategy(Strategy):
    name = "Few-Shot"
    description = "Include curated examples before the question"

    EXAMPLES = [
        {
            "q": "My Kubernetes pod keeps restarting. How do I debug?",
            "a": (
                "1. Check events: `kubectl describe pod <name>`\n"
                "2. View crash logs: `kubectl logs <name> --previous`\n"
                "3. Common causes: OOM, failing probes, missing configs.\n"
                "4. Fix: adjust limits/probes, then verify: `kubectl get pod <name> -w`.\n"
                "5. Prevent: set proper resource requests, add PodDisruptionBudgets."
            ),
        },
        {
            "q": "Docker build is very slow. What can I do?",
            "a": (
                "1. Check build context: `du -sh .` and add `.dockerignore`.\n"
                "2. Order Dockerfile layers: dependencies before source code.\n"
                "3. Use multi-stage builds to reduce final image size.\n"
                "4. Enable BuildKit: `DOCKER_BUILDKIT=1 docker build .`\n"
                "5. Prevent: set up CI caching with `--cache-from`."
            ),
        },
    ]

    def build_messages(self, problem: str) -> list[dict]:
        msgs: list[dict] = [
            {
                "role": "system",
                "content": (
                    "You are a senior DevOps engineer. Provide step-by-step "
                    "troubleshooting with exact commands and prevention tips."
                ),
            },
        ]
        for ex in self.EXAMPLES:
            msgs.append({"role": "user", "content": ex["q"]})
            msgs.append({"role": "assistant", "content": ex["a"]})
        msgs.append({"role": "user", "content": problem})
        return msgs


class ChainOfThoughtStrategy(Strategy):
    name = "Chain-of-Thought"
    description = "Structured 5-step reasoning framework"

    def build_messages(self, problem: str) -> list[dict]:
        system = (
            "You are a senior DevOps engineer. For every problem, follow these "
            "5 steps explicitly:\n\n"
            "Step 1 - OBSERVE: List all symptoms and note missing information.\n"
            "Step 2 - HYPOTHESISE: Generate 3+ ranked hypotheses with evidence.\n"
            "Step 3 - DIAGNOSE: Provide exact commands to test each hypothesis.\n"
            "Step 4 - REMEDIATE: Step-by-step fix with rollback plans.\n"
            "Step 5 - PREVENT: Monitoring, alerting, and process recommendations.\n\n"
            "Label each step clearly."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ]


class StructuredOutputStrategy(Strategy):
    name = "Structured JSON"
    description = "Request response in structured JSON format"

    def build_messages(self, problem: str) -> list[dict]:
        system = (
            "You are a DevOps troubleshooting API. ALWAYS respond with a JSON object "
            "matching this schema exactly:\n"
            "{\n"
            '  "severity": "LOW|MEDIUM|HIGH|CRITICAL",\n'
            '  "root_cause": "string",\n'
            '  "hypotheses": ["string", ...],\n'
            '  "diagnostic_commands": ["string", ...],\n'
            '  "fix_steps": [{"step": 1, "action": "string", "command": "string"}, ...],\n'
            '  "rollback": "string",\n'
            '  "prevention": ["string", ...]\n'
            "}\n\n"
            "Return ONLY valid JSON, no additional text or markdown."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ]


class RolePlayStrategy(Strategy):
    """Student exercise strategy -- now complete."""

    name = "Role-Play SRE"
    description = "Pretend you are a senior SRE who has seen this exact issue before"

    def build_messages(self, problem: str) -> list[dict]:
        system = (
            "Pretend you are a senior Site Reliability Engineer (SRE) at a large-scale "
            "tech company who has personally encountered and resolved this exact issue "
            "multiple times. Draw on your 'experience' to give highly specific, opinionated "
            "advice. Speak in first person, referencing past incidents.\n\n"
            "Structure your answer as:\n"
            "1. RECOGNITION -- 'I have seen this before when...'\n"
            "2. QUICK DIAGNOSIS -- The fastest way you learned to confirm the root cause.\n"
            "3. BATTLE-TESTED FIX -- The fix that worked every time, with exact commands.\n"
            "4. WAR STORY -- A brief anecdote about a time this went wrong.\n"
            "5. HARD-WON LESSONS -- Monitoring and prevention advice from your experience."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": problem},
        ]


# All strategies in order
ALL_STRATEGIES: list[Strategy] = [
    ZeroShotStrategy(),
    FewShotStrategy(),
    ChainOfThoughtStrategy(),
    StructuredOutputStrategy(),
    RolePlayStrategy(),
]


# ---------------------------------------------------------------------------
# Comparison Runner
# ---------------------------------------------------------------------------
class ComparisonRunner:
    """Runs all strategies against one or more problems and collects results."""

    def __init__(self, strategies: list[Strategy] | None = None, temperature: float = 0.3) -> None:
        self.strategies = strategies or ALL_STRATEGIES
        self.temperature = temperature

    def _call(self, messages: list[dict]) -> tuple[str, float]:
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1200,
            )
            text = resp.choices[0].message.content.strip()
        except Exception as exc:
            text = f"[LLM Error] {exc}"
        return text, time.time() - start

    def run(self, problem: str) -> list[StrategyResult]:
        results: list[StrategyResult] = []
        for strat in self.strategies:
            msgs = strat.build_messages(problem)
            text, elapsed = self._call(msgs)
            sc = score_response(text, elapsed)
            results.append(StrategyResult(name=strat.name, response=text, score=sc))
        return results

    def run_temperature_experiment(
        self, problem: str, temperatures: list[float] | None = None
    ) -> dict[float, list[StrategyResult]]:
        """Run a single strategy (CoT) across multiple temperatures."""
        temperatures = temperatures or [0.0, 0.3, 0.7, 1.0]
        cot = ChainOfThoughtStrategy()
        results: dict[float, list[StrategyResult]] = {}
        for temp in temperatures:
            self.temperature = temp
            msgs = cot.build_messages(problem)
            text, elapsed = self._call(msgs)
            sc = score_response(text, elapsed)
            results[temp] = [StrategyResult(name=f"CoT (t={temp})", response=text, score=sc)]
        self.temperature = 0.3  # reset
        return results


# ---------------------------------------------------------------------------
# Test Problems
# ---------------------------------------------------------------------------
PROBLEMS = [
    (
        "Kubernetes pods are OOMKilled after Java 17 upgrade. "
        "Memory limit is 2Gi, -Xmx1536m. Pods survive ~2h then die."
    ),
    (
        "Intermittent 502 errors behind NGINX Ingress on EKS. "
        "Backend pods pass readiness probes. Errors spike during rolling deploys."
    ),
    (
        "Terraform plan shows 47 resources to destroy and re-create after "
        "upgrading the AWS provider from 4.x to 5.x. No config changes were made."
    ),
]


# ---------------------------------------------------------------------------
# Dashboard Display
# ---------------------------------------------------------------------------
def display_dashboard(problem: str, results: list[StrategyResult]) -> None:
    """Print a Rich dashboard for a set of strategy results."""
    console.print(Panel(problem, title="Problem", border_style="cyan"))

    # -- Response panels side-by-side (or stacked) --------------------------
    panels = []
    for r in results:
        content = r.response[:600] + ("..." if len(r.response) > 600 else "")
        colour = "green" if r.score.total >= 7 else "yellow" if r.score.total >= 4 else "red"
        panels.append(Panel(
            Markdown(content),
            title=f"{r.name} (score: {r.score.total}/10)",
            border_style=colour,
            width=60,
        ))
    # Print two per row
    for i in range(0, len(panels), 2):
        row = panels[i : i + 2]
        console.print(Columns(row, equal=True, expand=True))

    # -- Comparison table ---------------------------------------------------
    tbl = Table(title="Quality Comparison", show_lines=True)
    tbl.add_column("Metric", style="bold")
    for r in results:
        tbl.add_column(r.name, justify="center")

    tbl.add_row("Score (/10)", *[str(r.score.total) for r in results])
    tbl.add_row("Has commands?", *["Yes" if r.score.has_commands else "No" for r in results])
    tbl.add_row("Has steps?", *["Yes" if r.score.has_steps else "No" for r in results])
    tbl.add_row("Has prevention?", *["Yes" if r.score.has_prevention else "No" for r in results])
    tbl.add_row("Length (chars)", *[str(r.score.response_length) for r in results])
    tbl.add_row("Valid JSON?", *["Yes" if r.score.is_valid_json else "No" for r in results])
    tbl.add_row("Time (s)", *[f"{r.score.elapsed:.1f}" for r in results])
    console.print(tbl)


def display_temperature_results(temp_results: dict[float, list[StrategyResult]]) -> None:
    """Print a table comparing temperature effects."""
    tbl = Table(title="Temperature Experiment (Chain-of-Thought)", show_lines=True)
    tbl.add_column("Metric", style="bold")
    for temp in sorted(temp_results.keys()):
        tbl.add_column(f"t={temp}", justify="center")

    temps_sorted = sorted(temp_results.keys())
    rows = {
        "Score (/10)": [],
        "Has commands?": [],
        "Has steps?": [],
        "Has prevention?": [],
        "Length (chars)": [],
        "Time (s)": [],
    }
    for temp in temps_sorted:
        r = temp_results[temp][0]
        rows["Score (/10)"].append(str(r.score.total))
        rows["Has commands?"].append("Yes" if r.score.has_commands else "No")
        rows["Has steps?"].append("Yes" if r.score.has_steps else "No")
        rows["Has prevention?"].append("Yes" if r.score.has_prevention else "No")
        rows["Length (chars)"].append(str(r.score.response_length))
        rows["Time (s)"].append(f"{r.score.elapsed:.1f}")

    for label, vals in rows.items():
        tbl.add_row(label, *vals)
    console.print(tbl)


def display_recommendations() -> None:
    """Print strategy selection recommendations."""
    recs = Table(title="Strategy Selection Guide", show_lines=True)
    recs.add_column("Strategy", style="bold green")
    recs.add_column("Best For")
    recs.add_column("Avoid When")

    recs.add_row(
        "Zero-Shot",
        "Simple, well-known questions; quick look-ups",
        "Complex multi-step issues; need for structured output",
    )
    recs.add_row(
        "Few-Shot",
        "Consistent formatting; domain-specific conventions; new team onboarding",
        "Token budget is very tight; examples do not match the domain",
    )
    recs.add_row(
        "Chain-of-Thought",
        "Complex incidents; root-cause analysis; post-mortems",
        "Simple one-liner answers; latency-sensitive automation",
    )
    recs.add_row(
        "Structured JSON",
        "Automation pipelines; integrating LLM output into tooling; dashboards",
        "Human-facing answers; creative problem-solving",
    )
    recs.add_row(
        "Role-Play SRE",
        "Opinionated advice; training scenarios; war-gaming exercises",
        "Need for neutral/balanced analysis; formal documentation",
    )
    console.print(recs)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------
def main() -> None:
    console.print(Panel(
        "[bold]Module 6 Exercise 4 -- Prompt Strategy Comparison Lab[/bold]",
        style="blue",
    ))

    # Show available strategies
    stbl = Table(title="Strategies Under Test")
    stbl.add_column("#", width=4, style="cyan")
    stbl.add_column("Strategy", style="bold green")
    stbl.add_column("Description")
    for i, s in enumerate(ALL_STRATEGIES, 1):
        stbl.add_row(str(i), s.name, s.description)
    console.print(stbl)

    runner = ComparisonRunner()

    # ---- Run each problem through all strategies --------------------------
    for idx, problem in enumerate(PROBLEMS, 1):
        console.rule(f"[bold]Problem {idx} of {len(PROBLEMS)}[/bold]")
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
            task = prog.add_task(f"Running 5 strategies on problem {idx}...", total=None)
            results = runner.run(problem)
            prog.update(task, completed=True)
        display_dashboard(problem, results)

    # ---- Temperature experiment -------------------------------------------
    console.rule("[bold magenta]Temperature Experiment[/bold magenta]")
    console.print("[dim]Running Chain-of-Thought at temperatures 0.0, 0.3, 0.7, 1.0[/dim]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as prog:
        prog.add_task("Running temperature experiment...", total=None)
        temp_results = runner.run_temperature_experiment(PROBLEMS[0])
    display_temperature_results(temp_results)

    # ---- Recommendations --------------------------------------------------
    console.rule("[bold]Recommendations[/bold]")
    display_recommendations()

    console.print(Panel(
        "[bold green]Lab complete![/bold green] Review the tables above to understand "
        "which strategy works best for different DevOps troubleshooting scenarios.",
        style="green",
    ))


if __name__ == "__main__":
    main()

"""
Module 6, Exercise 1 - SOLUTION: DevOps Prompt Library
======================================================

Complete solution implementing a reusable prompt template library for DevOps
troubleshooting. Includes all 5 base templates plus 2 custom templates
(COMPARISON and MIGRATION) that were assigned as student exercises.

Features:
  - 7 named prompt templates with variable substitution
  - render() method for filling templates with context-specific values
  - list_templates() for discoverability
  - Interactive CLI demo with Rich formatting
  - LLM integration to show real results from rendered prompts
"""

import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown

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
# Prompt Library
# ---------------------------------------------------------------------------
class DevOpsPromptLibrary:
    """A library of reusable, parameterised prompt templates for DevOps tasks."""

    TEMPLATES = {
        # 1. System-level persona prompt
        "SYSTEM": {
            "description": "Base system prompt that sets the DevOps expert persona",
            "template": (
                "You are a senior DevOps engineer with {years} years of experience "
                "specialising in {specialty}. You work primarily with {tools}. "
                "When answering questions:\n"
                "- Be precise and actionable\n"
                "- Include exact commands when possible\n"
                "- Warn about potential risks\n"
                "- Suggest monitoring and verification steps"
            ),
            "variables": ["years", "specialty", "tools"],
        },
        # 2. Error troubleshooting prompt
        "ERROR_TROUBLESHOOT": {
            "description": "Structured error troubleshooting prompt",
            "template": (
                "I am experiencing the following error in my {environment} environment:\n\n"
                "Error message:\n```\n{error_message}\n```\n\n"
                "Context:\n- Service: {service}\n- Last change: {last_change}\n"
                "- Impact: {impact}\n\n"
                "Please provide:\n"
                "1. Root cause analysis\n"
                "2. Step-by-step fix\n"
                "3. Commands to verify the fix\n"
                "4. Prevention measures"
            ),
            "variables": ["environment", "error_message", "service", "last_change", "impact"],
        },
        # 3. Complex multi-service issue prompt
        "COMPLEX_ISSUE": {
            "description": "Multi-service incident investigation prompt",
            "template": (
                "We have a complex incident affecting multiple services.\n\n"
                "Symptoms:\n{symptoms}\n\n"
                "Services involved: {services}\n"
                "Infrastructure: {infrastructure}\n"
                "Timeline: {timeline}\n\n"
                "Please help me:\n"
                "1. Identify the most likely failure chain\n"
                "2. Suggest diagnostic commands for each service\n"
                "3. Propose a remediation order\n"
                "4. Draft a post-incident review outline"
            ),
            "variables": ["symptoms", "services", "infrastructure", "timeline"],
        },
        # 4. Quick help / one-liner prompt
        "QUICK_HELP": {
            "description": "Quick command or one-liner help",
            "template": (
                "Give me the exact {tool} command to {action}. "
                "Target environment: {target}. "
                "Include any required flags and explain each flag briefly."
            ),
            "variables": ["tool", "action", "target"],
        },
        # 5. Structured JSON output prompt
        "STRUCTURED_JSON": {
            "description": "Request a structured JSON response for automation",
            "template": (
                "Analyse the following {resource_type} configuration and return "
                "a JSON object with these fields:\n"
                "- status: OK | WARNING | CRITICAL\n"
                "- issues: list of issue descriptions\n"
                "- recommendations: list of recommended changes\n"
                "- commands: list of remediation commands\n\n"
                "Configuration:\n```\n{config}\n```\n\n"
                "Return ONLY valid JSON, no additional text."
            ),
            "variables": ["resource_type", "config"],
        },
        # 6. Comparison prompt (STUDENT EXERCISE -- now complete)
        "COMPARISON": {
            "description": "Compare two DevOps approaches or tools side-by-side",
            "template": (
                "Compare {option_a} vs {option_b} for {use_case}.\n\n"
                "Context: {context}\n\n"
                "Please provide a detailed comparison covering:\n"
                "1. Key differences in architecture and approach\n"
                "2. Pros and cons of each option\n"
                "3. Performance and scalability considerations\n"
                "4. Operational complexity and learning curve\n"
                "5. Cost implications\n"
                "6. Your recommendation with justification"
            ),
            "variables": ["option_a", "option_b", "use_case", "context"],
        },
        # 7. Migration guide prompt (STUDENT EXERCISE -- now complete)
        "MIGRATION": {
            "description": "Step-by-step migration guide between technologies",
            "template": (
                "Create a detailed migration plan from {source} to {target}.\n\n"
                "Current setup:\n{current_setup}\n\n"
                "Requirements:\n- Zero-downtime: {zero_downtime}\n"
                "- Data volume: {data_volume}\n"
                "- Team size: {team_size}\n\n"
                "Provide:\n"
                "1. Pre-migration checklist\n"
                "2. Step-by-step migration procedure\n"
                "3. Rollback plan for each step\n"
                "4. Validation and smoke-test commands\n"
                "5. Post-migration monitoring recommendations"
            ),
            "variables": [
                "source", "target", "current_setup",
                "zero_downtime", "data_volume", "team_size",
            ],
        },
    }

    # ---- public helpers ----------------------------------------------------

    @classmethod
    def list_templates(cls) -> Table:
        """Return a Rich Table listing every available template."""
        table = Table(title="DevOps Prompt Library", show_lines=True)
        table.add_column("#", style="bold cyan", width=4)
        table.add_column("Name", style="bold green")
        table.add_column("Description")
        table.add_column("Variables", style="yellow")
        for idx, (name, meta) in enumerate(cls.TEMPLATES.items(), 1):
            table.add_row(
                str(idx),
                name,
                meta["description"],
                ", ".join(meta["variables"]),
            )
        return table

    @classmethod
    def render(cls, template_name: str, **variables: str) -> str:
        """Render a template by name, substituting the given variables."""
        if template_name not in cls.TEMPLATES:
            raise KeyError(
                f"Unknown template '{template_name}'. "
                f"Available: {', '.join(cls.TEMPLATES)}"
            )
        tmpl = cls.TEMPLATES[template_name]
        missing = [v for v in tmpl["variables"] if v not in variables]
        if missing:
            raise ValueError(
                f"Missing variables for {template_name}: {', '.join(missing)}"
            )
        return tmpl["template"].format(**variables)

    @classmethod
    def template_names(cls) -> list[str]:
        return list(cls.TEMPLATES.keys())


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Send a system + user message pair to the LLM and return the response."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM Error] {exc}"


# ---------------------------------------------------------------------------
# Interactive Demo
# ---------------------------------------------------------------------------
DEMO_VARS = {
    "SYSTEM": dict(years="10", specialty="Kubernetes and CI/CD", tools="AWS, Terraform, ArgoCD"),
    "ERROR_TROUBLESHOOT": dict(
        environment="production",
        error_message="FATAL: could not open relation mapping file: No space left on device",
        service="PostgreSQL 15",
        last_change="Deployed new analytics pipeline writing temp tables",
        impact="All writes failing, reads degraded",
    ),
    "COMPLEX_ISSUE": dict(
        symptoms="- API latency >5s\n- Kafka consumer lag growing\n- Database CPU at 98%",
        services="API Gateway, Order Service, Kafka, PostgreSQL",
        infrastructure="Kubernetes on AWS EKS (3 nodes, m5.xlarge)",
        timeline="Started 14:32 UTC after deploy of order-service v2.8.1",
    ),
    "QUICK_HELP": dict(tool="kubectl", action="find all pods in CrashLoopBackOff across all namespaces", target="production EKS cluster"),
    "STRUCTURED_JSON": dict(
        resource_type="Kubernetes Deployment",
        config=(
            "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web-app\nspec:\n"
            "  replicas: 1\n  template:\n    spec:\n      containers:\n      - name: web\n"
            "        image: web:latest\n        resources: {}"
        ),
    ),
    "COMPARISON": dict(
        option_a="Helm",
        option_b="Kustomize",
        use_case="managing Kubernetes manifests in a GitOps workflow",
        context="Team of 8, 25 microservices, ArgoCD for deployment",
    ),
    "MIGRATION": dict(
        source="self-managed Jenkins",
        target="GitHub Actions",
        current_setup="Jenkins 2.x on EC2, 15 pipelines, Groovy shared libs, 40 jobs",
        zero_downtime="yes, parallel run required",
        data_volume="N/A (CI/CD migration)",
        team_size="6 engineers",
    ),
}


def interactive_demo() -> None:
    """Run the interactive CLI demo."""
    console.print(Panel("[bold]Module 6 Exercise 1 -- DevOps Prompt Library[/bold]", style="blue"))
    console.print(DevOpsPromptLibrary.list_templates())

    names = DevOpsPromptLibrary.template_names()
    choice = IntPrompt.ask(
        "\nSelect a template number to demo (1-7), or 0 to run all",
        default=0,
    )

    templates_to_run = names if choice == 0 else [names[choice - 1]]

    # Render the SYSTEM template once for the system message
    system_prompt = DevOpsPromptLibrary.render("SYSTEM", **DEMO_VARS["SYSTEM"])

    for tname in templates_to_run:
        console.rule(f"[bold green]{tname}[/bold green]")
        rendered = DevOpsPromptLibrary.render(tname, **DEMO_VARS[tname])
        console.print(Panel(rendered, title="Rendered Prompt", border_style="cyan"))

        if tname == "SYSTEM":
            console.print("[dim]SYSTEM template is used as the system message, not sent alone.[/dim]")
            continue

        console.print("[yellow]Calling LLM...[/yellow]")
        result = call_llm(system_prompt, rendered)
        console.print(Panel(Markdown(result), title="LLM Response", border_style="green"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    interactive_demo()

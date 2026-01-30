"""
Module 6, Exercise 2 - SOLUTION: Few-Shot Prompt Builder
=========================================================

Complete solution for building few-shot prompts with a curated bank of
DevOps troubleshooting examples.  Includes the additional category examples
that were assigned as student exercises.

Features:
  - ExampleBank with 8+ pre-loaded DevOps examples across all categories
  - FewShotBuilder with smart example selection by category & relevance
  - Token counting via tiktoken to stay within configurable limits
  - Configurable n-shot control (0, 1, 2, 3 shots)
  - run_comparison() demonstrating 0-shot vs 3-shot quality difference
  - Rich panels for prompt visualisation and response display
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

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
# Try importing tiktoken; fall back to rough estimation if unavailable
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    console.print("[yellow]tiktoken not installed -- using rough token estimate (words * 1.3)[/yellow]")

    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Example:
    """A single few-shot example."""
    category: str
    question: str
    answer: str
    relevance_tags: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return count_tokens(self.question) + count_tokens(self.answer)


# ---------------------------------------------------------------------------
# Example Bank -- curated DevOps troubleshooting examples
# ---------------------------------------------------------------------------
class ExampleBank:
    """Pre-loaded bank of DevOps few-shot examples across categories."""

    def __init__(self) -> None:
        self.examples: list[Example] = self._load_defaults()

    @staticmethod
    def _load_defaults() -> list[Example]:
        return [
            # ---- Kubernetes (2 examples) -----------------------------------
            Example(
                category="kubernetes",
                question="My pod is stuck in CrashLoopBackOff. How do I debug it?",
                answer=(
                    "1. Check pod events: `kubectl describe pod <name> -n <ns>`\n"
                    "2. View logs from the last crash: `kubectl logs <name> -n <ns> --previous`\n"
                    "3. Common causes: missing config/secrets, OOM kills, failing health checks.\n"
                    "4. If OOM: increase `resources.limits.memory`.\n"
                    "5. Verify: `kubectl get pod <name> -w` until Running."
                ),
                relevance_tags=["pod", "crash", "restart", "logs"],
            ),
            Example(
                category="kubernetes",
                question="How do I troubleshoot a service that returns 503 errors?",
                answer=(
                    "1. Confirm endpoint health: `kubectl get endpoints <svc>`\n"
                    "2. Check backing pods: `kubectl get pods -l <selector>`\n"
                    "3. Test from inside the cluster: `kubectl run tmp --image=curlimages/curl -it --rm -- curl <svc>:<port>`\n"
                    "4. Review readiness probe configuration.\n"
                    "5. Inspect ingress/load-balancer logs for upstream connection errors."
                ),
                relevance_tags=["service", "503", "networking", "endpoint"],
            ),
            # ---- Docker (2 examples) ---------------------------------------
            Example(
                category="docker",
                question="Docker build fails with 'no space left on device'. What should I do?",
                answer=(
                    "1. Check disk usage: `docker system df`\n"
                    "2. Prune unused resources: `docker system prune -a --volumes`\n"
                    "3. Check build context size: `du -sh .` (add a `.dockerignore`).\n"
                    "4. Consider multi-stage builds to reduce layer size.\n"
                    "5. Monitor: `df -h /var/lib/docker`."
                ),
                relevance_tags=["build", "disk", "space", "prune"],
            ),
            Example(
                category="docker",
                question="Container runs locally but fails in production. How to debug?",
                answer=(
                    "1. Compare environment variables: `docker inspect <container>` vs production config.\n"
                    "2. Check resource limits (CPU/mem cgroups) in production.\n"
                    "3. Verify network access -- DNS resolution, outbound rules.\n"
                    "4. Test with the exact production image tag: `docker run --env-file prod.env <image>:<tag>`.\n"
                    "5. Check log driver differences that might hide output."
                ),
                relevance_tags=["production", "environment", "debug", "network"],
            ),
            # ---- CI/CD (2 examples) ----------------------------------------
            Example(
                category="cicd",
                question="GitHub Actions workflow is failing on the checkout step. How to fix?",
                answer=(
                    "1. Check the error message -- common: `Permission denied` or `ref not found`.\n"
                    "2. Verify the `GITHUB_TOKEN` permissions in workflow YAML (`permissions:` block).\n"
                    "3. For private repos ensure `actions/checkout@v4` uses a PAT or deploy key.\n"
                    "4. Pin action version to avoid breaking changes.\n"
                    "5. Test locally with `act` (https://github.com/nektos/act)."
                ),
                relevance_tags=["github", "actions", "checkout", "permissions"],
            ),
            Example(
                category="cicd",
                question="Jenkins pipeline is hanging during the build stage. How do I investigate?",
                answer=(
                    "1. Check the Jenkins console output for the last printed line.\n"
                    "2. SSH to the agent and check system resources: `top`, `df -h`.\n"
                    "3. Look for input steps that block: search for `input message:` in Jenkinsfile.\n"
                    "4. Check for zombie child processes: `ps aux | grep <build>`.\n"
                    "5. Add `timeout(time: 30, unit: 'MINUTES')` around the stage."
                ),
                relevance_tags=["jenkins", "pipeline", "hang", "timeout"],
            ),
            # ---- Networking (1 example) -- STUDENT EXERCISE ADDITION -------
            Example(
                category="networking",
                question="DNS resolution is failing inside Kubernetes pods. How to troubleshoot?",
                answer=(
                    "1. Exec into the pod: `kubectl exec -it <pod> -- nslookup kubernetes.default`\n"
                    "2. Check CoreDNS pods: `kubectl get pods -n kube-system -l k8s-app=kube-dns`\n"
                    "3. View CoreDNS logs: `kubectl logs -n kube-system -l k8s-app=kube-dns`\n"
                    "4. Verify `/etc/resolv.conf` inside the pod points to the cluster DNS IP.\n"
                    "5. Test upstream resolution: `kubectl exec <pod> -- nslookup google.com 8.8.8.8`."
                ),
                relevance_tags=["dns", "coredns", "resolve", "kubernetes", "network"],
            ),
            # ---- Monitoring (1 example) -- STUDENT EXERCISE ADDITION -------
            Example(
                category="monitoring",
                question="Prometheus is not scraping my new service metrics. What do I check?",
                answer=(
                    "1. Verify the ServiceMonitor/PodMonitor label selectors match: "
                    "`kubectl get servicemonitor -A -o yaml | grep matchLabels`\n"
                    "2. Check Prometheus targets page (`/targets`) for the scrape job.\n"
                    "3. Ensure the `/metrics` endpoint returns 200: `curl <pod-ip>:<port>/metrics`.\n"
                    "4. Verify network policies allow Prometheus to reach the pod.\n"
                    "5. Reload Prometheus config: `curl -X POST http://prometheus:9090/-/reload`."
                ),
                relevance_tags=["prometheus", "scrape", "metrics", "servicemonitor"],
            ),
        ]

    # ---- selection helpers -------------------------------------------------

    def get_by_category(self, category: str) -> list[Example]:
        """Return all examples in the given category."""
        return [e for e in self.examples if e.category == category]

    def get_by_relevance(self, tags: list[str], top_n: int = 3) -> list[Example]:
        """Return the top-N most relevant examples based on tag overlap."""
        scored = []
        for ex in self.examples:
            score = sum(1 for t in tags if t in ex.relevance_tags)
            scored.append((score, ex))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [ex for _, ex in scored[:top_n]]

    @property
    def categories(self) -> list[str]:
        return sorted(set(e.category for e in self.examples))


# ---------------------------------------------------------------------------
# Few-Shot Builder
# ---------------------------------------------------------------------------
class FewShotBuilder:
    """Construct few-shot prompts with token-aware example selection."""

    SYSTEM_MSG = (
        "You are a senior DevOps engineer. Answer troubleshooting questions "
        "with precise, step-by-step instructions and exact commands."
    )

    def __init__(self, bank: ExampleBank, max_tokens: int = 2000) -> None:
        self.bank = bank
        self.max_tokens = max_tokens

    def build_messages(
        self,
        question: str,
        n_shot: int = 3,
        category: str | None = None,
        relevance_tags: list[str] | None = None,
    ) -> list[dict]:
        """Build a chat-completion message list with up to *n_shot* examples."""
        # Select examples
        if relevance_tags:
            candidates = self.bank.get_by_relevance(relevance_tags, top_n=n_shot)
        elif category:
            candidates = self.bank.get_by_category(category)[:n_shot]
        else:
            candidates = self.bank.examples[:n_shot]

        # Trim to token budget
        messages: list[dict] = [{"role": "system", "content": self.SYSTEM_MSG}]
        budget = self.max_tokens - count_tokens(self.SYSTEM_MSG) - count_tokens(question) - 50
        selected: list[Example] = []
        for ex in candidates:
            if budget - ex.token_count < 0:
                break
            selected.append(ex)
            budget -= ex.token_count

        # Append examples as user/assistant pairs
        for ex in selected:
            messages.append({"role": "user", "content": ex.question})
            messages.append({"role": "assistant", "content": ex.answer})

        # Append the real question
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def messages_to_text(messages: list[dict]) -> str:
        """Render message list as readable text (for display)."""
        lines = []
        for m in messages:
            role = m["role"].upper()
            lines.append(f"[{role}]\n{m['content']}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------
def call_llm(messages: list[dict]) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"[LLM Error] {exc}"


# ---------------------------------------------------------------------------
# Comparison demo
# ---------------------------------------------------------------------------
def run_comparison() -> None:
    """Run 0-shot vs 3-shot on the same question and display results."""
    console.print(Panel("[bold]Module 6 Exercise 2 -- Few-Shot Builder[/bold]", style="blue"))

    bank = ExampleBank()
    builder = FewShotBuilder(bank)

    # Show the example bank summary
    tbl = Table(title="Example Bank Summary")
    tbl.add_column("Category", style="cyan")
    tbl.add_column("Count", style="green", justify="center")
    tbl.add_column("Sample Tags", style="yellow")
    for cat in bank.categories:
        exs = bank.get_by_category(cat)
        tags = ", ".join(sorted(set(t for e in exs for t in e.relevance_tags)))
        tbl.add_row(cat, str(len(exs)), tags)
    console.print(tbl)

    question = (
        "Our Kubernetes pods keep getting OOMKilled after deploying a new "
        "Java service. Memory usage spikes to 4Gi within minutes. "
        "How do we troubleshoot and fix this?"
    )

    console.rule("[bold]Test Question[/bold]")
    console.print(Panel(question, border_style="cyan"))

    # ---- 0-shot -----------------------------------------------------------
    console.rule("[bold red]0-Shot (no examples)[/bold red]")
    msgs_0 = builder.build_messages(question, n_shot=0)
    console.print(f"[dim]Prompt tokens (approx): {sum(count_tokens(m['content']) for m in msgs_0)}[/dim]")
    resp_0 = call_llm(msgs_0)
    console.print(Panel(Markdown(resp_0), title="0-Shot Response", border_style="red"))

    # ---- 3-shot -----------------------------------------------------------
    console.rule("[bold green]3-Shot (best examples)[/bold green]")
    msgs_3 = builder.build_messages(
        question,
        n_shot=3,
        relevance_tags=["pod", "crash", "kubernetes", "oom"],
    )
    console.print(f"[dim]Prompt tokens (approx): {sum(count_tokens(m['content']) for m in msgs_3)}[/dim]")

    # Show the constructed prompt
    console.print(Panel(builder.messages_to_text(msgs_3), title="Full 3-Shot Prompt", border_style="yellow"))

    resp_3 = call_llm(msgs_3)
    console.print(Panel(Markdown(resp_3), title="3-Shot Response", border_style="green"))

    # ---- Quick comparison table -------------------------------------------
    cmp = Table(title="Response Comparison")
    cmp.add_column("Metric", style="bold")
    cmp.add_column("0-Shot", justify="center")
    cmp.add_column("3-Shot", justify="center")
    cmp.add_row("Approx token count", str(count_tokens(resp_0)), str(count_tokens(resp_3)))
    cmp.add_row("Contains commands?", "Yes" if "`" in resp_0 else "No", "Yes" if "`" in resp_3 else "No")
    cmp.add_row("Numbered steps?", "Yes" if any(f"{i}." in resp_0 for i in range(1, 6)) else "No",
                "Yes" if any(f"{i}." in resp_3 for i in range(1, 6)) else "No")
    console.print(cmp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_comparison()

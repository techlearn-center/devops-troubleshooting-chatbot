"""
Exercise 2: Dynamic Few-Shot Prompt Builder
=============================================
Module 6 - Advanced Prompting | DevOps Troubleshooting Chatbot Course

WHAT YOU'LL LEARN:
    - What few-shot learning is and why it works
    - How examples teach the LLM a response pattern
    - Why 2-3 examples is usually optimal (consistency vs tokens)
    - How to manage an example bank and select relevant examples

ANALOGY:
    Imagine you're training a new hire at a help desk. You could:
      - Zero-shot:  "Just answer customer questions." (no examples -- good luck!)
      - One-shot:   "Here's one example of a great answer. Do it like this."
      - Few-shot:   "Here are 3 examples of great answers. See the pattern?"

    The more examples you show, the better the new hire understands the STYLE
    and DEPTH you expect. Few-shot prompting works the same way with LLMs.

HOW TO RUN:
    pip install openai python-dotenv rich
    python ex2_few_shot_builder.py

    For Ollama: set USE_OLLAMA=true then run.

DURATION: ~40 minutes
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
from rich.text import Text
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
# TOKEN ESTIMATION HELPER
# =============================================================================
# LLMs have a "context window" -- a maximum number of tokens they can process
# at once. A token is roughly 4 characters or ~0.75 words in English.
# If your prompt is too long (too many examples), the LLM will truncate or error.
# This helper gives a rough estimate so we can warn before exceeding limits.

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token for English text.

    Real tokenizers (like tiktoken) are more accurate, but this is good enough
    for our guardrail checks. Think of it like estimating driving time --
    you don't need GPS precision to know "that's too far."
    """
    return len(text) // 4


# =============================================================================
# EXAMPLE BANK CLASS
# =============================================================================
# An ExampleBank is like a filing cabinet: each drawer (category) holds
# example Q&A cards. When building a few-shot prompt, we pull cards from
# the right drawer to show the LLM the kind of answer we expect.

class ExampleBank:
    """
    Stores question/answer example pairs organized by category.

    Each example is a dict with:
        - "question": The user's question (what we show as the "input")
        - "answer":   The ideal response (what we show as the "output")
        - "category": Which DevOps domain this belongs to
    """

    def __init__(self):
        """Initialize the bank with pre-loaded DevOps examples."""
        self.examples = []
        self._load_default_examples()

    def _load_default_examples(self):
        """
        Pre-load the bank with curated DevOps Q&A examples.
        These are real-world questions and high-quality answers that
        demonstrate the response pattern we want the LLM to follow.
        """

        # --- TERRAFORM EXAMPLES ---
        self.add_example(
            category="terraform",
            question="My Terraform plan shows 'resource already exists' for an S3 bucket. What do I do?",
            answer=(
                "**Problem:** Terraform doesn't know about a resource that already exists in AWS.\n\n"
                "**Root Cause:** The S3 bucket was created outside of Terraform (manually or by another tool), "
                "so it's not in your Terraform state file.\n\n"
                "**Fix:**\n"
                "```bash\n"
                "# Import the existing resource into Terraform state\n"
                "terraform import aws_s3_bucket.my_bucket my-bucket-name\n"
                "# Then run plan again to verify\n"
                "terraform plan\n"
                "```\n\n"
                "**Prevention:** Always create cloud resources through Terraform, not the console. "
                "Use `terraform import` for any pre-existing resources."
            ),
        )
        self.add_example(
            category="terraform",
            question="Terraform is showing a 'cycle' error between two resources. How do I fix it?",
            answer=(
                "**Problem:** Two resources depend on each other, creating a circular dependency.\n\n"
                "**Root Cause:** Resource A references Resource B, and Resource B references Resource A. "
                "Terraform can't determine which to create first.\n\n"
                "**Fix:**\n"
                "1. Identify the cycle from the error message -- it lists the resources involved.\n"
                "2. Break the cycle by using `depends_on` explicitly or restructuring references.\n"
                "```hcl\n"
                "# Instead of direct reference, use a separate data source or output\n"
                "resource \"aws_security_group\" \"app\" {\n"
                "  name = \"app-sg\"\n"
                "  # Remove the direct reference that causes the cycle\n"
                "}\n"
                "```\n"
                "3. Consider splitting into separate `terraform apply` runs if needed.\n\n"
                "**Prevention:** Design your resource dependencies as a DAG (directed acyclic graph). "
                "Draw out the dependency arrows before writing the code."
            ),
        )

        # --- KUBERNETES EXAMPLES ---
        self.add_example(
            category="kubernetes",
            question="My pod is in CrashLoopBackOff. How do I debug it?",
            answer=(
                "**Problem:** The container starts, crashes, Kubernetes restarts it, it crashes again -- a loop.\n\n"
                "**Root Cause:** The application inside the container is exiting with an error. Common causes: "
                "missing environment variables, wrong command, failed health checks, or OOM (out of memory).\n\n"
                "**Fix:**\n"
                "```bash\n"
                "# Step 1: Check the pod's recent logs\n"
                "kubectl logs <pod-name> --previous\n"
                "# Step 2: Describe the pod for events and exit codes\n"
                "kubectl describe pod <pod-name>\n"
                "# Step 3: Check exit code -- 137 means OOM, 1 means app error\n"
                "# Step 4: If OOM, increase memory limits in your deployment YAML\n"
                "```\n\n"
                "**Prevention:** Always set resource requests/limits. Add proper health check probes. "
                "Test your container locally with `docker run` before deploying to K8s."
            ),
        )

        # --- DOCKER EXAMPLES ---
        self.add_example(
            category="docker",
            question="Docker build is slow and the image is 2GB. How do I optimize it?",
            answer=(
                "**Problem:** Large image size increases build time, pull time, and storage costs.\n\n"
                "**Root Cause:** Common causes include using a large base image, not leveraging "
                "layer caching, copying unnecessary files, and not using multi-stage builds.\n\n"
                "**Fix:**\n"
                "```dockerfile\n"
                "# Use multi-stage build (game changer!)\n"
                "FROM node:18 AS builder\n"
                "WORKDIR /app\n"
                "COPY package*.json ./\n"
                "RUN npm ci\n"
                "COPY . .\n"
                "RUN npm run build\n\n"
                "# Final stage -- tiny image with only what we need\n"
                "FROM node:18-alpine\n"
                "COPY --from=builder /app/dist ./dist\n"
                "CMD [\"node\", \"dist/main.js\"]\n"
                "```\n"
                "Also add a `.dockerignore` for `node_modules`, `.git`, etc.\n\n"
                "**Prevention:** Start every Dockerfile with multi-stage builds. "
                "Use Alpine or distroless base images. Run `docker image ls` regularly."
            ),
        )

        # --- CI/CD EXAMPLES ---
        self.add_example(
            category="cicd",
            question="My GitHub Actions workflow keeps failing on the 'npm test' step. How do I fix it?",
            answer=(
                "**Problem:** Tests pass locally but fail in CI.\n\n"
                "**Root Cause:** Environment differences between your machine and the CI runner. "
                "Common issues: different Node version, missing environment variables, "
                "database not available, or timezone differences.\n\n"
                "**Fix:**\n"
                "```yaml\n"
                "# .github/workflows/test.yml\n"
                "jobs:\n"
                "  test:\n"
                "    runs-on: ubuntu-latest\n"
                "    steps:\n"
                "      - uses: actions/checkout@v4\n"
                "      - uses: actions/setup-node@v4\n"
                "        with:\n"
                "          node-version-file: '.nvmrc'  # Match local version!\n"
                "      - run: npm ci                     # Not 'npm install'\n"
                "      - run: npm test\n"
                "        env:\n"
                "          CI: true\n"
                "          DATABASE_URL: ${{ secrets.TEST_DB_URL }}\n"
                "```\n\n"
                "**Prevention:** Use `.nvmrc` to pin the Node version. Use `npm ci` (not `npm install`) "
                "for reproducible installs. Add required env vars to the workflow."
            ),
        )
        self.add_example(
            category="cicd",
            question="How do I cache dependencies in GitHub Actions to speed up builds?",
            answer=(
                "**Problem:** Every CI run downloads all dependencies from scratch, wasting time.\n\n"
                "**Root Cause:** By default, each GitHub Actions run starts with a clean environment. "
                "Without caching, `npm install` or `pip install` downloads everything every time.\n\n"
                "**Fix:**\n"
                "```yaml\n"
                "- uses: actions/setup-node@v4\n"
                "  with:\n"
                "    node-version: 18\n"
                "    cache: 'npm'  # Built-in caching! Uses package-lock.json as key\n"
                "# For pip:\n"
                "- uses: actions/cache@v4\n"
                "  with:\n"
                "    path: ~/.cache/pip\n"
                "    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}\n"
                "```\n\n"
                "**Prevention:** Always enable dependency caching from day one. "
                "Use lock files (package-lock.json, poetry.lock) as cache keys so the "
                "cache is invalidated only when dependencies actually change."
            ),
        )

        # =================================================================
        # TODO (STUDENT EXERCISE): Add examples for a NEW category!
        # =================================================================
        # Ideas: "monitoring", "networking", "security", "linux", "git"
        #
        # Add at least 2 examples for your new category using:
        #   self.add_example(
        #       category="your_category",
        #       question="...",
        #       answer="..."
        #   )
        #
        # Make sure the answers follow the same pattern:
        #   **Problem:** ... **Root Cause:** ... **Fix:** (with commands) **Prevention:** ...

        # TODO: Add your first example for a new category here

        # TODO: Add your second example for the new category here

    def add_example(self, category: str, question: str, answer: str):
        """Add a single example to the bank."""
        self.examples.append({
            "category": category.lower(),
            "question": question,
            "answer": answer,
        })

    def get_categories(self) -> list:
        """Return a sorted list of all unique categories in the bank."""
        return sorted(set(ex["category"] for ex in self.examples))

    def get_by_category(self, category: str) -> list:
        """Return all examples in a specific category."""
        return [ex for ex in self.examples if ex["category"] == category.lower()]

    def get_all(self) -> list:
        """Return all examples regardless of category."""
        return self.examples


# =============================================================================
# FEW-SHOT BUILDER CLASS
# =============================================================================
# This is the engine that constructs few-shot prompts from the example bank.
# It picks the right examples, formats them, and wraps everything into a
# prompt that teaches the LLM by demonstration.

class FewShotBuilder:
    """
    Dynamically constructs few-shot prompts from an ExampleBank.

    The builder:
    1. Selects relevant examples from the bank (by category)
    2. Formats them into a clear input/output pattern
    3. Appends the actual user question at the end
    4. Checks token count to avoid exceeding context limits
    """

    def __init__(self, example_bank: ExampleBank, max_tokens: int = 3000):
        """
        Args:
            example_bank: The ExampleBank to pull examples from.
            max_tokens:   Rough limit for the prompt portion (not the response).
        """
        self.bank = example_bank
        self.max_tokens = max_tokens

    def build_prompt(self, question: str, category: str = None, num_shots: int = 2) -> str:
        """
        Build a few-shot prompt with the specified number of examples.

        Args:
            question:   The actual user question to answer.
            category:   Filter examples by category (None = use all).
            num_shots:  How many examples to include (0, 1, 2, or 3).

        Returns:
            The complete prompt string, ready to send to the LLM.
        """
        # --- Step 1: Select examples ---
        if category:
            pool = self.bank.get_by_category(category)
        else:
            pool = self.bank.get_all()

        # Take the requested number of examples (up to what's available)
        selected = pool[:num_shots]

        # --- Step 2: Build the prompt ---
        parts = []

        # Preamble: tell the LLM what format to follow
        if num_shots > 0:
            parts.append(
                "You are a DevOps troubleshooting expert. Below are examples of "
                "how to answer DevOps questions. Follow the same format and depth.\n"
            )

            # Add each example as an Input/Output pair
            for i, ex in enumerate(selected, 1):
                parts.append(f"--- Example {i} ---")
                parts.append(f"Question: {ex['question']}")
                parts.append(f"Answer: {ex['answer']}")
                parts.append("")  # blank line separator
        else:
            # Zero-shot: no examples, just ask directly
            parts.append("You are a DevOps troubleshooting expert.\n")

        # The actual question (the one we want answered)
        parts.append("--- Your Turn ---")
        parts.append(f"Question: {question}")
        parts.append("Answer:")

        prompt = "\n".join(parts)

        # --- Step 3: Token check ---
        token_est = estimate_tokens(prompt)
        if token_est > self.max_tokens:
            console.print(
                f"[yellow]Warning: Prompt is ~{token_est} tokens "
                f"(limit: {self.max_tokens}). Consider fewer examples.[/yellow]"
            )

        return prompt

    def get_shot_count_info(self, category: str = None) -> str:
        """Return how many examples are available for a category."""
        if category:
            count = len(self.bank.get_by_category(category))
        else:
            count = len(self.bank.get_all())
        return f"{count} examples available"


# =============================================================================
# LLM HELPER
# =============================================================================
def ask_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return the response."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[LLM Error] {e}"


# =============================================================================
# INTERACTIVE DEMO
# =============================================================================
def run_comparison_demo():
    """
    The main demo: compare 0-shot, 1-shot, and 3-shot responses side-by-side.
    This is the "aha moment" where students SEE the quality difference.
    """
    console.print(Panel(
        "[bold white]Exercise 2: Dynamic Few-Shot Prompt Builder[/bold white]\n\n"
        "Few-shot prompting = teaching by example.\n"
        "We'll compare 0-shot (no examples) vs 1-shot vs 3-shot on the same question.\n"
        "Watch how the quality and consistency improve with more examples!\n\n"
        f"[dim]Backend: {'Ollama (local)' if os.getenv('USE_OLLAMA', 'false').lower() == 'true' else 'OpenAI (cloud)'} | Model: {MODEL}[/dim]",
        title="Module 6 - Advanced Prompting",
        border_style="bright_blue",
    ))

    # Set up the bank and builder
    bank = ExampleBank()
    builder = FewShotBuilder(bank)

    # Show available categories
    categories = bank.get_categories()
    console.print("\n[bold]Available categories:[/bold]")
    for i, cat in enumerate(categories, 1):
        count = len(bank.get_by_category(cat))
        console.print(f"  {i}. [cyan]{cat}[/cyan] ({count} examples)")
    console.print(f"  {len(categories) + 1}. [cyan]all[/cyan] (use all categories)")
    console.print()

    # User picks a category
    cat_choice = IntPrompt.ask("Pick a category", default=len(categories) + 1)
    if cat_choice <= len(categories):
        selected_category = categories[cat_choice - 1]
    else:
        selected_category = None
    cat_label = selected_category or "all"

    # User enters a question
    console.print()
    default_q = "My Kubernetes deployment keeps restarting and I see OOMKilled in the events. What should I do?"
    question = Prompt.ask(
        "Enter a DevOps question (or press Enter for the default)",
        default=default_q,
    )

    # --- Run all 3 variations ---
    shot_configs = [0, 1, 3]
    results = []

    for n_shots in shot_configs:
        label = f"{n_shots}-shot"
        console.print(f"\n[bold yellow]Building {label} prompt (category: {cat_label})...[/bold yellow]")

        prompt = builder.build_prompt(question, category=selected_category, num_shots=n_shots)
        token_est = estimate_tokens(prompt)

        # Show the prompt that was built
        console.print(Panel(
            prompt,
            title=f"{label} Prompt (~{token_est} tokens)",
            border_style="yellow",
        ))

        # Call the LLM
        console.print(f"[dim]Sending {label} prompt to LLM...[/dim]")
        start = time.time()
        response = ask_llm(prompt)
        elapsed = time.time() - start

        results.append({
            "shots": n_shots,
            "label": label,
            "prompt_tokens": token_est,
            "response": response,
            "time": elapsed,
            "response_len": len(response),
        })

        console.print(Panel(
            response,
            title=f"{label} Response ({elapsed:.1f}s, {len(response)} chars)",
            border_style="green",
        ))

    # --- Summary Table ---
    console.print("\n")
    summary = Table(
        title="Few-Shot Comparison Summary",
        box=box.DOUBLE_EDGE,
        show_lines=True,
    )
    summary.add_column("Strategy", style="bold cyan")
    summary.add_column("Prompt Tokens (est.)", justify="right")
    summary.add_column("Response Length", justify="right")
    summary.add_column("Time (s)", justify="right")
    summary.add_column("Has Commands?", justify="center")
    summary.add_column("Has Structured Sections?", justify="center")

    for r in results:
        has_commands = "```" in r["response"] or "$ " in r["response"]
        has_sections = "**" in r["response"] or "##" in r["response"] or "Step" in r["response"]
        summary.add_row(
            r["label"],
            str(r["prompt_tokens"]),
            str(r["response_len"]),
            f"{r['time']:.1f}",
            "[green]Yes[/green]" if has_commands else "[red]No[/red]",
            "[green]Yes[/green]" if has_sections else "[red]No[/red]",
        )

    console.print(summary)

    console.print(Panel(
        "[bold]Key Takeaways:[/bold]\n\n"
        "1. [cyan]0-shot[/cyan] gives a generic answer -- fine for simple questions.\n"
        "2. [cyan]1-shot[/cyan] teaches the format -- the LLM mirrors the example structure.\n"
        "3. [cyan]3-shot[/cyan] locks in the pattern -- most consistent quality.\n\n"
        "More examples = more tokens = higher cost and latency.\n"
        "For most DevOps tasks, 2-3 examples hits the sweet spot.",
        title="What Did You Learn?",
        border_style="bright_green",
    ))


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_comparison_demo()

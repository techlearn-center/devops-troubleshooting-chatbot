#!/usr/bin/env python3
"""
Exercise 1: Build the Error Classifier
========================================
Module 7 (Capstone) - Exercise 1 of 4

WHAT YOU'LL BUILD:
    A component that automatically detects what category a DevOps question
    belongs to: terraform, kubernetes, docker, cicd, or general.

WHY THIS MATTERS:
    Think of classification like sorting mail into the right mailbox.
    If someone asks "my pod is crashing", we know that's a Kubernetes question.
    By classifying first, we can:
    1. Search ONLY the Kubernetes docs (faster + more relevant results)
    2. Use a Kubernetes-specific system prompt (better answers)
    3. Show the user which category their question falls into (transparency)

WHAT YOU'LL LEARN:
    - Keyword-based classification (simple but effective)
    - Confidence scoring (how sure are we about the category?)
    - How classification improves RAG retrieval quality
    - How to explain WHY a classification was chosen (transparency)

PREREQUISITES:
    pip install rich openai python-dotenv

SUPPORTS:
    - OpenAI (set OPENAI_API_KEY in .env)
    - Ollama (set USE_OLLAMA=true in .env)

RUN THIS FILE:
    python ex1_error_classifier.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
from typing import Dict, List, Optional, Tuple

# python-dotenv loads variables from a .env file into the environment.
# This is how we keep API keys out of our code (never hardcode secrets!).
from dotenv import load_dotenv

# Rich is a Python library for beautiful terminal output.
# We use it to make our exercises look professional and easy to read.
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt

# Load environment variables from .env file
# This looks for a file called ".env" in the current directory or parents
load_dotenv()

# Create a Rich console for pretty output
console = Console()


# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================
# Each DevOps category has:
#   - A description (what it covers)
#   - Keywords with weights (how strongly each keyword signals this category)
#
# WEIGHTS EXPLAINED:
#   - Weight 3 (strong): Very specific to this category (e.g., "kubectl" = kubernetes)
#   - Weight 2 (medium): Commonly associated (e.g., "container" = docker)
#   - Weight 1 (weak):   Could appear in multiple categories (e.g., "error")

CATEGORIES = {
    "terraform": {
        "description": (
            "Infrastructure as Code (IaC) using HashiCorp Terraform. "
            "Covers: state management, provider configuration, resource creation, "
            "modules, workspaces, plan/apply errors, and HCL syntax."
        ),
        "keywords": {
            # Strong signals (weight 3) - unique to Terraform
            "terraform": 3,
            "tf": 3,
            "hcl": 3,
            "tfstate": 3,
            "tfvars": 3,
            "tofu": 3,            # OpenTofu, a Terraform fork
            "terraform init": 3,
            "terraform plan": 3,
            "terraform apply": 3,
            "terraform destroy": 3,
            "terraform import": 3,
            # Medium signals (weight 2)
            "provider": 2,
            "resource": 2,
            "state file": 2,
            "state lock": 2,
            "module": 2,
            "workspace": 2,
            "backend": 2,
            "data source": 2,
            "variable": 2,
            "output": 2,
            "plan": 2,
            # Weak signals (weight 1) - could appear in other contexts
            "infrastructure": 1,
            "aws": 1,
            "azure": 1,
            "gcp": 1,
            "cloud": 1,
            "provisioning": 1,
        },
    },
    "kubernetes": {
        "description": (
            "Container orchestration with Kubernetes (K8s). "
            "Covers: pod errors, deployments, services, networking, "
            "ConfigMaps, Secrets, RBAC, Helm, and cluster management."
        ),
        "keywords": {
            # Strong signals (weight 3) - unique to Kubernetes
            "kubernetes": 3,
            "k8s": 3,
            "kubectl": 3,
            "kubelet": 3,
            "kube-proxy": 3,
            "kubeconfig": 3,
            "minikube": 3,
            "kind cluster": 3,
            "helm": 3,
            # Medium signals (weight 2)
            "pod": 2,
            "deployment": 2,
            "service": 2,
            "ingress": 2,
            "configmap": 2,
            "secret": 2,
            "namespace": 2,
            "node": 2,
            "daemonset": 2,
            "statefulset": 2,
            "replicaset": 2,
            "crashloopbackoff": 2,
            "imagepullbackoff": 2,
            "pending pod": 2,
            "cluster": 2,
            "rbac": 2,
            # Weak signals (weight 1)
            "container": 1,
            "scale": 1,
            "replica": 1,
            "yaml": 1,
            "manifest": 1,
            "orchestration": 1,
        },
    },
    "docker": {
        "description": (
            "Containerization with Docker. "
            "Covers: Dockerfile issues, build errors, container runtime problems, "
            "networking, volumes, Docker Compose, and image management."
        ),
        "keywords": {
            # Strong signals (weight 3) - unique to Docker
            "docker": 3,
            "dockerfile": 3,
            "docker-compose": 3,
            "docker compose": 3,
            "docker build": 3,
            "docker run": 3,
            "docker push": 3,
            "docker pull": 3,
            "dockerhub": 3,
            # Medium signals (weight 2)
            "container": 2,
            "image": 2,
            "volume": 2,
            "layer": 2,
            "registry": 2,
            "multi-stage": 2,
            "build context": 2,
            "entrypoint": 2,
            "cmd": 2,
            "expose": 2,
            "port mapping": 2,
            # Weak signals (weight 1)
            "build": 1,
            "cache": 1,
            "network": 1,
            "bridge": 1,
            "daemon": 1,
        },
    },
    "cicd": {
        "description": (
            "Continuous Integration and Continuous Deployment pipelines. "
            "Covers: GitHub Actions, Jenkins, GitLab CI, pipeline failures, "
            "deployment automation, testing, and artifact management."
        ),
        "keywords": {
            # Strong signals (weight 3) - unique to CI/CD
            "github actions": 3,
            "github action": 3,
            "workflow": 3,
            "pipeline": 3,
            "jenkins": 3,
            "gitlab ci": 3,
            "circleci": 3,
            "travis": 3,
            "ci/cd": 3,
            "cicd": 3,
            ".github/workflows": 3,
            # Medium signals (weight 2)
            "build step": 2,
            "deploy": 2,
            "artifact": 2,
            "runner": 2,
            "stage": 2,
            "job": 2,
            "trigger": 2,
            "action": 2,
            "continuous integration": 2,
            "continuous deployment": 2,
            "pull request check": 2,
            # Weak signals (weight 1)
            "automated": 1,
            "test": 1,
            "release": 1,
            "branch": 1,
            "merge": 1,
        },
    },
    "general": {
        "description": (
            "General DevOps questions that don't fit a specific category. "
            "Covers: best practices, architecture, tooling decisions, "
            "learning paths, and cross-cutting concerns."
        ),
        "keywords": {
            # General DevOps terms (all weight 1 - they don't strongly signal anything)
            "devops": 1,
            "best practice": 1,
            "recommend": 1,
            "compare": 1,
            "difference between": 1,
            "how to learn": 1,
            "getting started": 1,
            "architecture": 1,
            "monitoring": 1,
            "logging": 1,
            "security": 1,
        },
    },
}


# ============================================================================
# ERROR CLASSIFIER CLASS
# ============================================================================
class ErrorClassifier:
    """
    Classifies DevOps questions into categories using keyword matching.

    HOW IT WORKS (the mail sorting analogy):
        Imagine you work in a mail room. Each piece of mail (question) needs
        to go to the right department (category). You look for clues:
        - If it mentions "kubectl" -> definitely Kubernetes department
        - If it mentions "container" -> probably Docker, but could be Kubernetes
        - If it mentions "pipeline" -> CI/CD department

        We score each category based on how many of its keywords appear in the
        question, weighted by how specific each keyword is.

    CONFIDENCE SCORING:
        confidence = best_score / total_possible_score_for_that_category
        - 0.0 = no keywords matched at all
        - 0.5 = some keywords matched (moderate confidence)
        - 1.0 = many keywords matched (high confidence)

        If confidence is below a threshold, we fall back to "general".
    """

    def __init__(self, confidence_threshold: float = 0.05):
        """
        Initialize the ErrorClassifier.

        Args:
            confidence_threshold: Minimum confidence to assign a specific category.
                                  Below this, we return "general" instead.
                                  Default 0.05 (5%) - very permissive because even
                                  one strong keyword match should count.
        """
        # Store the threshold - if no category scores above this, return "general"
        self.confidence_threshold = confidence_threshold

        # Store our category definitions (defined above)
        self.categories = CATEGORIES

        # Pre-compute the maximum possible score for each category.
        # This is used to normalize confidence scores to 0.0-1.0 range.
        self.max_scores = {}
        for category, info in self.categories.items():
            # Sum all keyword weights = maximum possible score
            self.max_scores[category] = sum(info["keywords"].values())

    def classify(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a query into a DevOps category.

        This is the main method - give it a question, get back a category.

        Args:
            query: The user's question (e.g., "Why is my pod crashing?")

        Returns:
            A tuple of:
            - category (str): The best matching category ("kubernetes", "docker", etc.)
            - confidence (float): How confident we are (0.0 to 1.0)
            - all_scores (dict): Scores for ALL categories (for transparency)

        Example:
            >>> classifier = ErrorClassifier()
            >>> category, confidence, scores = classifier.classify("kubectl get pods fails")
            >>> print(category)    # "kubernetes"
            >>> print(confidence)  # 0.75
        """
        # Convert query to lowercase for case-insensitive matching
        # "My Pod Is CRASHING" -> "my pod is crashing"
        query_lower = query.lower()

        # Score each category by checking which keywords appear in the query
        scores = {}
        for category, info in self.categories.items():
            score = 0.0
            for keyword, weight in info["keywords"].items():
                # Check if this keyword appears in the query
                if keyword.lower() in query_lower:
                    score += weight
            scores[category] = score

        # Normalize scores to 0.0-1.0 range (confidence)
        # This makes scores comparable across categories with different numbers of keywords
        normalized_scores = {}
        for category, score in scores.items():
            max_score = self.max_scores[category]
            if max_score > 0:
                normalized_scores[category] = score / max_score
            else:
                normalized_scores[category] = 0.0

        # Find the category with the highest score
        best_category = max(scores, key=scores.get)
        best_score = scores[best_category]

        # Calculate confidence for the best category
        best_confidence = normalized_scores[best_category]

        # If confidence is too low, fall back to "general"
        # This handles questions like "hello" or "how are you"
        if best_confidence < self.confidence_threshold or best_score == 0:
            return "general", 0.0, normalized_scores

        return best_category, best_confidence, normalized_scores

    def get_category_info(self, category: str) -> str:
        """
        Get a description of what a category covers.

        This is useful for showing the user what kind of help they'll get.

        Args:
            category: Category name (e.g., "terraform")

        Returns:
            Description string, or "Unknown category" if not found.
        """
        if category in self.categories:
            return self.categories[category]["description"]
        return "Unknown category."

    def explain_classification(self, query: str) -> Dict:
        """
        Explain WHY a query was classified a certain way.

        This is important for transparency - users should understand
        why the bot thinks their question is about Kubernetes vs Docker.

        Args:
            query: The user's question

        Returns:
            A dict containing:
            - "category": the chosen category
            - "confidence": confidence score
            - "matched_keywords": dict of {category: [list of matched keywords]}
            - "explanation": human-readable explanation string
        """
        query_lower = query.lower()

        # Find all keyword matches for each category
        matched_keywords = {}
        for category, info in self.categories.items():
            matches = []
            for keyword, weight in info["keywords"].items():
                if keyword.lower() in query_lower:
                    matches.append((keyword, weight))
            if matches:
                matched_keywords[category] = matches

        # Get the classification result
        category, confidence, all_scores = self.classify(query)

        # Build a human-readable explanation
        explanation_parts = []
        explanation_parts.append(f"Query: \"{query}\"")
        explanation_parts.append(f"Classified as: {category.upper()} (confidence: {confidence:.1%})")
        explanation_parts.append("")

        if matched_keywords:
            explanation_parts.append("Keyword matches found:")
            for cat, matches in matched_keywords.items():
                match_strs = [f"  '{kw}' (weight {w})" for kw, w in matches]
                explanation_parts.append(f"  [{cat}]:")
                explanation_parts.extend([f"    {ms}" for ms in match_strs])
        else:
            explanation_parts.append("No keyword matches found - defaulting to 'general'.")

        return {
            "category": category,
            "confidence": confidence,
            "matched_keywords": matched_keywords,
            "all_scores": all_scores,
            "explanation": "\n".join(explanation_parts),
        }

    # ========================================================================
    # TODO: Add LLM-based classification as a fallback
    # ========================================================================
    def classify_with_llm(self, query: str) -> Tuple[str, float]:
        """
        TODO: Implement LLM-based classification for ambiguous queries.

        When keyword-based classification has low confidence (< 0.1), use
        an LLM to classify the query instead. This handles cases like:
        - "my deployment keeps failing" (could be kubernetes OR cicd)
        - "I'm getting a permission denied error" (could be anything)
        - "how do I set up auto-scaling?" (kubernetes? cloud? terraform?)

        INSTRUCTIONS:
        1. Check the USE_OLLAMA environment variable to pick the right backend
        2. Send the query to the LLM with a prompt asking it to classify into
           one of our categories: terraform, kubernetes, docker, cicd, general
        3. Parse the LLM's response to extract the category
        4. Return (category, confidence) where confidence is 0.7 for LLM results
           (we trust the LLM moderately but not 100%)

        HINT - Here's a prompt template you could use:
            "Classify this DevOps question into exactly ONE category.
             Categories: terraform, kubernetes, docker, cicd, general
             Question: {query}
             Reply with ONLY the category name, nothing else."

        Args:
            query: The user's question

        Returns:
            Tuple of (category, confidence)
        """
        # TODO: Implement this method!
        # Step 1: Check if USE_OLLAMA is set
        # use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"

        # Step 2: Build the classification prompt
        # prompt = f"Classify this DevOps question into exactly ONE category.\n"
        #          f"Categories: terraform, kubernetes, docker, cicd, general\n"
        #          f"Question: {query}\n"
        #          f"Reply with ONLY the category name, nothing else."

        # Step 3: Call the LLM (OpenAI or Ollama)
        # if use_ollama:
        #     ... call Ollama API ...
        # else:
        #     ... call OpenAI API ...

        # Step 4: Parse the response and return
        # return (parsed_category, 0.7)

        # For now, fall back to keyword-based classification
        category, confidence, _ = self.classify(query)
        return category, confidence


# ============================================================================
# TODO: Add a new category
# ============================================================================
# TODO: Add a "monitoring" or "networking" category to the CATEGORIES dict above.
#
# INSTRUCTIONS:
# 1. Pick either "monitoring" or "networking" (or invent your own!)
# 2. Add it to the CATEGORIES dictionary at the top of this file
# 3. Include:
#    - A "description" string explaining what it covers
#    - A "keywords" dict with at least 10 keywords and appropriate weights
#
# EXAMPLE for "monitoring":
#    "monitoring": {
#        "description": "Monitoring, alerting, and observability. Covers: ...",
#        "keywords": {
#            "prometheus": 3,
#            "grafana": 3,
#            "alertmanager": 3,
#            "datadog": 3,
#            ... etc ...
#        },
#    },
#
# After adding, test it by running this file and trying monitoring-related queries.


# ============================================================================
# DISPLAY HELPERS (using Rich for pretty output)
# ============================================================================

# Color mapping for each category (used in Rich output)
CATEGORY_COLORS = {
    "terraform": "purple",
    "kubernetes": "blue",
    "docker": "cyan",
    "cicd": "green",
    "general": "white",
}


def display_classification_result(
    query: str,
    classifier: ErrorClassifier
) -> None:
    """
    Display a beautiful classification result using Rich.

    Shows:
    - The query
    - The chosen category with a colored badge
    - Confidence bar
    - All category scores in a table
    - Matched keywords with their weights
    """
    # Get the full explanation
    result = classifier.explain_classification(query)
    category = result["category"]
    confidence = result["confidence"]
    all_scores = result["all_scores"]
    matched_keywords = result["matched_keywords"]

    # Get color for the category
    color = CATEGORY_COLORS.get(category, "white")

    # Build the main panel content
    lines = []
    lines.append(f"[bold]Query:[/bold] {query}")
    lines.append("")
    lines.append(
        f"[bold]Category:[/bold] [{color} bold]{category.upper()}[/{color} bold]"
    )

    # Confidence bar: [=====>     ] 55%
    bar_length = 20
    filled = int(confidence * bar_length)
    bar = "=" * filled + ">" + " " * (bar_length - filled - 1)
    confidence_color = "green" if confidence > 0.3 else "yellow" if confidence > 0.1 else "red"
    lines.append(
        f"[bold]Confidence:[/bold] [{confidence_color}][{bar}] {confidence:.1%}[/{confidence_color}]"
    )

    # Show the category description
    lines.append("")
    lines.append(f"[dim]{classifier.get_category_info(category)}[/dim]")

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Classification Result[/bold]",
        border_style=color,
    ))

    # Show scores table for all categories
    table = Table(title="All Category Scores", show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Bar", justify="left")
    table.add_column("Matched Keywords", style="dim")

    for cat in ["terraform", "kubernetes", "docker", "cicd", "general"]:
        score = all_scores.get(cat, 0.0)
        cat_color = CATEGORY_COLORS.get(cat, "white")

        # Mini bar
        mini_bar_len = 15
        mini_filled = int(score * mini_bar_len)
        mini_bar = "[" + "#" * mini_filled + "." * (mini_bar_len - mini_filled) + "]"

        # Matched keywords for this category
        if cat in matched_keywords:
            kw_list = ", ".join(
                [f"{kw}({w})" for kw, w in matched_keywords[cat]]
            )
        else:
            kw_list = "-"

        # Highlight the winning category
        if cat == category:
            table.add_row(
                f"[{cat_color} bold]>> {cat.upper()}[/{cat_color} bold]",
                f"[bold]{score:.2%}[/bold]",
                f"[{cat_color}]{mini_bar}[/{cat_color}]",
                kw_list,
            )
        else:
            table.add_row(
                f"[{cat_color}]{cat}[/{cat_color}]",
                f"{score:.2%}",
                f"[dim]{mini_bar}[/dim]",
                kw_list,
            )

    console.print(table)
    console.print()


# ============================================================================
# TEST QUERIES
# ============================================================================
# These test queries cover all categories and edge cases.
# Each tuple is: (query, expected_category)

TEST_QUERIES = [
    # Terraform queries
    ("terraform init fails with provider error", "terraform"),
    ("my terraform state file is locked", "terraform"),
    ("how do I import existing AWS resources into terraform", "terraform"),

    # Kubernetes queries
    ("my pod is stuck in CrashLoopBackOff", "kubernetes"),
    ("kubectl get pods shows ImagePullBackOff", "kubernetes"),
    ("how to set up an ingress controller in kubernetes", "kubernetes"),
    ("helm chart deployment fails with timeout", "kubernetes"),

    # Docker queries
    ("docker build fails with COPY file not found", "docker"),
    ("my docker container keeps exiting immediately", "docker"),
    ("docker-compose up shows port already in use", "docker"),

    # CI/CD queries
    ("github actions workflow fails on push", "cicd"),
    ("my pipeline build step times out", "cicd"),
    ("how do I set up continuous deployment with Jenkins", "cicd"),

    # General / ambiguous queries
    ("what is the best practice for DevOps", "general"),
    ("hello", "general"),
]


# ============================================================================
# MAIN - Interactive Demo
# ============================================================================
def main():
    """
    Run the Error Classifier in interactive mode.

    First runs all test queries, then lets the user type their own questions.
    """
    console.print(Panel.fit(
        "[bold cyan]Exercise 1: Error Classifier[/bold cyan]\n"
        "[dim]Module 7 - Complete Chatbot (Capstone)[/dim]\n\n"
        "This component classifies DevOps questions into categories\n"
        "so we can search the right part of our knowledge base.",
        border_style="cyan",
    ))

    # Create the classifier
    classifier = ErrorClassifier()

    # ---- Part 1: Run test queries ----
    console.print("\n[bold yellow]Part 1: Running Test Queries[/bold yellow]\n")

    # Track accuracy
    correct = 0
    total = len(TEST_QUERIES)

    for query, expected in TEST_QUERIES:
        category, confidence, _ = classifier.classify(query)
        is_correct = category == expected

        if is_correct:
            correct += 1
            icon = "[green]PASS[/green]"
        else:
            icon = "[red]FAIL[/red]"

        console.print(
            f"  {icon}  [{CATEGORY_COLORS.get(category, 'white')}]{category:12s}[/{CATEGORY_COLORS.get(category, 'white')}]"
            f"  (conf: {confidence:.0%})  {query}"
        )

    # Show accuracy
    accuracy = correct / total if total > 0 else 0
    acc_color = "green" if accuracy > 0.8 else "yellow" if accuracy > 0.5 else "red"
    console.print(
        f"\n  [{acc_color}]Accuracy: {correct}/{total} ({accuracy:.0%})[/{acc_color}]"
    )

    # ---- Part 2: Detailed view of one query ----
    console.print("\n[bold yellow]Part 2: Detailed Classification View[/bold yellow]\n")
    display_classification_result(
        "My kubernetes pod keeps crashing with OOMKilled error",
        classifier,
    )

    # ---- Part 3: Interactive mode ----
    console.print("[bold yellow]Part 3: Interactive Mode[/bold yellow]")
    console.print("[dim]Type a DevOps question and see how it gets classified.[/dim]")
    console.print("[dim]Type 'quit' to exit.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold green]Your question[/bold green]")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            if not user_input.strip():
                continue

            # Show detailed classification
            display_classification_result(user_input, classifier)

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

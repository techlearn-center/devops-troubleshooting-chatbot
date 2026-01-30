"""
Module 7 - Exercise 1: Error Classifier (COMPLETE SOLUTION)
============================================================

A hybrid error classifier that uses keyword-based classification with
confidence scoring, plus an LLM-based fallback for ambiguous queries.

Categories: kubernetes, docker, cicd, networking, monitoring, general

Features:
  - Keyword-based classification with weighted scoring
  - LLM fallback when keyword confidence is below threshold
  - Confidence scoring with explanation of matched keywords
  - Rich colored output with confidence bars
  - 20+ test queries with expected categories
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
from rich.text import Text
from rich.progress_bar import ProgressBar

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
# Category colour map
# ---------------------------------------------------------------------------
CATEGORY_COLORS = {
    "kubernetes": "bright_cyan",
    "docker": "bright_blue",
    "cicd": "bright_yellow",
    "networking": "bright_green",
    "monitoring": "bright_magenta",
    "general": "white",
}

# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    category: str
    confidence: float
    matched_keywords: list = field(default_factory=list)
    method: str = "keyword"  # "keyword" or "llm"

    def __str__(self) -> str:
        return (
            f"[{self.category.upper()}] "
            f"confidence={self.confidence:.2f} "
            f"method={self.method}"
        )


# ---------------------------------------------------------------------------
# ErrorClassifier
# ---------------------------------------------------------------------------
class ErrorClassifier:
    """Hybrid keyword + LLM error classifier for DevOps queries."""

    # Keyword dictionaries with weights (higher weight = stronger signal)
    KEYWORDS: dict[str, dict[str, float]] = {
        "kubernetes": {
            "pod": 1.0, "kubectl": 1.2, "deployment": 0.9, "k8s": 1.2,
            "kubelet": 1.2, "namespace": 0.9, "node": 0.6, "helm": 0.8,
            "kube": 1.0, "container": 0.5, "service mesh": 0.7,
            "ingress": 0.9, "replicaset": 1.0, "statefulset": 1.0,
            "daemonset": 1.0, "cronjob": 0.7, "configmap": 1.0,
            "secret": 0.5, "pvc": 0.9, "persistent volume": 0.9,
            "crashloopbackoff": 1.2, "imagepullbackoff": 1.2,
            "oomkilled": 1.0, "evicted": 0.9, "taint": 1.0,
        },
        "docker": {
            "docker": 1.2, "dockerfile": 1.2, "container": 0.7,
            "image": 0.6, "docker-compose": 1.2, "registry": 0.7,
            "build": 0.4, "layer": 0.6, "volume": 0.5,
            "docker run": 1.2, "docker pull": 1.2, "docker push": 1.2,
            "entrypoint": 1.0, "cmd": 0.3, "multistage": 1.0,
            "docker network": 1.0, "bridge": 0.4, "overlay": 0.4,
        },
        "cicd": {
            "pipeline": 0.9, "ci/cd": 1.2, "cicd": 1.2, "jenkins": 1.2,
            "github actions": 1.2, "gitlab ci": 1.2, "circleci": 1.2,
            "workflow": 0.6, "build": 0.4, "deploy": 0.5,
            "artifact": 0.7, "stage": 0.5, "runner": 0.8,
            "yaml": 0.3, "trigger": 0.5, "release": 0.5,
            "argocd": 0.9, "spinnaker": 1.0, "tekton": 1.0,
        },
        "networking": {
            "dns": 1.0, "tcp": 0.9, "udp": 0.9, "http": 0.6,
            "ssl": 0.9, "tls": 0.9, "certificate": 0.7,
            "firewall": 1.0, "port": 0.5, "load balancer": 0.9,
            "proxy": 0.7, "nginx": 0.8, "haproxy": 0.9,
            "latency": 0.7, "timeout": 0.6, "connection refused": 1.0,
            "502": 0.8, "503": 0.8, "504": 0.8, "cors": 0.9,
            "ip address": 0.7, "subnet": 0.9, "vpn": 0.8,
            "routing": 0.7, "iptables": 1.0,
        },
        # ------------- MONITORING (new category added) -------------
        "monitoring": {
            "prometheus": 1.2, "grafana": 1.2, "alertmanager": 1.2,
            "metrics": 0.9, "dashboard": 0.7, "alert": 0.7,
            "pagerduty": 1.2, "datadog": 1.2, "nagios": 1.0,
            "zabbix": 1.0, "observability": 1.0, "tracing": 0.8,
            "jaeger": 1.0, "elk": 0.9, "kibana": 1.0,
            "loki": 1.0, "fluentd": 1.0, "logstash": 1.0,
            "sla": 0.7, "slo": 0.7, "sli": 0.7,
            "uptime": 0.7, "scrape": 0.9, "exporter": 1.0,
        },
    }

    LLM_CONFIDENCE_THRESHOLD = 0.5  # below this, fall back to LLM

    # ------------------------------------------------------------------
    # Keyword-based classification
    # ------------------------------------------------------------------
    def _keyword_classify(self, query: str) -> ClassificationResult:
        """Score *query* against every category's keywords."""
        query_lower = query.lower()
        scores: dict[str, float] = {}
        matches: dict[str, list] = {}

        for category, kw_map in self.KEYWORDS.items():
            score = 0.0
            matched = []
            for keyword, weight in kw_map.items():
                if keyword in query_lower:
                    score += weight
                    matched.append(keyword)
            scores[category] = score
            matches[category] = matched

        total = sum(scores.values())
        if total == 0:
            return ClassificationResult("general", 0.0, [], "keyword")

        best_cat = max(scores, key=scores.get)  # type: ignore[arg-type]
        confidence = scores[best_cat] / (total + 1e-9)
        # Clamp to [0, 1]
        confidence = min(1.0, max(0.0, confidence))

        return ClassificationResult(
            category=best_cat,
            confidence=round(confidence, 3),
            matched_keywords=matches[best_cat],
            method="keyword",
        )

    # ------------------------------------------------------------------
    # LLM-based classification (fallback)
    # ------------------------------------------------------------------
    def _llm_classify(self, query: str) -> ClassificationResult:
        """Ask the LLM to classify an ambiguous query."""
        categories = list(self.KEYWORDS.keys()) + ["general"]
        prompt = (
            "You are a DevOps error classifier. Classify the following query "
            "into exactly ONE of these categories:\n"
            f"  {', '.join(categories)}\n\n"
            "Respond ONLY with valid JSON: "
            '{"category": "<cat>", "confidence": <0.0-1.0>}\n\n'
            f"Query: {query}"
        )
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=60,
            )
            text = response.choices[0].message.content.strip()
            # Attempt to parse JSON from response
            data = json.loads(text)
            cat = data.get("category", "general").lower()
            conf = float(data.get("confidence", 0.7))
            if cat not in categories:
                cat = "general"
            return ClassificationResult(
                category=cat,
                confidence=round(min(1.0, max(0.0, conf)), 3),
                matched_keywords=[],
                method="llm",
            )
        except Exception as exc:
            console.print(f"[dim]LLM fallback failed ({exc}); defaulting to general[/dim]")
            return ClassificationResult("general", 0.3, [], "llm")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(self, query: str) -> ClassificationResult:
        """Classify *query*, using LLM fallback if keyword confidence is low."""
        result = self._keyword_classify(query)
        if result.confidence < self.LLM_CONFIDENCE_THRESHOLD:
            llm_result = self._llm_classify(query)
            # Prefer LLM if it is more confident
            if llm_result.confidence > result.confidence:
                return llm_result
        return result

    def explain_classification(self, query: str) -> str:
        """Return a human-readable explanation of the classification."""
        result = self.classify(query)
        lines = [
            f"Query   : {query}",
            f"Category: {result.category.upper()}",
            f"Confidence: {result.confidence:.0%}",
            f"Method  : {result.method}",
        ]
        if result.matched_keywords:
            lines.append(f"Matched : {', '.join(result.matched_keywords)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------
def _confidence_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[green]" + "█" * filled + "[/green]" + "[dim]░[/dim]" * (width - filled)


def display_result(query: str, result: ClassificationResult, expected: Optional[str] = None):
    """Pretty-print a single classification result."""
    color = CATEGORY_COLORS.get(result.category, "white")
    match_icon = ""
    if expected:
        match_icon = " [green]OK[/green]" if result.category == expected else " [red]MISS[/red]"
    console.print(
        f"  [{color}]{result.category.upper():12s}[/{color}] "
        f"{_confidence_bar(result.confidence)} "
        f"{result.confidence:5.1%}  "
        f"[dim]({result.method})[/dim]{match_icon}  "
        f"[italic]{query}[/italic]"
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    ("My pod keeps restarting with CrashLoopBackOff", "kubernetes"),
    ("kubectl get pods shows ImagePullBackOff", "kubernetes"),
    ("How to set up a Kubernetes namespace?", "kubernetes"),
    ("Docker build fails at COPY step", "docker"),
    ("Cannot pull image from private docker registry", "docker"),
    ("docker-compose up returns exit code 137", "docker"),
    ("GitHub Actions workflow is failing on test step", "cicd"),
    ("Jenkins pipeline stuck at deploy stage", "cicd"),
    ("GitLab CI runner keeps timing out", "cicd"),
    ("DNS resolution failing inside container", "networking"),
    ("Getting 502 Bad Gateway from nginx reverse proxy", "networking"),
    ("SSL certificate expired, TLS handshake fails", "networking"),
    ("Connection refused on port 443", "networking"),
    ("Prometheus scrape targets are all down", "monitoring"),
    ("Grafana dashboard not showing any data", "monitoring"),
    ("Alertmanager not sending PagerDuty notifications", "monitoring"),
    ("Datadog agent reporting high latency metrics", "monitoring"),
    ("How to set up ELK stack for log aggregation?", "monitoring"),
    ("My application is slow but I don't know why", "general"),
    ("Best practices for DevOps team structure", "general"),
    ("Helm chart values not being applied to deployment", "kubernetes"),
    ("How to configure Jaeger distributed tracing?", "monitoring"),
    ("Pipeline artifact upload failing in CircleCI", "cicd"),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(
        Panel(
            "[bold]Module 7 - Exercise 1: Error Classifier[/bold]\n"
            "[dim]Hybrid keyword + LLM classification for DevOps queries[/dim]",
            border_style="bright_cyan",
        )
    )

    classifier = ErrorClassifier()

    # Legend
    console.print("\n[bold]Category legend:[/bold]")
    for cat, col in CATEGORY_COLORS.items():
        console.print(f"  [{col}]■ {cat.upper()}[/{col}]")
    console.print()

    # Run all test queries
    console.print("[bold underline]Classification Results[/bold underline]\n")
    correct = 0
    total = len(TEST_QUERIES)

    for query, expected in TEST_QUERIES:
        result = classifier.classify(query)
        display_result(query, result, expected)
        if result.category == expected:
            correct += 1

    # Summary
    console.print(f"\n[bold]Accuracy: {correct}/{total} ({correct/total:.0%})[/bold]\n")

    # Detailed explanation for one example
    console.print(Panel("[bold]Detailed Explanation Example[/bold]", border_style="yellow"))
    console.print(classifier.explain_classification(
        "Prometheus scrape targets down and Grafana dashboard empty"
    ))
    console.print()

    # Interactive mode
    console.print("[bold green]Interactive mode[/bold green] — type a query or 'quit' to exit.\n")
    while True:
        try:
            query = console.input("[bold]> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break
        result = classifier.classify(query)
        display_result(query, result)
        if result.matched_keywords:
            console.print(f"    [dim]Matched keywords: {', '.join(result.matched_keywords)}[/dim]")
        console.print()

    console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()

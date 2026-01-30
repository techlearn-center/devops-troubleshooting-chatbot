"""
Module 7 - Exercise 4: Complete DevOps Troubleshooting Chatbot (COMPLETE SOLUTION)
====================================================================================

THE GRAND FINALE -- a production-quality chatbot integrating every component:

  - Error Classifier    (keyword + LLM hybrid)
  - RAG Pipeline        (MMR retrieval, ChromaDB / in-memory fallback)
  - Prompt Engine       (presets, history, token management, debug mode)
  - LLM Generation      (streaming responses)

Slash commands:
  /help      - Show available commands
  /clear     - Clear conversation history
  /history   - Show conversation history
  /stats     - Show RAG pipeline & session statistics
  /category  - Manually set or auto-detect category
  /export    - Export conversation to a Markdown file
  /feedback  - Rate the last response (1-5 stars)
  /preset    - Switch prompt preset (expert, quick, teacher)
  /debug     - Toggle debug mode
  /quit      - Exit the chatbot

Features:
  - Streaming responses with Rich Live display
  - Category-colored response panels
  - Source documents shown after each response
  - Styled ASCII art welcome screen
  - Graceful error handling throughout
  - Self-contained and runnable standalone
  - Falls back to demo data if knowledge base not found
"""

import os
import sys
import json
import hashlib
import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client setup -- supports both OpenAI and Ollama
# ---------------------------------------------------------------------------
from openai import OpenAI

if os.getenv("USE_OLLAMA", "false").lower() == "true":
    client = OpenAI(
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key="ollama",
    )
    MODEL = os.getenv("OLLAMA_MODEL", "llama2")
else:
    client = OpenAI()
    MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Optional imports
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORY COLOURS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "kubernetes": "bright_cyan",
    "docker": "bright_blue",
    "cicd": "bright_yellow",
    "networking": "bright_green",
    "monitoring": "bright_magenta",
    "general": "white",
}

CATEGORY_EMOJI = {
    "kubernetes": "K8s",
    "docker": "Docker",
    "cicd": "CI/CD",
    "networking": "Net",
    "monitoring": "Mon",
    "general": "Gen",
}


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 1: ERROR CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    matched_keywords: list = field(default_factory=list)
    method: str = "keyword"


class ErrorClassifier:
    KEYWORDS = {
        "kubernetes": {
            "pod": 1.0, "kubectl": 1.2, "deployment": 0.9, "k8s": 1.2,
            "kubelet": 1.2, "namespace": 0.9, "helm": 0.8, "kube": 1.0,
            "ingress": 0.9, "replicaset": 1.0, "statefulset": 1.0,
            "daemonset": 1.0, "configmap": 1.0, "pvc": 0.9,
            "crashloopbackoff": 1.2, "imagepullbackoff": 1.2,
            "oomkilled": 1.0, "evicted": 0.9, "taint": 1.0,
        },
        "docker": {
            "docker": 1.2, "dockerfile": 1.2, "container": 0.7,
            "image": 0.6, "docker-compose": 1.2, "registry": 0.7,
            "layer": 0.6, "volume": 0.5, "entrypoint": 1.0,
            "multistage": 1.0,
        },
        "cicd": {
            "pipeline": 0.9, "ci/cd": 1.2, "cicd": 1.2, "jenkins": 1.2,
            "github actions": 1.2, "gitlab ci": 1.2, "circleci": 1.2,
            "workflow": 0.6, "artifact": 0.7, "runner": 0.8,
            "argocd": 0.9, "tekton": 1.0,
        },
        "networking": {
            "dns": 1.0, "tcp": 0.9, "ssl": 0.9, "tls": 0.9,
            "certificate": 0.7, "firewall": 1.0, "load balancer": 0.9,
            "proxy": 0.7, "nginx": 0.8, "latency": 0.7,
            "timeout": 0.6, "connection refused": 1.0,
            "502": 0.8, "503": 0.8, "504": 0.8, "cors": 0.9,
        },
        "monitoring": {
            "prometheus": 1.2, "grafana": 1.2, "alertmanager": 1.2,
            "metrics": 0.9, "dashboard": 0.7, "alert": 0.7,
            "pagerduty": 1.2, "datadog": 1.2, "observability": 1.0,
            "jaeger": 1.0, "elk": 0.9, "kibana": 1.0, "loki": 1.0,
            "fluentd": 1.0, "exporter": 1.0,
        },
    }

    LLM_THRESHOLD = 0.5

    def _keyword_classify(self, query: str) -> ClassificationResult:
        query_lower = query.lower()
        scores, matches = {}, {}
        for cat, kw_map in self.KEYWORDS.items():
            score = 0.0
            matched = []
            for kw, w in kw_map.items():
                if kw in query_lower:
                    score += w
                    matched.append(kw)
            scores[cat] = score
            matches[cat] = matched
        total = sum(scores.values())
        if total == 0:
            return ClassificationResult("general", 0.0, [], "keyword")
        best = max(scores, key=scores.get)
        conf = min(1.0, scores[best] / (total + 1e-9))
        return ClassificationResult(best, round(conf, 3), matches[best], "keyword")

    def _llm_classify(self, query: str) -> ClassificationResult:
        cats = list(self.KEYWORDS.keys()) + ["general"]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": (
                    f"Classify into ONE of {cats}. JSON only: "
                    '{"category":"<cat>","confidence":<0-1>}\n\n' + query
                )}],
                temperature=0, max_tokens=60,
            )
            data = json.loads(resp.choices[0].message.content.strip())
            cat = data.get("category", "general").lower()
            if cat not in cats:
                cat = "general"
            return ClassificationResult(cat, round(float(data.get("confidence", 0.7)), 3), [], "llm")
        except Exception:
            return ClassificationResult("general", 0.3, [], "llm")

    def classify(self, query: str) -> ClassificationResult:
        result = self._keyword_classify(query)
        if result.confidence < self.LLM_THRESHOLD:
            llm_result = self._llm_classify(query)
            if llm_result.confidence > result.confidence:
                return llm_result
        return result


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 2: RAG PIPELINE (simplified, self-contained)
# ═══════════════════════════════════════════════════════════════════════════

DEMO_KB = [
    {"title": "Kubernetes CrashLoopBackOff", "category": "kubernetes",
     "content": "CrashLoopBackOff means a pod keeps crashing after restarting. Common causes: application errors, missing config maps or secrets, insufficient resources (OOMKilled), incorrect command. Debug: 1) kubectl logs <pod> --previous 2) kubectl describe pod <pod> 3) Check resource limits 4) Verify env vars."},
    {"title": "Kubernetes ImagePullBackOff", "category": "kubernetes",
     "content": "ImagePullBackOff occurs when Kubernetes cannot pull a container image. Causes: wrong image name/tag, private registry without imagePullSecrets, network issues, registry rate limits. Fix: verify image name, create imagePullSecret, check node network connectivity."},
    {"title": "Kubernetes OOMKilled", "category": "kubernetes",
     "content": "OOMKilled means the container exceeded its memory limit. Fix by increasing memory limits, profiling for leaks, or optimising usage. Use kubectl top pod to monitor. Set both requests and limits in the pod spec."},
    {"title": "Docker Build Optimization", "category": "docker",
     "content": "Docker build slow? Use multi-stage builds, order instructions so changing steps come last, leverage build cache. COPY requirements.txt first, then RUN pip install, then COPY the rest. Use .dockerignore to exclude unnecessary files."},
    {"title": "Docker Compose Networking", "category": "docker",
     "content": "Containers in docker-compose share a default bridge network. Services reach each other by service name. Common issues: port conflicts, DNS delays, dependencies not ready. Use depends_on with healthchecks."},
    {"title": "Docker Exit Code 137", "category": "docker",
     "content": "Exit code 137 = SIGKILL, usually OOM. Increase Docker daemon memory or container --memory flag. Check dmesg for OOM killer. On Docker Desktop, increase memory in Settings > Resources."},
    {"title": "GitHub Actions Debugging", "category": "cicd",
     "content": "GitHub Actions failures: YAML syntax errors, expired secrets, disk space, flaky tests. Debug locally with 'act', enable ACTIONS_STEP_DEBUG secret. Check Annotations tab. Cache dependencies to speed up."},
    {"title": "Jenkins Pipeline Issues", "category": "cicd",
     "content": "Jenkins issues: Groovy sandbox, stale workspace, missing agents. Use Replay for quick iteration. Check /var/log/jenkins/jenkins.log. Clean workspace with deleteDir(), update plugins, verify agent labels."},
    {"title": "DNS Resolution in Containers", "category": "networking",
     "content": "DNS failures in containers: check /etc/resolv.conf, verify kube-dns/CoreDNS pods running, check network policies not blocking UDP 53. Test with nslookup or dig. In Docker, check DNS settings in daemon.json."},
    {"title": "502 Bad Gateway Nginx", "category": "networking",
     "content": "502 from Nginx = upstream down or misconfigured. Check upstream service running, proxy_pass URL correct, timeout settings (proxy_read_timeout, proxy_connect_timeout), and buffer sizes."},
    {"title": "SSL/TLS Certificate Errors", "category": "networking",
     "content": "TLS errors: expired cert, wrong CN/SAN, untrusted CA, protocol mismatch. Verify with openssl s_client -connect host:443. Renew with certbot. Check full chain served. For K8s, use cert-manager."},
    {"title": "Prometheus Targets Down", "category": "monitoring",
     "content": "Prometheus targets DOWN: verify /metrics endpoint responds, check network connectivity, verify scrape config labels match ServiceMonitor/PodMonitor. Check Status > Targets in Prometheus UI."},
    {"title": "Grafana No Data", "category": "monitoring",
     "content": "Grafana no data: verify data source connection, check time range, ensure Prometheus collects the metric. Test with simple query like up{}. Check variable selectors. Verify Grafana can reach Prometheus."},
    {"title": "Alertmanager Notifications", "category": "monitoring",
     "content": "Alertmanager not sending: check receivers config, verify route matching, check inhibit_rules, verify external connectivity (SMTP, PagerDuty API). Test with amtool config routes test <labels>."},
    {"title": "ELK Stack Logging", "category": "monitoring",
     "content": "ELK setup: allocate heap for ES (half RAM, max 32GB), use ILM for retention, configure Logstash pipelines. For K8s, use Fluent Bit (EFK stack). Set up Kibana index patterns after data flows."},
]


class EmbeddingProvider:
    def __init__(self):
        self._model = SentenceTransformer("all-MiniLM-L6-v2") if HAS_ST else None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model:
            return self._model.encode(texts, show_progress_bar=False).tolist()
        dim = 384
        result = []
        for t in texts:
            h = hashlib.sha256(t.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim*2), 2)]
            vec += [0.0] * (dim - len(vec))
            result.append(vec)
        return result


@dataclass
class RetrievedChunk:
    content: str
    title: str
    category: str
    score: float


class RAGPipeline:
    def __init__(self):
        self.embedder = EmbeddingProvider()
        self.documents = DEMO_KB
        self._docs: list[dict] = []
        self._vecs: list[list[float]] = []
        self._ingest()

    def _ingest(self):
        for doc in self.documents:
            self._docs.append(doc)
        texts = [d["content"] for d in self._docs]
        self._vecs = self.embedder.embed(texts)

    def rebuild_index(self):
        self._docs.clear()
        self._vecs.clear()
        self._ingest()

    @staticmethod
    def _cosine(a, b):
        a_, b_ = np.array(a), np.array(b)
        return float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-10))

    def retrieve_mmr(self, query: str, k: int = 3, category: Optional[str] = None,
                     lambda_mult: float = 0.7) -> list[RetrievedChunk]:
        q_emb = self.embedder.embed([query])[0]
        candidates = []
        for i, (doc, vec) in enumerate(zip(self._docs, self._vecs)):
            if category and doc.get("category") != category:
                continue
            sim = self._cosine(q_emb, vec)
            candidates.append((i, sim))
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:k * 3]  # fetch_k

        selected, remaining = [], list(range(len(candidates)))
        for _ in range(min(k, len(candidates))):
            best_idx, best_score = None, -float("inf")
            for ri in remaining:
                ci, rel = candidates[ri]
                if selected:
                    max_sim = max(self._cosine(self._vecs[ci], self._vecs[candidates[si][0]]) for si in selected)
                else:
                    max_sim = 0.0
                score = lambda_mult * rel - (1 - lambda_mult) * max_sim
                if score > best_score:
                    best_score = score
                    best_idx = ri
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        results = []
        for si in selected:
            ci, sim = candidates[si]
            doc = self._docs[ci]
            results.append(RetrievedChunk(
                content=doc["content"], title=doc["title"],
                category=doc.get("category", "general"), score=round(sim, 4),
            ))
        return results

    def stats(self) -> dict:
        cats = {}
        for d in self._docs:
            c = d.get("category", "general")
            cats[c] = cats.get(c, 0) + 1
        return {"total_documents": len(self._docs), "categories": cats}


# ═══════════════════════════════════════════════════════════════════════════
# COMPONENT 3: PROMPT ENGINE
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "expert": (
        "You are a senior DevOps engineer and SRE with 15+ years of experience across "
        "Kubernetes, Docker, CI/CD, cloud platforms, monitoring, and networking. "
        "Provide detailed, production-ready solutions with specific commands and best practices. "
        "Mention potential pitfalls and security considerations."
    ),
    "quick": (
        "You are a concise DevOps assistant. Give short, actionable answers with bullet points. "
        "Focus on exact commands or steps needed. One-liner solutions preferred."
    ),
    "teacher": (
        "You are a patient DevOps instructor teaching juniors. Explain step by step with analogies. "
        "After the solution, explain WHY it works. Suggest topics for further study."
    ),
}


def _count_tokens(text: str) -> int:
    if HAS_TIKTOKEN:
        return len(_enc.encode(text))
    return len(text) // 4


class PromptEngine:
    MAX_HISTORY_TOKENS = 2000
    MAX_TOTAL_TOKENS = 4096

    def __init__(self, preset: str = "expert", debug: bool = False):
        self.preset = preset
        self.debug = debug
        self.history: list[dict] = []
        self._system = PRESETS.get(preset, PRESETS["expert"])

    def set_preset(self, name: str):
        if name in PRESETS:
            self._system = PRESETS[name]
            self.preset = name
            return True
        return False

    def add_history(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history.clear()

    def _trimmed_history(self) -> list[dict]:
        trimmed = list(self.history)
        total = sum(_count_tokens(m["content"]) for m in trimmed)
        while total > self.MAX_HISTORY_TOKENS and len(trimmed) > 2:
            removed = trimmed.pop(0)
            total -= _count_tokens(removed["content"])
        return trimmed

    def get_history_summary(self) -> str:
        if not self.history:
            return ""
        text = "\n".join(f"{m['role']}: {m['content']}" for m in self.history[-8:])
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Summarize this DevOps conversation in 2 sentences."},
                    {"role": "user", "content": text},
                ],
                temperature=0.2, max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return text[:200]

    def assemble(self, query: str, context: Optional[str] = None,
                 category: Optional[str] = None) -> list[dict]:
        system_parts = [self._system]
        if category:
            system_parts.append(f"\nQuery category: {category.upper()}. Focus on this area.")
        msgs = [{"role": "system", "content": "\n".join(system_parts)}]
        msgs.extend(self._trimmed_history())
        if context:
            msgs.append({"role": "system", "content":
                         "Knowledge base context (use to answer):\n\n" + context})
        msgs.append({"role": "user", "content": query})
        if self.debug:
            self._show_debug(msgs)
        return msgs

    def _show_debug(self, msgs):
        console.print(Panel("[bold yellow]DEBUG: Prompt Assembly[/bold yellow]", border_style="yellow"))
        total = 0
        for m in msgs:
            t = _count_tokens(m["content"])
            total += t
            color = {"system": "cyan", "user": "green", "assistant": "yellow"}.get(m["role"], "white")
            console.print(f"  [{color}]{m['role'].upper()}[/{color}] ({t} tok) {m['content'][:120]}...")
        console.print(f"  [bold]Total: ~{total} tokens[/bold]\n")


# ═══════════════════════════════════════════════════════════════════════════
# THE COMPLETE CHATBOT
# ═══════════════════════════════════════════════════════════════════════════

WELCOME_ART = r"""
[bright_cyan]
  ____             ___                ____ _           _   _           _
 |  _ \  _____   _/ _ \ _ __  ___   / ___| |__   __ _| |_| |__   ___ | |_
 | | | |/ _ \ \ / / | | | '_ \/ __| | |   | '_ \ / _` | __| '_ \ / _ \| __|
 | |_| |  __/\ V /| |_| | |_) \__ \ | |___| | | | (_| | |_| |_) | (_) | |_
 |____/ \___| \_/  \___/| .__/|___/  \____|_| |_|\__,_|\__|_.__/ \___/ \__|
                         |_|
[/bright_cyan]
[bold]DevOps Troubleshooting Chatbot[/bold] [dim]v1.0 - Module 7 Capstone[/dim]
[dim]Type /help for commands. Powered by {model}.[/dim]
"""

HELP_TEXT = """
[bold]Available Commands:[/bold]

  [cyan]/help[/cyan]       Show this help message
  [cyan]/clear[/cyan]      Clear conversation history
  [cyan]/history[/cyan]    Show conversation history
  [cyan]/stats[/cyan]      Show pipeline & session statistics
  [cyan]/category[/cyan]   Show current category or set manually (/category docker)
  [cyan]/export[/cyan]     Export conversation to Markdown file
  [cyan]/feedback[/cyan]   Rate the last response (1-5 stars)
  [cyan]/preset[/cyan]     Switch prompt preset (/preset quick)
  [cyan]/debug[/cyan]      Toggle debug mode
  [cyan]/quit[/cyan]       Exit the chatbot
"""


class DevOpsChatbot:
    """The grand finale -- all components integrated."""

    def __init__(self):
        self.classifier = ErrorClassifier()
        self.rag = RAGPipeline()
        self.prompt_engine = PromptEngine(preset="expert")
        self.current_category: Optional[str] = None
        self.last_response: str = ""
        self.query_count: int = 0
        self.feedback_file = Path("chatbot_feedback.json")
        self.session_start = datetime.datetime.now()

    # ------------------------------------------------------------------
    # Slash command handlers
    # ------------------------------------------------------------------
    def _cmd_help(self):
        console.print(HELP_TEXT)

    def _cmd_clear(self):
        self.prompt_engine.clear_history()
        self.current_category = None
        console.print("[green]Conversation history cleared.[/green]")

    def _cmd_history(self):
        if not self.prompt_engine.history:
            console.print("[dim]No conversation history yet.[/dim]")
            return
        table = Table(title="Conversation History", border_style="cyan", show_lines=True)
        table.add_column("#", width=4, justify="right")
        table.add_column("Role", width=10)
        table.add_column("Content", max_width=80)
        for i, msg in enumerate(self.prompt_engine.history, 1):
            color = "green" if msg["role"] == "user" else "yellow"
            content = msg["content"][:150] + ("..." if len(msg["content"]) > 150 else "")
            table.add_row(str(i), f"[{color}]{msg['role']}[/{color}]", content)
        console.print(table)

    def _cmd_stats(self):
        rag_stats = self.rag.stats()
        elapsed = datetime.datetime.now() - self.session_start
        table = Table(title="Session & Pipeline Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Session Duration", str(elapsed).split(".")[0])
        table.add_row("Queries Answered", str(self.query_count))
        table.add_row("History Turns", str(len(self.prompt_engine.history)))
        table.add_row("Current Category", self.current_category or "auto")
        table.add_row("Prompt Preset", self.prompt_engine.preset)
        table.add_row("Debug Mode", "ON" if self.prompt_engine.debug else "OFF")
        table.add_row("LLM Model", MODEL)
        table.add_row("---", "---")
        table.add_row("KB Documents", str(rag_stats["total_documents"]))
        for cat, count in sorted(rag_stats["categories"].items()):
            color = CATEGORY_COLORS.get(cat, "white")
            table.add_row(f"  [{color}]{cat}[/{color}]", str(count))
        console.print(table)

    def _cmd_category(self, arg: str):
        if arg:
            valid = list(CATEGORY_COLORS.keys())
            if arg in valid:
                self.current_category = arg
                color = CATEGORY_COLORS[arg]
                console.print(f"[{color}]Category set to: {arg.upper()}[/{color}]")
            elif arg == "auto":
                self.current_category = None
                console.print("[green]Category set to auto-detect.[/green]")
            else:
                console.print(f"[red]Unknown category. Valid: {', '.join(valid)}, auto[/red]")
        else:
            if self.current_category:
                color = CATEGORY_COLORS.get(self.current_category, "white")
                console.print(f"Current category: [{color}]{self.current_category.upper()}[/{color}]")
            else:
                console.print("Category: [green]auto-detect[/green]")

    def _cmd_export(self):
        """Export the conversation history to a Markdown file."""
        if not self.prompt_engine.history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"devops_chat_export_{timestamp}.md"
        lines = [
            f"# DevOps Chatbot Conversation Export",
            f"",
            f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Preset:** {self.prompt_engine.preset}",
            f"**Model:** {MODEL}",
            f"**Queries:** {self.query_count}",
            f"",
            f"---",
            f"",
        ]
        for msg in self.prompt_engine.history:
            if msg["role"] == "user":
                lines.append(f"## User")
                lines.append(f"")
                lines.append(msg["content"])
                lines.append(f"")
            elif msg["role"] == "assistant":
                lines.append(f"## Assistant")
                lines.append(f"")
                lines.append(msg["content"])
                lines.append(f"")
                lines.append(f"---")
                lines.append(f"")

        content = "\n".join(lines)
        Path(filename).write_text(content, encoding="utf-8")
        console.print(f"[green]Conversation exported to: {filename}[/green]")

    def _cmd_feedback(self, arg: str):
        """Store feedback (1-5 stars) for the last response."""
        if not self.last_response:
            console.print("[yellow]No response to rate yet.[/yellow]")
            return
        if not arg:
            console.print("Usage: /feedback <1-5>  (e.g., /feedback 5)")
            return
        try:
            rating = int(arg)
            if rating < 1 or rating > 5:
                raise ValueError
        except ValueError:
            console.print("[red]Please provide a rating between 1 and 5.[/red]")
            return

        # Load existing feedback
        feedback_data = []
        if self.feedback_file.exists():
            try:
                feedback_data = json.loads(self.feedback_file.read_text(encoding="utf-8"))
            except Exception:
                feedback_data = []

        # Find the last user query
        last_query = ""
        for msg in reversed(self.prompt_engine.history):
            if msg["role"] == "user":
                last_query = msg["content"]
                break

        feedback_data.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "query": last_query[:200],
            "response_preview": self.last_response[:200],
            "rating": rating,
            "preset": self.prompt_engine.preset,
            "model": MODEL,
        })

        self.feedback_file.write_text(json.dumps(feedback_data, indent=2), encoding="utf-8")
        stars = "[yellow]" + "*" * rating + "[/yellow]" + "[dim]" + "*" * (5 - rating) + "[/dim]"
        console.print(f"  Feedback recorded: {stars} ({rating}/5)")
        console.print(f"  [dim]Saved to {self.feedback_file}[/dim]")

    def _cmd_preset(self, arg: str):
        if not arg:
            console.print(f"Current preset: [bold]{self.prompt_engine.preset}[/bold]")
            console.print(f"Available: {', '.join(PRESETS.keys())}")
            return
        if self.prompt_engine.set_preset(arg):
            console.print(f"[green]Switched to '{arg}' preset.[/green]")
        else:
            console.print(f"[red]Unknown preset. Available: {', '.join(PRESETS.keys())}[/red]")

    def _cmd_debug(self):
        self.prompt_engine.debug = not self.prompt_engine.debug
        state = "ON" if self.prompt_engine.debug else "OFF"
        console.print(f"[yellow]Debug mode: {state}[/yellow]")

    # ------------------------------------------------------------------
    # Process slash commands
    # ------------------------------------------------------------------
    def handle_command(self, user_input: str) -> bool:
        """Handle a slash command. Returns True if handled."""
        parts = user_input.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        commands = {
            "/help": lambda: self._cmd_help(),
            "/clear": lambda: self._cmd_clear(),
            "/history": lambda: self._cmd_history(),
            "/stats": lambda: self._cmd_stats(),
            "/category": lambda: self._cmd_category(arg),
            "/export": lambda: self._cmd_export(),
            "/feedback": lambda: self._cmd_feedback(arg),
            "/preset": lambda: self._cmd_preset(arg),
            "/debug": lambda: self._cmd_debug(),
            "/quit": lambda: None,
            "/exit": lambda: None,
        }

        if cmd in commands:
            commands[cmd]()
            return True
        return False

    # ------------------------------------------------------------------
    # Core: process a user query
    # ------------------------------------------------------------------
    def process_query(self, query: str):
        """Full pipeline: classify -> retrieve -> assemble -> generate."""

        # 1. Classify
        if self.current_category:
            classification = ClassificationResult(self.current_category, 1.0, [], "manual")
        else:
            classification = self.classifier.classify(query)

        cat = classification.category
        color = CATEGORY_COLORS.get(cat, "white")
        tag = CATEGORY_EMOJI.get(cat, cat)
        console.print(
            f"  [{color}][{tag}][/{color}] "
            f"[dim]confidence={classification.confidence:.0%} "
            f"method={classification.method}[/dim]"
        )

        # 2. Retrieve context (RAG)
        chunks = self.rag.retrieve_mmr(query, k=3, category=cat if cat != "general" else None)
        context = ""
        if chunks:
            context = "\n\n---\n".join(
                f"[{c.title}]\n{c.content}" for c in chunks
            )

        # 3. Assemble prompt
        messages = self.prompt_engine.assemble(query, context=context, category=cat)

        # 4. Generate response (streaming)
        console.print()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.4,
                max_tokens=800,
                stream=True,
            )
            with Live(console=console, refresh_per_second=12) as live:
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        full_response += delta.content
                        panel = Panel(
                            Markdown(full_response),
                            title=f"[{color}]{tag}[/{color}] Response",
                            border_style=color,
                            padding=(1, 2),
                        )
                        live.update(panel)
        except Exception as exc:
            console.print(f"[red]Error generating response: {exc}[/red]")
            full_response = f"I encountered an error: {exc}. Please check your API key and model configuration."
            console.print(Panel(full_response, title="Error", border_style="red"))

        # 5. Record history
        self.prompt_engine.add_history("user", query)
        self.prompt_engine.add_history("assistant", full_response)
        self.last_response = full_response
        self.query_count += 1

        # 6. Show sources
        if chunks:
            console.print()
            console.print("[dim]Sources:[/dim]")
            for c in chunks:
                src_color = CATEGORY_COLORS.get(c.category, "white")
                console.print(
                    f"  [{src_color}]>[/{src_color}] {c.title} "
                    f"[dim]({c.category}, score: {c.score:.3f})[/dim]"
                )
        console.print()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        """Launch the interactive chatbot."""
        console.print(WELCOME_ART.format(model=MODEL))
        console.print(Panel(
            "[bold]Welcome to the DevOps Troubleshooting Chatbot![/bold]\n\n"
            "Ask me anything about Kubernetes, Docker, CI/CD, networking, or monitoring.\n"
            "I will classify your query, retrieve relevant knowledge, and provide\n"
            "a detailed, actionable answer.\n\n"
            "[dim]Type /help for commands, or just ask a question.[/dim]",
            border_style="bright_cyan",
            padding=(1, 3),
        ))

        while True:
            try:
                user_input = console.input("\n[bold bright_cyan]You>[/bold bright_cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye! Happy troubleshooting![/dim]")
                break

            if not user_input:
                continue

            # Check for slash commands
            if user_input.startswith("/"):
                if user_input.lower() in ("/quit", "/exit"):
                    console.print("[dim]Goodbye! Happy troubleshooting![/dim]")
                    break
                if self.handle_command(user_input):
                    continue
                console.print(f"[red]Unknown command: {user_input.split()[0]}. Type /help for commands.[/red]")
                continue

            # Process the query
            self.process_query(user_input)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    try:
        chatbot = DevOpsChatbot()
        chatbot.run()
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted. Goodbye![/dim]")
    except Exception as exc:
        console.print(f"\n[red bold]Fatal error: {exc}[/red bold]")
        console.print("[dim]Please check your environment setup (.env file, API keys, dependencies).[/dim]")
        raise


if __name__ == "__main__":
    main()

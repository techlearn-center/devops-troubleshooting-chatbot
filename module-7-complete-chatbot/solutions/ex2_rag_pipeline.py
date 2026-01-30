"""
Module 7 - Exercise 2: RAG Pipeline (COMPLETE SOLUTION)
========================================================

A full Retrieval-Augmented Generation pipeline for DevOps troubleshooting.

Features:
  - HuggingFace all-MiniLM-L6-v2 embeddings
  - ChromaDB persistent vector store
  - MMR (Maximal Marginal Relevance) retrieval for diverse results
  - rebuild_index() method for re-indexing from scratch
  - Category-filtered retrieval
  - Stats display: total docs, docs per category, total chunks
  - Rich progress bars during loading
  - Interactive query mode with source display
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
from openai import OpenAI

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

# ---------------------------------------------------------------------------
# Third-party imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# ---------------------------------------------------------------------------
# Demo knowledge base (used when real docs are not present)
# ---------------------------------------------------------------------------
DEMO_KNOWLEDGE_BASE: list[dict] = [
    {"title": "Kubernetes CrashLoopBackOff", "category": "kubernetes",
     "content": "CrashLoopBackOff means a pod keeps crashing after restarting. Common causes include: "
                "application errors, missing config maps or secrets, insufficient resources (OOMKilled), "
                "or incorrect command/entrypoint. Debug steps: 1) kubectl logs <pod> --previous  "
                "2) kubectl describe pod <pod>  3) Check resource limits  4) Verify environment variables."},
    {"title": "Kubernetes ImagePullBackOff", "category": "kubernetes",
     "content": "ImagePullBackOff occurs when Kubernetes cannot pull a container image. Causes: wrong image "
                "name/tag, private registry without imagePullSecrets, network issues, or registry rate limits. "
                "Fix: verify image name, create/attach imagePullSecret, check node network connectivity."},
    {"title": "Kubernetes OOMKilled", "category": "kubernetes",
     "content": "OOMKilled means the container exceeded its memory limit and was terminated by the kernel. "
                "Fix by increasing memory limits in the pod spec, profiling the application for memory leaks, "
                "or optimising memory usage.  Use 'kubectl top pod' to monitor real-time usage."},
    {"title": "Docker Build Cache Issues", "category": "docker",
     "content": "Docker build cache can cause stale layers. Use '--no-cache' to force rebuild. Order "
                "Dockerfile instructions so that frequently-changing steps come last. Use multi-stage builds "
                "to reduce image size.  COPY requirements.txt first, then RUN pip install before COPY . ."},
    {"title": "Docker Compose Networking", "category": "docker",
     "content": "Containers in docker-compose share a default bridge network. Services reach each other "
                "by service name. Common issues: port conflicts on the host, DNS resolution delays, and "
                "containers starting before dependencies are ready (use depends_on + healthchecks)."},
    {"title": "Docker Exit Code 137", "category": "docker",
     "content": "Exit code 137 means the container was killed by SIGKILL, usually due to out-of-memory. "
                "Increase Docker daemon memory limits or the container's --memory flag. Check host 'dmesg' "
                "for OOM killer messages.  On Docker Desktop, increase memory in Settings > Resources."},
    {"title": "GitHub Actions Workflow Failures", "category": "cicd",
     "content": "Common GitHub Actions failures: syntax errors in YAML, expired secrets, runner out of disk "
                "space, or flaky tests. Debug with 'act' locally, enable ACTIONS_STEP_DEBUG, and review the "
                "'Annotations' tab.  Cache dependencies to speed up workflows."},
    {"title": "Jenkins Pipeline Troubleshooting", "category": "cicd",
     "content": "Jenkins pipeline issues: Groovy sandbox restrictions, stale workspace, agent not available. "
                "Use 'Replay' to iterate quickly. Check /var/log/jenkins/jenkins.log. Common fix: clean "
                "workspace with deleteDir(), update plugins, and verify agent labels match."},
    {"title": "DNS Resolution Inside Containers", "category": "networking",
     "content": "DNS failures inside containers: check /etc/resolv.conf points to the right nameserver, "
                "verify kube-dns or CoreDNS pods are running (kubectl get pods -n kube-system), and check "
                "network policies aren't blocking UDP port 53. Test with 'nslookup' or 'dig' inside the pod."},
    {"title": "502 Bad Gateway Nginx", "category": "networking",
     "content": "502 Bad Gateway from Nginx usually means the upstream server is down or misconfigured. "
                "Check: upstream service is running, proxy_pass URL is correct, upstream timeout settings "
                "(proxy_read_timeout, proxy_connect_timeout), and buffer sizes."},
    {"title": "SSL/TLS Certificate Errors", "category": "networking",
     "content": "TLS errors: expired certificate, wrong CN/SAN, untrusted CA, or protocol mismatch. "
                "Verify with 'openssl s_client -connect host:443'. Renew with certbot or your CA. "
                "Check the full chain is served.  For Kubernetes, use cert-manager for auto-renewal."},
    {"title": "Prometheus Scrape Targets Down", "category": "monitoring",
     "content": "Prometheus targets showing as DOWN: verify the target's /metrics endpoint responds, "
                "check network connectivity from Prometheus to the target, verify scrape_interval and "
                "scrape_timeout, and check ServiceMonitor/PodMonitor labels match. Use the Prometheus UI "
                "Status > Targets page for details."},
    {"title": "Grafana Dashboard No Data", "category": "monitoring",
     "content": "Grafana dashboards showing 'No data': verify the data source connection in Settings, "
                "check the time range picker, ensure Prometheus is actually collecting the metric "
                "(query in Prometheus UI first), and check variable selectors.  Test with a simple query "
                "like up{} to isolate the issue."},
    {"title": "Alertmanager Not Sending Notifications", "category": "monitoring",
     "content": "Alertmanager not firing: check alertmanager.yml receivers config, verify route matching, "
                "check inhibit_rules aren't silencing alerts, verify external connectivity (e.g., SMTP, "
                "PagerDuty API). Use amtool to test route matching: amtool config routes test <labels>."},
    {"title": "ELK Stack Setup", "category": "monitoring",
     "content": "ELK (Elasticsearch, Logstash, Kibana) setup tips: allocate enough heap for ES "
                "(half of RAM, max 32GB), use index lifecycle management for retention, configure Logstash "
                "pipelines for parsing, and set up Kibana index patterns. For Kubernetes, consider Fluentd "
                "or Fluent Bit as the log shipper (EFK stack)."},
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    content: str
    title: str
    category: str
    score: float  # similarity score (higher = more relevant)
    chunk_id: str = ""


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------
class EmbeddingProvider:
    """Wraps sentence-transformers for embedding generation."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if HAS_SENTENCE_TRANSFORMERS:
            self._model = SentenceTransformer(model_name)
        else:
            self._model = None
            console.print("[yellow]sentence-transformers not installed; using hash-based mock embeddings[/yellow]")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return self._model.encode(texts, show_progress_bar=False).tolist()
        # Fallback: deterministic pseudo-embeddings (for demo only)
        dim = 384
        result = []
        for t in texts:
            h = hashlib.sha256(t.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim * 2), 2)]
            vec += [0.0] * (dim - len(vec))
            result.append(vec)
        return result


# ---------------------------------------------------------------------------
# DevOpsRAGPipeline
# ---------------------------------------------------------------------------
class DevOpsRAGPipeline:
    """Full RAG pipeline: ingest, embed, store, retrieve, generate."""

    CHUNK_SIZE = 300  # approximate token count per chunk
    CHUNK_OVERLAP = 50

    def __init__(self, persist_dir: str = "./chroma_devops_db"):
        self.persist_dir = persist_dir
        self.embedder = EmbeddingProvider()
        self.documents: list[dict] = []
        self._init_chroma()

    # ------------------------------------------------------------------
    # ChromaDB initialisation
    # ------------------------------------------------------------------
    def _init_chroma(self):
        if HAS_CHROMA:
            self._chroma_client = chromadb.Client(ChromaSettings(
                persist_directory=self.persist_dir,
                anonymized_telemetry=False,
            ))
            self.collection = self._chroma_client.get_or_create_collection(
                name="devops_kb",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            # In-memory fallback when chromadb is not installed
            self.collection = None
            self._inmemory_docs: list[dict] = []
            self._inmemory_vecs: list[list[float]] = []
            console.print("[yellow]chromadb not installed; using in-memory vector store[/yellow]")

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_text(text: str, size: int = 300, overlap: int = 50) -> list[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + size
            chunks.append(" ".join(words[start:end]))
            start += size - overlap
        return chunks if chunks else [text]

    # ------------------------------------------------------------------
    # Ingest documents
    # ------------------------------------------------------------------
    def ingest(self, documents: Optional[list[dict]] = None):
        """Load documents into the vector store."""
        docs = documents or DEMO_KNOWLEDGE_BASE
        self.documents = docs

        all_chunks: list[str] = []
        all_metas: list[dict] = []
        all_ids: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Ingesting documents...", total=len(docs))
            for doc in docs:
                chunks = self._chunk_text(doc["content"])
                for i, chunk in enumerate(chunks):
                    cid = hashlib.md5(f"{doc['title']}_{i}".encode()).hexdigest()
                    all_chunks.append(chunk)
                    all_metas.append({
                        "title": doc["title"],
                        "category": doc.get("category", "general"),
                        "chunk_index": i,
                    })
                    all_ids.append(cid)
                progress.advance(task)

        # Embed
        console.print("[dim]Generating embeddings...[/dim]")
        embeddings = self.embedder.embed(all_chunks)

        # Store
        if self.collection is not None:
            self.collection.upsert(
                ids=all_ids,
                documents=all_chunks,
                metadatas=all_metas,
                embeddings=embeddings,
            )
        else:
            self._inmemory_docs = [
                {"id": aid, "document": doc, "metadata": meta}
                for aid, doc, meta in zip(all_ids, all_chunks, all_metas)
            ]
            self._inmemory_vecs = embeddings

        console.print(f"[green]Ingested {len(all_chunks)} chunks from {len(docs)} documents.[/green]")

    # ------------------------------------------------------------------
    # rebuild_index  (exercise requirement)
    # ------------------------------------------------------------------
    def rebuild_index(self, documents: Optional[list[dict]] = None):
        """Delete existing collection and re-ingest everything."""
        console.print("[yellow]Rebuilding index from scratch...[/yellow]")
        if HAS_CHROMA:
            try:
                self._chroma_client.delete_collection("devops_kb")
            except Exception:
                pass
            self.collection = self._chroma_client.get_or_create_collection(
                name="devops_kb",
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._inmemory_docs = []
            self._inmemory_vecs = []
        self.ingest(documents)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        a_np, b_np = np.array(a), np.array(b)
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-10))

    def _retrieve_basic(self, query_emb: list[float], k: int, category: Optional[str]) -> list[dict]:
        """Basic nearest-neighbour retrieval."""
        if self.collection is not None:
            where_filter = {"category": category} if category else None
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=k,
                where=where_filter,
            )
            items = []
            for i in range(len(results["ids"][0])):
                items.append({
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - (results["distances"][0][i] if results["distances"] else 0),
                })
            return items
        else:
            # In-memory fallback
            scored = []
            for doc_rec, vec in zip(self._inmemory_docs, self._inmemory_vecs):
                if category and doc_rec["metadata"].get("category") != category:
                    continue
                sim = self._cosine_sim(query_emb, vec)
                scored.append({**doc_rec, "score": sim})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:k]

    # ------------------------------------------------------------------
    # MMR retrieval  (exercise requirement)
    # ------------------------------------------------------------------
    def retrieve_mmr(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 10,
        lambda_mult: float = 0.7,
        category: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Maximal Marginal Relevance retrieval for diverse results.

        lambda_mult closer to 1 => more relevance-focused
        lambda_mult closer to 0 => more diversity-focused
        """
        query_emb = self.embedder.embed([query])[0]

        # Fetch a broader set first
        candidates = self._retrieve_basic(query_emb, fetch_k, category)
        if not candidates:
            return []

        # Re-embed candidate texts for MMR scoring
        cand_texts = [c["document"] for c in candidates]
        cand_embs = self.embedder.embed(cand_texts)

        selected_indices: list[int] = []
        remaining = list(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = None
            best_score = -float("inf")
            for idx in remaining:
                relevance = self._cosine_sim(query_emb, cand_embs[idx])
                # Max similarity to already-selected docs
                if selected_indices:
                    max_sim = max(
                        self._cosine_sim(cand_embs[idx], cand_embs[s])
                        for s in selected_indices
                    )
                else:
                    max_sim = 0.0
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)

        results = []
        for idx in selected_indices:
            c = candidates[idx]
            results.append(RetrievedChunk(
                content=c["document"],
                title=c["metadata"]["title"],
                category=c["metadata"].get("category", "general"),
                score=round(c["score"], 4),
                chunk_id=c.get("id", ""),
            ))
        return results

    # ------------------------------------------------------------------
    # Simple retrieval (non-MMR)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 4, category: Optional[str] = None) -> list[RetrievedChunk]:
        query_emb = self.embedder.embed([query])[0]
        items = self._retrieve_basic(query_emb, k, category)
        return [
            RetrievedChunk(
                content=it["document"],
                title=it["metadata"]["title"],
                category=it["metadata"].get("category", "general"),
                score=round(it["score"], 4),
                chunk_id=it.get("id", ""),
            )
            for it in items
        ]

    # ------------------------------------------------------------------
    # Generate answer using RAG
    # ------------------------------------------------------------------
    def query(self, question: str, k: int = 4, category: Optional[str] = None, use_mmr: bool = True) -> str:
        """End-to-end RAG: retrieve context, then generate an answer."""
        if use_mmr:
            chunks = self.retrieve_mmr(question, k=k, category=category)
        else:
            chunks = self.retrieve(question, k=k, category=category)

        if not chunks:
            return "No relevant documents found in the knowledge base."

        context = "\n\n---\n".join(
            f"[{c.title} | {c.category}]\n{c.content}" for c in chunks
        )

        messages = [
            {"role": "system", "content": (
                "You are an expert DevOps troubleshooting assistant. "
                "Answer the user's question using ONLY the provided context. "
                "If the context doesn't contain the answer, say so honestly. "
                "Provide clear, actionable steps."
            )},
            {"role": "user", "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Provide a clear, step-by-step answer."
            )},
        ]

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """Return index statistics."""
        cat_counts: dict[str, int] = {}
        total_chunks = 0

        if self.collection is not None:
            all_data = self.collection.get()
            total_chunks = len(all_data["ids"])
            for meta in all_data["metadatas"]:
                cat = meta.get("category", "general")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
        else:
            total_chunks = len(self._inmemory_docs)
            for d in self._inmemory_docs:
                cat = d["metadata"].get("category", "general")
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": total_chunks,
            "categories": cat_counts,
        }

    def display_stats(self):
        s = self.stats()
        table = Table(title="RAG Pipeline Statistics", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total Documents", str(s["total_documents"]))
        table.add_row("Total Chunks", str(s["total_chunks"]))
        for cat, count in sorted(s["categories"].items()):
            table.add_row(f"  [{cat}]", str(count))
        console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    console.print(Panel(
        "[bold]Module 7 - Exercise 2: RAG Pipeline[/bold]\n"
        "[dim]Retrieval-Augmented Generation for DevOps troubleshooting[/dim]",
        border_style="bright_cyan",
    ))

    pipeline = DevOpsRAGPipeline(persist_dir="./chroma_devops_m7")

    # Ingest demo knowledge base
    pipeline.ingest()
    console.print()
    pipeline.display_stats()
    console.print()

    # Demo queries
    demo_queries = [
        ("My pod keeps crashing with OOMKilled error", None),
        ("Docker build is slow, how to use cache?", "docker"),
        ("Prometheus targets are all showing as down", "monitoring"),
        ("GitHub Actions workflow keeps failing", "cicd"),
    ]

    for q, cat in demo_queries:
        console.print(Panel(f"[bold]Q:[/bold] {q}" + (f"  [dim](category filter: {cat})[/dim]" if cat else ""),
                            border_style="yellow"))
        chunks = pipeline.retrieve_mmr(q, k=3, category=cat)
        for i, chunk in enumerate(chunks, 1):
            console.print(f"  [cyan]Source {i}:[/cyan] {chunk.title} [dim](score: {chunk.score:.3f})[/dim]")
            console.print(f"    [dim]{chunk.content[:120]}...[/dim]")
        console.print()

    # Interactive mode
    console.print("[bold green]Interactive query mode[/bold green] — type a question or 'quit' to exit.\n")
    while True:
        try:
            question = console.input("[bold]Question> [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        console.print("[dim]Retrieving and generating...[/dim]")
        answer = pipeline.query(question, use_mmr=True)
        console.print(Panel(answer, title="Answer", border_style="green"))

        # Show sources
        chunks = pipeline.retrieve_mmr(question, k=3)
        console.print("[dim]Sources:[/dim]")
        for c in chunks:
            console.print(f"  [cyan]{c.title}[/cyan] ({c.category}) — score {c.score:.3f}")
        console.print()

    console.print("[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exercise 4: Assemble the Complete Chatbot
==========================================
Module 7 (Capstone) - Exercise 4 of 4

THE GRAND FINALE!

WHAT YOU'LL BUILD:
    A fully working DevOps troubleshooting chatbot that combines EVERYTHING
    from Modules 1-6 and Exercises 1-3:
    - Error classification (Exercise 1) -> knows what category
    - RAG retrieval (Exercise 2) -> finds relevant docs
    - Prompt engineering (Exercise 3) -> builds optimal prompts
    - LLM generation (Module 1) -> generates the answer
    - Rich CLI (Module 2) -> beautiful terminal interface

    The complete flow:
    User types question
      -> Classify the question (terraform? kubernetes? docker? cicd?)
      -> Retrieve relevant docs from knowledge base (filtered by category)
      -> Build an optimized prompt (question + context + history)
      -> Send to LLM (OpenAI or Ollama)
      -> Stream the response with a beautiful UI
      -> Show sources and category badge

WHY THIS MATTERS:
    This is how REAL production chatbots work! Companies like GitHub Copilot,
    Cursor, and AWS Q all use this same pattern:
    Classify -> Retrieve -> Prompt -> Generate -> Display

WHAT YOU'LL LEARN:
    - How to compose multiple components into a working application
    - The main loop: input -> classify -> retrieve -> prompt -> generate -> display
    - Adding slash commands for power users
    - Error handling and graceful degradation
    - Streaming responses for better UX

PREREQUISITES:
    pip install rich openai chromadb sentence-transformers python-dotenv tiktoken

SUPPORTS:
    - OpenAI (set OPENAI_API_KEY in .env)
    - Ollama (set USE_OLLAMA=true in .env, run: ollama serve)

RUN THIS FILE:
    python ex4_complete_chatbot.py

THIS FILE IS SELF-CONTAINED:
    It includes simplified versions of all components so it runs standalone.
    You don't need the other exercise files (but they show the full versions).
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from datetime import datetime

# Rich library for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.live import Live
from rich.spinner import Spinner
from rich.markdown import Markdown

# python-dotenv for loading API keys from .env file
from dotenv import load_dotenv

# Load environment variables FIRST (before we check for API keys)
load_dotenv()

# ChromaDB for vector storage (try to import, gracefully handle if missing)
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

# Sentence transformers for embeddings (try to import)
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

# tiktoken for token counting
try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# Rich console - our main output tool
console = Console()


# ============================================================================
# ASCII ART WELCOME SCREEN
# ============================================================================
WELCOME_ART = r"""
    ____             ____               ____        __
   / __ \___  _   __/ __ \____  _____  / __ )____  / /_
  / / / / _ \| | / / / / / __ \/ ___/ / __  / __ \/ __/
 / /_/ /  __/| |/ / /_/ / /_/ (__  ) / /_/ / /_/ / /_
/_____/\___/ |___/\____/ .___/____/ /_____/\____/\__/
                       /_/
"""

WELCOME_SUBTITLE = (
    "Your AI-powered DevOps troubleshooting assistant\n"
    "Ask about Terraform, Kubernetes, Docker, or CI/CD issues!"
)


# ============================================================================
# CATEGORY DEFINITIONS (simplified from Exercise 1)
# ============================================================================
CATEGORY_KEYWORDS = {
    "terraform": {
        "keywords": {
            "terraform": 3, "tf": 3, "hcl": 3, "tfstate": 3, "tfvars": 3,
            "terraform init": 3, "terraform plan": 3, "terraform apply": 3,
            "provider": 2, "state file": 2, "state lock": 2, "module": 2,
            "workspace": 2, "backend": 2, "infrastructure": 1, "aws": 1,
        },
        "color": "purple",
        "emoji": "TF",
    },
    "kubernetes": {
        "keywords": {
            "kubernetes": 3, "k8s": 3, "kubectl": 3, "kubelet": 3, "helm": 3,
            "pod": 2, "deployment": 2, "service": 2, "ingress": 2, "namespace": 2,
            "configmap": 2, "crashloopbackoff": 2, "imagepullbackoff": 2,
            "node": 2, "cluster": 2, "container": 1, "yaml": 1,
        },
        "color": "blue",
        "emoji": "K8",
    },
    "docker": {
        "keywords": {
            "docker": 3, "dockerfile": 3, "docker-compose": 3, "docker compose": 3,
            "docker build": 3, "docker run": 3, "dockerhub": 3,
            "container": 2, "image": 2, "volume": 2, "layer": 2,
            "registry": 2, "entrypoint": 2, "build": 1, "cache": 1,
        },
        "color": "cyan",
        "emoji": "DK",
    },
    "cicd": {
        "keywords": {
            "github actions": 3, "workflow": 3, "pipeline": 3, "jenkins": 3,
            "gitlab ci": 3, "ci/cd": 3, "cicd": 3, ".github/workflows": 3,
            "deploy": 2, "artifact": 2, "runner": 2, "stage": 2,
            "continuous": 2, "action": 2, "test": 1, "release": 1,
        },
        "color": "green",
        "emoji": "CI",
    },
    "general": {
        "keywords": {
            "devops": 1, "best practice": 1, "recommend": 1, "compare": 1,
            "how to": 1, "what is": 1, "explain": 1,
        },
        "color": "white",
        "emoji": "GN",
    },
}

# Category descriptions for the /category command
CATEGORY_DESCRIPTIONS = {
    "terraform": "Infrastructure as Code - state, providers, modules, HCL",
    "kubernetes": "Container orchestration - pods, deployments, services, networking",
    "docker": "Containerization - Dockerfile, build, compose, images",
    "cicd": "CI/CD pipelines - GitHub Actions, Jenkins, GitLab CI",
    "general": "General DevOps questions and best practices",
}


# ============================================================================
# BUILT-IN ERROR CLASSIFIER
# ============================================================================
class ErrorClassifier:
    """
    Simplified error classifier (full version in Exercise 1).
    Classifies DevOps questions into categories using keyword matching.
    """

    def classify(self, query: str) -> Tuple[str, float]:
        """Classify a query into a category with confidence score."""
        query_lower = query.lower()
        scores = {}

        for category, info in CATEGORY_KEYWORDS.items():
            score = 0
            for keyword, weight in info["keywords"].items():
                if keyword.lower() in query_lower:
                    score += weight
            scores[category] = score

        # Find the best category
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]

        if best_score == 0:
            return "general", 0.0

        # Calculate confidence
        max_possible = sum(CATEGORY_KEYWORDS[best_cat]["keywords"].values())
        confidence = best_score / max_possible if max_possible > 0 else 0.0

        return best_cat, confidence


# ============================================================================
# BUILT-IN RAG (simplified - full version in Exercise 2)
# ============================================================================
class SimpleRAG:
    """
    Simplified RAG component for the chatbot.
    Uses ChromaDB + HuggingFace embeddings if available, otherwise uses demo data.

    This is a self-contained version so ex4 runs without ex2.
    """

    def __init__(self):
        """Initialize the RAG component."""
        self.has_rag = HAS_CHROMADB and HAS_EMBEDDINGS
        self.collection = None
        self.embedding_model = None

        if self.has_rag:
            try:
                self._init_vector_store()
            except Exception as e:
                console.print(f"[yellow]RAG init failed ({e}), using demo mode[/yellow]")
                self.has_rag = False

        if not self.has_rag:
            console.print("[yellow]Running in demo mode (no vector store)[/yellow]")
            self._init_demo_data()

    def _init_vector_store(self):
        """Initialize ChromaDB and embedding model."""
        # Load embedding model (runs locally, no API key needed)
        console.print("[cyan]Loading embedding model...[/cyan]")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize ChromaDB (in-memory for simplicity in this exercise)
        self.chroma_client = chromadb.Client(
            Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="devops_chatbot"
        )

        # Try to load knowledge base
        kb_path = Path(__file__).parent.parent.parent / "knowledge-base"
        if kb_path.exists():
            self._load_knowledge_base(kb_path)
        else:
            self._load_demo_docs()

    def _load_knowledge_base(self, kb_path: Path):
        """Load actual knowledge base markdown files."""
        md_files = list(kb_path.rglob("*.md"))
        if not md_files:
            self._load_demo_docs()
            return

        console.print(f"[cyan]Loading {len(md_files)} knowledge base files...[/cyan]")

        ids = []
        texts = []
        metadatas = []

        for i, md_file in enumerate(md_files):
            try:
                content = md_file.read_text(encoding="utf-8")

                # Determine category from path
                try:
                    relative = md_file.relative_to(kb_path)
                    category = relative.parts[0] if len(relative.parts) > 1 else "general"
                except ValueError:
                    category = "general"

                # Simple chunking: split on double newlines
                chunks = content.split("\n\n")
                for j, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    if len(chunk) > 50:  # Skip tiny chunks
                        ids.append(f"doc_{i}_{j}")
                        texts.append(chunk)
                        metadatas.append({
                            "category": category,
                            "source_file": md_file.name,
                        })
            except Exception:
                pass

        if texts:
            # Generate embeddings
            console.print("[cyan]Generating embeddings...[/cyan]")
            embeddings = self.embedding_model.encode(texts).tolist()

            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            console.print(f"[green]Indexed {len(texts)} chunks from knowledge base[/green]")
        else:
            self._load_demo_docs()

    def _load_demo_docs(self):
        """Load demo documents into ChromaDB."""
        demo_docs = [
            ("CrashLoopBackOff means your container keeps crashing. Check logs: kubectl logs <pod> --previous. Common causes: OOMKilled (exit 137), missing env vars, failed liveness probe.", "kubernetes", "pod-errors.md"),
            ("Terraform state lock error: Another process holds the lock. Fix: terraform force-unlock LOCK_ID. Prevention: use remote state with locking.", "terraform", "state-errors.md"),
            ("Docker build COPY failed: File is outside build context or in .dockerignore. Fix: check build context path, check .dockerignore, use relative paths.", "docker", "build-errors.md"),
            ("GitHub Actions permission denied: Add permissions block to workflow YAML. Check GITHUB_TOKEN scope. Secrets not available in fork PRs.", "cicd", "actions-errors.md"),
            ("ImagePullBackOff: Kubernetes can't pull your container image. Check: image name/tag correct, registry accessible, imagePullSecrets configured for private registries.", "kubernetes", "image-errors.md"),
            ("Terraform provider error: Run terraform init to download providers. Check required_providers version constraints. Set auth via env vars.", "terraform", "provider-errors.md"),
            ("Docker container exits immediately: Check CMD/ENTRYPOINT runs a long-lived process. Add -it for interactive. Check logs: docker logs <container>.", "docker", "container-errors.md"),
            ("Kubernetes OOMKilled (exit code 137): Container exceeded memory limit. Fix: increase resources.limits.memory in pod spec. Monitor with kubectl top pod.", "kubernetes", "oom-errors.md"),
            ("Terraform destroy fails: Resources have dependencies. Use -target flag to destroy specific resources. Check for manual changes outside Terraform.", "terraform", "destroy-errors.md"),
            ("Pipeline timeout: Check if tests hang, increase timeout value, add caching for dependencies, parallelize build steps.", "cicd", "pipeline-errors.md"),
        ]

        if self.has_rag and self.embedding_model:
            ids = [f"demo_{i}" for i in range(len(demo_docs))]
            texts = [d[0] for d in demo_docs]
            metadatas = [{"category": d[1], "source_file": d[2]} for d in demo_docs]

            embeddings = self.embedding_model.encode(texts).tolist()
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            console.print(f"[green]Loaded {len(demo_docs)} demo documents[/green]")

    def _init_demo_data(self):
        """Initialize with simple keyword-based demo data (no vector store)."""
        self.demo_docs = {
            "terraform": [
                {"content": "Terraform state lock error: Another process holds the lock. Fix with: terraform force-unlock LOCK_ID. Always use remote state with locking enabled.", "source_file": "state-errors.md"},
                {"content": "Terraform provider errors: Run 'terraform init' to download providers. Check required_providers version. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.", "source_file": "provider-errors.md"},
            ],
            "kubernetes": [
                {"content": "CrashLoopBackOff: Container keeps crashing. Debug: kubectl logs <pod> --previous, kubectl describe pod <pod>. Causes: app crash, OOMKilled (exit 137), missing ConfigMap/Secret.", "source_file": "pod-errors.md"},
                {"content": "ImagePullBackOff: Can't pull container image. Check image name/tag, registry access, imagePullSecrets for private registries.", "source_file": "image-errors.md"},
            ],
            "docker": [
                {"content": "Docker build COPY failed: File outside build context or in .dockerignore. Use relative paths, check .dockerignore, verify file exists.", "source_file": "build-errors.md"},
                {"content": "Docker container exits immediately: Ensure CMD runs a foreground process. Use 'docker logs' to check errors. Add '-it' for interactive mode.", "source_file": "container-errors.md"},
            ],
            "cicd": [
                {"content": "GitHub Actions permission denied: Add permissions block (contents: write, pull-requests: write). Check GITHUB_TOKEN scope. Secrets unavailable in forks.", "source_file": "actions-errors.md"},
                {"content": "Pipeline build timeout: Check for hanging tests, increase timeout, add dependency caching, parallelize build steps.", "source_file": "pipeline-errors.md"},
            ],
        }

    def retrieve(self, query: str, category: Optional[str] = None, k: int = 3) -> List[Dict]:
        """Retrieve relevant documents for a query."""
        if self.has_rag and self.collection and self.collection.count() > 0:
            return self._retrieve_vector(query, category, k)
        else:
            return self._retrieve_demo(query, category, k)

    def _retrieve_vector(self, query: str, category: Optional[str], k: int) -> List[Dict]:
        """Retrieve using vector search (ChromaDB)."""
        query_embedding = self.embedding_model.encode(query).tolist()

        where_filter = None
        if category and category != "general":
            where_filter = {"category": category}

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fallback without filter
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                dist = results["distances"][0][i] if results["distances"] else 0
                formatted.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "relevance": 1.0 / (1.0 + dist),
                })
        return formatted

    def _retrieve_demo(self, query: str, category: Optional[str], k: int) -> List[Dict]:
        """Retrieve using simple keyword matching (fallback when no vector store)."""
        results = []
        search_cats = [category] if category and category in self.demo_docs else self.demo_docs.keys()

        for cat in search_cats:
            if cat not in self.demo_docs:
                continue
            for doc in self.demo_docs[cat]:
                # Simple relevance: count matching words
                query_words = set(query.lower().split())
                doc_words = set(doc["content"].lower().split())
                overlap = len(query_words & doc_words)
                if overlap > 0:
                    results.append({
                        "content": doc["content"],
                        "metadata": {"category": cat, "source_file": doc["source_file"]},
                        "relevance": overlap / len(query_words) if query_words else 0,
                    })

        # Sort by relevance and return top k
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:k]

    def get_stats(self) -> Dict:
        """Get stats about the indexed documents."""
        if self.has_rag and self.collection:
            return {
                "mode": "Vector Store (ChromaDB)",
                "total_documents": self.collection.count(),
            }
        else:
            total = sum(len(docs) for docs in getattr(self, "demo_docs", {}).values())
            return {
                "mode": "Demo Mode (keyword matching)",
                "total_documents": total,
            }


# ============================================================================
# SYSTEM PROMPT
# ============================================================================
SYSTEM_PROMPT = (
    "You are a senior DevOps troubleshooting expert. You help users diagnose and fix "
    "issues with Terraform, Kubernetes, Docker, and CI/CD pipelines.\n\n"
    "Guidelines:\n"
    "1. Start with the most likely cause and solution\n"
    "2. Include exact commands the user can run\n"
    "3. Explain WHY the error happens, not just how to fix it\n"
    "4. Use code blocks for commands and config examples\n"
    "5. If you need more info, ask clarifying questions\n"
    "6. Be concise but thorough\n"
)


# ============================================================================
# THE COMPLETE CHATBOT
# ============================================================================
class DevOpsChatbot:
    """
    The Complete DevOps Troubleshooting Chatbot!

    This class ties together ALL the components:
    - ErrorClassifier: Detects what category the question is about
    - SimpleRAG: Retrieves relevant documentation
    - Prompt assembly: Combines question + context + history
    - LLM generation: Calls OpenAI or Ollama to generate answers
    - Rich display: Beautiful terminal output with panels and colors

    THE MAIN FLOW:
    1. User types a question
    2. ErrorClassifier determines the category (terraform/k8s/docker/cicd)
    3. SimpleRAG retrieves relevant docs (filtered by category)
    4. We assemble a prompt: system prompt + context + history + question
    5. We send it to the LLM (OpenAI or Ollama)
    6. We stream the response to the terminal
    7. We show the category badge and source documents
    """

    def __init__(self):
        """
        Initialize all chatbot components.

        This is where we wire everything together - like plugging in
        all the components of a stereo system.
        """
        # ---- Determine LLM backend ----
        self.use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"

        if self.use_ollama:
            self.model = os.getenv("OLLAMA_MODEL", "llama2")
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        else:
            self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            # Validate API key exists
            if not os.getenv("OPENAI_API_KEY"):
                console.print(
                    "[yellow]Warning: OPENAI_API_KEY not set. "
                    "Set it in .env or use USE_OLLAMA=true for local LLM.[/yellow]"
                )

        # ---- Initialize components ----
        # 1. Error Classifier (from Exercise 1)
        self.classifier = ErrorClassifier()

        # 2. RAG pipeline (simplified from Exercise 2)
        console.print("[cyan]Initializing RAG engine...[/cyan]")
        self.rag = SimpleRAG()

        # 3. Conversation history (managed like Exercise 3)
        self.history: List[Dict[str, str]] = []
        self.max_history = 10  # Keep last 10 exchanges

        # 4. LLM client
        self._init_llm()

        # 5. Session stats
        self.stats = {
            "questions_asked": 0,
            "categories_seen": {},
            "session_start": datetime.now(),
        }

        # Store last response sources for /sources command
        self.last_sources: List[Dict] = []
        self.last_category: str = "general"

    def _init_llm(self):
        """Initialize the LLM client (OpenAI or Ollama)."""
        if self.use_ollama:
            # Ollama uses HTTP requests - no special client needed
            import requests
            self.ollama_session = requests.Session()
            console.print(f"[green]LLM: Ollama ({self.model})[/green]")
        else:
            # OpenAI uses their Python client
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                console.print(f"[green]LLM: OpenAI ({self.model})[/green]")
            except ImportError:
                console.print("[red]OpenAI package not installed: pip install openai[/red]")
                self.openai_client = None
            except Exception as e:
                console.print(f"[red]OpenAI init failed: {e}[/red]")
                self.openai_client = None

    def _call_llm(self, messages: List[Dict]) -> Generator[str, None, None]:
        """
        Call the LLM and yield response chunks (streaming).

        This is the core LLM interaction. It supports both backends:
        - OpenAI: Uses their streaming API
        - Ollama: Uses HTTP streaming

        Args:
            messages: List of message dicts [{"role": "...", "content": "..."}]

        Yields:
            Response text chunks as they arrive
        """
        if self.use_ollama:
            yield from self._call_ollama(messages)
        else:
            yield from self._call_openai(messages)

    def _call_openai(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Call OpenAI API with streaming."""
        if not hasattr(self, "openai_client") or self.openai_client is None:
            yield "Error: OpenAI client not initialized. Check your API key."
            return

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            yield f"\n\nError calling OpenAI: {e}"

    def _call_ollama(self, messages: List[Dict]) -> Generator[str, None, None]:
        """Call Ollama API with streaming."""
        try:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": 0.7},
            }

            response = self.ollama_session.post(url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]

        except Exception as e:
            yield f"\n\nError calling Ollama: {e}"
            yield "\nMake sure Ollama is running: ollama serve"

    def process_question(self, question: str) -> Generator[str, None, None]:
        """
        Process a user question through the full pipeline.

        THIS IS THE HEART OF THE CHATBOT - the complete flow:
        1. Classify the question
        2. Retrieve relevant context
        3. Build the prompt
        4. Generate the response
        5. Update history and stats

        Args:
            question: The user's question

        Yields:
            Response text chunks (for streaming display)
        """
        # ---- Step 1: Classify the question ----
        category, confidence = self.classifier.classify(question)
        self.last_category = category

        # Track stats
        self.stats["questions_asked"] += 1
        self.stats["categories_seen"][category] = (
            self.stats["categories_seen"].get(category, 0) + 1
        )

        # Show the classification
        cat_info = CATEGORY_KEYWORDS.get(category, {})
        color = cat_info.get("color", "white")
        badge = cat_info.get("emoji", "??")
        console.print(
            f"  [dim]Category:[/dim] [{color} bold][{badge}] {category}[/{color} bold] "
            f"[dim](confidence: {confidence:.0%})[/dim]"
        )

        # ---- Step 2: Retrieve relevant docs ----
        context_docs = self.rag.retrieve(question, category=category, k=3)
        self.last_sources = context_docs

        if context_docs:
            console.print(
                f"  [dim]Found {len(context_docs)} relevant docs[/dim]"
            )
        else:
            console.print("  [dim]No relevant docs found, using LLM knowledge[/dim]")

        # ---- Step 3: Build the prompt ----
        # Format context for the prompt
        context_str = ""
        if context_docs:
            context_parts = []
            for i, doc in enumerate(context_docs, 1):
                source = doc.get("metadata", {}).get("source_file", "unknown")
                context_parts.append(
                    f"--- Source: {source} ---\n{doc['content']}"
                )
            context_str = "\n\n".join(context_parts)

        # Format history
        history_str = "No previous conversation."
        if self.history:
            history_lines = []
            for msg in self.history[-10:]:  # Last 10 messages
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                history_lines.append(f"{role}: {msg['content']}")
            history_str = "\n".join(history_lines)

        # Assemble the user prompt
        if context_str:
            user_prompt = (
                f"Use the following documentation to answer the user's question.\n\n"
                f"DOCUMENTATION CONTEXT:\n{context_str}\n\n"
                f"CONVERSATION HISTORY:\n{history_str}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"Provide a helpful, accurate response:"
            )
        else:
            user_prompt = (
                f"CONVERSATION HISTORY:\n{history_str}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"Provide a helpful response based on your DevOps knowledge:"
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # ---- Step 4: Generate the response ----
        full_response = ""
        for chunk in self._call_llm(messages):
            full_response += chunk
            yield chunk

        # ---- Step 5: Update history ----
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": full_response})

        # Trim history if too long
        max_entries = self.max_history * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    # ========================================================================
    # SLASH COMMANDS
    # ========================================================================

    def cmd_help(self) -> None:
        """Display help information."""
        help_text = (
            "[bold]Available Commands:[/bold]\n\n"
            "  [green]/help[/green]      - Show this help message\n"
            "  [green]/clear[/green]     - Clear conversation history\n"
            "  [green]/history[/green]   - Show conversation history\n"
            "  [green]/stats[/green]     - Show session statistics\n"
            "  [green]/category[/green]  - Show all categories and descriptions\n"
            "  [green]/sources[/green]   - Show sources for the last response\n"
            "  [green]quit[/green]       - Exit the chatbot\n\n"
            "[bold]Tips:[/bold]\n"
            "  - Just type your DevOps question naturally\n"
            "  - The bot auto-detects the category (Terraform, K8s, Docker, CI/CD)\n"
            "  - Follow-up questions use conversation history for context"
        )
        console.print(Panel(help_text, title="[bold]Help[/bold]", border_style="green"))

    def cmd_clear(self) -> None:
        """Clear conversation history."""
        self.history = []
        self.last_sources = []
        console.print("[yellow]Conversation history cleared.[/yellow]")

    def cmd_history(self) -> None:
        """Show conversation history."""
        if not self.history:
            console.print("[dim]No conversation history yet.[/dim]")
            return

        for msg in self.history:
            if msg["role"] == "user":
                console.print(f"  [green bold]You:[/green bold] {msg['content']}")
            else:
                # Truncate long assistant responses for display
                content = msg["content"]
                if len(content) > 150:
                    content = content[:150] + "..."
                console.print(f"  [blue bold]Bot:[/blue bold] {content}")

    def cmd_stats(self) -> None:
        """Show session statistics."""
        elapsed = datetime.now() - self.stats["session_start"]
        rag_stats = self.rag.get_stats()

        table = Table(title="Session Statistics", show_header=True)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("Questions Asked", str(self.stats["questions_asked"]))
        table.add_row("Session Duration", str(elapsed).split(".")[0])
        table.add_row("LLM Backend", f"{'Ollama' if self.use_ollama else 'OpenAI'} ({self.model})")
        table.add_row("RAG Mode", rag_stats["mode"])
        table.add_row("Indexed Documents", str(rag_stats["total_documents"]))
        table.add_row("History Length", f"{len(self.history)} messages")

        # Category breakdown
        if self.stats["categories_seen"]:
            cat_str = ", ".join(
                f"{cat}: {count}"
                for cat, count in sorted(
                    self.stats["categories_seen"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            table.add_row("Categories Used", cat_str)

        console.print(table)

    def cmd_category(self) -> None:
        """Show all available categories."""
        table = Table(title="DevOps Categories", show_header=True)
        table.add_column("Badge", justify="center")
        table.add_column("Category", style="bold")
        table.add_column("Description")
        table.add_column("Keywords (sample)", style="dim")

        for cat, desc in CATEGORY_DESCRIPTIONS.items():
            info = CATEGORY_KEYWORDS.get(cat, {})
            color = info.get("color", "white")
            badge = info.get("emoji", "??")

            # Get top 5 keywords
            keywords = list(info.get("keywords", {}).keys())[:5]
            kw_str = ", ".join(keywords)

            table.add_row(
                f"[{color} bold][{badge}][/{color} bold]",
                f"[{color}]{cat}[/{color}]",
                desc,
                kw_str,
            )

        console.print(table)

    def cmd_sources(self) -> None:
        """Show sources for the last response."""
        if not self.last_sources:
            console.print("[dim]No sources available. Ask a question first![/dim]")
            return

        table = Table(title="Sources for Last Response", show_header=True)
        table.add_column("#", justify="center", style="bold")
        table.add_column("Source File", style="cyan")
        table.add_column("Category", style="bold")
        table.add_column("Relevance", justify="right")
        table.add_column("Content Preview", style="dim")

        for i, doc in enumerate(self.last_sources, 1):
            meta = doc.get("metadata", {})
            source = meta.get("source_file", "unknown")
            category = meta.get("category", "unknown")
            relevance = doc.get("relevance", 0)
            color = CATEGORY_KEYWORDS.get(category, {}).get("color", "white")

            # Truncate content for preview
            content = doc.get("content", "")
            if len(content) > 80:
                content = content[:80] + "..."

            table.add_row(
                str(i),
                source,
                f"[{color}]{category}[/{color}]",
                f"{relevance:.2%}",
                content,
            )

        console.print(table)

    # ========================================================================
    # TODO: Add /export command to save conversation to markdown
    # ========================================================================
    def cmd_export(self) -> None:
        """
        TODO: Export the conversation to a markdown file.

        INSTRUCTIONS:
        1. Create a filename with timestamp (e.g., "chat_2024-01-15_14-30-00.md")
        2. Write a markdown file with:
           - Header with date/time and model info
           - Each exchange formatted as:
             ## User
             <question>
             ## Assistant
             <response>
        3. Save to the current directory
        4. Print the filename so the user knows where it was saved

        HINT:
            filename = f"chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
            with open(filename, 'w') as f:
                f.write("# DevOps Chatbot Conversation\n\n")
                for msg in self.history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    f.write(f"## {role}\n{msg['content']}\n\n")
        """
        # TODO: Implement this command!
        console.print("[yellow]/export not implemented yet - this is your TODO![/yellow]")
        console.print("[dim]Save the conversation history to a markdown file.[/dim]")

    # ========================================================================
    # TODO: Add /feedback command for user to rate responses
    # ========================================================================
    def cmd_feedback(self) -> None:
        """
        TODO: Let the user rate the last response.

        INSTRUCTIONS:
        1. Ask the user for a rating (1-5 stars)
        2. Optionally ask for a text comment
        3. Save the feedback to a JSON file (append mode)
        4. Each entry should include:
           - timestamp
           - the question
           - the response (or first 200 chars)
           - the rating (1-5)
           - the comment (if any)
           - the category

        WHY FEEDBACK MATTERS:
            Collecting user feedback lets you identify:
            - Which categories get poor ratings (need better docs)
            - Which types of questions the bot struggles with
            - How to improve the system prompt or RAG pipeline

        HINT:
            feedback_file = "chatbot_feedback.json"
            feedback_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": self.history[-2]["content"] if len(self.history) >= 2 else "",
                "rating": int(rating),
                "comment": comment,
                "category": self.last_category,
            }
            # Load existing, append, save
        """
        # TODO: Implement this command!
        console.print("[yellow]/feedback not implemented yet - this is your TODO![/yellow]")
        console.print("[dim]Rate the last response to help improve the chatbot.[/dim]")


# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    """
    Run the complete DevOps Troubleshooting Chatbot.

    This is the entry point - it shows the welcome screen,
    initializes the chatbot, and runs the main interaction loop.
    """
    # ---- Welcome Screen ----
    console.print(Panel(
        f"[bold cyan]{WELCOME_ART}[/bold cyan]\n"
        f"[bold white]{WELCOME_SUBTITLE}[/bold white]\n\n"
        f"[dim]Type /help for commands, or just ask a DevOps question![/dim]",
        border_style="cyan",
        padding=(0, 2),
    ))

    # ---- Show backend info ----
    use_ollama = os.getenv("USE_OLLAMA", "").lower() == "true"
    if use_ollama:
        backend_info = f"Ollama ({os.getenv('OLLAMA_MODEL', 'llama2')})"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            backend_info = f"OpenAI ({os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')})"
        else:
            backend_info = "OpenAI (API key not set - responses may fail)"

    console.print(f"[dim]Backend: {backend_info}[/dim]")
    console.print()

    # ---- Initialize Chatbot ----
    try:
        console.print("[cyan]Initializing chatbot components...[/cyan]\n")
        chatbot = DevOpsChatbot()
        console.print("\n[green bold]Chatbot is ready! Start asking questions.[/green bold]\n")
    except Exception as e:
        console.print(f"[red]Failed to initialize chatbot: {e}[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Check .env file has OPENAI_API_KEY (or USE_OLLAMA=true)")
        console.print("  2. Run: pip install rich openai chromadb sentence-transformers python-dotenv")
        console.print("  3. For Ollama: run 'ollama serve' first")
        return

    # ---- Slash command mapping ----
    # Maps command strings to methods on the chatbot
    commands = {
        "/help": chatbot.cmd_help,
        "/clear": chatbot.cmd_clear,
        "/history": chatbot.cmd_history,
        "/stats": chatbot.cmd_stats,
        "/category": chatbot.cmd_category,
        "/sources": chatbot.cmd_sources,
        "/export": chatbot.cmd_export,
        "/feedback": chatbot.cmd_feedback,
    }

    # ---- Main Chat Loop ----
    while True:
        try:
            # Get user input with a styled prompt
            console.print()
            user_input = Prompt.ask("[bold green]You[/bold green]")

            # Handle empty input
            if not user_input.strip():
                continue

            # Handle quit
            if user_input.lower() in ["quit", "exit", "q", "/quit", "/exit"]:
                # Show session summary
                if chatbot.stats["questions_asked"] > 0:
                    console.print("\n[bold]Session Summary:[/bold]")
                    chatbot.cmd_stats()
                console.print("\n[cyan]Thanks for using DevOps Bot! Goodbye![/cyan]")
                break

            # Handle slash commands
            cmd_key = user_input.strip().lower().split()[0]
            if cmd_key in commands:
                commands[cmd_key]()
                continue

            # ---- Process the question through the full pipeline ----
            console.print()  # Blank line before response

            # Show a spinner while classifying and retrieving
            # (The actual LLM response will stream below)
            console.print("[dim]Processing...[/dim]")

            # Stream the response
            console.print(f"\n[bold blue]Assistant[/bold blue]")

            response_text = ""
            try:
                for chunk in chatbot.process_question(user_input):
                    # Print each chunk as it arrives (streaming effect)
                    console.print(chunk, end="")
                    response_text += chunk
            except Exception as e:
                console.print(f"\n[red]Error generating response: {e}[/red]")
                console.print("[dim]Try again or check your LLM configuration.[/dim]")
                continue

            console.print()  # Newline after response

            # Show source documents summary (inline, compact)
            if chatbot.last_sources:
                sources = chatbot.last_sources
                source_names = [
                    s.get("metadata", {}).get("source_file", "?")
                    for s in sources
                ]
                cat_color = CATEGORY_KEYWORDS.get(
                    chatbot.last_category, {}
                ).get("color", "white")

                console.print(
                    f"\n  [dim]Sources: {', '.join(source_names)} | "
                    f"Category: [{cat_color}]{chatbot.last_category}[/{cat_color}] | "
                    f"Type /sources for details[/dim]"
                )

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            console.print("\n\n[cyan]Interrupted. Type 'quit' to exit or keep chatting.[/cyan]")
            continue

        except Exception as e:
            # Catch-all for unexpected errors
            console.print(f"\n[red]Unexpected error: {e}[/red]")
            console.print("[dim]The chatbot will continue running. Try again.[/dim]")
            continue


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()

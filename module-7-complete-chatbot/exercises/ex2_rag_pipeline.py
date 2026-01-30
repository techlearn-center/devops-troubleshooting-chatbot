#!/usr/bin/env python3
"""
Exercise 2: Build the Complete RAG Pipeline
=============================================
Module 7 (Capstone) - Exercise 2 of 4

WHAT YOU'LL BUILD:
    The full RAG (Retrieval-Augmented Generation) pipeline that:
    1. LOADS markdown documents from the knowledge base
    2. CHUNKS them into smaller pieces (for better search)
    3. EMBEDS them as vectors (numbers that capture meaning)
    4. STORES them in a vector database (ChromaDB)
    5. RETRIEVES the most relevant chunks for any question

WHY THIS MATTERS:
    RAG is what makes our chatbot ACCURATE. Without RAG, the LLM can only
    use its training data (which might be outdated or wrong). With RAG,
    we feed it OUR documentation, so answers are grounded in real docs.

    Think of it like an open-book exam vs. a closed-book exam:
    - Without RAG: The LLM answers from memory (might hallucinate)
    - With RAG: The LLM looks up the answer in our docs (accurate!)

WHAT YOU'LL LEARN:
    - How all the RAG pieces from Modules 3-5 fit together
    - The complete flow: Load -> Chunk -> Embed -> Store -> Retrieve
    - How to build a pipeline class that manages the full lifecycle
    - Testing retrieval quality with known questions

PREREQUISITES:
    pip install rich chromadb sentence-transformers langchain langchain-community

KNOWLEDGE BASE LOCATION:
    ../../knowledge-base/  (relative to this file)
    Contains subdirectories: terraform/, kubernetes/, docker/, cicd/
    Each has markdown files with troubleshooting content.

RUN THIS FILE:
    python ex2_rag_pipeline.py
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt
from rich.text import Text
from rich.syntax import Syntax

# ChromaDB is our vector database - it stores embeddings and enables similarity search.
# Think of it like a smart filing cabinet that finds documents by MEANING, not just keywords.
import chromadb
from chromadb.config import Settings

# sentence-transformers converts text into vectors (embeddings).
# The model "all-MiniLM-L6-v2" is small, fast, and free - runs 100% locally.
# It turns text like "pod crashing" into a list of 384 numbers that capture its meaning.
from sentence_transformers import SentenceTransformer

# LangChain helps us load and split documents.
# RecursiveCharacterTextSplitter is smart about splitting - it tries to keep
# paragraphs together, then sentences, then words (in that order).
from langchain.text_splitter import RecursiveCharacterTextSplitter

# python-dotenv for environment variables
from dotenv import load_dotenv

load_dotenv()

# Rich console
console = Console()


# ============================================================================
# CONSTANTS
# ============================================================================

# Path to the knowledge base (relative to this exercise file)
# Going up from exercises/ -> module-7-complete-chatbot/ -> root -> knowledge-base/
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent.parent / "knowledge-base"

# Embedding model name - this runs locally, no API key needed!
# all-MiniLM-L6-v2 produces 384-dimensional vectors
# It's a great balance of speed and quality for English text
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB storage path (persists the database to disk)
CHROMA_PERSIST_DIR = Path(__file__).parent / "data" / "chroma_db"

# Chunking parameters
# chunk_size: max characters per chunk (1000 chars ~= 200 words ~= 250 tokens)
# chunk_overlap: characters shared between adjacent chunks (prevents cutting mid-thought)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Category colors for Rich output
CATEGORY_COLORS = {
    "terraform": "purple",
    "kubernetes": "blue",
    "docker": "cyan",
    "cicd": "green",
    "unknown": "white",
}


# ============================================================================
# RAG PIPELINE CLASS
# ============================================================================
class DevOpsRAGPipeline:
    """
    Complete RAG Pipeline for the DevOps Troubleshooting Chatbot.

    THE BIG PICTURE:
        User asks a question
            -> We EMBED the question into a vector
            -> We SEARCH our vector database for similar vectors
            -> We RETRIEVE the matching document chunks
            -> We pass those chunks as CONTEXT to the LLM
            -> The LLM generates an answer grounded in our docs

    LIFECYCLE:
        1. __init__() - sets up embedding model and ChromaDB
        2. load_knowledge_base() - reads markdown files from disk
        3. chunk_documents() - splits large docs into smaller pieces
        4. build_index() - embeds chunks and stores in ChromaDB
        5. retrieve() - finds relevant chunks for a query
        6. get_stats() - shows what's in the database
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: str = EMBEDDING_MODEL,
        collection_name: str = "devops_docs",
    ):
        """
        Initialize the RAG pipeline.

        This sets up two main components:
        1. The embedding model (converts text to vectors)
        2. ChromaDB (stores and searches those vectors)

        Args:
            persist_directory: Where to store the ChromaDB database on disk.
                               If None, uses a default path next to this file.
            embedding_model:   Name of the sentence-transformers model to use.
            collection_name:   Name for the ChromaDB collection (like a table name).
        """
        self.collection_name = collection_name

        # ---- Step 1: Set up the persist directory ----
        # ChromaDB saves its data to disk so we don't have to re-index every time
        if persist_directory:
            self.persist_dir = Path(persist_directory)
        else:
            self.persist_dir = CHROMA_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # ---- Step 2: Load the embedding model ----
        # This downloads the model on first run (~80MB), then caches it locally.
        # It runs entirely on your CPU - no GPU or API key needed!
        console.print(f"[cyan]Loading embedding model: {embedding_model}...[/cyan]")
        self.embedding_model = SentenceTransformer(embedding_model)
        console.print("[green]Embedding model loaded![/green]")

        # ---- Step 3: Initialize ChromaDB ----
        # PersistentClient saves to disk (survives program restarts)
        # anonymized_telemetry=False disables data collection
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create our collection (like getting or creating a database table)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DevOps troubleshooting knowledge base"},
        )

        console.print(
            f"[green]ChromaDB ready! Collection '{collection_name}' has "
            f"{self.collection.count()} documents.[/green]"
        )

    def load_knowledge_base(self, path: Optional[str] = None) -> List[Dict]:
        """
        Load all markdown files from the knowledge base directory.

        This is the FIRST step in the RAG pipeline. We read every .md file
        and extract its text content along with metadata (category, filename).

        The knowledge base is organized like:
            knowledge-base/
                terraform/
                    state-management.md
                    provider-errors.md
                kubernetes/
                    pod-errors.md
                    networking.md
                docker/
                    build-errors.md
                cicd/
                    github-actions.md

        Args:
            path: Path to knowledge base directory. If None, uses default.

        Returns:
            List of dicts, each with:
            - "content": The text content of the file
            - "source_file": The filename (e.g., "pod-errors.md")
            - "category": The subdirectory name (e.g., "kubernetes")
            - "full_path": The full file path
        """
        kb_path = Path(path) if path else KNOWLEDGE_BASE_PATH

        if not kb_path.exists():
            console.print(f"[red]Knowledge base not found at: {kb_path}[/red]")
            console.print("[yellow]Creating demo documents instead...[/yellow]")
            return self._create_demo_documents()

        documents = []

        # Use Rich progress bar to show loading progress
        md_files = list(kb_path.rglob("*.md"))

        if not md_files:
            console.print(f"[yellow]No .md files found in {kb_path}[/yellow]")
            return self._create_demo_documents()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Loading knowledge base...", total=len(md_files)
            )

            for md_file in md_files:
                try:
                    # Read the file content
                    content = md_file.read_text(encoding="utf-8")

                    # Determine the category from the directory structure
                    # e.g., knowledge-base/kubernetes/pod-errors.md -> "kubernetes"
                    try:
                        relative = md_file.relative_to(kb_path)
                        if len(relative.parts) > 1:
                            category = relative.parts[0]  # First subdirectory
                        else:
                            category = "general"
                    except ValueError:
                        category = "general"

                    documents.append({
                        "content": content,
                        "source_file": md_file.name,
                        "category": category,
                        "full_path": str(md_file),
                    })

                    progress.advance(task)

                except Exception as e:
                    console.print(f"[red]Error loading {md_file}: {e}[/red]")
                    progress.advance(task)

        console.print(
            f"[green]Loaded {len(documents)} documents from {kb_path}[/green]"
        )
        return documents

    def _create_demo_documents(self) -> List[Dict]:
        """
        Create demo documents if the knowledge base directory is not found.
        This ensures the exercise works even without the full knowledge base.
        """
        demo_docs = [
            {
                "content": (
                    "# Terraform State Lock Error\n\n"
                    "## Problem\n"
                    "When running `terraform apply`, you get: "
                    "'Error acquiring the state lock'.\n\n"
                    "## Cause\n"
                    "Another Terraform process is running, or a previous process "
                    "crashed without releasing the lock.\n\n"
                    "## Solution\n"
                    "1. Wait for the other process to finish\n"
                    "2. If crashed: `terraform force-unlock LOCK_ID`\n"
                    "3. Check who holds the lock in your state backend (S3, etc.)\n"
                ),
                "source_file": "state-errors.md",
                "category": "terraform",
                "full_path": "demo/terraform/state-errors.md",
            },
            {
                "content": (
                    "# Kubernetes Pod CrashLoopBackOff\n\n"
                    "## Problem\n"
                    "Your pod status shows CrashLoopBackOff.\n\n"
                    "## Cause\n"
                    "The container keeps crashing and Kubernetes keeps restarting it.\n\n"
                    "## Diagnosis\n"
                    "1. Check logs: `kubectl logs <pod-name> --previous`\n"
                    "2. Describe pod: `kubectl describe pod <pod-name>`\n"
                    "3. Check events: `kubectl get events --sort-by=.lastTimestamp`\n\n"
                    "## Common Causes\n"
                    "- Application error (check logs for stack trace)\n"
                    "- Missing environment variables or ConfigMaps\n"
                    "- OOMKilled (exit code 137) - increase memory limits\n"
                    "- Failed liveness probe - check health endpoint\n"
                ),
                "source_file": "pod-errors.md",
                "category": "kubernetes",
                "full_path": "demo/kubernetes/pod-errors.md",
            },
            {
                "content": (
                    "# Docker Build COPY Failed\n\n"
                    "## Problem\n"
                    "`docker build` fails with 'COPY failed: file not found'.\n\n"
                    "## Cause\n"
                    "The file you're trying to COPY is outside the build context, "
                    "or it's listed in .dockerignore.\n\n"
                    "## Solution\n"
                    "1. Check your build context: `docker build .` uses current dir\n"
                    "2. Check .dockerignore for excluded files\n"
                    "3. Use relative paths in COPY (relative to build context)\n"
                    "4. Make sure the file exists: `ls -la <filename>`\n"
                ),
                "source_file": "build-errors.md",
                "category": "docker",
                "full_path": "demo/docker/build-errors.md",
            },
            {
                "content": (
                    "# GitHub Actions Permission Denied\n\n"
                    "## Problem\n"
                    "Your GitHub Actions workflow fails with 'Permission denied'.\n\n"
                    "## Common Causes\n"
                    "1. **GITHUB_TOKEN permissions**: Add permissions block\n"
                    "   ```yaml\n"
                    "   permissions:\n"
                    "     contents: write\n"
                    "     pull-requests: write\n"
                    "   ```\n"
                    "2. **Secret not available**: Secrets aren't available in forks\n"
                    "3. **File permissions**: Add `chmod +x script.sh` step\n"
                    "4. **Branch protection**: Check branch protection rules\n"
                ),
                "source_file": "github-actions-errors.md",
                "category": "cicd",
                "full_path": "demo/cicd/github-actions-errors.md",
            },
            {
                "content": (
                    "# Kubernetes ImagePullBackOff\n\n"
                    "## Problem\n"
                    "Pod status shows ImagePullBackOff or ErrImagePull.\n\n"
                    "## Cause\n"
                    "Kubernetes cannot pull the container image.\n\n"
                    "## Solutions\n"
                    "1. Check image name and tag: `kubectl describe pod <name>`\n"
                    "2. Verify image exists: `docker pull <image>`\n"
                    "3. For private registries, create an imagePullSecret:\n"
                    "   ```bash\n"
                    "   kubectl create secret docker-registry regcred \\\n"
                    "     --docker-server=<registry> \\\n"
                    "     --docker-username=<user> \\\n"
                    "     --docker-password=<pass>\n"
                    "   ```\n"
                    "4. Add imagePullSecrets to your pod spec\n"
                ),
                "source_file": "image-pull-errors.md",
                "category": "kubernetes",
                "full_path": "demo/kubernetes/image-pull-errors.md",
            },
            {
                "content": (
                    "# Terraform Provider Configuration\n\n"
                    "## Problem\n"
                    "Terraform fails with provider-related errors during init or plan.\n\n"
                    "## Common Errors\n"
                    "1. **Provider not found**: Run `terraform init` to download\n"
                    "2. **Version constraint**: Check required_providers block\n"
                    "3. **Auth failure**: Set credentials via env vars or provider block\n\n"
                    "## AWS Provider Setup\n"
                    "```hcl\n"
                    "provider \"aws\" {\n"
                    "  region = \"us-east-1\"\n"
                    "}\n"
                    "```\n"
                    "Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env vars.\n"
                ),
                "source_file": "provider-config.md",
                "category": "terraform",
                "full_path": "demo/terraform/provider-config.md",
            },
        ]
        console.print(f"[green]Created {len(demo_docs)} demo documents.[/green]")
        return demo_docs

    def chunk_documents(
        self,
        documents: List[Dict],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval.

        WHY CHUNKING?
            A full document might be 5000 characters. If a user asks about
            "pod CrashLoopBackOff", we don't want to return the ENTIRE
            Kubernetes troubleshooting guide - just the relevant section.

            By splitting into ~1000 character chunks, we can return just
            the paragraph about CrashLoopBackOff.

        MARKDOWN-AWARE SPLITTING:
            We use separators that respect markdown structure:
            1. First try to split on ## headers (major sections)
            2. Then ### headers (subsections)
            3. Then double newlines (paragraphs)
            4. Then single newlines
            5. Then spaces (last resort)

        Args:
            documents:     List of document dicts from load_knowledge_base()
            chunk_size:    Maximum characters per chunk
            chunk_overlap: Characters shared between adjacent chunks

        Returns:
            List of chunk dicts, each with:
            - "content": The chunk text
            - "source_file": Original filename
            - "category": Original category
            - "chunk_index": Position of this chunk within its document
        """
        # Create a markdown-aware text splitter
        # The separators list is tried in order - it prefers splitting on headers
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n## ",    # Split on H2 headers first (major sections)
                "\n### ",   # Then H3 headers (subsections)
                "\n\n",     # Then paragraph breaks
                "\n",       # Then line breaks
                " ",        # Then spaces
                "",         # Last resort: character by character
            ],
        )

        all_chunks = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Chunking documents...", total=len(documents)
            )

            for doc in documents:
                # Split this document into chunks
                chunks = text_splitter.split_text(doc["content"])

                for i, chunk_text in enumerate(chunks):
                    all_chunks.append({
                        "content": chunk_text,
                        "source_file": doc["source_file"],
                        "category": doc["category"],
                        "chunk_index": i,
                    })

                progress.advance(task)

        console.print(
            f"[green]Created {len(all_chunks)} chunks from "
            f"{len(documents)} documents[/green]"
        )
        return all_chunks

    def build_index(self, chunks: List[Dict]) -> int:
        """
        Embed chunks and store them in ChromaDB.

        This is where the magic happens! We:
        1. Convert each chunk's text into a vector (embedding)
        2. Store the vector + text + metadata in ChromaDB
        3. ChromaDB builds an index for fast similarity search

        WHAT'S AN EMBEDDING?
            An embedding is a list of numbers (e.g., 384 numbers) that
            represents the MEANING of text. Similar texts have similar
            numbers. This lets us find relevant docs by comparing numbers
            instead of matching keywords.

            "pod is crashing"     -> [0.12, -0.34, 0.56, ...]
            "container keeps failing" -> [0.11, -0.33, 0.55, ...]  (similar!)
            "terraform state"     -> [-0.45, 0.67, -0.12, ...]  (different!)

        Args:
            chunks: List of chunk dicts from chunk_documents()

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            console.print("[yellow]No chunks to index![/yellow]")
            return 0

        # Clear existing documents first (fresh index)
        existing_count = self.collection.count()
        if existing_count > 0:
            console.print(
                f"[yellow]Clearing {existing_count} existing documents...[/yellow]"
            )
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "DevOps troubleshooting knowledge base"},
            )

        # Prepare batch data
        ids = []        # Unique ID for each chunk (required by ChromaDB)
        texts = []      # The actual text content
        metadatas = []  # Metadata (category, source file, chunk index)
        embeddings = [] # The vector representations

        # Generate embeddings for all chunks
        console.print("[cyan]Generating embeddings...[/cyan]")
        all_texts = [chunk["content"] for chunk in chunks]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Embedding chunks...", total=len(chunks)
            )

            # Embed in batches for efficiency
            batch_size = 32
            all_embeddings = []
            for i in range(0, len(all_texts), batch_size):
                batch = all_texts[i : i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                all_embeddings.extend(batch_embeddings.tolist())
                progress.advance(task, advance=len(batch))

        # Prepare data for ChromaDB
        for i, chunk in enumerate(chunks):
            ids.append(f"chunk_{i}")
            texts.append(chunk["content"])
            metadatas.append({
                "category": chunk["category"],
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
            })

        # Add to ChromaDB in batches
        console.print("[cyan]Storing in ChromaDB...[/cyan]")
        add_batch_size = 100
        for i in range(0, len(ids), add_batch_size):
            end = min(i + add_batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=all_embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

        final_count = self.collection.count()
        console.print(f"[green]Indexed {final_count} chunks successfully![/green]")
        return final_count

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a query.

        HOW RETRIEVAL WORKS:
            1. We embed the query into a vector (same model as indexing)
            2. ChromaDB finds the k nearest vectors (most similar meaning)
            3. We return those chunks with their similarity scores

        CATEGORY FILTERING:
            If we know the question is about Kubernetes (from Exercise 1's
            classifier), we can filter to ONLY search Kubernetes docs.
            This makes retrieval faster AND more accurate.

        Args:
            query:    The user's question
            category: Optional category filter (e.g., "kubernetes")
            k:        Number of results to return (default: 5)

        Returns:
            List of result dicts, each with:
            - "content": The chunk text
            - "metadata": Dict with category, source_file, chunk_index
            - "distance": How far this result is from the query (lower = better)
            - "relevance_score": Normalized 0-1 score (higher = better)
        """
        if self.collection.count() == 0:
            console.print("[yellow]No documents in the index! Run build_index first.[/yellow]")
            return []

        # Embed the query using the same model we used for indexing
        # This is crucial - query and documents must use the SAME embedding model!
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build optional category filter
        # ChromaDB uses a "where" clause for metadata filtering
        where_filter = None
        if category and category != "general":
            where_filter = {"category": category}

        # Query ChromaDB for similar chunks
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            # If category filter fails (e.g., no docs in that category),
            # retry without the filter
            console.print(
                f"[yellow]Category filter '{category}' failed, searching all docs...[/yellow]"
            )
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = results["distances"][0][i] if results["distances"] else 0
                # Convert distance to a 0-1 relevance score
                # ChromaDB uses L2 distance (lower = more similar)
                # We convert: relevance = 1 / (1 + distance)
                relevance = 1.0 / (1.0 + distance)

                formatted.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": distance,
                    "relevance_score": relevance,
                })

        return formatted

    def get_stats(self) -> Dict:
        """
        Get statistics about the indexed knowledge base.

        Returns a dict with document count, category breakdown, etc.
        Useful for debugging and monitoring the pipeline.
        """
        total_count = self.collection.count()

        stats = {
            "total_documents": total_count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_dir),
            "embedding_model": EMBEDDING_MODEL,
            "categories": {},
        }

        # Count documents per category by querying metadata
        if total_count > 0:
            # Get all documents to count categories
            try:
                all_docs = self.collection.get(
                    include=["metadatas"],
                    limit=total_count,
                )
                if all_docs["metadatas"]:
                    for meta in all_docs["metadatas"]:
                        cat = meta.get("category", "unknown")
                        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
            except Exception:
                stats["categories"] = {"error": "Could not retrieve category stats"}

        return stats

    # ========================================================================
    # TODO: Implement MMR (Maximal Marginal Relevance) retrieval
    # ========================================================================
    def retrieve_mmr(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 5,
        diversity: float = 0.3,
    ) -> List[Dict]:
        """
        TODO: Implement MMR retrieval for diverse results.

        WHAT IS MMR?
            Standard retrieval returns the k MOST SIMILAR chunks. But these
            might all say the same thing (redundant). MMR balances:
            - Relevance: How similar is this chunk to the query?
            - Diversity: How different is this chunk from already-selected chunks?

            MMR score = (1 - diversity) * relevance - diversity * max_similarity_to_selected

        WHY IT MATTERS:
            If the user asks "how to fix pod errors", standard retrieval might
            return 5 chunks all about CrashLoopBackOff. With MMR, we'd get
            chunks about CrashLoopBackOff, ImagePullBackOff, OOMKilled, etc.

        INSTRUCTIONS:
        1. Retrieve more candidates than needed (e.g., k * 3)
        2. Select the first result (most relevant)
        3. For each remaining slot:
           a. Score each candidate: (1-diversity) * relevance - diversity * similarity_to_selected
           b. Pick the candidate with the highest MMR score
           c. Add it to selected results
        4. Return the k selected results

        HINT:
            You can compute similarity between two embeddings using cosine similarity:
            from numpy import dot
            from numpy.linalg import norm
            cos_sim = dot(a, b) / (norm(a) * norm(b))

        Args:
            query:     The user's question
            category:  Optional category filter
            k:         Number of results to return
            diversity: 0.0 = pure relevance, 1.0 = pure diversity

        Returns:
            List of result dicts (same format as retrieve())
        """
        # TODO: Implement MMR retrieval!
        # For now, fall back to standard retrieval
        console.print("[yellow]MMR not implemented yet - using standard retrieval[/yellow]")
        return self.retrieve(query, category=category, k=k)

    # ========================================================================
    # TODO: Implement rebuild_index()
    # ========================================================================
    def rebuild_index(self, knowledge_base_path: Optional[str] = None) -> int:
        """
        TODO: Clear the existing index and rebuild from scratch.

        This is useful when:
        - Knowledge base documents have been updated
        - You want to change chunking parameters
        - The index seems corrupted

        INSTRUCTIONS:
        1. Delete the existing ChromaDB collection
        2. Recreate an empty collection
        3. Load the knowledge base using load_knowledge_base()
        4. Chunk the documents using chunk_documents()
        5. Build the index using build_index()
        6. Return the number of indexed chunks

        Args:
            knowledge_base_path: Path to knowledge base (or None for default)

        Returns:
            Number of chunks indexed
        """
        # TODO: Implement this method!
        # Step 1: Delete existing collection
        # self.chroma_client.delete_collection(self.collection_name)

        # Step 2: Recreate empty collection
        # self.collection = self.chroma_client.create_collection(...)

        # Step 3: Load documents
        # docs = self.load_knowledge_base(knowledge_base_path)

        # Step 4: Chunk documents
        # chunks = self.chunk_documents(docs)

        # Step 5: Build index
        # count = self.build_index(chunks)

        # Step 6: Return count
        # return count

        console.print("[yellow]rebuild_index() not implemented yet![/yellow]")
        return 0


# ============================================================================
# DISPLAY HELPERS
# ============================================================================

def display_stats(stats: Dict) -> None:
    """Display pipeline statistics in a Rich table."""
    table = Table(title="RAG Pipeline Statistics", show_header=True)
    table.add_column("Property", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Documents", str(stats["total_documents"]))
    table.add_row("Collection Name", stats["collection_name"])
    table.add_row("Persist Directory", stats["persist_directory"])
    table.add_row("Embedding Model", stats["embedding_model"])

    # Category breakdown
    if stats["categories"]:
        cat_str = "\n".join(
            f"  [{CATEGORY_COLORS.get(cat, 'white')}]{cat}[/{CATEGORY_COLORS.get(cat, 'white')}]: {count}"
            for cat, count in sorted(stats["categories"].items())
        )
        table.add_row("Categories", cat_str)

    console.print(table)


def display_results(query: str, results: List[Dict]) -> None:
    """Display retrieval results in a Rich panel."""
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Query:[/bold] {query}")
    console.print(f"[dim]Found {len(results)} results:[/dim]\n")

    for i, result in enumerate(results, 1):
        metadata = result.get("metadata", {})
        category = metadata.get("category", "unknown")
        source = metadata.get("source_file", "unknown")
        relevance = result.get("relevance_score", 0)
        distance = result.get("distance", 0)
        color = CATEGORY_COLORS.get(category, "white")

        # Truncate content for display
        content = result["content"]
        if len(content) > 300:
            content = content[:300] + "..."

        # Build the panel title with category badge and relevance score
        title = (
            f"[bold]Result {i}[/bold] | "
            f"[{color}]{category}[/{color}] | "
            f"{source} | "
            f"Relevance: {relevance:.2%}"
        )

        console.print(Panel(
            content,
            title=title,
            border_style=color,
            width=90,
        ))


# ============================================================================
# MAIN - Interactive Demo
# ============================================================================
def main():
    """
    Run the RAG Pipeline interactive demo.

    1. Loads the knowledge base
    2. Chunks and indexes documents
    3. Shows stats
    4. Lets the user query and see results
    """
    console.print(Panel.fit(
        "[bold cyan]Exercise 2: Complete RAG Pipeline[/bold cyan]\n"
        "[dim]Module 7 - Complete Chatbot (Capstone)[/dim]\n\n"
        "This builds the full RAG pipeline:\n"
        "  Load -> Chunk -> Embed -> Store -> Retrieve",
        border_style="cyan",
    ))

    # ---- Step 1: Create the pipeline ----
    console.print("\n[bold yellow]Step 1: Initializing Pipeline[/bold yellow]\n")
    pipeline = DevOpsRAGPipeline()

    # ---- Step 2: Load documents ----
    console.print("\n[bold yellow]Step 2: Loading Knowledge Base[/bold yellow]\n")
    documents = pipeline.load_knowledge_base()

    # ---- Step 3: Chunk documents ----
    console.print("\n[bold yellow]Step 3: Chunking Documents[/bold yellow]\n")
    chunks = pipeline.chunk_documents(documents)

    # ---- Step 4: Build the index ----
    console.print("\n[bold yellow]Step 4: Building Vector Index[/bold yellow]\n")
    pipeline.build_index(chunks)

    # ---- Step 5: Show stats ----
    console.print("\n[bold yellow]Step 5: Pipeline Statistics[/bold yellow]\n")
    stats = pipeline.get_stats()
    display_stats(stats)

    # ---- Step 6: Test with sample queries ----
    console.print("\n[bold yellow]Step 6: Testing Retrieval[/bold yellow]\n")

    test_queries = [
        ("my pod is in CrashLoopBackOff", "kubernetes"),
        ("terraform state lock error", "terraform"),
        ("docker build COPY file not found", "docker"),
        ("github actions permission denied", "cicd"),
    ]

    for query, category in test_queries:
        console.print(f"\n[bold]Testing:[/bold] \"{query}\" (category: {category})")

        # Retrieve without category filter
        results = pipeline.retrieve(query, k=2)
        if results:
            top = results[0]
            console.print(
                f"  [green]Top result:[/green] {top['metadata'].get('source_file', '?')} "
                f"(relevance: {top['relevance_score']:.2%})"
            )
        else:
            console.print("  [red]No results found[/red]")

        # Retrieve WITH category filter
        filtered = pipeline.retrieve(query, category=category, k=2)
        if filtered:
            top = filtered[0]
            console.print(
                f"  [green]Filtered ({category}):[/green] {top['metadata'].get('source_file', '?')} "
                f"(relevance: {top['relevance_score']:.2%})"
            )

    # ---- Step 7: Interactive mode ----
    console.print("\n[bold yellow]Step 7: Interactive Query Mode[/bold yellow]")
    console.print("[dim]Type a DevOps question to search the knowledge base.[/dim]")
    console.print("[dim]Prefix with a category to filter, e.g.: 'k8s: pod crashing'[/dim]")
    console.print("[dim]Type 'stats' to see index stats, or 'quit' to exit.[/dim]\n")

    # Category shorthand mapping
    category_shortcuts = {
        "tf:": "terraform",
        "k8s:": "kubernetes",
        "docker:": "docker",
        "cicd:": "cicd",
        "ci:": "cicd",
    }

    while True:
        try:
            user_input = Prompt.ask("[bold green]Search[/bold green]")

            if user_input.lower() in ["quit", "exit", "q"]:
                console.print("[cyan]Goodbye![/cyan]")
                break

            if user_input.lower() == "stats":
                display_stats(pipeline.get_stats())
                continue

            if not user_input.strip():
                continue

            # Check for category prefix
            category_filter = None
            query = user_input
            for prefix, cat in category_shortcuts.items():
                if user_input.lower().startswith(prefix):
                    category_filter = cat
                    query = user_input[len(prefix):].strip()
                    console.print(f"[dim]Filtering by category: {cat}[/dim]")
                    break

            # Retrieve and display
            start_time = time.time()
            results = pipeline.retrieve(query, category=category_filter, k=5)
            elapsed = time.time() - start_time

            console.print(f"[dim]Retrieved {len(results)} results in {elapsed:.3f}s[/dim]")
            display_results(query, results)

        except KeyboardInterrupt:
            console.print("\n[cyan]Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()

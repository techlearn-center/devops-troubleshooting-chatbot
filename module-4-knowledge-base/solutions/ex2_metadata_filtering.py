#!/usr/bin/env python3
"""
SOLUTION: Exercise 2 - Metadata Filtering
==========================================

Indexes documents with metadata into ChromaDB and demonstrates
filtered vs. unfiltered similarity search by category.

RUN THIS:
    python module-4-knowledge-base/solutions/ex2_metadata_filtering.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class FilterableKB:
    """
    Knowledge base with category-based metadata filtering.
    Uses HuggingFace embeddings (free, local) and ChromaDB.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    def index_documents(self, documents: list):
        """Store documents with metadata in ChromaDB (in-memory)."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
        )
        print(f"Indexed {len(documents)} documents")

    def search(self, query: str, k: int = 3, category: str = None) -> list:
        """
        Similarity search with optional category filter.

        Args:
            query: Search text.
            k: Number of results.
            category: If set, restricts results to this category.

        Returns:
            List of (Document, score) tuples.
        """
        filter_dict = {"category": category} if category else None
        return self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )

    @staticmethod
    def print_results(results: list, title: str = "Results"):
        """Pretty-print a list of (Document, score) tuples."""
        print(f"\n{'=' * 55}")
        print(f"  {title}")
        print(f"{'=' * 55}")

        if not results:
            print("  (no results)")
            return

        for i, item in enumerate(results, 1):
            doc, score = item if isinstance(item, tuple) else (item, None)
            cat = doc.metadata.get("category", "N/A")
            src = doc.metadata.get("source", "N/A")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"\n  [{i}] category={cat}  source={src}")
            if score is not None:
                print(f"      score: {score:.4f}")
            print(f"      {preview}...")


# --- Sample documents covering multiple categories ---
SAMPLE_DOCS = [
    Document(
        page_content=(
            "CrashLoopBackOff means the container keeps crashing and restarting. "
            "Check logs with: kubectl logs <pod> --previous"
        ),
        metadata={"source": "k8s/pod-errors.md", "category": "kubernetes"},
    ),
    Document(
        page_content=(
            "OOMKilled (exit code 137) means the container exceeded its memory limit. "
            "Increase resources.limits.memory in the pod spec."
        ),
        metadata={"source": "k8s/pod-errors.md", "category": "kubernetes"},
    ),
    Document(
        page_content=(
            "ImagePullBackOff: Kubernetes cannot pull the container image. "
            "Verify image name, check registry auth, and ensure imagePullSecrets are configured."
        ),
        metadata={"source": "k8s/pod-errors.md", "category": "kubernetes"},
    ),
    Document(
        page_content=(
            "Terraform state lock error: another process is running. "
            "Wait or run: terraform force-unlock <LOCK_ID>"
        ),
        metadata={"source": "tf/errors.md", "category": "terraform"},
    ),
    Document(
        page_content=(
            "Resource already exists error in Terraform. Import the resource: "
            "terraform import aws_instance.example i-1234567890abcdef0"
        ),
        metadata={"source": "tf/errors.md", "category": "terraform"},
    ),
    Document(
        page_content=(
            "Docker COPY failed: file not found. The file must be inside the "
            "build context. Check .dockerignore and use relative paths."
        ),
        metadata={"source": "docker/build-errors.md", "category": "docker"},
    ),
    Document(
        page_content=(
            "Docker no space left on device. Clean up with: docker system prune -a "
            "and use multi-stage builds to reduce image size."
        ),
        metadata={"source": "docker/build-errors.md", "category": "docker"},
    ),
    Document(
        page_content=(
            "GitHub Actions 'Resource not accessible by integration'. "
            "Add explicit permissions block to the job: permissions: contents: read."
        ),
        metadata={"source": "cicd/github-actions.md", "category": "cicd"},
    ),
]


if __name__ == "__main__":
    kb = FilterableKB()
    kb.index_documents(SAMPLE_DOCS)

    # --- Unfiltered search ---
    results = kb.search("my pod keeps crashing", k=3)
    kb.print_results(results, "Unfiltered: 'my pod keeps crashing'")

    # --- Filtered to kubernetes only ---
    results = kb.search("error", k=3, category="kubernetes")
    kb.print_results(results, "Kubernetes only: 'error'")

    # --- Filtered to terraform only ---
    results = kb.search("error", k=3, category="terraform")
    kb.print_results(results, "Terraform only: 'error'")

    # --- Filtered to docker only ---
    results = kb.search("build problem", k=2, category="docker")
    kb.print_results(results, "Docker only: 'build problem'")

    print("\nDone. Metadata filtering lets you scope searches to specific categories.")

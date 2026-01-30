#!/usr/bin/env python3
"""
SOLUTION: Exercise 5 - Similarity Search
=========================================

Complete similarity search solution with ChromaDB.

RUN THIS:
    python module-3-rag-fundamentals/solutions/ex5_similarity_search.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


class DevOpsSearchEngine:
    """
    Semantic search engine for DevOps documentation.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None

    def index_documents(self, documents: list):
        """Index documents into vector store."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print(f"✅ Indexed {len(documents)} documents")

    def search(self, query: str, k: int = 3, category: str = None) -> list:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results
            category: Optional category filter

        Returns:
            List of (document, score) tuples
        """
        filter_dict = {"category": category} if category else None

        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )
        return results

    def search_mmr(self, query: str, k: int = 3, diversity: float = 0.5) -> list:
        """Search with MMR for diverse results."""
        results = self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k * 3,
            lambda_mult=diversity
        )
        return results

    def print_results(self, results: list, show_content: bool = True):
        """Pretty print search results."""
        for i, item in enumerate(results, 1):
            if isinstance(item, tuple):
                doc, score = item
            else:
                doc, score = item, None

            source = doc.metadata.get('source', 'unknown')
            category = doc.metadata.get('category', 'N/A')

            print(f"\n[{i}] {source}")
            print(f"    Category: {category}")
            if score is not None:
                print(f"    Score: {score:.4f}")
            if show_content:
                preview = doc.page_content[:150].replace('\n', ' ')
                print(f"    Preview: {preview}...")


# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="CrashLoopBackOff means the container keeps crashing. Check logs with kubectl logs.",
        metadata={"source": "k8s/errors.md", "category": "kubernetes"}
    ),
    Document(
        page_content="OOMKilled exit code 137 means out of memory. Increase memory limits.",
        metadata={"source": "k8s/resources.md", "category": "kubernetes"}
    ),
    Document(
        page_content="Terraform state lock error. Use terraform force-unlock to release.",
        metadata={"source": "tf/state.md", "category": "terraform"}
    ),
    Document(
        page_content="Docker multi-stage builds reduce image size. Use FROM ... AS builder.",
        metadata={"source": "docker/builds.md", "category": "docker"}
    ),
]


if __name__ == "__main__":
    # Create search engine
    engine = DevOpsSearchEngine()
    engine.index_documents(SAMPLE_DOCS)

    # Test searches
    print("\n" + "=" * 60)
    print("SEARCH TESTS")
    print("=" * 60)

    # Basic search
    print("\n🔍 Query: 'my pod keeps crashing'")
    results = engine.search("my pod keeps crashing", k=2)
    engine.print_results(results)

    # Filtered search
    print("\n🔍 Query: 'error' (Kubernetes only)")
    results = engine.search("error", k=2, category="kubernetes")
    engine.print_results(results)

    # MMR search
    print("\n🔍 Query: 'troubleshooting' (MMR - diverse)")
    results = engine.search_mmr("troubleshooting", k=3)
    engine.print_results([(r, None) for r in results])

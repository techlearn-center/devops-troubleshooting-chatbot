#!/usr/bin/env python3
"""
SOLUTION: Exercise 4 - Embeddings Visualized
=============================================

Complete embeddings solution with similarity analysis.

RUN THIS:
    python module-3-rag-fundamentals/solutions/ex4_embeddings_visualized.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingAnalyzer:
    """
    Analyze and compare text embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"📦 Loading embedding model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.cache = {}

    def embed(self, text: str) -> list:
        """Get embedding for text (cached)."""
        if text not in self.cache:
            self.cache[text] = self.embeddings.embed_query(text)
        return self.cache[text]

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        vec1 = self.embed(text1)
        vec2 = self.embed(text2)

        if HAS_NUMPY:
            a, b = np.array(vec1), np.array(vec2)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        else:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5
            return dot / (mag1 * mag2)

    def similarity_matrix(self, texts: list) -> list:
        """Create similarity matrix for multiple texts."""
        n = len(texts)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                matrix[i][j] = self.similarity(texts[i], texts[j])

        return matrix

    def find_most_similar(self, query: str, candidates: list, top_k: int = 3) -> list:
        """Find most similar texts to query."""
        scores = [(text, self.similarity(query, text)) for text in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def print_similarity_matrix(self, texts: list, labels: list = None):
        """Print formatted similarity matrix."""
        if labels is None:
            labels = [f"Text {i+1}" for i in range(len(texts))]

        matrix = self.similarity_matrix(texts)

        # Header
        print("\n" + " " * 15 + "  ".join(f"{l[:8]:>8}" for l in labels))
        print("-" * (15 + 10 * len(labels)))

        # Rows
        for i, label in enumerate(labels):
            row = "  ".join(f"{matrix[i][j]:>8.3f}" for j in range(len(texts)))
            print(f"{label[:12]:<12}   {row}")


if __name__ == "__main__":
    analyzer = EmbeddingAnalyzer()

    # Test texts
    texts = [
        "Kubernetes pod crash",
        "Container keeps restarting",
        "CrashLoopBackOff error",
        "Terraform state lock",
        "Docker build optimization",
    ]

    labels = ["K8s crash", "Container", "CrashLoop", "TF state", "Docker"]

    # Similarity matrix
    print("\n📊 Similarity Matrix:")
    analyzer.print_similarity_matrix(texts, labels)

    # Find similar to query
    query = "my application keeps crashing in kubernetes"
    print(f"\n🔍 Query: \"{query}\"")
    print("\nMost similar texts:")

    results = analyzer.find_most_similar(query, texts)
    for text, score in results:
        bar = "█" * int(score * 20)
        print(f"   {score:.3f} {bar} {text}")

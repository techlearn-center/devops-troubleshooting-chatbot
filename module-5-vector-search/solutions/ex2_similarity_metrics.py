#!/usr/bin/env python3
"""
SOLUTION: Exercise 2 - Similarity Metrics
==========================================

Implement and compare cosine similarity, Euclidean distance, and dot product.
Shows when and why different metrics give different rankings.

RUN THIS:
    python module-5-vector-search/solutions/ex2_similarity_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


# --- Raw metric implementations ---

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors. Range: [-1, 1]."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two vectors. Range: [0, inf). Lower = more similar."""
    return float(np.linalg.norm(a - b))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Raw dot product. Higher = more similar (assumes normalized vectors)."""
    return float(np.dot(a, b))


# --- DevOps text pairs for comparison ---

TEXT_PAIRS = [
    # (label, text_a, text_b)
    ("Semantically close",
     "Kubernetes pod stuck in CrashLoopBackOff",
     "Container keeps crashing and restarting in K8s"),

    ("Same topic, different angle",
     "How to debug CrashLoopBackOff errors",
     "Kubernetes pod resource limits and OOMKilled"),

    ("Different topics",
     "Kubernetes pod stuck in CrashLoopBackOff",
     "Terraform state lock error during apply"),

    ("Completely unrelated",
     "Kubernetes pod stuck in CrashLoopBackOff",
     "Best Italian restaurant near downtown"),
]


class SimilarityAnalyzer:
    """Analyze text pairs using multiple similarity metrics."""

    METRICS = {
        "Cosine":    cosine_similarity,
        "Euclidean": euclidean_distance,
        "Dot Prod":  dot_product,
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = HuggingFaceEmbeddings(model_name=model_name)

    def _embed(self, text: str) -> np.ndarray:
        return np.array(self.model.embed_query(text))

    def compare_metrics(self, text_a: str, text_b: str) -> dict:
        """Run all three metrics on a single text pair."""
        vec_a = self._embed(text_a)
        vec_b = self._embed(text_b)
        return {name: fn(vec_a, vec_b) for name, fn in self.METRICS.items()}

    def analyze_pairs(self, pairs: list[tuple[str, str, str]]) -> list[dict]:
        """Analyze a list of (label, text_a, text_b) tuples."""
        results = []
        for label, text_a, text_b in pairs:
            scores = self.compare_metrics(text_a, text_b)
            results.append({"label": label, **scores})
        return results


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    print(f"\n{'Relationship':<30} {'Cosine':>10} {'Euclidean':>10} {'Dot Prod':>10}")
    print("-" * 62)
    for r in results:
        print(f"{r['label']:<30} {r['Cosine']:>10.4f} {r['Euclidean']:>10.4f} {r['Dot Prod']:>10.4f}")


def print_ranking_comparison(results: list[dict]):
    """Show how each metric ranks the pairs (most to least similar)."""
    print("\nRankings (most similar first):")
    print("-" * 62)

    # Cosine: higher is better
    by_cosine = sorted(results, key=lambda r: r["Cosine"], reverse=True)
    print("  Cosine:    ", " > ".join(r["label"][:20] for r in by_cosine))

    # Euclidean: lower is better
    by_euclid = sorted(results, key=lambda r: r["Euclidean"])
    print("  Euclidean: ", " > ".join(r["label"][:20] for r in by_euclid))

    # Dot product: higher is better
    by_dot = sorted(results, key=lambda r: r["Dot Prod"], reverse=True)
    print("  Dot Prod:  ", " > ".join(r["label"][:20] for r in by_dot))

    # Check if all rankings agree
    cosine_order = [r["label"] for r in by_cosine]
    euclid_order = [r["label"] for r in by_euclid]
    dot_order    = [r["label"] for r in by_dot]

    if cosine_order == euclid_order == dot_order:
        print("\n  All metrics agree on ranking.")
    else:
        print("\n  Metrics DISAGREE on ranking -- this happens when vector")
        print("  magnitudes vary. Cosine ignores magnitude; Euclidean and")
        print("  dot product do not.")


if __name__ == "__main__":
    analyzer = SimilarityAnalyzer()

    print("\n" + "=" * 62)
    print("SIMILARITY METRICS COMPARISON")
    print("=" * 62)

    results = analyzer.analyze_pairs(TEXT_PAIRS)
    print_comparison_table(results)
    print_ranking_comparison(results)

    # Explain the metrics
    print("\n" + "=" * 62)
    print("METRIC CHEAT SHEET")
    print("=" * 62)
    print("""
  Cosine Similarity
    - Measures angle between vectors (ignores magnitude)
    - Range: [-1, 1]. Best for text similarity.
    - ChromaDB distance function: "cosine"

  Euclidean Distance (L2)
    - Straight-line distance. Lower = more similar.
    - Sensitive to vector magnitude.
    - ChromaDB distance function: "l2"

  Dot Product
    - Combines direction AND magnitude.
    - Equivalent to cosine when vectors are normalized.
    - ChromaDB distance function: "ip" (inner product)

  Rule of thumb: Use cosine for text search. It is the default
  in most vector databases and works well with sentence embeddings.
""")

#!/usr/bin/env python3
"""
SOLUTION: Exercise 1 - Embedding Model Comparison
==================================================

Compare different HuggingFace embedding models on DevOps texts.
Shows dimension differences, encoding speed, and similarity quality.

RUN THIS:
    python module-5-vector-search/solutions/ex1_embedding_models.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


# DevOps texts for comparison
DEVOPS_TEXTS = [
    "Kubernetes pod stuck in CrashLoopBackOff after deployment",
    "Container keeps restarting with OOMKilled exit code 137",
    "Terraform state lock error prevents apply from running",
    "Docker build fails during npm install step",
    "Jenkins pipeline timeout during integration tests",
    "Nginx returns 502 Bad Gateway under heavy load",
    "SSL certificate expired causing HTTPS failures",
    "GitLab CI runner out of disk space on shared runner",
]

# Query-document pairs to test similarity quality
SIMILARITY_PAIRS = [
    ("my pod keeps crashing",           "Kubernetes pod stuck in CrashLoopBackOff after deployment"),
    ("out of memory container",         "Container keeps restarting with OOMKilled exit code 137"),
    ("terraform cannot acquire lock",   "Terraform state lock error prevents apply from running"),
    ("build broken npm",                "Docker build fails during npm install step"),
]


class EmbeddingComparator:
    """Compare embedding models on DevOps-specific text."""

    MODELS = {
        "MiniLM-L6":  "all-MiniLM-L6-v2",
        "MPNet-base": "all-mpnet-base-v2",
    }

    def __init__(self):
        self.models = {}
        for label, name in self.MODELS.items():
            print(f"Loading {label} ({name})...")
            self.models[label] = HuggingFaceEmbeddings(model_name=name)

    def compare_models(self, texts: list[str]) -> dict:
        """Encode texts with each model. Return dimensions and timing."""
        results = {}
        for label, model in self.models.items():
            start = time.perf_counter()
            embeddings = model.embed_documents(texts)
            elapsed = time.perf_counter() - start

            results[label] = {
                "dimensions": len(embeddings[0]),
                "num_texts": len(texts),
                "time_sec": round(elapsed, 4),
                "per_text_ms": round(elapsed / len(texts) * 1000, 2),
            }
        return results

    def compare_similarity(self, pairs: list[tuple[str, str]]) -> dict:
        """Compare cosine similarity scores across models for query-doc pairs."""
        results = {label: [] for label in self.models}

        for query, document in pairs:
            for label, model in self.models.items():
                q_vec = np.array(model.embed_query(query))
                d_vec = np.array(model.embed_query(document))
                cos_sim = float(np.dot(q_vec, d_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)))
                results[label].append(round(cos_sim, 4))

        return results


def print_comparison_table(perf: dict):
    """Print model performance comparison table."""
    print(f"\n{'Model':<14} {'Dims':<8} {'Texts':<8} {'Total (s)':<12} {'Per Text (ms)':<14}")
    print("-" * 56)
    for label, stats in perf.items():
        print(f"{label:<14} {stats['dimensions']:<8} {stats['num_texts']:<8} "
              f"{stats['time_sec']:<12} {stats['per_text_ms']:<14}")


def print_similarity_table(sim_results: dict, pairs: list[tuple[str, str]]):
    """Print similarity comparison table."""
    labels = list(sim_results.keys())
    print(f"\n{'Query':<36} ", end="")
    for label in labels:
        print(f"{label:<14}", end="")
    print(f"  {'Diff':<8}")
    print("-" * 72)

    for i, (query, _) in enumerate(pairs):
        short_q = query[:34]
        scores = [sim_results[label][i] for label in labels]
        diff = abs(scores[0] - scores[1])
        print(f"{short_q:<36} ", end="")
        for s in scores:
            print(f"{s:<14.4f}", end="")
        print(f"  {diff:<8.4f}")

    # Averages
    print("-" * 72)
    print(f"{'Average':<36} ", end="")
    avgs = []
    for label in labels:
        avg = np.mean(sim_results[label])
        avgs.append(avg)
        print(f"{avg:<14.4f}", end="")
    print(f"  {abs(avgs[0] - avgs[1]):<8.4f}")


if __name__ == "__main__":
    comparator = EmbeddingComparator()

    # Performance comparison
    print("\n" + "=" * 56)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 56)
    perf = comparator.compare_models(DEVOPS_TEXTS)
    print_comparison_table(perf)

    # Similarity quality comparison
    print("\n" + "=" * 72)
    print("SIMILARITY QUALITY COMPARISON")
    print("=" * 72)
    sim_results = comparator.compare_similarity(SIMILARITY_PAIRS)
    print_similarity_table(sim_results, SIMILARITY_PAIRS)

    # Summary
    print("\nKey Takeaways:")
    print("  - MPNet-base: 768 dims, higher quality, slower encoding")
    print("  - MiniLM-L6:  384 dims, good quality, faster encoding")
    print("  - For DevOps chatbot, MiniLM-L6 is usually sufficient")

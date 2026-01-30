#!/usr/bin/env python3
"""
SOLUTION: Exercise 4 - Retrieval Quality Evaluation
=====================================================

Evaluate how well the vector search retrieves relevant documents.
Measures precision at different k values using a labeled test set.

RUN THIS:
    python module-5-vector-search/solutions/ex4_retrieval_quality.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions


# --- Knowledge base (documents with category labels) ---

KNOWLEDGE_BASE = [
    # Kubernetes
    {"id": "k8s-001", "text": "CrashLoopBackOff: container keeps crashing. Check kubectl logs --previous.", "category": "kubernetes"},
    {"id": "k8s-002", "text": "OOMKilled exit code 137: container out of memory. Increase resource limits.", "category": "kubernetes"},
    {"id": "k8s-003", "text": "ImagePullBackOff: cannot pull image. Verify image name and registry credentials.", "category": "kubernetes"},
    {"id": "k8s-004", "text": "Pod stuck in Pending: insufficient resources or node selector mismatch.", "category": "kubernetes"},
    {"id": "k8s-005", "text": "Service not reachable: check selector labels match pod labels and port mapping.", "category": "kubernetes"},
    # Terraform
    {"id": "tf-001", "text": "Terraform state lock: another process holds the lock. Use force-unlock.", "category": "terraform"},
    {"id": "tf-002", "text": "Terraform import: bring existing resources under management with terraform import.", "category": "terraform"},
    {"id": "tf-003", "text": "Terraform plan shows unexpected destroy: check for removed resources in config.", "category": "terraform"},
    # Docker
    {"id": "docker-001", "text": "Multi-stage builds reduce image size. Separate build and runtime stages.", "category": "docker"},
    {"id": "docker-002", "text": "Docker layer caching: order Dockerfile commands from least to most changing.", "category": "docker"},
    # CI/CD
    {"id": "ci-001", "text": "Jenkins pipeline timeout: increase timeout directive or parallelize stages.", "category": "cicd"},
    {"id": "ci-002", "text": "GitHub Actions workflow fails: check runner OS, permissions, and secret availability.", "category": "cicd"},
    # Networking
    {"id": "net-001", "text": "Nginx 502 Bad Gateway: upstream server unreachable. Check backend health.", "category": "networking"},
    {"id": "net-002", "text": "SSL certificate expired: renew with certbot or update in load balancer config.", "category": "networking"},
]


# --- Test cases: (query, expected_category) ---

TEST_CASES = [
    ("my pod keeps crashing and restarting",            "kubernetes"),
    ("container killed out of memory",                  "kubernetes"),
    ("cannot pull docker image in cluster",             "kubernetes"),
    ("terraform cannot acquire state lock",             "terraform"),
    ("import existing AWS resources into terraform",    "terraform"),
    ("reduce docker image file size",                   "docker"),
    ("speed up docker builds with caching",             "docker"),
    ("jenkins build times out",                         "cicd"),
    ("github actions workflow permission denied",       "cicd"),
    ("nginx returning 502 errors",                      "networking"),
    ("https certificate expired",                       "networking"),
    ("pod stuck in pending state",                      "kubernetes"),
]


class RetrievalEvaluator:
    """Evaluate retrieval quality with precision metrics."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.Client()
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self._build_collection()

    def _build_collection(self):
        """Index the knowledge base into ChromaDB."""
        try:
            self.client.delete_collection("eval_collection")
        except Exception:
            pass

        collection = self.client.create_collection(
            name="eval_collection",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            ids=[d["id"] for d in KNOWLEDGE_BASE],
            documents=[d["text"] for d in KNOWLEDGE_BASE],
            metadatas=[{"category": d["category"]} for d in KNOWLEDGE_BASE],
        )
        print(f"Indexed {collection.count()} documents")
        return collection

    def evaluate(self, test_cases: list[tuple[str, str]], k: int = 3) -> dict:
        """
        Evaluate precision@k for each test case.

        Precision@k = (relevant docs in top-k) / k
        A retrieved doc is "relevant" if its category matches the expected category.
        """
        precisions = []
        details = []

        for query, expected_cat in test_cases:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
            )
            categories = [m["category"] for m in results["metadatas"][0]]
            relevant = sum(1 for c in categories if c == expected_cat)
            precision = relevant / k

            precisions.append(precision)
            details.append({
                "query": query,
                "expected": expected_cat,
                "retrieved": categories,
                "precision": precision,
            })

        return {
            "k": k,
            "mean_precision": sum(precisions) / len(precisions),
            "perfect_count": sum(1 for p in precisions if p == 1.0),
            "total": len(test_cases),
            "details": details,
        }

    def compare_k_values(
        self, test_cases: list[tuple[str, str]], k_values: list[int] = None
    ) -> list[dict]:
        """Run evaluation at multiple k values."""
        if k_values is None:
            k_values = [1, 3, 5, 10]

        results = []
        for k in k_values:
            # Cap k at collection size
            effective_k = min(k, self.collection.count())
            result = self.evaluate(test_cases, k=effective_k)
            results.append(result)
        return results


def print_quality_report(eval_result: dict):
    """Print detailed evaluation results."""
    k = eval_result["k"]
    print(f"\n{'Query':<48} {'Expected':<14} {'P@{}'.format(k):>6}  {'Retrieved'}")
    print("-" * 100)

    for d in eval_result["details"]:
        status = "PASS" if d["precision"] == 1.0 else "MISS"
        cats = ", ".join(d["retrieved"])
        print(f"{d['query'][:46]:<48} {d['expected']:<14} {d['precision']:>5.2f}  {cats}  [{status}]")

    print("-" * 100)
    print(f"Mean Precision@{k}: {eval_result['mean_precision']:.3f}  "
          f"| Perfect: {eval_result['perfect_count']}/{eval_result['total']}")


def print_k_comparison(k_results: list[dict]):
    """Print precision across different k values."""
    print(f"\n{'k':<6} {'Mean P@k':>10} {'Perfect':>10} {'Total':>8}")
    print("-" * 36)
    for r in k_results:
        print(f"{r['k']:<6} {r['mean_precision']:>10.3f} "
              f"{r['perfect_count']:>7}/{r['total']:<4}")


if __name__ == "__main__":
    print("=" * 60)
    print("RETRIEVAL QUALITY EVALUATION")
    print("=" * 60)

    evaluator = RetrievalEvaluator()

    # Detailed report at k=3
    print("\n--- Detailed Results (k=3) ---")
    result = evaluator.evaluate(TEST_CASES, k=3)
    print_quality_report(result)

    # Compare across k values
    print("\n--- Precision vs. k ---")
    k_results = evaluator.compare_k_values(TEST_CASES)
    print_k_comparison(k_results)

    # Interpretation
    print(f"""
Interpretation:
  - P@1 tells you: "Is the top result relevant?" (strict)
  - P@3 tells you: "Are most of the top-3 relevant?" (balanced)
  - P@5+ dilutes precision because we have few docs per category.

  Higher P@1 = good for single-answer use cases (chatbot).
  Higher P@3 = good for context-stuffing in RAG pipelines.

  To improve retrieval quality:
  - Add more diverse documents per category
  - Try a larger embedding model (all-mpnet-base-v2)
  - Use metadata filters to narrow the search space
  - Fine-tune embeddings on your domain vocabulary
""")

# Module 5: Vector Search & Embeddings

**Time Required: 2 hours**

Understanding embeddings and vector search is crucial for building effective RAG systems. Let's dive deep into how semantic search actually works.

---

## Learning Objectives

By the end of this module, you will:
- Understand what embeddings are and how they work
- Know different similarity metrics and when to use them
- Configure ChromaDB for optimal retrieval
- Tune search parameters for better results

---

## What Are Embeddings?

### The Intuition

Embeddings convert text into numbers (vectors) that capture meaning:

```
TEXT                              VECTOR (simplified to 3D)
====                              ========================

"Terraform error"            -->  [0.8, 0.2, 0.1]
"Infrastructure issue"       -->  [0.75, 0.25, 0.15]  <- Similar!
"Kubernetes pod crash"       -->  [0.1, 0.9, 0.3]     <- Different

The closer the vectors, the more similar the meaning!
```

### Real Embeddings

Real embedding models produce vectors with 384-1536 dimensions:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed some DevOps text
texts = [
    "Terraform state lock error",
    "Infrastructure state is locked",
    "Kubernetes pod is crashing"
]

embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384) - 3 texts, 384 dimensions
```

---

## Similarity Metrics

### Cosine Similarity (Most Common)

Measures the angle between vectors:

```
Vector A: [1, 0]
Vector B: [0.7, 0.7]
Vector C: [0, 1]

Cosine Similarity:
- A and B: 0.707 (somewhat similar)
- A and C: 0.0 (orthogonal/different)
- B and C: 0.707 (somewhat similar)

       ^
     C |     B
       |   /
       | /
       +-------> A

Range: -1 (opposite) to 1 (identical)
For normalized vectors: 0 to 1
```

### Euclidean Distance

Measures straight-line distance:

```python
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Smaller distance = more similar
```

### Which to Use?

| Metric | Best For | Notes |
|--------|----------|-------|
| Cosine | Text similarity | Most common for embeddings |
| Euclidean | General purpose | Works well when magnitude matters |
| Dot Product | Normalized vectors | Fast, same as cosine when normalized |

---

## ChromaDB Deep Dive

### Creating a Collection

```python
import chromadb
from chromadb.config import Settings

# Persistent storage
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection with specific settings
collection = client.create_collection(
    name="devops_knowledge",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
```

### Adding Documents

```python
# Add documents with embeddings and metadata
collection.add(
    documents=["Terraform state error fix", "Kubernetes pod debugging"],
    metadatas=[
        {"category": "terraform", "type": "error"},
        {"category": "kubernetes", "type": "debugging"}
    ],
    ids=["doc1", "doc2"]
)
```

### Querying with Filters

```python
# Basic search
results = collection.query(
    query_texts=["How to fix state errors"],
    n_results=5
)

# Filtered search (Terraform only)
results = collection.query(
    query_texts=["How to fix state errors"],
    n_results=5,
    where={"category": "terraform"}
)

# Complex filters
results = collection.query(
    query_texts=["debugging issues"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$in": ["terraform", "kubernetes"]}},
            {"type": "error"}
        ]
    }
)
```

---

## Exercise 1: Understanding Embeddings

Create `exercises/ex1_embeddings.py`:

```python
"""
Exercise 1: Exploring Embeddings
================================
Goal: Understand how embeddings represent semantic similarity
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # DevOps-related sentences
    sentences = [
        # Terraform related
        "Terraform state lock error",
        "Infrastructure state is locked by another process",
        "Cannot acquire state lock",

        # Kubernetes related
        "Kubernetes pod is in CrashLoopBackOff",
        "Container keeps restarting",
        "Pod failed to start",

        # Unrelated
        "The weather is nice today",
        "I like pizza",
    ]

    # Generate embeddings
    embeddings = model.encode(sentences)

    # TODO: Calculate similarity matrix
    # Compare each sentence to every other sentence

    print("Similarity Matrix:")
    print("-" * 60)

    for i, sent1 in enumerate(sentences):
        for j, sent2 in enumerate(sentences):
            if i < j:  # Only upper triangle
                sim = cosine_similarity(embeddings[i], embeddings[j])
                if sim > 0.5:  # Only show similar pairs
                    print(f"Similarity: {sim:.3f}")
                    print(f"  '{sent1[:40]}...'")
                    print(f"  '{sent2[:40]}...'")
                    print()


if __name__ == "__main__":
    main()
```

---

## Exercise 2: Retrieval Quality Analysis

```python
"""
Exercise 2: Analyzing Retrieval Quality
=======================================
Goal: Understand and improve retrieval accuracy
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class RetrievalAnalyzer:
    """Analyze and improve retrieval quality."""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def search_with_scores(self, query: str, k: int = 5) -> list:
        """
        Search and return results with relevance scores.

        Returns:
            List of (document, score) tuples
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k
        )
        return results

    def analyze_query(self, query: str, expected_category: str = None):
        """
        Analyze retrieval quality for a query.

        Args:
            query: Search query
            expected_category: Category we expect to retrieve
        """
        results = self.search_with_scores(query, k=5)

        print(f"\nQuery: {query}")
        print(f"Expected category: {expected_category or 'any'}")
        print("-" * 50)

        correct = 0
        for i, (doc, score) in enumerate(results):
            category = doc.metadata.get("category", "unknown")
            is_correct = (expected_category is None or
                         category == expected_category)
            correct += is_correct

            status = "✓" if is_correct else "✗"
            print(f"{i+1}. [{status}] Score: {score:.3f} | Category: {category}")
            print(f"   {doc.page_content[:80]}...")

        accuracy = correct / len(results) if results else 0
        print(f"\nAccuracy: {accuracy:.0%} ({correct}/{len(results)})")

        return accuracy


# Test queries with expected results
TEST_QUERIES = [
    ("How to fix Terraform state lock", "terraform"),
    ("Kubernetes pod crash loop", "kubernetes"),
    ("Docker build fails", "docker"),
    ("GitHub Actions workflow error", "cicd"),
]
```

---

## Tuning Retrieval Parameters

### Number of Results (k)

```python
# Too few: might miss relevant docs
results = search(query, k=1)  # Risky!

# Too many: includes irrelevant docs
results = search(query, k=20)  # Dilutes context

# Sweet spot for most cases
results = search(query, k=3)  # or k=5
```

### Score Thresholding

```python
def search_with_threshold(query: str, min_score: float = 0.5):
    """Only return results above a relevance threshold."""
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=10
    )
    return [(doc, score) for doc, score in results if score >= min_score]
```

### Hybrid Search (Coming in advanced usage)

Combine keyword and semantic search:

```python
# Pseudo-code for hybrid search
def hybrid_search(query: str, k: int = 5):
    # Semantic search
    semantic_results = vectorstore.similarity_search(query, k=k*2)

    # Keyword search (BM25)
    keyword_results = keyword_index.search(query, k=k*2)

    # Combine and re-rank
    combined = merge_results(semantic_results, keyword_results)
    return combined[:k]
```

---

## Key Takeaways

1. **Embeddings capture meaning** - Similar text = similar vectors
2. **Cosine similarity is standard** - Use it for text similarity
3. **Chunk size affects retrieval** - Test different sizes
4. **Filter by metadata** - Narrow down to relevant categories
5. **Score thresholds prevent noise** - Only return confident matches

---

## What's Next?

In **Module 6**, we'll explore advanced prompting techniques:
- Few-shot learning for DevOps
- Chain-of-thought for complex troubleshooting
- Structured output for actionable responses

```bash
cd ../module-6-advanced-prompting
```

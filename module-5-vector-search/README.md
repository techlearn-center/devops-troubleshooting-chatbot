# Module 5: Vector Search & Embeddings Deep Dive

**Time Required: 2-3 hours**

In Module 3 we introduced embeddings and similarity search. Now we'll go deeper: comparing embedding models, understanding different similarity metrics, using ChromaDB's native API, and measuring retrieval quality to make your RAG system better.

---

## Learning Objectives

By the end of this module, you will:
- Compare different embedding models and know when to use which
- Understand cosine, euclidean, and dot product similarity (with math!)
- Use ChromaDB's native API for fine-grained control
- Measure and improve retrieval quality
- Tune search parameters for optimal results

---

## Prerequisites

```bash
# Install required packages
pip install chromadb sentence-transformers numpy
pip install langchain langchain-community
pip install langchain-openai        # If using OpenAI embeddings
```

---

## Module 3 vs Module 5

```
MODULE 3 (Basics):                    MODULE 5 (Deep Dive):
══════════════════                    ═════════════════════

✅ What are embeddings?                ✅ Comparing embedding MODELS
✅ Basic similarity concept            ✅ Similarity MATH explained
✅ ChromaDB via LangChain              ✅ ChromaDB NATIVE API
✅ Simple search                       ✅ Retrieval QUALITY metrics
                                       ✅ Parameter TUNING
                                       ✅ Advanced FILTERING
                                       ✅ HNSW algorithm explained
```

---

## Embedding Models Compared

Not all embedding models are equal. Here's how they differ:

```
EMBEDDING MODEL COMPARISON:
═══════════════════════════

┌──────────────────────────┬────────┬─────────┬──────────┬──────────┐
│ Model                    │ Dims   │ Speed   │ Quality  │ Cost     │
├──────────────────────────┼────────┼─────────┼──────────┼──────────┤
│ all-MiniLM-L6-v2         │ 384    │ ⚡ Fast  │ ★★★☆☆   │ FREE     │
│ all-mpnet-base-v2        │ 768    │ 🐢 Med  │ ★★★★☆   │ FREE     │
│ bge-small-en-v1.5        │ 384    │ ⚡ Fast  │ ★★★★☆   │ FREE     │
│ text-embedding-3-small   │ 1536   │ 🌐 API  │ ★★★★☆   │ $0.02/1M │
│ text-embedding-3-large   │ 3072   │ 🌐 API  │ ★★★★★   │ $0.13/1M │
└──────────────────────────┴────────┴─────────┴──────────┴──────────┘

RECOMMENDATIONS:
- Learning/Development: all-MiniLM-L6-v2 (free, fast, good enough)
- Production (budget): bge-small-en-v1.5 (free, excellent quality)
- Production (best): text-embedding-3-small (cheap, high quality)
```

### How Embedding Dimensions Affect Quality

```
MORE DIMENSIONS = MORE NUANCE (but diminishing returns)
══════════════════════════════════════════════════════

384 dimensions (MiniLM):
┌─────────────────────────────────────────────┐
│ Can distinguish: Terraform vs Kubernetes     │
│ Might confuse: Similar error messages        │
└─────────────────────────────────────────────┘

768 dimensions (mpnet):
┌──────────────────────────────────────────────────────────┐
│ Can distinguish: Terraform state lock vs provider error   │
│ Better at: Subtle meaning differences                     │
└──────────────────────────────────────────────────────────┘

1536 dimensions (OpenAI):
┌────────────────────────────────────────────────────────────────────┐
│ Can distinguish: Very subtle semantic differences                   │
│ Better at: Complex technical queries                                │
│ But: Requires API calls, costs money                               │
└────────────────────────────────────────────────────────────────────┘

For DevOps troubleshooting: 384 dimensions is usually enough!
```

---

## Similarity Metrics Explained

### Cosine Similarity (Most Common for Text)

Measures the **angle** between two vectors, ignoring their length.

```
COSINE SIMILARITY - STEP BY STEP:
══════════════════════════════════

Given two vectors:
    A = [3, 4]
    B = [4, 3]

Step 1: Dot product (multiply matching elements, sum)
    A · B = (3×4) + (4×3) = 12 + 12 = 24

Step 2: Magnitudes (length of each vector)
    |A| = √(3² + 4²) = √(9 + 16) = √25 = 5
    |B| = √(4² + 3²) = √(16 + 9) = √25 = 5

Step 3: Divide
    cosine = 24 / (5 × 5) = 24/25 = 0.96

Result: 0.96 → Very similar! (1.0 = identical)

VISUAL:
        ▲
      4 │    • B(4,3)
        │   /
      3 │  / • A(3,4)     Small angle = High similarity
        │ /
        │/
        └──────────►
        0  1  2  3  4
```

### Euclidean Distance

Measures the **straight-line distance** between two points.

```
EUCLIDEAN DISTANCE - STEP BY STEP:
══════════════════════════════════

Given two vectors:
    A = [3, 4]
    B = [4, 3]

Step 1: Subtract
    A - B = [3-4, 4-3] = [-1, 1]

Step 2: Square each
    [-1, 1]² = [1, 1]

Step 3: Sum and square root
    √(1 + 1) = √2 ≈ 1.41

Result: 1.41 → Small distance = similar

NOTE: Unlike cosine, smaller distance = MORE similar!

VISUAL:
        ▲
      4 │    • B(4,3)
        │    |
      3 │    • A(3,4)     Short distance = Similar
        │
        └──────────►
        0  1  2  3  4
```

### Dot Product

Measures both **angle and magnitude**.

```
DOT PRODUCT:
════════════

A · B = (3×4) + (4×3) = 24

Higher dot product = more similar
(when vectors are normalized, same as cosine similarity)
```

### Which Metric When?

```
┌─────────────────────┬──────────────────────────────────────────────┐
│ Metric              │ Use When                                      │
├─────────────────────┼──────────────────────────────────────────────┤
│ Cosine Similarity   │ Text search (default choice)                  │
│                     │ - Ignores document length                     │
│                     │ - Focus on meaning, not volume                │
│                     │                                                │
│ Euclidean Distance  │ When magnitude matters                        │
│                     │ - Comparing things of similar scale            │
│                     │ - General-purpose similarity                   │
│                     │                                                │
│ Dot Product         │ Already-normalized vectors                    │
│                     │ - Fastest computation                          │
│                     │ - Same as cosine when normalized               │
└─────────────────────┴──────────────────────────────────────────────┘

FOR DEVOPS RAG: Use cosine similarity (it's the default in ChromaDB)
```

---

## ChromaDB Native API

In Module 3 we used ChromaDB through LangChain. Now let's use it directly for more control.

### Why Use the Native API?

```
LANGCHAIN WRAPPER:                   CHROMADB NATIVE API:
══════════════════                   ════════════════════

✅ Simple, fewer lines               ✅ Full control over settings
✅ Integrates with chains            ✅ Complex filters
❌ Limited configuration             ✅ Direct collection management
❌ Hidden complexity                  ✅ Batch operations
                                      ✅ Update/delete specific docs

Use LangChain for:                   Use Native API for:
  Quick prototypes                     Production systems
  Simple RAG chains                    Advanced filtering
  Learning basics                      Fine-tuning retrieval
```

### Creating Collections

```python
import chromadb

# Create a persistent client (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection with cosine similarity
collection = client.get_or_create_collection(
    name="devops_knowledge",
    metadata={"hnsw:space": "cosine"}  # Similarity metric
)
```

### Adding Documents

```python
collection.add(
    documents=[
        "CrashLoopBackOff means the container keeps crashing.",
        "Terraform state lock error occurs during concurrent runs.",
        "Docker build cache can speed up builds significantly."
    ],
    metadatas=[
        {"category": "kubernetes", "severity": "high"},
        {"category": "terraform", "severity": "medium"},
        {"category": "docker", "severity": "low"}
    ],
    ids=["k8s_001", "tf_001", "docker_001"]
)
```

### Advanced Filtering

```python
# Simple filter
results = collection.query(
    query_texts=["pod crash error"],
    n_results=5,
    where={"category": "kubernetes"}
)

# Complex filter with AND/OR
results = collection.query(
    query_texts=["deployment error"],
    n_results=5,
    where={
        "$and": [
            {"category": {"$in": ["kubernetes", "docker"]}},
            {"severity": {"$eq": "high"}}
        ]
    }
)

# Filter by document content
results = collection.query(
    query_texts=["debugging"],
    n_results=5,
    where_document={"$contains": "kubectl"}
)
```

---

## How ChromaDB Finds Vectors (HNSW)

ChromaDB uses an algorithm called **HNSW** (Hierarchical Navigable Small World) for fast search.

```
HNSW EXPLAINED SIMPLY:
══════════════════════

BRUTE FORCE (slow): Compare query to EVERY vector
    Query ──► Compare with doc 1 ✓
    Query ──► Compare with doc 2 ✓
    Query ──► Compare with doc 3 ✓
    ...
    Query ──► Compare with doc 1,000,000 ✓
    Time: O(n) - checks every single document!

HNSW (fast): Navigate through a graph of connected vectors
    Layer 2 (few nodes):    A ─── B ─── C
                                  │
    Layer 1 (more nodes):   D ─ E ─ F ─ G ─ H
                                │       │
    Layer 0 (all nodes):    I J K L M N O P Q R S T

    The query starts at the top layer and "zooms in"
    to find the nearest neighbors quickly!

    Time: O(log n) - much faster!

WHY YOU SHOULD CARE:
- With 1,000 docs: Both are fast (doesn't matter)
- With 100,000 docs: HNSW is ~100x faster
- With 1,000,000 docs: HNSW is ~1000x faster
```

---

## Measuring Retrieval Quality

How do you know if your RAG system is finding the RIGHT documents?

```
RETRIEVAL QUALITY METRICS:
══════════════════════════

PRECISION: Of the docs returned, how many are relevant?
─────────────────────────────────────────────────────
    Returned: [Doc1✓, Doc2✓, Doc3✗, Doc4✓, Doc5✗]
    Precision = 3 relevant / 5 returned = 60%

RECALL: Of all relevant docs, how many were found?
─────────────────────────────────────────────────────
    All relevant docs: [Doc1, Doc2, Doc4, Doc7, Doc9]
    Found: [Doc1, Doc2, Doc4]
    Recall = 3 found / 5 total relevant = 60%

THE TRADE-OFF:
─────────────────────────────────────────────────────
    High k (many results): High recall, lower precision
    Low k (few results):   High precision, lower recall

    ┌──────────────────────────────────────────┐
    │ k=1:  Precision: 100%  Recall: 20%      │ Very precise but misses docs
    │ k=3:  Precision: 80%   Recall: 50%      │ Good balance ✓
    │ k=5:  Precision: 60%   Recall: 70%      │ Good balance ✓
    │ k=10: Precision: 40%   Recall: 90%      │ Finds most but noisy
    │ k=20: Precision: 25%   Recall: 95%      │ Very noisy
    └──────────────────────────────────────────┘
```

---

## Tuning Search Parameters

### The k Value

```python
# Too few: might miss relevant docs
results = search(query, k=1)   # ❌ Risky

# Too many: dilutes context with noise
results = search(query, k=20)  # ❌ Wasteful

# Sweet spot for DevOps RAG
results = search(query, k=3)   # ✅ Good default
results = search(query, k=5)   # ✅ Also good
```

### Score Thresholding

```python
def search_with_threshold(vectorstore, query, min_score=0.5, k=10):
    """Only return results above a relevance threshold."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return [(doc, score) for doc, score in results if score >= min_score]

# Prevents returning irrelevant docs when nothing matches well
```

### MMR (Maximum Marginal Relevance)

```
MMR REDUCES REDUNDANCY:
═══════════════════════

WITHOUT MMR:
    Query: "pod error"
    Result 1: "CrashLoopBackOff in pod web-app-1"
    Result 2: "CrashLoopBackOff in pod web-app-2"    ← Redundant!
    Result 3: "CrashLoopBackOff in pod web-app-3"    ← Redundant!

WITH MMR:
    Query: "pod error"
    Result 1: "CrashLoopBackOff in pod web-app-1"
    Result 2: "ImagePullBackOff in pod api-server"    ← Different error!
    Result 3: "OOMKilled in pod worker-node"          ← Different error!

MMR picks DIVERSE results that cover more ground!
```

---

## Exercises

### Exercise 1: Embedding Model Comparison
Compare different models on DevOps text.
→ `exercises/ex1_embedding_models.py`

### Exercise 2: Similarity Metrics
Hands-on cosine vs euclidean vs dot product.
→ `exercises/ex2_similarity_metrics.py`

### Exercise 3: ChromaDB Native API
Use ChromaDB directly with advanced features.
→ `exercises/ex3_chromadb_direct.py`

### Exercise 4: Retrieval Quality
Measure and improve search accuracy.
→ `exercises/ex4_retrieval_quality.py`

---

## Key Takeaways

```
┌──────────────────────────────────────────────────────────────────────┐
│                         KEY TAKEAWAYS                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. MiniLM is great for learning; consider bge/OpenAI for prod      │
│                                                                       │
│  2. COSINE SIMILARITY is the default for text search                │
│                                                                       │
│  3. ChromaDB native API gives fine-grained control                  │
│                                                                       │
│  4. MEASURE retrieval quality with precision & recall               │
│                                                                       │
│  5. USE MMR for diverse, non-redundant results                      │
│                                                                       │
│  6. k=3-5 is the sweet spot for most DevOps RAG systems            │
│                                                                       │
│  7. Score thresholds prevent returning irrelevant results           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What's Next?

In **Module 6**, we'll explore advanced prompting techniques:
- Few-shot learning for DevOps troubleshooting
- Chain-of-thought for complex debugging
- Structured output for actionable responses

```bash
cd ../module-6-advanced-prompting
```

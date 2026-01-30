#!/usr/bin/env python3
"""
Exercise 5: Similarity Search with ChromaDB
============================================

GOAL: Learn how to search a vector database and understand results.

WHAT YOU'LL LEARN:
- How to query ChromaDB
- Understanding similarity scores
- Filtering search results
- Tuning retrieval for better RAG

HOW SIMILARITY SEARCH WORKS:
---------------------------

    Step 1: Your query becomes a vector
    ┌──────────────────────┐         ┌─────────────────────────┐
    │ "fix pod crash"      │  ─────► │ [0.82, -0.15, 0.43, ...] │
    └──────────────────────┘         └─────────────────────────┘

    Step 2: Compare with all stored vectors
    ┌─────────────────────────────────────────────────────────────┐
    │ ChromaDB                                                     │
    │  ┌─────────────────────────────────────────────────────────┐ │
    │  │ Doc 1: [0.79, -0.12, 0.45, ...] → similarity = 0.95    │ │
    │  │ Doc 2: [0.11, 0.65, -0.33, ...] → similarity = 0.23    │ │
    │  │ Doc 3: [0.75, -0.18, 0.41, ...] → similarity = 0.89    │ │
    │  └─────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘

    Step 3: Return top-k most similar
    ┌─────────────────────────────────────────────────────────────┐
    │ Results (k=2):                                               │
    │   1. Doc 1 (similarity: 0.95) - "CrashLoopBackOff guide..."│
    │   2. Doc 3 (similarity: 0.89) - "Pod troubleshooting..."   │
    └─────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install chromadb langchain langchain-community sentence-transformers

RUN THIS:
    python module-3-rag-fundamentals/exercises/ex5_similarity_search.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# Sample DevOps documentation
SAMPLE_DOCS = [
    # Kubernetes docs
    Document(
        page_content="""## CrashLoopBackOff Error

The CrashLoopBackOff error means your container keeps crashing and Kubernetes
keeps trying to restart it. This creates a loop of crashes and restarts.

### Debugging Steps
1. Check pod logs: kubectl logs <pod-name>
2. Check previous container logs: kubectl logs <pod-name> --previous
3. Describe pod for events: kubectl describe pod <pod-name>

### Common Causes
- Application error on startup
- Missing environment variables
- Failed health checks
- Out of memory (OOMKilled)""",
        metadata={"source": "kubernetes/pod_errors.md", "category": "kubernetes"}
    ),
    Document(
        page_content="""## ImagePullBackOff Error

ImagePullBackOff occurs when Kubernetes cannot pull the container image.

### Common Causes
1. Image doesn't exist in registry
2. Wrong image name or tag
3. Private registry without credentials
4. Network issues

### Solutions
- Verify image: docker pull <image>
- Check spelling of image name
- Create imagePullSecret for private registries""",
        metadata={"source": "kubernetes/pod_errors.md", "category": "kubernetes"}
    ),
    Document(
        page_content="""## OOMKilled (Exit Code 137)

OOMKilled means your container exceeded its memory limit and was killed.

### Solution
Increase memory limits in your pod spec:

```yaml
resources:
  limits:
    memory: "512Mi"
  requests:
    memory: "256Mi"
```

### Prevention
- Profile your application's memory usage
- Set appropriate memory limits
- Implement memory-efficient code""",
        metadata={"source": "kubernetes/resources.md", "category": "kubernetes"}
    ),
    # Terraform docs
    Document(
        page_content="""## Terraform State Lock Error

The state lock error occurs when another Terraform process is running
or a previous run was interrupted.

### Error Message
Error: Error locking state: Error acquiring the state lock

### Solutions
1. Wait for other process to finish
2. Force unlock: terraform force-unlock <LOCK_ID>
3. Check CI/CD pipelines for running jobs

### Warning
Only force-unlock if you're SURE no other process is running!""",
        metadata={"source": "terraform/errors.md", "category": "terraform"}
    ),
    Document(
        page_content="""## Terraform Import Command

The terraform import command brings existing infrastructure under
Terraform management.

### Usage
terraform import <resource_type>.<name> <resource_id>

### Example
terraform import aws_instance.web i-1234567890abcdef0

### After Import
1. Write the corresponding resource block
2. Run terraform plan to verify
3. Adjust configuration to match actual resource""",
        metadata={"source": "terraform/commands.md", "category": "terraform"}
    ),
    # Docker docs
    Document(
        page_content="""## Docker Build Best Practices

Optimize your Docker builds for faster CI/CD pipelines.

### Multi-stage Builds
Use multi-stage builds to reduce image size:

```dockerfile
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM node:18-alpine
COPY --from=builder /app/dist ./dist
CMD ["node", "dist/index.js"]
```

### Layer Caching
Order commands from least to most frequently changing.""",
        metadata={"source": "docker/best_practices.md", "category": "docker"}
    ),
]


def create_vector_store():
    """
    Create a ChromaDB vector store with sample documents.
    """
    print("\n" + "=" * 60)
    print("CREATING VECTOR STORE")
    print("=" * 60)

    # Initialize embeddings
    print("\n📦 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    print("💾 Creating ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=SAMPLE_DOCS,
        embedding=embeddings,
        collection_name="devops_docs"
    )

    print(f"✅ Stored {len(SAMPLE_DOCS)} documents")
    return vectorstore


def demo_basic_search(vectorstore):
    """
    Demonstrate basic similarity search.
    """
    print("\n" + "=" * 60)
    print("BASIC SIMILARITY SEARCH")
    print("=" * 60)

    query = "my pod keeps crashing and restarting"
    print(f"\n🔍 Query: \"{query}\"")

    # Simple search - returns top k documents
    results = vectorstore.similarity_search(query, k=3)

    print(f"\n📚 Top {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n   {i}. Source: {doc.metadata.get('source', 'unknown')}")
        print(f"      Category: {doc.metadata.get('category', 'unknown')}")
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"      Preview: {preview}...")


def demo_search_with_scores(vectorstore):
    """
    Demonstrate search with similarity scores.
    """
    print("\n" + "=" * 60)
    print("SEARCH WITH SIMILARITY SCORES")
    print("=" * 60)

    query = "how to fix memory issues in container"
    print(f"\n🔍 Query: \"{query}\"")

    # Search with scores - returns (document, score) tuples
    results = vectorstore.similarity_search_with_score(query, k=5)

    print(f"\n📊 Results with scores:")
    print(f"\n   {'Score':<10} {'Source':<35} {'Preview'}")
    print("   " + "-" * 80)

    for doc, score in results:
        source = doc.metadata.get('source', 'unknown')[:32]
        preview = doc.page_content[:40].replace('\n', ' ')

        # Visualize score
        # Note: ChromaDB returns DISTANCE (lower = more similar)
        # Convert to similarity for display
        similarity = 1 / (1 + score)  # Rough conversion
        bar = "█" * int(similarity * 20)

        print(f"   {score:<10.4f} {source:<35} {preview}...")


def demo_filter_by_metadata(vectorstore):
    """
    Demonstrate filtering search results by metadata.
    """
    print("\n" + "=" * 60)
    print("FILTERING BY METADATA")
    print("=" * 60)

    query = "how to fix errors"
    print(f"\n🔍 Query: \"{query}\"")

    # Search ALL categories
    print("\n   [All Categories]")
    all_results = vectorstore.similarity_search(query, k=3)
    for doc in all_results:
        cat = doc.metadata.get('category', 'unknown')
        source = doc.metadata.get('source', 'unknown')
        print(f"   - [{cat}] {source}")

    # Filter: Only Kubernetes docs
    print("\n   [Filtered: Kubernetes only]")
    k8s_results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"category": "kubernetes"}
    )
    for doc in k8s_results:
        source = doc.metadata.get('source', 'unknown')
        print(f"   - {source}")

    # Filter: Only Terraform docs
    print("\n   [Filtered: Terraform only]")
    tf_results = vectorstore.similarity_search(
        query,
        k=3,
        filter={"category": "terraform"}
    )
    for doc in tf_results:
        source = doc.metadata.get('source', 'unknown')
        print(f"   - {source}")

    print("""
    METADATA FILTERING USE CASES:
    ─────────────────────────────
    • Filter by category (kubernetes, terraform, docker)
    • Filter by date/version
    • Filter by author or team
    • Filter by environment (prod, staging)
    """)


def demo_retriever_options(vectorstore):
    """
    Demonstrate different retriever configurations.
    """
    print("\n" + "=" * 60)
    print("RETRIEVER CONFIGURATIONS")
    print("=" * 60)

    query = "pod error troubleshooting"

    # Option 1: Basic retriever
    print("\n📋 Option 1: Basic Retriever (k=3)")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    for doc in docs:
        print(f"   - {doc.metadata.get('source', 'unknown')}")

    # Option 2: MMR (Maximum Marginal Relevance)
    # This reduces redundancy in results
    print("\n📋 Option 2: MMR Retriever (diverse results)")
    print("   MMR = Maximum Marginal Relevance")
    print("   Reduces redundancy by selecting diverse documents")

    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,  # Fetch more, then diversify
        }
    )
    docs_mmr = retriever_mmr.invoke(query)
    for doc in docs_mmr:
        print(f"   - {doc.metadata.get('source', 'unknown')}")

    # Option 3: With score threshold
    print("\n📋 Option 3: Score Threshold (only high-quality matches)")
    print("   Only returns results above a similarity threshold")

    retriever_threshold = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,  # Minimum similarity
            "k": 5
        }
    )
    docs_threshold = retriever_threshold.invoke(query)
    print(f"   Found {len(docs_threshold)} docs above threshold")
    for doc in docs_threshold:
        print(f"   - {doc.metadata.get('source', 'unknown')}")


def demo_compare_queries(vectorstore):
    """
    Compare different query phrasings and their results.
    """
    print("\n" + "=" * 60)
    print("COMPARING DIFFERENT QUERIES")
    print("=" * 60)

    queries = [
        "pod keeps crashing",
        "CrashLoopBackOff",
        "container restart loop kubernetes",
        "my application won't start",
    ]

    print("\n   All these queries are about the same problem.")
    print("   Let's see what results each one gets:\n")

    for query in queries:
        results = vectorstore.similarity_search_with_score(query, k=1)
        if results:
            doc, score = results[0]
            source = doc.metadata.get('source', 'unknown')
            print(f"   Query: \"{query}\"")
            print(f"   Best match: {source} (score: {score:.4f})\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 5: Similarity Search with ChromaDB")
    print("=" * 60)

    # Create vector store
    vectorstore = create_vector_store()

    # Run demos
    demo_basic_search(vectorstore)
    demo_search_with_scores(vectorstore)
    demo_filter_by_metadata(vectorstore)
    demo_retriever_options(vectorstore)
    demo_compare_queries(vectorstore)

    print("\n" + "=" * 60)
    print("SIMILARITY SEARCH BEST PRACTICES")
    print("=" * 60)
    print("""
    ┌────────────────────────────────────────────────────────────────┐
    │                  SEARCH BEST PRACTICES                          │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. CHOOSE THE RIGHT k VALUE                                   │
    │     - Too low (k=1): Might miss relevant docs                  │
    │     - Too high (k=10): Includes irrelevant noise               │
    │     - Sweet spot: k=3-5 for most RAG systems                   │
    │                                                                 │
    │  2. USE METADATA FILTERING                                     │
    │     - Filter by category when you know the topic               │
    │     - Reduces noise and improves relevance                     │
    │                                                                 │
    │  3. CONSIDER MMR FOR DIVERSITY                                 │
    │     - Prevents getting 3 very similar documents                │
    │     - Gives broader coverage of the topic                      │
    │                                                                 │
    │  4. USE SCORE THRESHOLDS CAREFULLY                             │
    │     - Prevents returning irrelevant results                    │
    │     - But might return nothing if threshold too high           │
    │                                                                 │
    │  5. TEST YOUR QUERIES                                          │
    │     - Try different phrasings                                   │
    │     - Check what your users actually type                      │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘

    RECOMMENDED SETTINGS FOR DEVOPS RAG:
    ────────────────────────────────────
    retriever = vectorstore.as_retriever(
        search_type="mmr",        # Diverse results
        search_kwargs={
            "k": 4,               # Return 4 documents
            "fetch_k": 10,        # Consider 10 candidates
        }
    )
    """)

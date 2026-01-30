#!/usr/bin/env python3
"""
Exercise 3: ChromaDB Native API -- Direct Database Operations
==============================================================

GOAL: Learn to use ChromaDB directly (NOT through LangChain) so you
      understand what's happening under the hood and can leverage
      advanced features like complex filtering, metadata updates,
      and multiple distance functions.

WHAT YOU'LL LEARN:
- ChromaDB's native Python API (without LangChain wrapper)
- Creating and managing collections
- Adding documents with metadata and custom IDs
- Advanced filtering with $and, $or, $in, $gt, $lt operators
- Updating and deleting documents
- Configuring distance functions (cosine, l2, ip) per collection

WHY USE CHROMADB DIRECTLY?
──────────────────────────
LangChain's Chroma wrapper is convenient but hides many features:

    ┌───────────────────────────────────────────────────────────────────┐
    │                                                                   │
    │  LangChain Chroma Wrapper                                        │
    │  ─────────────────────────                                       │
    │  - Simple API: from_documents(), similarity_search()             │
    │  - Good for quick prototyping                                    │
    │  - Limited filtering options                                     │
    │  - No direct update/delete by ID                                 │
    │  - Fixed distance function                                       │
    │                                                                   │
    │  ChromaDB Native API                                             │
    │  ────────────────────                                            │
    │  - Full control over collections                                 │
    │  - Advanced where/where_document filtering                       │
    │  - Update documents by ID                                        │
    │  - Delete by ID or filter                                        │
    │  - Choose distance function per collection                       │
    │  - Peek, count, get operations                                   │
    │  - Multi-tenant support                                          │
    │                                                                   │
    └───────────────────────────────────────────────────────────────────┘

CHROMADB ARCHITECTURE:
──────────────────────

    ┌───────────────────────────────────────────────────────────────────┐
    │  ChromaDB Client                                                  │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
    │  │ Collection:  │  │ Collection:  │  │ Collection:  │           │
    │  │ "k8s_docs"   │  │ "tf_docs"    │  │ "docker_docs"│           │
    │  │              │  │              │  │              │           │
    │  │ distance:    │  │ distance:    │  │ distance:    │           │
    │  │  cosine      │  │  l2          │  │  ip          │           │
    │  │              │  │              │  │              │           │
    │  │ Documents:   │  │ Documents:   │  │ Documents:   │           │
    │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │           │
    │  │  │ ID     │  │  │  │ ID     │  │  │  │ ID     │  │           │
    │  │  │ Text   │  │  │  │ Text   │  │  │  │ Text   │  │           │
    │  │  │ Vector │  │  │  │ Vector │  │  │  │ Vector │  │           │
    │  │  │Metadata│  │  │  │Metadata│  │  │  │Metadata│  │           │
    │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │           │
    │  │  ┌────────┐  │  │  ┌────────┐  │  │  ┌────────┐  │           │
    │  │  │ ...    │  │  │  │ ...    │  │  │  │ ...    │  │           │
    │  │  └────────┘  │  │  └────────┘  │  │  └────────┘  │           │
    │  └──────────────┘  └──────────────┘  └──────────────┘           │
    └───────────────────────────────────────────────────────────────────┘

    Each COLLECTION is like a table in a database:
    - Has its own distance function (cosine, l2, or ip)
    - Stores documents, embeddings, metadata, and unique IDs
    - Can be queried independently

DATA MODEL (per document):
──────────────────────────
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  ID: "doc_001"            <- Unique string identifier           │
    │  Document: "CrashLoop..."  <- The original text                 │
    │  Embedding: [0.23, ...]    <- The vector (auto-generated or     │
    │                                provided manually)               │
    │  Metadata: {               <- Key-value pairs for filtering     │
    │    "category": "kubernetes",                                    │
    │    "severity": "high",                                          │
    │    "created_at": 1700000000                                     │
    │  }                                                              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install chromadb sentence-transformers

RUN THIS:
    python module-5-vector-search/exercises/ex3_chromadb_direct.py
"""

import os
import sys
from pathlib import Path

# =============================================================================
# PROJECT SETUP
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ChromaDB -- the vector database itself (no LangChain wrapper!)
import chromadb

# SentenceTransformer -- to generate embeddings manually
# (ChromaDB can auto-embed with its default model, but we want control)
from sentence_transformers import SentenceTransformer


# =============================================================================
# SAMPLE DATA: DEVOPS KNOWLEDGE BASE
# =============================================================================
# Each document has: id, text, category, severity, and source.
# This simulates a real DevOps troubleshooting knowledge base.

DEVOPS_DOCUMENTS = [
    # ─── Kubernetes Documents ──────────────────────────────────────────
    {
        "id": "k8s_001",
        "text": "CrashLoopBackOff means your pod's container crashes on startup and "
                "Kubernetes keeps restarting it. Check logs with kubectl logs <pod> "
                "and describe pod events with kubectl describe pod <pod>.",
        "metadata": {
            "category": "kubernetes",
            "severity": "high",
            "topic": "pod-errors",
            "year": 2024,
        }
    },
    {
        "id": "k8s_002",
        "text": "OOMKilled (Exit Code 137) occurs when a container exceeds its memory "
                "limit. Increase the memory limit in pod spec under resources.limits.memory "
                "or optimize your application's memory usage.",
        "metadata": {
            "category": "kubernetes",
            "severity": "high",
            "topic": "pod-errors",
            "year": 2024,
        }
    },
    {
        "id": "k8s_003",
        "text": "ImagePullBackOff happens when Kubernetes cannot pull the container image. "
                "Verify the image name and tag, check registry credentials, and ensure "
                "network connectivity to the container registry.",
        "metadata": {
            "category": "kubernetes",
            "severity": "medium",
            "topic": "pod-errors",
            "year": 2024,
        }
    },
    {
        "id": "k8s_004",
        "text": "Kubernetes NetworkPolicy defines rules for pod-to-pod communication. "
                "By default, all pods can communicate. NetworkPolicies restrict traffic "
                "based on labels, namespaces, and ports.",
        "metadata": {
            "category": "kubernetes",
            "severity": "medium",
            "topic": "networking",
            "year": 2024,
        }
    },
    {
        "id": "k8s_005",
        "text": "Horizontal Pod Autoscaler (HPA) automatically scales the number of "
                "pod replicas based on CPU or memory usage. Configure with "
                "kubectl autoscale deployment <name> --min=2 --max=10 --cpu-percent=80.",
        "metadata": {
            "category": "kubernetes",
            "severity": "low",
            "topic": "scaling",
            "year": 2023,
        }
    },

    # ─── Terraform Documents ───────────────────────────────────────────
    {
        "id": "tf_001",
        "text": "Terraform state lock error occurs when another process holds the lock. "
                "Wait for the other process to complete, or force-unlock with "
                "terraform force-unlock <LOCK_ID>. Only force-unlock if certain no "
                "other process is running.",
        "metadata": {
            "category": "terraform",
            "severity": "high",
            "topic": "state",
            "year": 2024,
        }
    },
    {
        "id": "tf_002",
        "text": "Terraform import brings existing cloud resources under Terraform "
                "management. Use terraform import <resource_type>.<name> <resource_id>. "
                "After import, write the corresponding resource block and run plan.",
        "metadata": {
            "category": "terraform",
            "severity": "medium",
            "topic": "state",
            "year": 2024,
        }
    },
    {
        "id": "tf_003",
        "text": "Terraform modules allow you to organize and reuse infrastructure code. "
                "Create a module directory with main.tf, variables.tf, and outputs.tf. "
                "Reference modules using module blocks with source paths.",
        "metadata": {
            "category": "terraform",
            "severity": "low",
            "topic": "modules",
            "year": 2023,
        }
    },

    # ─── Docker Documents ──────────────────────────────────────────────
    {
        "id": "docker_001",
        "text": "Docker multi-stage builds reduce image size dramatically. Use multiple "
                "FROM statements to separate build and runtime stages. Only copy the "
                "necessary artifacts from the build stage to the final image.",
        "metadata": {
            "category": "docker",
            "severity": "low",
            "topic": "optimization",
            "year": 2024,
        }
    },
    {
        "id": "docker_002",
        "text": "Docker container networking: use bridge for single-host, overlay for "
                "multi-host Swarm, and host for maximum performance. Expose ports with "
                "-p host:container. Link containers by name on the same network.",
        "metadata": {
            "category": "docker",
            "severity": "medium",
            "topic": "networking",
            "year": 2023,
        }
    },

    # ─── CI/CD Documents ───────────────────────────────────────────────
    {
        "id": "cicd_001",
        "text": "GitHub Actions workflow fails at build step: check the error logs in "
                "the Actions tab, verify dependency versions in package.json, ensure "
                "environment secrets are configured in repository settings.",
        "metadata": {
            "category": "cicd",
            "severity": "high",
            "topic": "troubleshooting",
            "year": 2024,
        }
    },
    {
        "id": "cicd_002",
        "text": "Jenkins pipeline best practices: use declarative syntax, define stages "
                "clearly, store Jenkinsfile in version control, use shared libraries "
                "for common functionality, and implement proper error handling.",
        "metadata": {
            "category": "cicd",
            "severity": "low",
            "topic": "best-practices",
            "year": 2023,
        }
    },
]


# =============================================================================
# DEMO 1: CLIENT AND COLLECTION BASICS
# =============================================================================

def demo_client_and_collections():
    """
    Create a ChromaDB client and manage collections.

    A "collection" in ChromaDB is like a table in a relational database.
    Each collection stores documents, embeddings, and metadata.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: ChromaDB Client and Collection Management")
    print("=" * 70)

    # ─── Create Client ─────────────────────────────────────────────────
    # EphemeralClient() stores everything in memory (lost when process ends).
    # For persistence, use: chromadb.PersistentClient(path="./chroma_data")
    print("""
    ChromaDB Client Types:
    ──────────────────────
    1. EphemeralClient()         - In-memory only (great for testing)
    2. PersistentClient(path=)   - Saves to disk (for production)
    3. HttpClient(host=, port=)  - Connect to remote ChromaDB server
    """)

    print("  Creating in-memory (ephemeral) client...")
    client = chromadb.Client()  # Same as EphemeralClient()
    print("  Client created!")

    # ─── Create a Collection ───────────────────────────────────────────
    # get_or_create_collection: creates if doesn't exist, gets if it does.
    # This is idempotent -- safe to run multiple times.
    print("\n  Creating collection 'devops_kb'...")
    collection = client.get_or_create_collection(
        name="devops_kb",
        # metadata configures the collection itself
        metadata={
            # Distance function: how similarity is measured
            # Options: "cosine" (default), "l2" (Euclidean), "ip" (dot product)
            "hnsw:space": "cosine"
        }
    )
    print(f"  Collection created: '{collection.name}'")
    print(f"  Distance function: cosine (default)")

    # ─── List Collections ──────────────────────────────────────────────
    all_collections = client.list_collections()
    print(f"\n  All collections: {[c.name for c in all_collections]}")

    # ─── Collection Info ───────────────────────────────────────────────
    print(f"  Document count: {collection.count()}")

    # ─── Delete and Recreate (for clean demo) ──────────────────────────
    # delete_collection removes the entire collection and its data
    client.delete_collection("devops_kb")
    print("\n  Deleted collection 'devops_kb' (clean slate for next demo)")

    return client


# =============================================================================
# DEMO 2: ADDING DOCUMENTS WITH EMBEDDINGS
# =============================================================================

def demo_add_documents(client):
    """
    Add documents to ChromaDB with pre-computed embeddings and metadata.

    ChromaDB can auto-generate embeddings using its built-in model,
    but we compute them ourselves for full control over the model choice.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Adding Documents with Metadata")
    print("=" * 70)

    # ─── Load Our Embedding Model ──────────────────────────────────────
    print("\n  Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded!")

    # ─── Create Collection ─────────────────────────────────────────────
    collection = client.get_or_create_collection(
        name="devops_kb",
        metadata={"hnsw:space": "cosine"}
    )

    # ─── Prepare Data ──────────────────────────────────────────────────
    # ChromaDB expects parallel lists: ids, documents, embeddings, metadatas
    ids = []
    documents = []
    embeddings = []
    metadatas = []

    print(f"\n  Preparing {len(DEVOPS_DOCUMENTS)} documents...")

    for doc in DEVOPS_DOCUMENTS:
        ids.append(doc["id"])
        documents.append(doc["text"])
        metadatas.append(doc["metadata"])

        # Generate embedding for this document
        vec = model.encode(doc["text"])
        embeddings.append(vec.tolist())  # ChromaDB expects plain lists, not numpy arrays

    # ─── Add All Documents at Once (Batch Insert) ──────────────────────
    # Batch insert is MUCH faster than adding one at a time.
    # ChromaDB processes them all in a single operation.
    print(f"  Adding {len(ids)} documents to collection...")

    collection.add(
        ids=ids,                # Unique string IDs
        documents=documents,    # Original text (stored for retrieval)
        embeddings=embeddings,  # Pre-computed vectors
        metadatas=metadatas,    # Key-value metadata for filtering
    )

    print(f"  Done! Collection now has {collection.count()} documents.")

    # ─── Verify with Peek ──────────────────────────────────────────────
    # peek() returns a few documents as a sanity check
    print("\n  Peeking at first 3 documents:")
    peek_result = collection.peek(limit=3)
    for i in range(min(3, len(peek_result["ids"]))):
        doc_id = peek_result["ids"][i]
        doc_text = peek_result["documents"][i][:60] + "..."
        doc_meta = peek_result["metadatas"][i]
        print(f"    [{doc_id}] {doc_text}")
        print(f"             metadata: {doc_meta}")

    print("""
    DATA STRUCTURE IN CHROMADB:
    ──────────────────────────
    Each document is stored as:

    ┌──────────────────────────────────────────────────────────────────┐
    │ ID:        "k8s_001"                                             │
    │ Document:  "CrashLoopBackOff means your pod's container..."     │
    │ Embedding: [0.234, -0.567, 0.891, ..., 0.123]  (384 floats)    │
    │ Metadata:  {"category": "kubernetes", "severity": "high", ...}  │
    └──────────────────────────────────────────────────────────────────┘
    """)

    return collection, model


# =============================================================================
# DEMO 3: QUERYING WITH THE NATIVE API
# =============================================================================

def demo_querying(collection, model):
    """
    Query ChromaDB using its native query API.

    The native API gives you more control over results compared to
    LangChain's similarity_search().
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Querying ChromaDB (Native API)")
    print("=" * 70)

    # ─── Basic Query ───────────────────────────────────────────────────
    query_text = "my pod keeps crashing and restarting"
    print(f"\n  Query: \"{query_text}\"")

    # Generate query embedding (must use same model as documents!)
    query_embedding = model.encode(query_text).tolist()

    # The query() method returns results with distances
    results = collection.query(
        query_embeddings=[query_embedding],  # List of query vectors (can batch!)
        n_results=5,                         # How many results to return
        include=["documents", "metadatas", "distances"],  # What to include
    )

    # ─── Parse Results ─────────────────────────────────────────────────
    # Results are nested lists because query() supports batch queries.
    # results["ids"][0] = IDs for first query
    # results["distances"][0] = distances for first query
    # etc.
    print(f"\n  Top {len(results['ids'][0])} results:")
    print(f"  " + "-" * 60)

    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        document = results["documents"][0][i]
        metadata = results["metadatas"][0][i]

        # ChromaDB returns DISTANCE, not similarity.
        # For cosine: distance = 1 - cosine_similarity
        # So: similarity = 1 - distance
        similarity = 1 - distance

        short_doc = document[:60] + "..." if len(document) > 60 else document
        print(f"    {i+1}. [{doc_id}] (similarity: {similarity:.4f}, distance: {distance:.4f})")
        print(f"       Category: {metadata['category']} | Severity: {metadata['severity']}")
        print(f"       {short_doc}")

    print("""
    UNDERSTANDING CHROMADB DISTANCES:
    ─────────────────────────────────
    ChromaDB returns DISTANCE (lower = more similar), not similarity.

    For "cosine" space:
        distance = 1 - cosine_similarity
        distance = 0      -> cosine_similarity = 1.0 (identical)
        distance = 1      -> cosine_similarity = 0.0 (unrelated)
        distance = 2      -> cosine_similarity = -1.0 (opposite)

    For "l2" space:
        distance = Euclidean distance squared (L2 squared)
        distance = 0      -> identical
        distance = large   -> very different

    For "ip" space (inner product / dot product):
        distance = 1 - dot_product  (for normalized vectors)
        Behaves like cosine when vectors are normalized.

    ┌────────────────────────────────────────────────────────────────────┐
    │  GOTCHA: LangChain's similarity_search_with_score() returns      │
    │  these raw distances, NOT similarity scores! Many beginners      │
    │  get confused thinking higher = better, but it's the opposite.  │
    └────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# DEMO 4: ADVANCED FILTERING
# =============================================================================

def demo_advanced_filtering(collection, model):
    """
    Demonstrate ChromaDB's powerful filtering capabilities.

    ChromaDB supports two types of filters:
    1. where: Filter by metadata fields
    2. where_document: Filter by document text content
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: Advanced Filtering ($and, $or, $in, comparison operators)")
    print("=" * 70)

    query_text = "troubleshoot errors"
    query_embedding = model.encode(query_text).tolist()

    # ─── Filter 1: Simple Equality ─────────────────────────────────────
    print(f"\n  FILTER 1: Simple equality -- category = 'kubernetes'")
    print("  " + "-" * 60)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"category": "kubernetes"},  # Simple key-value filter
        include=["documents", "metadatas", "distances"],
    )

    print(f"  Found {len(results['ids'][0])} Kubernetes docs:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        cat = results["metadatas"][0][i]["category"]
        topic = results["metadatas"][0][i]["topic"]
        print(f"    [{doc_id}] category={cat}, topic={topic}")

    # ─── Filter 2: $and Operator ───────────────────────────────────────
    print(f"\n  FILTER 2: $and -- category='kubernetes' AND severity='high'")
    print("  " + "-" * 60)
    print("""    Syntax:
    {"$and": [
        {"category": "kubernetes"},
        {"severity": "high"}
    ]}
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={
            "$and": [
                {"category": "kubernetes"},
                {"severity": "high"}
            ]
        },
        include=["documents", "metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} high-severity Kubernetes docs:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        print(f"    [{doc_id}] severity={meta['severity']}, topic={meta['topic']}")

    # ─── Filter 3: $or Operator ────────────────────────────────────────
    print(f"\n  FILTER 3: $or -- category='kubernetes' OR category='docker'")
    print("  " + "-" * 60)
    print("""    Syntax:
    {"$or": [
        {"category": "kubernetes"},
        {"category": "docker"}
    ]}
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={
            "$or": [
                {"category": "kubernetes"},
                {"category": "docker"}
            ]
        },
        include=["metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} Kubernetes or Docker docs:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        cat = results["metadatas"][0][i]["category"]
        print(f"    [{doc_id}] category={cat}")

    # ─── Filter 4: $in Operator ────────────────────────────────────────
    print(f"\n  FILTER 4: $in -- category in ['kubernetes', 'terraform']")
    print("  " + "-" * 60)
    print("""    Syntax:
    {"category": {"$in": ["kubernetes", "terraform"]}}
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"category": {"$in": ["kubernetes", "terraform"]}},
        include=["metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} K8s or Terraform docs:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        cat = results["metadatas"][0][i]["category"]
        print(f"    [{doc_id}] category={cat}")

    # ─── Filter 5: Comparison Operators ────────────────────────────────
    print(f"\n  FILTER 5: Comparison -- year >= 2024")
    print("  " + "-" * 60)
    print("""    Syntax:
    {"year": {"$gte": 2024}}

    Available comparison operators:
      $gt   - greater than
      $gte  - greater than or equal
      $lt   - less than
      $lte  - less than or equal
      $ne   - not equal
      $eq   - equal (same as simple key-value)
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={"year": {"$gte": 2024}},
        include=["metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} docs from 2024+:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        year = results["metadatas"][0][i]["year"]
        cat = results["metadatas"][0][i]["category"]
        print(f"    [{doc_id}] year={year}, category={cat}")

    # ─── Filter 6: Combined Complex Filter ─────────────────────────────
    print(f"\n  FILTER 6: Complex -- (K8s OR Terraform) AND severity='high' AND year>=2024")
    print("  " + "-" * 60)
    print("""    Syntax:
    {"$and": [
        {"$or": [
            {"category": "kubernetes"},
            {"category": "terraform"}
        ]},
        {"severity": "high"},
        {"year": {"$gte": 2024}}
    ]}
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where={
            "$and": [
                {"$or": [
                    {"category": "kubernetes"},
                    {"category": "terraform"}
                ]},
                {"severity": "high"},
                {"year": {"$gte": 2024}}
            ]
        },
        include=["metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} docs matching complex filter:")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        meta = results["metadatas"][0][i]
        print(f"    [{doc_id}] cat={meta['category']}, sev={meta['severity']}, year={meta['year']}")

    # ─── Filter 7: where_document (text content filter) ────────────────
    print(f"\n  FILTER 7: where_document -- text contains 'kubectl'")
    print("  " + "-" * 60)
    print("""    Syntax:
    where_document={"$contains": "kubectl"}

    This searches the document TEXT, not metadata!
    Useful for finding docs that mention specific commands or tools.
    """)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        where_document={"$contains": "kubectl"},
        include=["documents", "metadatas"],
    )

    print(f"  Found {len(results['ids'][0])} docs containing 'kubectl':")
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        doc_preview = results["documents"][0][i][:60] + "..."
        print(f"    [{doc_id}] {doc_preview}")

    print("""
    FILTER CHEAT SHEET:
    ───────────────────
    ┌───────────────────────────────┬──────────────────────────────────┐
    │ Filter                        │ Syntax                           │
    ├───────────────────────────────┼──────────────────────────────────┤
    │ Equality                      │ {"key": "value"}                 │
    │ Not equal                     │ {"key": {"$ne": "value"}}        │
    │ In list                       │ {"key": {"$in": ["a", "b"]}}     │
    │ Not in list                   │ {"key": {"$nin": ["a", "b"]}}    │
    │ Greater than                  │ {"key": {"$gt": 5}}              │
    │ Greater or equal              │ {"key": {"$gte": 5}}             │
    │ Less than                     │ {"key": {"$lt": 5}}              │
    │ Less or equal                 │ {"key": {"$lte": 5}}             │
    │ AND                           │ {"$and": [{...}, {...}]}         │
    │ OR                            │ {"$or": [{...}, {...}]}          │
    │ Text contains                 │ where_document={"$contains": x}  │
    │ Text not contains             │ where_document={"$not_contains":}│
    └───────────────────────────────┴──────────────────────────────────┘
    """)


# =============================================================================
# DEMO 5: UPDATE AND DELETE DOCUMENTS
# =============================================================================

def demo_update_and_delete(collection, model):
    """
    Update existing documents and delete documents from ChromaDB.

    In a real DevOps knowledge base, documentation changes frequently.
    You need to update existing docs and remove outdated ones.
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: Updating and Deleting Documents")
    print("=" * 70)

    print(f"\n  Current document count: {collection.count()}")

    # ─── GET: Retrieve Specific Documents by ID ────────────────────────
    print("\n  GET: Retrieve document by ID")
    print("  " + "-" * 60)

    result = collection.get(
        ids=["k8s_001"],
        include=["documents", "metadatas"]
    )

    print(f"  Document k8s_001:")
    print(f"    Text: {result['documents'][0][:70]}...")
    print(f"    Metadata: {result['metadatas'][0]}")

    # ─── UPDATE: Modify an Existing Document ───────────────────────────
    print("\n  UPDATE: Modify document k8s_001")
    print("  " + "-" * 60)

    new_text = (
        "CrashLoopBackOff means your pod's container crashes on startup and "
        "Kubernetes keeps restarting it. UPDATED: Also check for failing "
        "liveness probes and readiness probes. Debug with: "
        "kubectl logs <pod> --previous and kubectl describe pod <pod>."
    )
    new_embedding = model.encode(new_text).tolist()

    collection.update(
        ids=["k8s_001"],
        documents=[new_text],
        embeddings=[new_embedding],
        metadatas=[{
            "category": "kubernetes",
            "severity": "high",
            "topic": "pod-errors",
            "year": 2024,
            "updated": True,  # Track that this doc was updated
        }]
    )

    # Verify the update
    updated = collection.get(ids=["k8s_001"], include=["documents", "metadatas"])
    print(f"  Updated text: {updated['documents'][0][:70]}...")
    print(f"  Updated metadata: {updated['metadatas'][0]}")
    print(f"  Document count (unchanged): {collection.count()}")

    # ─── UPSERT: Insert or Update ──────────────────────────────────────
    print("\n  UPSERT: Insert-or-Update (idempotent)")
    print("  " + "-" * 60)
    print("  upsert() creates the document if it doesn't exist,")
    print("  or updates it if it does. Safe to call multiple times.")

    new_doc_text = (
        "Kubernetes Ingress Controller manages external access to services. "
        "Common controllers: nginx-ingress, traefik, and HAProxy. Configure "
        "TLS termination, path-based routing, and host-based routing."
    )

    collection.upsert(
        ids=["k8s_006"],  # New ID -- will be created
        documents=[new_doc_text],
        embeddings=[model.encode(new_doc_text).tolist()],
        metadatas=[{
            "category": "kubernetes",
            "severity": "medium",
            "topic": "networking",
            "year": 2024,
        }]
    )

    print(f"  Upserted k8s_006 (new document)")
    print(f"  Document count: {collection.count()}")

    # ─── DELETE: Remove Documents ──────────────────────────────────────
    print("\n  DELETE: Remove documents")
    print("  " + "-" * 60)

    # Delete by ID
    print("  Deleting k8s_006 by ID...")
    collection.delete(ids=["k8s_006"])
    print(f"  Document count after delete: {collection.count()}")

    # Delete by filter (be careful -- this deletes ALL matching docs!)
    print("\n  Delete by filter example (showing syntax only):")
    print("""
    # Delete all 2023 documents:
    collection.delete(
        where={"year": {"$lte": 2023}}
    )

    # Delete all low-severity docs:
    collection.delete(
        where={"severity": "low"}
    )
    """)

    print("""
    ┌────────────────────────────────────────────────────────────────────┐
    │  CRUD OPERATIONS SUMMARY:                                          │
    │                                                                    │
    │  collection.add()     - Insert new documents (fails if ID exists) │
    │  collection.get()     - Retrieve by ID(s) or filter               │
    │  collection.update()  - Modify existing documents (by ID)         │
    │  collection.upsert()  - Insert or update (idempotent, safest)     │
    │  collection.delete()  - Remove by ID(s) or filter                 │
    │  collection.query()   - Semantic search with optional filters     │
    │  collection.peek()    - Quick look at a few documents             │
    │  collection.count()   - Total document count                      │
    │                                                                    │
    │  TIP: In production, prefer upsert() over add() + update()       │
    │  because it's idempotent -- safe to retry on failure.             │
    └────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# DEMO 6: DISTANCE FUNCTIONS COMPARISON
# =============================================================================

def demo_distance_functions(client, model):
    """
    Create three collections with different distance functions and
    compare search results.

    This demonstrates how the choice of distance function affects
    which documents are considered "most similar".
    """
    print("\n" + "=" * 70)
    print("  DEMO 6: Different Distance Functions (cosine, l2, ip)")
    print("=" * 70)

    print("""
    ChromaDB supports three distance functions per collection:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ Function │ Name in ChromaDB │ What It Measures    │ Lower = Better │
    ├──────────┼──────────────────┼─────────────────────┼────────────────┤
    │ Cosine   │ "cosine"         │ 1 - cos(angle)      │ Yes            │
    │ L2       │ "l2"             │ Euclidean dist^2    │ Yes            │
    │ IP       │ "ip"             │ 1 - dot_product     │ Yes (normalized)│
    └──────────┴──────────────────┴─────────────────────┴────────────────┘

    Note: ChromaDB returns DISTANCES (lower = more similar) for all three.
    The raw metric values are transformed so that lower is always better.
    """)

    # A smaller set of docs for clear comparison
    docs = DEVOPS_DOCUMENTS[:6]  # First 6 documents
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metas = [d["metadata"] for d in docs]
    embeddings = [model.encode(t).tolist() for t in texts]

    # ─── Create Three Collections with Different Distance Functions ────
    distance_functions = ["cosine", "l2", "ip"]
    collections = {}

    for dist_fn in distance_functions:
        col_name = f"test_{dist_fn}"

        # Delete if exists from a previous run
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        col = client.create_collection(
            name=col_name,
            metadata={"hnsw:space": dist_fn}
        )

        col.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )

        collections[dist_fn] = col
        print(f"  Created collection '{col_name}' with {dist_fn} distance")

    # ─── Query All Three ───────────────────────────────────────────────
    query = "my kubernetes container keeps crashing"
    query_vec = model.encode(query).tolist()

    print(f"\n  Query: \"{query}\"")
    print(f"\n  Results from each distance function:\n")

    for dist_fn in distance_functions:
        col = collections[dist_fn]
        results = col.query(
            query_embeddings=[query_vec],
            n_results=4,
            include=["distances", "metadatas"]
        )

        print(f"  [{dist_fn.upper()}] distance function:")
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            distance = results["distances"][0][i]
            cat = results["metadatas"][0][i]["category"]
            print(f"    {i+1}. [{doc_id}] distance={distance:.6f}  (cat={cat})")
        print()

    # Check if rankings differ
    rankings = {}
    for dist_fn in distance_functions:
        col = collections[dist_fn]
        results = col.query(
            query_embeddings=[query_vec],
            n_results=4,
            include=["distances"]
        )
        rankings[dist_fn] = results["ids"][0]

    print("  RANKING COMPARISON (top 4 documents):")
    print(f"  " + "-" * 60)
    print(f"    {'Rank':<6} {'Cosine':<12} {'L2':<12} {'IP':<12}")
    print(f"    {'─'*6} {'─'*12} {'─'*12} {'─'*12}")
    for rank in range(4):
        cos_id = rankings["cosine"][rank]
        l2_id = rankings["l2"][rank]
        ip_id = rankings["ip"][rank]
        print(f"    #{rank+1:<5} {cos_id:<12} {l2_id:<12} {ip_id:<12}")

    # Clean up test collections
    for dist_fn in distance_functions:
        try:
            client.delete_collection(f"test_{dist_fn}")
        except Exception:
            pass

    print("""
    WHICH DISTANCE FUNCTION TO USE?
    ────────────────────────────────
    - "cosine" (DEFAULT): Best general-purpose choice. Ignores vector
      magnitude, focuses on direction. Use this unless you have a
      specific reason not to.

    - "l2" (Euclidean): Sensitive to magnitude. Use when vector lengths
      carry meaningful information (rare with text embeddings).

    - "ip" (Inner Product): Fastest computation. Equivalent to cosine
      when vectors are normalized. Use with normalized embeddings
      for best performance.

    RECOMMENDATION FOR YOUR DEVOPS CHATBOT:
    ────────────────────────────────────────
    Use "cosine" to start. Switch to "ip" with normalized embeddings
    when you need maximum query speed at scale.
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 5, EXERCISE 3: ChromaDB Native API")
    print("  Direct database operations without LangChain")
    print("=" * 70)

    # Demo 1: Client and collection management
    client = demo_client_and_collections()

    # Demo 2: Adding documents with embeddings and metadata
    collection, model = demo_add_documents(client)

    # Demo 3: Querying with native API
    demo_querying(collection, model)

    # Demo 4: Advanced filtering
    demo_advanced_filtering(collection, model)

    # Demo 5: Updating and deleting documents
    demo_update_and_delete(collection, model)

    # Demo 6: Distance function comparison
    demo_distance_functions(client, model)

    # ─── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  1. CHROMADB NATIVE API gives you full control                        │
    │     - Create/delete/list collections                                  │
    │     - CRUD operations (add, get, update, upsert, delete)             │
    │     - Advanced query filtering                                        │
    │     - Per-collection distance function configuration                  │
    │                                                                        │
    │  2. COLLECTIONS are like database tables                               │
    │     - Each has its own distance function                               │
    │     - Use separate collections for different doc types if needed      │
    │     - Or use one collection with metadata filters                     │
    │                                                                        │
    │  3. FILTERING is powerful and composable                               │
    │     - $and, $or for combining conditions                              │
    │     - $in, $nin for list membership                                   │
    │     - $gt, $gte, $lt, $lte for numeric comparisons                    │
    │     - where_document for text content search                          │
    │     - Combine vector similarity + metadata filters for best results  │
    │                                                                        │
    │  4. ALWAYS USE THE SAME EMBEDDING MODEL for documents and queries     │
    │     - Mixing models produces garbage results                          │
    │     - Store the model name in collection metadata for reference       │
    │                                                                        │
    │  5. CHROMADB RETURNS DISTANCES (lower = more similar)                  │
    │     - Cosine: distance = 1 - cosine_similarity                        │
    │     - L2: distance = euclidean_distance_squared                       │
    │     - This catches beginners who expect higher = better               │
    │                                                                        │
    │  6. PREFER UPSERT OVER ADD+UPDATE                                     │
    │     - Idempotent: safe to retry on failure                            │
    │     - Simplifies your document ingestion pipeline                     │
    │                                                                        │
    │  NEXT: Exercise 4 measures retrieval QUALITY -- precision, recall,    │
    │        and how to tune k values and search strategies for your        │
    │        DevOps chatbot.                                                │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """)

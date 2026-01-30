#!/usr/bin/env python3
"""
SOLUTION: Exercise 3 - ChromaDB Native API
============================================

Full CRUD operations using the ChromaDB Python client directly
(no LangChain wrapper). Demonstrates collections, metadata filters,
distance functions, updates, and deletes.

RUN THIS:
    python module-5-vector-search/solutions/ex3_chromadb_direct.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.utils import embedding_functions


# --- Sample DevOps knowledge base ---

DOCUMENTS = [
    {
        "id": "k8s-001",
        "text": "CrashLoopBackOff means the container keeps crashing and restarting. "
                "Check logs with kubectl logs <pod> --previous.",
        "metadata": {"category": "kubernetes", "severity": "high", "tool": "kubectl"},
    },
    {
        "id": "k8s-002",
        "text": "OOMKilled (exit code 137) means the container ran out of memory. "
                "Increase memory limits in the pod spec.",
        "metadata": {"category": "kubernetes", "severity": "high", "tool": "kubectl"},
    },
    {
        "id": "k8s-003",
        "text": "ImagePullBackOff means Kubernetes cannot pull the container image. "
                "Verify the image name, tag, and registry credentials.",
        "metadata": {"category": "kubernetes", "severity": "medium", "tool": "kubectl"},
    },
    {
        "id": "tf-001",
        "text": "Terraform state lock error: another process holds the lock. "
                "Use terraform force-unlock <LOCK_ID> to release it.",
        "metadata": {"category": "terraform", "severity": "medium", "tool": "terraform"},
    },
    {
        "id": "tf-002",
        "text": "Terraform import lets you bring existing infrastructure under management. "
                "Run terraform import <resource_type>.<name> <id>.",
        "metadata": {"category": "terraform", "severity": "low", "tool": "terraform"},
    },
    {
        "id": "docker-001",
        "text": "Multi-stage Docker builds reduce image size. Use FROM ... AS builder "
                "to separate build and runtime stages.",
        "metadata": {"category": "docker", "severity": "low", "tool": "docker"},
    },
    {
        "id": "ci-001",
        "text": "Jenkins pipeline timeout: increase the timeout directive or split "
                "long-running stages into parallel steps.",
        "metadata": {"category": "cicd", "severity": "medium", "tool": "jenkins"},
    },
]


def create_collection_demo(client: chromadb.Client):
    """Create a collection with a specific distance function."""
    # Use HuggingFace sentence-transformers for embeddings
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Create collection -- cosine is best for text similarity
    collection = client.get_or_create_collection(
        name="devops_knowledge",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # distance function
    )
    print(f"Collection created: {collection.name}")
    print(f"  Distance function: cosine")
    return collection


def add_documents_demo(collection):
    """Add documents with metadata and explicit IDs."""
    collection.add(
        ids=[d["id"] for d in DOCUMENTS],
        documents=[d["text"] for d in DOCUMENTS],
        metadatas=[d["metadata"] for d in DOCUMENTS],
    )
    print(f"Added {collection.count()} documents")


def query_basic(collection):
    """Basic semantic search."""
    results = collection.query(
        query_texts=["my pod keeps crashing"],
        n_results=3,
    )
    print("\nBasic query: 'my pod keeps crashing'")
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"  [{dist:.4f}] ({meta['category']}) {doc[:80]}...")


def query_with_filters(collection):
    """Query with metadata filters using $and and $in operators."""
    # Filter: kubernetes only
    results = collection.query(
        query_texts=["error troubleshooting"],
        n_results=3,
        where={"category": "kubernetes"},
    )
    print("\nFiltered query (kubernetes only):")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  ({meta['severity']}) {doc[:80]}...")

    # Filter: $and -- kubernetes AND high severity
    results = collection.query(
        query_texts=["container problem"],
        n_results=3,
        where={"$and": [
            {"category": "kubernetes"},
            {"severity": "high"},
        ]},
    )
    print("\nFiltered query (kubernetes AND high severity):")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['id'] if 'id' in meta else ''}] {doc[:80]}...")

    # Filter: $in -- multiple categories
    results = collection.query(
        query_texts=["infrastructure issue"],
        n_results=3,
        where={"category": {"$in": ["terraform", "docker"]}},
    )
    print("\nFiltered query (terraform OR docker):")
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  ({meta['category']}) {doc[:80]}...")


def update_documents_demo(collection):
    """Update an existing document's text and metadata."""
    print("\nBefore update:")
    result = collection.get(ids=["docker-001"])
    print(f"  {result['documents'][0][:80]}...")

    collection.update(
        ids=["docker-001"],
        documents=[
            "Multi-stage Docker builds reduce image size significantly. Use FROM ... AS builder "
            "for the build stage, then COPY --from=builder to keep only runtime artifacts."
        ],
        metadatas=[{"category": "docker", "severity": "medium", "tool": "docker"}],
    )

    print("After update:")
    result = collection.get(ids=["docker-001"])
    print(f"  {result['documents'][0][:80]}...")


def delete_documents_demo(collection):
    """Delete documents by ID and by filter."""
    print(f"\nCount before delete: {collection.count()}")

    # Delete by ID
    collection.delete(ids=["ci-001"])
    print(f"Deleted ci-001. Count: {collection.count()}")

    # Delete by metadata filter
    collection.delete(where={"category": "terraform"})
    print(f"Deleted terraform docs. Count: {collection.count()}")


def distance_functions_demo(client: chromadb.Client):
    """Show the three distance functions ChromaDB supports."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    query = "container crashes"
    docs = [d["text"] for d in DOCUMENTS[:4]]
    ids = [d["id"] for d in DOCUMENTS[:4]]

    print("\nDistance function comparison for: '" + query + "'")
    print(f"{'Rank':<6} {'Cosine':>10} {'L2':>10} {'IP':>10}  {'Document':<50}")
    print("-" * 90)

    scores = {"cosine": {}, "l2": {}, "ip": {}}
    for space in ["cosine", "l2", "ip"]:
        col_name = f"demo_{space}"
        # Clean slate
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

        col = client.create_collection(
            name=col_name,
            embedding_function=ef,
            metadata={"hnsw:space": space},
        )
        col.add(ids=ids, documents=docs)
        result = col.query(query_texts=[query], n_results=4)

        for doc_id, dist in zip(result["ids"][0], result["distances"][0]):
            scores[space][doc_id] = dist

        # Cleanup
        client.delete_collection(col_name)

    # Print combined table
    for i, doc_id in enumerate(ids):
        print(f"  {doc_id:<6} {scores['cosine'].get(doc_id, 0):>10.4f} "
              f"{scores['l2'].get(doc_id, 0):>10.4f} "
              f"{scores['ip'].get(doc_id, 0):>10.4f}  "
              f"{docs[i][:48]}...")


if __name__ == "__main__":
    # In-memory client (no persistence needed for demo)
    client = chromadb.Client()

    print("=" * 60)
    print("CHROMADB NATIVE API - FULL CRUD DEMO")
    print("=" * 60)

    # CREATE
    print("\n--- CREATE ---")
    collection = create_collection_demo(client)
    add_documents_demo(collection)

    # READ (queries)
    print("\n--- READ (Queries) ---")
    query_basic(collection)
    query_with_filters(collection)

    # UPDATE
    print("\n--- UPDATE ---")
    update_documents_demo(collection)

    # DELETE
    print("\n--- DELETE ---")
    delete_documents_demo(collection)

    # Distance function comparison
    print("\n--- DISTANCE FUNCTIONS ---")
    distance_functions_demo(client)

    print("\n" + "=" * 60)
    print("Demo complete. All operations used chromadb directly (no LangChain).")

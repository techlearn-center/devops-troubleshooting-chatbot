#!/usr/bin/env python3
"""
Exercise 2: Metadata Filtering with ChromaDB
==============================================

GOAL:
-----
Learn how to store documents in a vector database (ChromaDB) **with
metadata**, then use that metadata to filter search results at query time.

WHY THIS MATTERS:
-----------------
Imagine a user asks: "How do I fix a CrashLoopBackOff error?"

Without metadata filtering, the vector search might return documents from
every category -- Terraform state errors, Docker build tips, CI/CD
workflows -- because some of those documents also mention "errors" or
"fix".

With metadata filtering, you can tell the search: "Only look in the
kubernetes category."  This dramatically improves result quality.

    +------------------------------------------------------------------+
    |             UNFILTERED vs FILTERED SEARCH                         |
    +------------------------------------------------------------------+
    |                                                                    |
    |  UNFILTERED SEARCH:                                                |
    |  Query: "fix CrashLoopBackOff"                                     |
    |                                                                    |
    |    Result 1: kubernetes/pod-errors.md     (relevant!)              |
    |    Result 2: docker/runtime.md            (somewhat relevant)      |
    |    Result 3: terraform/errors.md          (NOT relevant)           |
    |    Result 4: cicd/github-actions.md       (NOT relevant)           |
    |                                                                    |
    |  FILTERED SEARCH (category = "kubernetes"):                        |
    |  Query: "fix CrashLoopBackOff"                                     |
    |                                                                    |
    |    Result 1: kubernetes/pod-errors.md     (relevant!)              |
    |    Result 2: kubernetes/debugging.md      (relevant!)              |
    |    Result 3: kubernetes/networking.md     (somewhat relevant)      |
    |                                                                    |
    |  --> Filtered results are ALL from the right domain!               |
    |                                                                    |
    +------------------------------------------------------------------+

WHAT IS CHROMADB?
-----------------
ChromaDB is an open-source vector database.  It stores:
  - Documents (the actual text)
  - Embeddings (numerical representations of meaning)
  - Metadata (key-value pairs like category, topic, etc.)

You can then search by:
  - Semantic similarity (find docs with similar meaning)
  - Metadata filters    (only look in certain categories)
  - A combination of both!

    +------------------------------------------------------------------+
    |                    CHROMADB STRUCTURE                              |
    +------------------------------------------------------------------+
    |                                                                    |
    |  Collection: "devops_knowledge"                                    |
    |  +----------------------------------------------------------+     |
    |  |  ID       | Document        | Embedding   | Metadata     |     |
    |  |----------|-----------------|-------------|--------------|     |
    |  | doc_0    | "CrashLoop..."  | [0.2, 0.8]  | category:k8s |     |
    |  | doc_1    | "Terraform..."  | [0.9, 0.1]  | category:tf  |     |
    |  | doc_2    | "Docker bui..." | [0.5, 0.4]  | category:dk  |     |
    |  | ...      | ...             | ...         | ...          |     |
    |  +----------------------------------------------------------+     |
    |                                                                    |
    +------------------------------------------------------------------+

WHAT ARE EMBEDDINGS?
--------------------
Embeddings are lists of numbers that capture the *meaning* of text.
Similar texts get similar numbers.

    Text: "Kubernetes pod error"  -->  [0.23, 0.87, -0.12, ...]
    Text: "K8s container crash"   -->  [0.25, 0.84, -0.10, ...]  (similar!)
    Text: "Terraform variables"   -->  [0.91, 0.05, 0.67, ...]   (different!)

We use HuggingFace's "all-MiniLM-L6-v2" model for embeddings because:
  - It is FREE and runs locally (no API key needed)
  - It is small and fast (~22 million parameters)
  - It produces 384-dimensional vectors
  - It is good enough for most retrieval tasks

CONCEPTS INTRODUCED:
--------------------
- ChromaDB collections, add(), query()
- HuggingFace SentenceTransformers for embedding
- Metadata filtering with "where" clauses
- Comparing filtered vs unfiltered search quality

PREREQUISITES:
--------------
- Exercise 1 completed (we reuse load_knowledge_base)
- pip install chromadb sentence-transformers
- No API keys needed (everything runs locally)

RUNNING THIS EXERCISE:
----------------------
    cd devops-troubleshooting-chatbot
    python module-4-knowledge-base/exercises/ex2_metadata_filtering.py

NOTE: The first run downloads the embedding model (~90 MB).
      Subsequent runs load from cache and are much faster.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# =============================================================================
# PATH SETUP
# =============================================================================
# Same pattern as Exercise 1: navigate up to the project root.
#
# Path(__file__)  -> this file (ex2_metadata_filtering.py)
# .parent         -> exercises/
# .parent         -> module-4-knowledge-base/
# .parent         -> project root
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# IMPORT OUR OWN MODULE FROM EXERCISE 1
# =============================================================================
# We reuse the loader we built in Exercise 1.  This is a key software
# engineering principle: don't repeat yourself (DRY).
# =============================================================================
from module_4_knowledge_base_exercises_helper import load_knowledge_base_helper

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
# chromadb        -- vector database for storing and searching embeddings
# SentenceTransformer -- model that converts text into embedding vectors
# =============================================================================
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    print("ERROR: chromadb is not installed.")
    print("  Fix: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("  Fix: pip install sentence-transformers")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"

# The embedding model we use.  This is a free, local model from HuggingFace.
# It produces 384-dimensional vectors and is a great default choice.
EMBEDDING_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ChromaDB collection name
COLLECTION_NAME = "devops_kb_exercise2"


# #############################################################################
#
#   HELPER: Inline loader (fallback if ex1 import fails)
#
# #############################################################################


def _load_knowledge_base_inline(kb_dir: Path) -> List[Dict[str, Any]]:
    """
    Inline fallback loader so this exercise is self-contained.

    In a real project you would import from a shared module.  Here we
    duplicate just enough logic so this file works standalone.
    """
    documents = []
    for md_file in sorted(kb_dir.rglob("*.md")):
        try:
            content = md_file.read_text(encoding="utf-8")
            relative = md_file.relative_to(kb_dir)
            parts = relative.parts
            category = parts[0] if len(parts) > 1 else "general"
            documents.append({
                "content": content,
                "metadata": {
                    "source": str(md_file),
                    "filename": md_file.name,
                    "category": category,
                    "topic": md_file.stem,
                    "char_count": len(content),
                }
            })
        except Exception as e:
            print(f"  WARNING: Could not load {md_file.name}: {e}")
    documents.sort(key=lambda d: (d["metadata"]["category"], d["metadata"]["filename"]))
    return documents


# Determine which loader to use
# (try importing from ex1, fall back to inline)
try:
    # This import path depends on how you run the script.
    # If it fails, the inline loader works just as well.
    from ex1_kb_loader import load_knowledge_base
except ImportError:
    load_knowledge_base = _load_knowledge_base_inline


# #############################################################################
#
#   CORE FUNCTIONS
#
# #############################################################################


def create_embedding_function(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Create and return a SentenceTransformer embedding model.

    HOW SENTENCE TRANSFORMERS WORK:
    --------------------------------
    SentenceTransformer models are neural networks trained specifically to
    produce meaningful embeddings for sentences and paragraphs.

    The model:
      1. Tokenizes the input text (splits into subword tokens)
      2. Passes tokens through a transformer neural network
      3. Pools the output into a single fixed-size vector
      4. Returns the vector (384 floats for MiniLM-L6-v2)

        "Kubernetes pod crash"
              |
              v
        [Tokenize] --> ["Kubernetes", "pod", "crash"]
              |
              v
        [Transformer Neural Network]
              |
              v
        [Pooling Layer]
              |
              v
        [0.23, 0.87, -0.12, ..., 0.45]   (384 numbers)

    Args:
        model_name: Name of the HuggingFace model to use.

    Returns:
        A SentenceTransformer model instance.
    """
    print(f"\n  Loading embedding model: {model_name}")
    print("  (First run downloads the model; subsequent runs use cache)")
    start = time.time()

    model = SentenceTransformer(model_name)

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    return model


def build_collection(
    documents: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """
    Create a ChromaDB collection and populate it with documents + metadata.

    This function does FOUR things:
      1. Creates an in-memory ChromaDB client
      2. Creates (or recreates) a collection
      3. Embeds each document using the SentenceTransformer model
      4. Adds documents, embeddings, and metadata to the collection

    WHY IN-MEMORY?
    We use an in-memory client here for simplicity and speed.  In Exercise 3
    you will learn to use a persistent (on-disk) database.

    HOW ChromaDB.add() WORKS:
    -------------------------
    collection.add(
        ids=["doc_0", "doc_1", ...],          # unique string IDs
        documents=["text of doc 0", ...],     # the actual text
        embeddings=[[0.1, 0.2, ...], ...],    # embedding vectors
        metadatas=[{"category": "k8s"}, ...], # metadata dicts
    )

    All four lists must have the same length.  Each index i corresponds
    to one document.

    Args:
        documents:       List of document dicts from the KB loader.
        embedding_model: A SentenceTransformer model for generating embeddings.
        collection_name: Name for the ChromaDB collection.

    Returns:
        A populated ChromaDB Collection object.
    """
    # -------------------------------------------------------------------------
    # Step 1: Create an in-memory ChromaDB client
    # -------------------------------------------------------------------------
    # chromadb.Client() creates a temporary database that lives only in RAM.
    # When the script exits, the data is gone.  This is fine for learning.
    # -------------------------------------------------------------------------
    client = chromadb.Client(
        ChromaSettings(anonymized_telemetry=False)
    )

    # -------------------------------------------------------------------------
    # Step 2: Create (or recreate) the collection
    # -------------------------------------------------------------------------
    # get_or_create_collection is idempotent: if the collection already exists,
    # it returns the existing one; otherwise it creates a new one.
    #
    # We delete first to ensure a clean slate for this exercise.
    # -------------------------------------------------------------------------
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection didn't exist -- that's fine

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "DevOps KB with metadata filtering (Exercise 2)"}
    )

    # -------------------------------------------------------------------------
    # Step 3: Generate embeddings for all documents
    # -------------------------------------------------------------------------
    # We embed the full content of each document.  In Exercise 3 we will learn
    # to chunk documents first (split into smaller pieces).
    #
    # NOTE: For very large documents, embedding the full text may exceed the
    # model's maximum token length.  The model silently truncates, which is
    # okay for this exercise.
    # -------------------------------------------------------------------------
    print(f"\n  Generating embeddings for {len(documents)} documents...")
    start = time.time()

    texts = [doc["content"] for doc in documents]
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    embeddings_list = embeddings.tolist()  # Convert numpy array to Python list

    elapsed = time.time() - start
    print(f"  Embeddings generated in {elapsed:.1f}s")

    # -------------------------------------------------------------------------
    # Step 4: Add everything to the collection
    # -------------------------------------------------------------------------
    # We prepare parallel lists: ids, documents, embeddings, metadatas.
    #
    # Metadata must be a dict of simple types: str, int, float, bool.
    # Nested dicts or lists are NOT supported by ChromaDB metadata.
    # -------------------------------------------------------------------------
    ids = []
    doc_texts = []
    metadatas = []

    for i, doc in enumerate(documents):
        ids.append(f"doc_{i}")
        doc_texts.append(doc["content"])

        # Build metadata dict -- only include ChromaDB-compatible types
        metadatas.append({
            "category": doc["metadata"]["category"],     # str
            "topic": doc["metadata"]["topic"],           # str
            "filename": doc["metadata"]["filename"],     # str
            "char_count": doc["metadata"]["char_count"], # int
        })

    collection.add(
        ids=ids,
        documents=doc_texts,
        embeddings=embeddings_list,
        metadatas=metadatas,
    )

    print(f"  Added {collection.count()} documents to collection '{collection_name}'")
    return collection


def search_unfiltered(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    query: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the collection WITHOUT any metadata filter.

    This returns the top-N most semantically similar documents regardless
    of category, topic, or any other metadata field.

    How ChromaDB.query() works:

        results = collection.query(
            query_embeddings=[query_vector],   # the question as a vector
            n_results=5,                       # how many results to return
            include=["documents", "metadatas", "distances"],
        )

        results is a dict with keys:
        {
            "ids":       [["doc_3", "doc_7", ...]],   # nested list!
            "documents": [["text of doc_3", ...]],
            "metadatas": [[{"category":"k8s"}, ...]],
            "distances": [[0.42, 0.58, ...]],
        }

        NOTE the double nesting: results["documents"][0] is the list of
        document texts.  The outer list corresponds to the number of
        query vectors (we only pass one, so index [0]).

    Args:
        collection:      The ChromaDB collection to search.
        embedding_model: Model to embed the query.
        query:           The search query string.
        n_results:       Number of results to return.

    Returns:
        List of result dicts, each with "content", "metadata", "distance".
    """
    # Embed the query text
    query_embedding = embedding_model.encode(query).tolist()

    # Search the collection (no "where" filter)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Format results into a clean list
    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i][:200],  # Preview only
                "metadata": results["metadatas"][0][i],
                "distance": round(results["distances"][0][i], 4),
            })

    return formatted


def search_filtered(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    query: str,
    category: str,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search the collection WITH a metadata category filter.

    This only returns documents whose metadata "category" field matches
    the provided value.  The search is still semantic (by meaning), but
    restricted to a subset of documents.

    HOW THE "where" FILTER WORKS:
    -----------------------------
    ChromaDB supports filtering with the "where" parameter:

        where={"category": "kubernetes"}

    This tells ChromaDB: "Before computing similarity, discard any
    document whose 'category' metadata is not 'kubernetes'."

    You can also use operators:

        where={"char_count": {"$gt": 1000}}       # greater than
        where={"category":   {"$in": ["k8s", "docker"]}}  # in list
        where={"topic":      {"$ne": "deprecated"}}       # not equal

    And combine with $and / $or:

        where={
            "$and": [
                {"category": "kubernetes"},
                {"char_count": {"$gt": 500}}
            ]
        }

    Args:
        collection:      The ChromaDB collection to search.
        embedding_model: Model to embed the query.
        query:           The search query string.
        category:        Category to filter by (e.g., "kubernetes").
        n_results:       Number of results to return.

    Returns:
        List of result dicts, each with "content", "metadata", "distance".
    """
    # Embed the query text
    query_embedding = embedding_model.encode(query).tolist()

    # Search with a "where" filter on the category field
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"category": category},  # <-- THE KEY DIFFERENCE
        include=["documents", "metadatas", "distances"],
    )

    # Format results
    formatted = []
    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            formatted.append({
                "content": results["documents"][0][i][:200],
                "metadata": results["metadatas"][0][i],
                "distance": round(results["distances"][0][i], 4),
            })

    return formatted


def print_results(results: List[Dict[str, Any]], label: str) -> None:
    """
    Pretty-print a list of search results.

    Args:
        results: List of result dicts from search functions.
        label:   A heading to display above the results.
    """
    print(f"\n    {label}")
    print("    " + "-" * 52)

    if not results:
        print("      (no results)")
        return

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        dist = r["distance"]
        category = meta.get("category", "?")
        filename = meta.get("filename", "?")
        # Truncate content preview for readability
        preview = r["content"][:80].replace("\n", " ").strip()

        print(f"      {i}. [{category}] {filename}  (distance: {dist})")
        print(f"         \"{preview}...\"")

    print()


def compare_searches(
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    query: str,
    expected_category: str,
    n_results: int = 5,
) -> None:
    """
    Run both unfiltered and filtered search for the same query, then
    compare the results side-by-side.

    This is the heart of the exercise: it shows concretely how metadata
    filtering improves retrieval quality.

    Args:
        collection:        The ChromaDB collection.
        embedding_model:   The embedding model.
        query:             The search query.
        expected_category: The category we expect to be most relevant.
        n_results:         Number of results to fetch.
    """
    print(f"\n  {'=' * 56}")
    print(f"  Query: \"{query}\"")
    print(f"  Expected category: {expected_category}")
    print(f"  {'=' * 56}")

    # --- Unfiltered search ---
    unfiltered = search_unfiltered(collection, embedding_model, query, n_results)
    print_results(unfiltered, "UNFILTERED RESULTS (no metadata filter):")

    # Count how many unfiltered results are from the expected category
    unfiltered_relevant = sum(
        1 for r in unfiltered
        if r["metadata"].get("category") == expected_category
    )

    # --- Filtered search ---
    filtered = search_filtered(
        collection, embedding_model, query, expected_category, n_results
    )
    print_results(filtered, f"FILTERED RESULTS (category = \"{expected_category}\"):")

    # --- Comparison summary ---
    print(f"    COMPARISON:")
    print(f"      Unfiltered: {unfiltered_relevant}/{len(unfiltered)} results from '{expected_category}'")
    print(f"      Filtered  : {len(filtered)}/{len(filtered)} results from '{expected_category}'")

    if len(filtered) > 0 and unfiltered_relevant < len(unfiltered):
        print(f"      --> Filtering improved precision!")
    elif unfiltered_relevant == len(unfiltered):
        print(f"      --> Unfiltered already returned all '{expected_category}' results.")
        print(f"          (The query was specific enough on its own.)")
    print()


# #############################################################################
#
#   DEMO / TEST
#
# #############################################################################

if __name__ == "__main__":
    """
    Demo: Load the KB into ChromaDB with metadata, then compare
    filtered vs unfiltered search results.
    """

    print("\n" + "=" * 60)
    print("  Exercise 2 -- Metadata Filtering with ChromaDB")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load documents from the knowledge base
    # =========================================================================
    print("\n  [Step 1/4] Loading knowledge base documents...")
    documents = load_knowledge_base(KNOWLEDGE_BASE_DIR)
    print(f"  Loaded {len(documents)} documents")

    if not documents:
        print("  ERROR: No documents loaded.  Check knowledge-base/ directory.")
        sys.exit(1)

    # =========================================================================
    # Step 2: Initialize the embedding model
    # =========================================================================
    print("\n  [Step 2/4] Initializing embedding model...")
    embedding_model = create_embedding_function(EMBEDDING_MODEL_NAME)

    # =========================================================================
    # Step 3: Build the ChromaDB collection
    # =========================================================================
    print("\n  [Step 3/4] Building ChromaDB collection...")
    collection = build_collection(documents, embedding_model, COLLECTION_NAME)

    # =========================================================================
    # Step 4: Compare filtered vs unfiltered searches
    # =========================================================================
    print("\n  [Step 4/4] Comparing filtered vs unfiltered searches...\n")

    # --- Test queries with expected categories ---
    #
    # Each tuple: (query_text, expected_category)
    #
    # These represent real questions a DevOps engineer might ask.
    # We test whether filtering to the right category helps.
    test_queries = [
        (
            "My pod keeps crashing with CrashLoopBackOff",
            "kubernetes",
        ),
        (
            "Terraform state lock error how to fix",
            "terraform",
        ),
        (
            "Docker build fails with file not found COPY error",
            "docker",
        ),
        (
            "GitHub Actions workflow permission denied error",
            "cicd",
        ),
    ]

    for query, expected_cat in test_queries:
        compare_searches(
            collection,
            embedding_model,
            query,
            expected_cat,
            n_results=3,
        )

    # =========================================================================
    # Bonus: Demonstrate advanced filtering
    # =========================================================================
    print("\n  " + "=" * 56)
    print("    BONUS: Advanced Metadata Filters")
    print("  " + "=" * 56)

    # Example: filter by topic instead of category
    query_embedding = embedding_model.encode("out of memory killed container").tolist()

    print("\n    Filter by topic = 'pod-errors':")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"topic": "pod-errors"},
        include=["documents", "metadatas", "distances"],
    )
    if results["metadatas"] and results["metadatas"][0]:
        for i, meta in enumerate(results["metadatas"][0]):
            dist = results["distances"][0][i]
            print(f"      {i+1}. [{meta['category']}] {meta['filename']}  (dist: {dist:.4f})")

    # Example: filter by document size (char_count > 3000)
    print("\n    Filter by char_count > 3000:")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"char_count": {"$gt": 3000}},
        include=["metadatas", "distances"],
    )
    if results["metadatas"] and results["metadatas"][0]:
        for i, meta in enumerate(results["metadatas"][0]):
            dist = results["distances"][0][i]
            print(f"      {i+1}. [{meta['category']}] {meta['filename']} "
                  f"({meta['char_count']} chars, dist: {dist:.4f})")

    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    print("  EXERCISE 2 COMPLETE")
    print("=" * 60)
    print("""
  KEY TAKEAWAYS:
  --------------
  1. ChromaDB stores documents alongside embeddings AND metadata.
     The metadata is just a flat dict of simple types (str, int, float, bool).

  2. The "where" parameter in collection.query() lets you filter results
     by metadata BEFORE computing similarity.  This is very efficient.

  3. Filtering by category dramatically improves precision -- you only
     get results from the domain the user is asking about.

  4. HuggingFace SentenceTransformers provide FREE, LOCAL embeddings.
     No API key needed.  The "all-MiniLM-L6-v2" model is a great default.

  5. ChromaDB also supports advanced filters:
       - Comparison: $gt, $gte, $lt, $lte
       - Membership: $in, $nin
       - Logical:    $and, $or
       - Negation:   $ne

  6. In a real chatbot, you might auto-detect the category from the
     user's question (e.g., by keyword matching or an LLM classifier)
     and then apply the filter automatically.

  NEXT STEP:
  ----------
  Proceed to Exercise 3 (ex3_kb_builder.py) to build a complete
  pipeline: load -> chunk -> embed -> store persistently.
""")

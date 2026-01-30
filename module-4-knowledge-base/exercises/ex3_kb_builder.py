#!/usr/bin/env python3
"""
Exercise 3: Complete Knowledge Base Builder
=============================================

GOAL:
-----
Build a complete, production-ready pipeline that takes raw Markdown
documents and turns them into a searchable vector database persisted
on disk.  This is the full "indexing" side of RAG.

THE FULL PIPELINE:
------------------

    +------------------------------------------------------------------+
    |              COMPLETE KB BUILDER PIPELINE                         |
    +------------------------------------------------------------------+
    |                                                                    |
    |  [1. LOAD]                                                         |
    |    |  Read .md files from knowledge-base/                          |
    |    |  Attach metadata (category, topic, filename)                  |
    |    v                                                               |
    |  [2. CHUNK]                                                        |
    |    |  Split long documents into smaller pieces                     |
    |    |  Use Markdown-aware splitting (respect headers)               |
    |    |  Overlap chunks so context is not lost at boundaries          |
    |    v                                                               |
    |  [3. EMBED]                                                        |
    |    |  Convert each chunk into a 384-dim vector                     |
    |    |  Use SentenceTransformer (local, free)                        |
    |    v                                                               |
    |  [4. STORE]                                                        |
    |    |  Save chunks + embeddings + metadata in ChromaDB              |
    |    |  Use PersistentClient so data survives restarts               |
    |    v                                                               |
    |  [5. VALIDATE]                                                     |
    |       Verify document count, run a test query                      |
    |       Print statistics about the indexed knowledge base            |
    |                                                                    |
    +------------------------------------------------------------------+

WHY CHUNKING MATTERS:
---------------------
Our Markdown documents range from 2,000 to 10,000+ characters.  Embedding
models and LLM context windows have limits, so we split documents into
smaller, self-contained pieces called "chunks".

    BEFORE CHUNKING:
    ================
    One big document (8,000 chars):
    +----------------------------------------------------------+
    | # Terraform Errors                                        |
    | ## Error: Resource Already Exists                          |
    | ... (2000 chars of content) ...                            |
    | ## Error: State Lock                                       |
    | ... (2000 chars of content) ...                            |
    | ## Error: Provider Not Found                               |
    | ... (2000 chars of content) ...                            |
    | ## Error: Cycle Detected                                   |
    | ... (2000 chars of content) ...                            |
    +----------------------------------------------------------+

    AFTER CHUNKING (chunk_size=1000, overlap=200):
    ===============================================
    +------------------+  +------------------+  +------------------+
    | Chunk 1          |  | Chunk 2          |  | Chunk 3          |
    | # Terraform...   |  | ## Error: State  |  | ## Error: Prov...|
    | ## Error: Res... |  | Lock             |  | ...              |
    | ...              |  | ...              |  |                  |
    | (1000 chars)     |  | (1000 chars)     |  | (1000 chars)     |
    +------------------+  +------------------+  +------------------+
           ^                    ^
           |                    |
           +--- 200 chars overlap (shared context between chunks)

    WHY OVERLAP?
    Without overlap, a sentence at the boundary between two chunks
    gets split in half.  With overlap, both adjacent chunks contain
    the boundary text, so it can be found regardless of which chunk
    is retrieved.

MARKDOWN-AWARE SPLITTING:
-------------------------
We split on Markdown structural elements IN THIS ORDER:

    1. "\\n## "   -- H2 headers (major sections)
    2. "\\n### "  -- H3 headers (subsections)
    3. "\\n\\n"   -- Blank lines (paragraph boundaries)
    4. "\\n"      -- Any newline
    5. " "        -- Spaces (last resort)
    6. ""         -- Character level (emergency only)

This ensures chunks respect document structure.  A chunk boundary
between two H2 sections is much better than one in the middle of
a code block.

PERSISTENT STORAGE:
-------------------
In Exercise 2 we used an in-memory ChromaDB client.  Data disappeared
when the script ended.  Here we use PersistentClient, which writes
to disk.  This means:
  - You only need to build the index ONCE
  - The chatbot (Exercise 4) can load the pre-built index instantly
  - Re-indexing is optional (only when documents change)

    In-Memory:  chromadb.Client()              -- data lost on exit
    Persistent: chromadb.PersistentClient(path) -- data saved to disk

DUAL BACKEND SUPPORT (OpenAI + Ollama):
---------------------------------------
This exercise supports both embedding backends:

  LOCAL (default):
    - Uses SentenceTransformer("all-MiniLM-L6-v2")
    - Free, no API key, runs on your machine
    - 384-dimensional vectors
    - Set EMBEDDING_PROVIDER=local in .env (or just leave default)

  OPENAI:
    - Uses OpenAI's text-embedding-ada-002
    - Requires OPENAI_API_KEY, costs money
    - 1536-dimensional vectors
    - Set EMBEDDING_PROVIDER=openai in .env

CONCEPTS INTRODUCED:
--------------------
- Markdown-aware text chunking
- Chunk overlap for boundary preservation
- Persistent vector storage with ChromaDB PersistentClient
- Batch processing for efficiency
- Index validation and statistics
- Dual embedding backend (local SentenceTransformer / OpenAI)

PREREQUISITES:
--------------
- Exercises 1 and 2 completed
- pip install chromadb sentence-transformers langchain
- For OpenAI embeddings: OPENAI_API_KEY in .env

RUNNING THIS EXERCISE:
----------------------
    cd devops-troubleshooting-chatbot
    python module-4-knowledge-base/exercises/ex3_kb_builder.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# =============================================================================
# PATH SETUP
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    print("ERROR: chromadb is not installed.  Run: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers is not installed.")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"

# Where to persist the ChromaDB database on disk.
# This directory will be created automatically if it doesn't exist.
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")

# Collection name inside ChromaDB
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "devops_knowledge")

# Embedding provider: "local" (free) or "openai" (paid)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")

# Local embedding model name (only used if EMBEDDING_PROVIDER=local)
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Chunking parameters
CHUNK_SIZE = 1000     # Maximum characters per chunk
CHUNK_OVERLAP = 200   # Characters of overlap between adjacent chunks


# #############################################################################
#
#   STEP 1: DOCUMENT LOADING
#
# #############################################################################


def load_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    """
    Load all Markdown documents from the knowledge base directory.

    This is the same logic as Exercise 1, included here so this file is
    fully self-contained.

    Args:
        kb_dir: Path to the knowledge-base/ directory.

    Returns:
        List of document dicts with "content" and "metadata" keys.
    """
    print(f"\n  [LOAD] Scanning: {kb_dir}")

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base not found: {kb_dir}")

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
            print(f"         Loaded: {category}/{md_file.name} ({len(content):,} chars)")

        except Exception as e:
            print(f"         WARNING: Skipping {md_file.name}: {e}")

    print(f"  [LOAD] Total documents: {len(documents)}")
    return documents


# #############################################################################
#
#   STEP 2: MARKDOWN-AWARE CHUNKING
#
# #############################################################################


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a text string into overlapping chunks using Markdown-aware
    separators.

    HOW THE ALGORITHM WORKS (simplified):
    --------------------------------------
    1. Try to split on the first separator ("\\n## " -- H2 headers)
    2. For each resulting piece:
       - If it fits within chunk_size, keep it as a chunk
       - If it is too big, recursively split on the NEXT separator
    3. After splitting, add overlap: copy the last `chunk_overlap`
       characters of chunk N to the beginning of chunk N+1

    SEPARATOR PRIORITY:
    -------------------
    The separators list defines the hierarchy:

        separators = [
            "\\n## ",    # Best: split between H2 sections
            "\\n### ",   # Good: split between H3 subsections
            "\\n\\n",    # OK:   split between paragraphs
            "\\n",       # Meh:  split between lines
            " ",         # Bad:  split between words
            "",          # Worst: split between characters
        ]

    Args:
        text:          The full document text to chunk.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Returns:
        List of chunk strings.

    Example:
        >>> chunks = chunk_text("Hello world. This is a test.", chunk_size=15, chunk_overlap=5)
        >>> len(chunks)
        2
    """
    # -------------------------------------------------------------------------
    # Define the separator hierarchy (most preferred first)
    # -------------------------------------------------------------------------
    separators = [
        "\n## ",    # H2 headers -- major sections
        "\n### ",   # H3 headers -- subsections
        "\n\n",     # Blank lines -- paragraph breaks
        "\n",       # Newlines -- line breaks
        " ",        # Spaces -- word boundaries
        "",         # Empty string -- character level (emergency)
    ]

    def _split_recursive(text: str, sep_index: int = 0) -> List[str]:
        """
        Recursively split text using the separator hierarchy.

        This inner function tries separators from most preferred to
        least preferred until the chunks are small enough.
        """
        # Base case: text already fits in one chunk
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        # Base case: we've exhausted all separators
        if sep_index >= len(separators):
            # Hard split at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size):
                piece = text[i:i + chunk_size]
                if piece.strip():
                    chunks.append(piece)
            return chunks

        separator = separators[sep_index]

        # Split using the current separator
        if separator == "":
            # Empty separator = split every character (last resort)
            pieces = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        elif separator in text:
            pieces = text.split(separator)
        else:
            # This separator doesn't appear in the text; try the next one
            return _split_recursive(text, sep_index + 1)

        # Merge small pieces together, and recursively split big pieces
        chunks = []
        current = ""

        for piece in pieces:
            # Would adding this piece exceed the chunk size?
            candidate = current + separator + piece if current else piece

            if len(candidate) <= chunk_size:
                # It fits -- accumulate
                current = candidate
            else:
                # It doesn't fit -- save current and start a new chunk
                if current.strip():
                    chunks.append(current.strip())

                # If the piece itself is too big, recursively split it
                if len(piece) > chunk_size:
                    sub_chunks = _split_recursive(piece, sep_index + 1)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = piece

        # Don't forget the last accumulated chunk
        if current.strip():
            chunks.append(current.strip())

        return chunks

    # -------------------------------------------------------------------------
    # Run the recursive splitting
    # -------------------------------------------------------------------------
    raw_chunks = _split_recursive(text)

    # -------------------------------------------------------------------------
    # Add overlap between adjacent chunks
    # -------------------------------------------------------------------------
    # Overlap ensures that text near chunk boundaries is present in BOTH
    # the preceding and following chunk, so it can be found regardless
    # of which chunk is retrieved.
    #
    #   Chunk 1: "...end of section A. Start of section B..."
    #                                  ^^^^^^^^^^^^^^^^^^^
    #                                  This overlap text also appears
    #                                  at the beginning of Chunk 2
    #
    #   Chunk 2: "Start of section B... rest of section B..."
    # -------------------------------------------------------------------------
    if chunk_overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev = raw_chunks[i - 1]
            # Take the last `chunk_overlap` characters of the previous chunk
            overlap_text = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped.append(overlap_text + " " + raw_chunks[i])
        return overlapped

    return raw_chunks


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk ALL documents in the knowledge base.

    Each chunk inherits the metadata of its parent document, plus extra
    metadata:
      - chunk_index: which chunk this is (0, 1, 2, ...)
      - total_chunks: how many chunks the parent document was split into
      - chunk_char_count: character count of THIS chunk (not the full doc)

    METADATA INHERITANCE:
    ---------------------

        Original Document:
        {
            "content": "... 8000 chars ...",
            "metadata": {"category": "terraform", "topic": "errors", ...}
        }

                       |
                       v  chunk_text() splits into 8 chunks

        Chunk 0:                              Chunk 7:
        {                                     {
            "content": "... 1000 chars ...",       "content": "... 800 chars ...",
            "metadata": {                          "metadata": {
                "category": "terraform",               "category": "terraform",
                "topic": "errors",                     "topic": "errors",
                "chunk_index": 0,                      "chunk_index": 7,
                "total_chunks": 8,                     "total_chunks": 8,
                ...                                    ...
            }                                      }
        }                                     }

    Args:
        documents:     List of document dicts (from load_documents).
        chunk_size:    Max chars per chunk.
        chunk_overlap: Overlap chars between adjacent chunks.

    Returns:
        List of chunk dicts with inherited + augmented metadata.
    """
    print(f"\n  [CHUNK] Parameters: size={chunk_size}, overlap={chunk_overlap}")

    all_chunks = []

    for doc in documents:
        # Split this document's content into chunks
        text_chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)

        # Create a chunk dict for each piece, inheriting parent metadata
        for i, chunk_text_str in enumerate(text_chunks):
            chunk_doc = {
                "content": chunk_text_str,
                "metadata": {
                    # Inherit from parent
                    "source": doc["metadata"]["source"],
                    "filename": doc["metadata"]["filename"],
                    "category": doc["metadata"]["category"],
                    "topic": doc["metadata"]["topic"],
                    # Add chunk-specific metadata
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "chunk_char_count": len(chunk_text_str),
                }
            }
            all_chunks.append(chunk_doc)

        print(f"         {doc['metadata']['category']}/{doc['metadata']['filename']}"
              f" -> {len(text_chunks)} chunks")

    print(f"  [CHUNK] Total chunks: {len(all_chunks)} "
          f"(from {len(documents)} documents)")
    return all_chunks


# #############################################################################
#
#   STEP 3: EMBEDDING
#
# #############################################################################


def create_embedding_model() -> Any:
    """
    Create an embedding model based on the EMBEDDING_PROVIDER setting.

    Returns either a SentenceTransformer (local) or a wrapper around
    OpenAI's embedding API.

    DUAL BACKEND ARCHITECTURE:
    --------------------------

        .env: EMBEDDING_PROVIDER=local
              |
              v
        SentenceTransformer("all-MiniLM-L6-v2")
        - Free, runs on your CPU/GPU
        - Produces 384-dim vectors
        - No internet needed after first download

        .env: EMBEDDING_PROVIDER=openai
              |
              v
        OpenAI Embeddings API
        - Requires OPENAI_API_KEY
        - Costs ~$0.0001 per 1K tokens
        - Produces 1536-dim vectors
        - Requires internet

    Returns:
        An object with an .encode(texts) method that returns embeddings.
    """
    print(f"\n  [EMBED] Provider: {EMBEDDING_PROVIDER}")

    if EMBEDDING_PROVIDER == "openai":
        # -----------------------------------------------------------------
        # OpenAI Embeddings
        # -----------------------------------------------------------------
        try:
            from openai import OpenAI
        except ImportError:
            print("  ERROR: openai package not installed.  Run: pip install openai")
            sys.exit(1)

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key or api_key == "sk-your-api-key-here":
            print("  ERROR: OPENAI_API_KEY not set in .env")
            print("  Either set the key or switch to EMBEDDING_PROVIDER=local")
            sys.exit(1)

        client = OpenAI()

        # We wrap the OpenAI client in a simple class that has an .encode()
        # method, matching the SentenceTransformer interface.
        class OpenAIEmbedder:
            """Wrapper to give OpenAI embeddings a .encode() interface."""

            def __init__(self, client):
                self.client = client
                self.model = "text-embedding-ada-002"
                self.dimension = 1536

            def encode(self, texts, show_progress_bar=False):
                """Embed a list of texts using OpenAI's API."""
                import numpy as np

                if isinstance(texts, str):
                    texts = [texts]

                # OpenAI API accepts up to 8191 tokens per text
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                embeddings = [item.embedding for item in response.data]
                return np.array(embeddings)

        embedder = OpenAIEmbedder(client)
        print(f"  [EMBED] Model: OpenAI {embedder.model} ({embedder.dimension}-dim)")
        return embedder

    else:
        # -----------------------------------------------------------------
        # Local Embeddings (SentenceTransformer)
        # -----------------------------------------------------------------
        print(f"  [EMBED] Loading model: {LOCAL_EMBEDDING_MODEL}")
        print("           (First run downloads ~90 MB; subsequent runs use cache)")
        start = time.time()

        model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)

        elapsed = time.time() - start
        dim = model.get_sentence_embedding_dimension()
        print(f"  [EMBED] Model loaded in {elapsed:.1f}s ({dim}-dim vectors)")
        return model


def embed_chunks(
    chunks: List[Dict[str, Any]],
    embedding_model,
    batch_size: int = 64,
) -> List[List[float]]:
    """
    Generate embeddings for all chunks in batches.

    WHY BATCH?
    ----------
    Embedding one text at a time is slow because of per-call overhead.
    By batching (sending multiple texts at once), we reduce overhead and
    enable the model to use parallelism (e.g., GPU batching).

        Single:  text_1 -> model -> emb_1, text_2 -> model -> emb_2, ...
                 (N calls, slow)

        Batched: [text_1, text_2, ..., text_64] -> model -> [emb_1, ..., emb_64]
                 (N/64 calls, fast!)

    Args:
        chunks:          List of chunk dicts (each has "content").
        embedding_model: Model with .encode() method.
        batch_size:      Number of texts to embed at once.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    print(f"\n  [EMBED] Generating embeddings for {len(chunks)} chunks "
          f"(batch_size={batch_size})...")
    start = time.time()

    texts = [chunk["content"] for chunk in chunks]
    all_embeddings = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)

        # Convert numpy arrays to plain Python lists
        for emb in batch_embeddings:
            all_embeddings.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))

        # Progress indicator
        done = min(i + batch_size, len(texts))
        print(f"           Embedded {done}/{len(texts)} chunks...")

    elapsed = time.time() - start
    print(f"  [EMBED] All embeddings generated in {elapsed:.1f}s")
    return all_embeddings


# #############################################################################
#
#   STEP 4: PERSISTENT STORAGE
#
# #############################################################################


def store_in_chromadb(
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    force_rebuild: bool = False,
) -> chromadb.Collection:
    """
    Store chunks and embeddings in a persistent ChromaDB database.

    PERSISTENT vs IN-MEMORY:
    ------------------------

        IN-MEMORY (Exercise 2):
        client = chromadb.Client()
        - Data lives in RAM only
        - Lost when script exits
        - Good for experiments

        PERSISTENT (this exercise):
        client = chromadb.PersistentClient(path="data/chroma_db")
        - Data written to disk
        - Survives restarts
        - Used by the chatbot in Exercise 4

    The persistent database creates files like:

        data/chroma_db/
        +-- chroma.sqlite3          <-- metadata & document text
        +-- <uuid>/
            +-- data_level0.bin     <-- embedding vectors
            +-- header.bin          <-- index metadata
            +-- length.bin          <-- vector lengths

    Args:
        chunks:          List of chunk dicts.
        embeddings:      List of embedding vectors.
        persist_dir:     Directory to store the database.
        collection_name: Name of the ChromaDB collection.
        force_rebuild:   If True, delete existing data and rebuild.

    Returns:
        The populated ChromaDB collection.
    """
    print(f"\n  [STORE] Persist directory: {persist_dir}")

    # Create the directory if it doesn't exist
    persist_dir.mkdir(parents=True, exist_ok=True)

    # If force_rebuild, delete existing database
    if force_rebuild and persist_dir.exists():
        print("  [STORE] Force rebuild: clearing existing database...")
        # Remove all files in the persist directory
        for item in persist_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    # Create a persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Delete existing collection if force rebuild
    if force_rebuild:
        try:
            client.delete_collection(collection_name)
            print(f"  [STORE] Deleted existing collection '{collection_name}'")
        except Exception:
            pass  # Collection didn't exist

    # Create (or get) the collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "DevOps troubleshooting knowledge base"},
    )

    # Check if already populated
    existing_count = collection.count()
    if existing_count > 0 and not force_rebuild:
        print(f"  [STORE] Collection already has {existing_count} documents.")
        print(f"           Use force_rebuild=True to rebuild.")
        return collection

    # -------------------------------------------------------------------------
    # Add chunks in batches
    # -------------------------------------------------------------------------
    # ChromaDB can handle large batches, but we chunk the inserts for
    # progress reporting and to avoid potential memory issues.
    # -------------------------------------------------------------------------
    BATCH_SIZE = 100
    total = len(chunks)

    print(f"  [STORE] Adding {total} chunks to collection '{collection_name}'...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)

        ids = [f"chunk_{i}" for i in range(batch_start, batch_end)]
        docs = [chunks[i]["content"] for i in range(batch_start, batch_end)]
        embs = embeddings[batch_start:batch_end]
        metas = []

        for i in range(batch_start, batch_end):
            meta = chunks[i]["metadata"].copy()
            # Ensure all metadata values are ChromaDB-compatible types
            # (str, int, float, bool only -- no lists, dicts, or None)
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = "unknown"
                elif isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metas.append(clean_meta)

        collection.add(
            ids=ids,
            documents=docs,
            embeddings=embs,
            metadatas=metas,
        )

        print(f"           Stored {batch_end}/{total} chunks...")

    print(f"  [STORE] Done! Collection now has {collection.count()} documents.")
    return collection


# #############################################################################
#
#   STEP 5: VALIDATION
#
# #############################################################################


def validate_collection(
    collection: chromadb.Collection,
    embedding_model,
) -> None:
    """
    Validate the built knowledge base by running test queries and
    printing detailed statistics.

    This step is crucial in any data pipeline: always verify the output!

    Args:
        collection:      The populated ChromaDB collection.
        embedding_model: The embedding model (for test queries).
    """
    print(f"\n  [VALIDATE] Collection: '{collection.name}'")
    print(f"  [VALIDATE] Document count: {collection.count()}")

    # -------------------------------------------------------------------------
    # Gather category statistics from the collection
    # -------------------------------------------------------------------------
    # ChromaDB doesn't have a native "GROUP BY" operation, so we query
    # all metadatas and aggregate in Python.
    # -------------------------------------------------------------------------
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data["metadatas"]

    if not metadatas:
        print("  [VALIDATE] WARNING: No documents in collection!")
        return

    # Count by category
    category_counts = {}
    for meta in metadatas:
        cat = meta.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n  [VALIDATE] Chunks per category:")
    for cat in sorted(category_counts.keys()):
        count = category_counts[cat]
        bar = "#" * min(count, 40)  # Cap bar at 40 chars
        print(f"             {cat.ljust(12)}: {count:3d}  {bar}")

    # -------------------------------------------------------------------------
    # Run test queries
    # -------------------------------------------------------------------------
    test_queries = [
        ("pod CrashLoopBackOff error", "kubernetes"),
        ("terraform state lock", "terraform"),
        ("docker build COPY failed", "docker"),
        ("GitHub Actions permission denied", "cicd"),
    ]

    print(f"\n  [VALIDATE] Running {len(test_queries)} test queries:")
    print("  " + "-" * 54)

    all_passed = True
    for query, expected_cat in test_queries:
        query_emb = embedding_model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_emb],
            n_results=1,
            include=["metadatas", "distances"],
        )

        if results["metadatas"] and results["metadatas"][0]:
            top_meta = results["metadatas"][0][0]
            top_dist = results["distances"][0][0]
            actual_cat = top_meta.get("category", "?")
            passed = actual_cat == expected_cat
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False

            print(f"    {status}: \"{query}\"")
            print(f"           Top result: [{actual_cat}] {top_meta.get('filename', '?')}"
                  f"  (dist: {top_dist:.4f})")
        else:
            print(f"    FAIL: \"{query}\" -- no results returned")
            all_passed = False

    print("  " + "-" * 54)
    overall = "ALL PASSED" if all_passed else "SOME FAILED"
    print(f"  [VALIDATE] Test queries: {overall}")


# #############################################################################
#
#   MAIN PIPELINE
#
# #############################################################################


def build_knowledge_base(
    kb_dir: Path = KNOWLEDGE_BASE_DIR,
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    force_rebuild: bool = False,
) -> Tuple[chromadb.Collection, Any]:
    """
    Execute the complete knowledge base build pipeline.

    This is the main entry point that orchestrates all five steps:
      1. Load documents
      2. Chunk documents
      3. Create embedding model
      4. Embed chunks
      5. Store in ChromaDB (persistent)

    Args:
        kb_dir:          Path to the knowledge-base/ directory.
        persist_dir:     Directory for persistent ChromaDB storage.
        collection_name: ChromaDB collection name.
        chunk_size:      Max characters per chunk.
        chunk_overlap:   Overlap characters between chunks.
        force_rebuild:   If True, rebuild from scratch.

    Returns:
        Tuple of (collection, embedding_model).
    """
    print("\n  " + "=" * 56)
    print("    BUILDING KNOWLEDGE BASE")
    print("  " + "=" * 56)

    total_start = time.time()

    # Step 1: Load
    documents = load_documents(kb_dir)

    # Step 2: Chunk
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)

    # Step 3: Create embedding model
    embedding_model = create_embedding_model()

    # Step 4: Embed
    embeddings = embed_chunks(chunks, embedding_model)

    # Step 5: Store
    collection = store_in_chromadb(
        chunks, embeddings, persist_dir, collection_name, force_rebuild
    )

    total_elapsed = time.time() - total_start
    print(f"\n  PIPELINE COMPLETE in {total_elapsed:.1f}s")

    return collection, embedding_model


# #############################################################################
#
#   DEMO / TEST
#
# #############################################################################

if __name__ == "__main__":
    """
    Demo: Build the complete knowledge base and validate it.
    """

    print("\n" + "=" * 60)
    print("  Exercise 3 -- Complete Knowledge Base Builder")
    print("=" * 60)

    # Check if knowledge base directory exists
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"\n  ERROR: Knowledge base directory not found: {KNOWLEDGE_BASE_DIR}")
        print("  Make sure you run from the project root.")
        sys.exit(1)

    # =========================================================================
    # Build the knowledge base (force rebuild for demo)
    # =========================================================================
    collection, embedding_model = build_knowledge_base(
        kb_dir=KNOWLEDGE_BASE_DIR,
        persist_dir=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        force_rebuild=True,
    )

    # =========================================================================
    # Validate
    # =========================================================================
    validate_collection(collection, embedding_model)

    # =========================================================================
    # Interactive test: let the user try a query
    # =========================================================================
    print(f"\n  " + "=" * 56)
    print("    TRY IT: Interactive Query Test")
    print("  " + "=" * 56)
    print("  Type a DevOps question and see the top 3 results.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            query = input("  Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        query_emb = embedding_model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )):
                print(f"\n    Result {i+1} [{meta.get('category', '?')}/"
                      f"{meta.get('filename', '?')}] (dist: {dist:.4f}):")
                preview = doc[:200].replace("\n", " ").strip()
                print(f"    \"{preview}...\"")
        else:
            print("    No results found.")

        print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("  EXERCISE 3 COMPLETE")
    print("=" * 60)
    print(f"""
  KEY TAKEAWAYS:
  --------------
  1. The full RAG indexing pipeline is: LOAD -> CHUNK -> EMBED -> STORE.
     Each step transforms the data into a more useful form.

  2. Markdown-aware chunking respects document structure (headers,
     paragraphs).  Splitting on "## " is much better than splitting
     on a fixed character count.

  3. Chunk overlap (e.g., 200 chars) ensures text at chunk boundaries
     is not lost.  Both adjacent chunks contain the boundary text.

  4. ChromaDB PersistentClient saves the database to disk.  You only
     need to build the index ONCE -- the chatbot loads it instantly.

  5. Always validate your pipeline!  Run test queries and check that
     the top results come from the expected category.

  6. Both SentenceTransformer (free, local) and OpenAI (paid, cloud)
     work as embedding backends.  Choose based on your needs.

  FILES CREATED:
  --------------
  Persistent database: {CHROMA_PERSIST_DIR}
  Collection name:     {COLLECTION_NAME}

  NEXT STEP:
  ----------
  Proceed to Exercise 4 (ex4_kb_query.py) to build an interactive
  RAG chatbot powered by this knowledge base.
""")

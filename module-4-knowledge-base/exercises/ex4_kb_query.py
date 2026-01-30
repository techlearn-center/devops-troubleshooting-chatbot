#!/usr/bin/env python3
"""
Exercise 4: Knowledge Base Query System (RAG Chatbot)
======================================================

GOAL:
-----
Build an interactive RAG chatbot powered by the knowledge base you built
in Exercise 3.  This is the "query" side of RAG -- the part that takes
a user question, finds relevant documents, and generates an informed
answer using an LLM.

THE COMPLETE RAG QUERY FLOW:
----------------------------

    +------------------------------------------------------------------+
    |               RAG QUERY PIPELINE (this exercise)                  |
    +------------------------------------------------------------------+
    |                                                                    |
    |  User asks: "My pod is in CrashLoopBackOff, how do I fix it?"     |
    |       |                                                            |
    |       v                                                            |
    |  [1. EMBED QUERY]                                                  |
    |       |  Convert the question to a 384-dim vector                  |
    |       v                                                            |
    |  [2. RETRIEVE]                                                     |
    |       |  Search ChromaDB for similar chunks                        |
    |       |  Optionally filter by category (e.g., "kubernetes")        |
    |       |  Return top-K most relevant chunks                         |
    |       v                                                            |
    |  [3. BUILD PROMPT]                                                 |
    |       |  Combine the user's question with retrieved context        |
    |       |  Add system instructions for the LLM                       |
    |       v                                                            |
    |  [4. GENERATE]                                                     |
    |       |  Send the prompt to the LLM (OpenAI or Ollama)             |
    |       |  LLM produces an answer grounded in the context            |
    |       v                                                            |
    |  [5. RESPOND]                                                      |
    |       |  Display the answer to the user                            |
    |       |  Show which source documents were used                     |
    |       |  Ready for the next question                               |
    |       v                                                            |
    |  [Loop back to step 1 for interactive mode]                        |
    |                                                                    |
    +------------------------------------------------------------------+

HOW RAG IMPROVES OVER PLAIN LLM:
---------------------------------

    +---------------------------+    +---------------------------+
    |    WITHOUT RAG            |    |    WITH RAG               |
    +---------------------------+    +---------------------------+
    |                           |    |                           |
    | User: "Fix CrashLoop"    |    | User: "Fix CrashLoop"    |
    |         |                 |    |         |                 |
    |         v                 |    |         v                 |
    |       [LLM]              |    |    [Search KB]            |
    |         |                 |    |         |                 |
    |         v                 |    |    Found: pod-errors.md  |
    |  Generic answer based    |    |    Found: debugging.md   |
    |  on training data only.  |    |         |                 |
    |  May be outdated or      |    |         v                 |
    |  hallucinated.           |    |  [LLM + Context]         |
    |                           |    |         |                 |
    +---------------------------+    |         v                 |
                                     |  Specific, accurate      |
                                     |  answer with exact        |
                                     |  kubectl commands and     |
                                     |  exit code meanings.      |
                                     |                           |
                                     +---------------------------+

SOURCE TRACKING:
----------------
This chatbot tells the user WHERE the answer came from.  After each
response, it lists the source documents and their relevance scores.
This is critical for trust and debugging:

    Answer: ... kubectl logs my-pod --previous ...

    Sources:
      1. [kubernetes] pod-errors.md  (relevance: 0.82)
      2. [kubernetes] debugging.md   (relevance: 0.71)

DUAL BACKEND SUPPORT:
---------------------
This exercise supports both LLM backends:

    OpenAI (default):
      - Set OPENAI_API_KEY in .env
      - Uses gpt-3.5-turbo (or whatever OPENAI_MODEL says)
      - Best quality, costs money

    Ollama (free, local):
      - Set USE_OLLAMA=true in .env
      - Set OLLAMA_MODEL=llama2 (or mistral, etc.)
      - Free, runs locally, needs 8GB+ RAM
      - Start Ollama first: ollama serve

CATEGORY FILTERING:
-------------------
The user can optionally restrict search to a specific category:

    /filter kubernetes   -- only search kubernetes docs
    /filter off          -- disable filtering (search all)

This uses the same ChromaDB "where" filter from Exercise 2.

CONCEPTS INTRODUCED:
--------------------
- RAG query pipeline (embed -> retrieve -> prompt -> generate)
- Prompt engineering for RAG (context injection)
- Source attribution (showing where answers come from)
- Interactive chat loop with command handling
- Dual LLM backend (OpenAI API + Ollama HTTP)
- Category filtering at query time

PREREQUISITES:
--------------
- Exercise 3 completed (knowledge base built and persisted)
- For OpenAI: OPENAI_API_KEY in .env
- For Ollama: ollama serve running, USE_OLLAMA=true in .env

RUNNING THIS EXERCISE:
----------------------
    cd devops-troubleshooting-chatbot

    # First build the KB (if not done already):
    python module-4-knowledge-base/exercises/ex3_kb_builder.py

    # Then run the chatbot:
    python module-4-knowledge-base/exercises/ex4_kb_query.py

INTERACTIVE COMMANDS:
    /filter <category>  -- set category filter (terraform, kubernetes, docker, cicd)
    /filter off         -- disable category filter
    /sources            -- show sources from last answer
    /rebuild            -- rebuild the knowledge base index
    /help               -- show available commands
    quit / exit / q     -- exit the chatbot
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
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
    print("ERROR: chromadb not installed.  Run: pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed.")
    print("  Run: pip install sentence-transformers")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================

KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
CHROMA_PERSIST_DIR = PROJECT_ROOT / os.getenv("CHROMA_PERSIST_DIR", "data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "devops_knowledge")
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM settings
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Retrieval settings
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))  # chars, not tokens

# Valid categories for filtering
VALID_CATEGORIES = {"terraform", "kubernetes", "docker", "cicd"}


# #############################################################################
#
#   PROMPT TEMPLATES
#
# #############################################################################

# -------------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------------
# This tells the LLM what role to play and how to behave.  It is sent
# as the first message in every conversation.
# -------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful DevOps expert assistant. Your role is to help users troubleshoot common DevOps issues related to Terraform, Kubernetes, Docker, and CI/CD pipelines.

Guidelines:
1. Be concise but thorough -- provide actionable solutions
2. Start with the most likely cause and solution
3. Include relevant commands the user can run
4. Explain WHY something failed, not just how to fix it
5. If the provided context contains a relevant solution, use it
6. If you need more information, ask clarifying questions
7. Use code blocks for commands and configuration examples
8. When the context does not cover the question, say so honestly"""

# -------------------------------------------------------------------------
# RAG PROMPT TEMPLATE
# -------------------------------------------------------------------------
# This template injects retrieved context into the user's prompt so the
# LLM can ground its answer in our knowledge base.
#
# The {context} placeholder gets replaced with relevant document chunks.
# The {question} placeholder gets replaced with the user's question.
# -------------------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """Use the following context from our DevOps knowledge base to help answer the user's question. If the context contains relevant solutions, explain them clearly with specific commands.

CONTEXT FROM KNOWLEDGE BASE:
{context}

USER QUESTION: {question}

Provide a helpful, accurate response. Include specific commands and code examples when relevant. If the context does not fully answer the question, say so and provide general guidance based on your knowledge."""

# -------------------------------------------------------------------------
# FALLBACK PROMPT (no context found)
# -------------------------------------------------------------------------
FALLBACK_PROMPT_TEMPLATE = """The user has a DevOps question, but no relevant documents were found in the knowledge base. Answer based on your general knowledge.

USER QUESTION: {question}

Provide a helpful response. Note that this answer is based on general knowledge, not the project's knowledge base."""


# #############################################################################
#
#   KNOWLEDGE BASE CONNECTION
#
# #############################################################################


def connect_to_knowledge_base(
    persist_dir: Path = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Tuple[chromadb.Collection, Any]:
    """
    Connect to the existing persistent ChromaDB database and load the
    embedding model.

    This does NOT rebuild the index -- it just opens the existing database
    created by Exercise 3.  If the database does not exist, it raises an
    error with instructions.

    Args:
        persist_dir:     Path to the ChromaDB persist directory.
        collection_name: Name of the collection to open.

    Returns:
        Tuple of (ChromaDB collection, embedding model).

    Raises:
        FileNotFoundError: If the persist directory does not exist.
    """
    print("  Connecting to knowledge base...")

    # -------------------------------------------------------------------------
    # Check that the database exists
    # -------------------------------------------------------------------------
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"ChromaDB database not found at: {persist_dir}\n"
            f"You need to build the knowledge base first!\n"
            f"Run: python module-4-knowledge-base/exercises/ex3_kb_builder.py"
        )

    # -------------------------------------------------------------------------
    # Open the persistent ChromaDB client
    # -------------------------------------------------------------------------
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # -------------------------------------------------------------------------
    # Get the collection
    # -------------------------------------------------------------------------
    try:
        collection = client.get_collection(collection_name)
        doc_count = collection.count()
        print(f"  Connected! Collection '{collection_name}' has {doc_count} chunks.")
    except Exception:
        # Collection doesn't exist -- maybe the build hasn't been run
        print(f"  WARNING: Collection '{collection_name}' not found.")
        print(f"  Building knowledge base from scratch...")
        # Fall back to building from scratch
        collection, embedding_model = _build_fresh(client, collection_name)
        return collection, embedding_model

    # -------------------------------------------------------------------------
    # Load the embedding model (must match what was used to build the index)
    # -------------------------------------------------------------------------
    print(f"  Loading embedding model: {LOCAL_EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    print(f"  Embedding model ready.")

    return collection, embedding_model


def _build_fresh(client, collection_name: str):
    """
    Emergency fallback: build the knowledge base if it doesn't exist.
    This imports and runs the builder from Exercise 3.
    """
    # Import the builder (inline to avoid circular import)
    try:
        from ex3_kb_builder import build_knowledge_base
        collection, embedding_model = build_knowledge_base(
            force_rebuild=True,
        )
        return collection, embedding_model
    except ImportError:
        # ex3 not importable -- do a minimal build inline
        print("  Building minimal index inline...")

        embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
        kb_dir = KNOWLEDGE_BASE_DIR

        documents = []
        for md_file in sorted(kb_dir.rglob("*.md")):
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

        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DevOps knowledge base"},
        )

        texts = [d["content"] for d in documents]
        embeddings = embedding_model.encode(texts).tolist()

        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{
            "category": d["metadata"]["category"],
            "topic": d["metadata"]["topic"],
            "filename": d["metadata"]["filename"],
            "char_count": d["metadata"]["char_count"],
        } for d in documents]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        print(f"  Built minimal index with {collection.count()} documents.")
        return collection, embedding_model


# #############################################################################
#
#   RETRIEVAL
#
# #############################################################################


def retrieve_context(
    query: str,
    collection: chromadb.Collection,
    embedding_model,
    n_results: int = TOP_K_RESULTS,
    category_filter: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant document chunks from the knowledge base.

    This is the "R" in RAG: Retrieval.  It:
      1. Embeds the user's query into a vector
      2. Searches ChromaDB for the most similar chunks
      3. Optionally filters by category
      4. Formats the results into a context string for the LLM
      5. Returns both the formatted context AND the raw sources

    HOW SIMILARITY SEARCH WORKS:
    ----------------------------

        Query: "pod CrashLoopBackOff"
               |
               v
        Embed: [0.23, 0.87, -0.12, ..., 0.45]
               |
               v
        ChromaDB compares this vector to ALL stored vectors
        using cosine distance (lower = more similar):
               |
               v
        Results (sorted by distance):
          0.15  kubernetes/pod-errors.md  chunk_3
          0.28  kubernetes/debugging.md   chunk_1
          0.45  docker/runtime.md         chunk_2
          0.67  terraform/errors.md       chunk_5
          0.89  cicd/jenkins.md           chunk_0

    Args:
        query:           The user's question.
        collection:      ChromaDB collection to search.
        embedding_model: Model to embed the query.
        n_results:       Number of results to retrieve.
        category_filter: Optional category to filter by.

    Returns:
        Tuple of:
          - Formatted context string (for the LLM prompt)
          - List of source dicts (for attribution display)
    """
    # -------------------------------------------------------------------------
    # Step 1: Embed the query
    # -------------------------------------------------------------------------
    query_embedding = embedding_model.encode(query).tolist()

    # -------------------------------------------------------------------------
    # Step 2: Build the where filter (if category filtering is active)
    # -------------------------------------------------------------------------
    where_filter = None
    if category_filter and category_filter in VALID_CATEGORIES:
        where_filter = {"category": category_filter}

    # -------------------------------------------------------------------------
    # Step 3: Query ChromaDB
    # -------------------------------------------------------------------------
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # -------------------------------------------------------------------------
    # Step 4: Format results
    # -------------------------------------------------------------------------
    sources = []
    context_parts = []
    total_chars = 0

    if results["documents"] and results["documents"][0]:
        for i in range(len(results["documents"][0])):
            doc_text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            # Compute a "relevance score" from distance (higher = better)
            # ChromaDB uses L2 distance by default; smaller = more similar.
            # We convert to a 0-1 score for human readability.
            relevance = max(0.0, 1.0 - distance)

            source = {
                "content": doc_text,
                "category": metadata.get("category", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "topic": metadata.get("topic", "unknown"),
                "distance": round(distance, 4),
                "relevance": round(relevance, 2),
            }
            sources.append(source)

            # Add to context string (with character budget)
            source_label = f"{metadata.get('category', '?')}/{metadata.get('filename', '?')}"
            doc_section = f"--- Source: {source_label} ---\n{doc_text}"

            if total_chars + len(doc_section) > MAX_CONTEXT_CHARS:
                # Truncate to fit within our budget
                remaining = MAX_CONTEXT_CHARS - total_chars - 50
                if remaining > 100:
                    doc_section = doc_section[:remaining] + "\n[truncated...]"
                    context_parts.append(doc_section)
                break

            context_parts.append(doc_section)
            total_chars += len(doc_section)

    context = "\n\n".join(context_parts) if context_parts else ""
    return context, sources


# #############################################################################
#
#   LLM GENERATION
#
# #############################################################################


def call_llm(
    question: str,
    context: str,
    use_ollama: bool = USE_OLLAMA,
) -> str:
    """
    Send the question + context to the LLM and get a response.

    This function supports both OpenAI (cloud) and Ollama (local).
    The backend is selected by the USE_OLLAMA environment variable.

    HOW THE PROMPT IS CONSTRUCTED:
    ------------------------------

        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": RAG_PROMPT_TEMPLATE
                                           .format(context=..., question=...)}
        ]

        The system message sets the LLM's behavior.
        The user message contains both the retrieved context AND the question.

    OPENAI vs OLLAMA:
    -----------------

        OpenAI:
          client = OpenAI()   # reads OPENAI_API_KEY from env
          response = client.chat.completions.create(
              model="gpt-3.5-turbo",
              messages=messages,
          )
          answer = response.choices[0].message.content

        Ollama:
          POST http://localhost:11434/api/chat
          body = {"model": "llama2", "messages": messages, "stream": false}
          answer = response.json()["message"]["content"]

    Args:
        question: The user's question.
        context:  Retrieved context from the knowledge base.
        use_ollama: If True, use Ollama; otherwise use OpenAI.

    Returns:
        The LLM's response as a string.
    """
    # -------------------------------------------------------------------------
    # Build the prompt
    # -------------------------------------------------------------------------
    if context:
        user_message = RAG_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
        )
    else:
        user_message = FALLBACK_PROMPT_TEMPLATE.format(
            question=question,
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # -------------------------------------------------------------------------
    # Call the appropriate backend
    # -------------------------------------------------------------------------
    if use_ollama:
        # =====================================================================
        # OLLAMA BACKEND (free, local)
        # =====================================================================
        # Ollama exposes a REST API at localhost:11434.
        # We POST to /api/chat with the messages array.
        #
        # Make sure Ollama is running: ollama serve
        # And you have a model pulled: ollama pull llama2
        # =====================================================================
        import requests

        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                    }
                },
                timeout=120,  # LLMs can be slow, especially locally
            )
            response.raise_for_status()
            return response.json()["message"]["content"]

        except requests.exceptions.ConnectionError:
            return (
                "[ERROR] Cannot connect to Ollama at "
                f"{OLLAMA_BASE_URL}.\n"
                "Make sure Ollama is running: ollama serve\n"
                "And you have a model: ollama pull " + OLLAMA_MODEL
            )
        except Exception as e:
            return f"[ERROR] Ollama call failed: {e}"

    else:
        # =====================================================================
        # OPENAI BACKEND (paid, cloud)
        # =====================================================================
        # Uses the openai Python package.
        # Requires OPENAI_API_KEY to be set in environment.
        # =====================================================================
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        except ImportError:
            return (
                "[ERROR] openai package not installed.\n"
                "Run: pip install openai\n"
                "Or switch to Ollama: set USE_OLLAMA=true in .env"
            )
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                return (
                    "[ERROR] OpenAI API key issue.\n"
                    "Check OPENAI_API_KEY in your .env file.\n"
                    "Or switch to Ollama: set USE_OLLAMA=true in .env"
                )
            return f"[ERROR] OpenAI call failed: {e}"


# #############################################################################
#
#   COMPLETE RAG QUERY FUNCTION
#
# #############################################################################


def ask_knowledge_base(
    question: str,
    collection: chromadb.Collection,
    embedding_model,
    n_results: int = TOP_K_RESULTS,
    category_filter: Optional[str] = None,
    show_sources: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Ask a question and get an answer grounded in the knowledge base.

    This is the main RAG function that ties everything together:
      1. Retrieve relevant context from ChromaDB
      2. Send question + context to the LLM
      3. Return the answer and source information

    Args:
        question:        The user's question.
        collection:      ChromaDB collection.
        embedding_model: Embedding model for query encoding.
        n_results:       Number of documents to retrieve.
        category_filter: Optional category to restrict search.
        show_sources:    If True, print source attribution.

    Returns:
        Tuple of (answer_string, list_of_source_dicts).
    """
    # Step 1: Retrieve
    context, sources = retrieve_context(
        query=question,
        collection=collection,
        embedding_model=embedding_model,
        n_results=n_results,
        category_filter=category_filter,
    )

    # Step 2: Generate
    answer = call_llm(question, context)

    # Step 3: Show sources (if requested)
    if show_sources and sources:
        print("\n  Sources used:")
        for i, src in enumerate(sources[:3], 1):  # Show top 3
            print(f"    {i}. [{src['category']}] {src['filename']}"
                  f"  (relevance: {src['relevance']})")

    return answer, sources


# #############################################################################
#
#   INTERACTIVE CHAT LOOP
#
# #############################################################################


def print_help():
    """Print available commands for the interactive chat."""
    print("""
  Available commands:
  -------------------
  /filter <category>  Set category filter (terraform, kubernetes, docker, cicd)
  /filter off         Disable category filter
  /sources            Show detailed sources from last answer
  /rebuild            Rebuild the knowledge base index
  /help               Show this help message
  quit / exit / q     Exit the chatbot
  """)


def run_interactive_chat(
    collection: chromadb.Collection,
    embedding_model,
) -> None:
    """
    Run an interactive chat loop.

    The user types questions, and the chatbot responds using RAG.
    Special commands start with "/" (e.g., /filter, /sources, /help).

    This is a simplified version of the final chatbot in Module 7.
    It demonstrates the core interaction pattern:

        while True:
            question = input()
            if is_command(question):
                handle_command(question)
            else:
                answer = ask_knowledge_base(question)
                print(answer)

    Args:
        collection:      ChromaDB collection.
        embedding_model: Embedding model.
    """
    # -------------------------------------------------------------------------
    # State variables
    # -------------------------------------------------------------------------
    category_filter = None    # None = no filter, or "kubernetes", etc.
    last_sources = []         # Sources from the most recent answer

    # -------------------------------------------------------------------------
    # Welcome banner
    # -------------------------------------------------------------------------
    backend = f"Ollama ({OLLAMA_MODEL})" if USE_OLLAMA else f"OpenAI ({OPENAI_MODEL})"
    doc_count = collection.count()

    print("\n" + "=" * 60)
    print("  DevOps Knowledge Base Chatbot")
    print("=" * 60)
    print(f"""
  Backend     : {backend}
  Knowledge   : {doc_count} indexed chunks
  Filter      : {'off' if not category_filter else category_filter}

  Ask any DevOps question about Terraform, Kubernetes, Docker, or CI/CD.
  Type /help for commands, or 'quit' to exit.
""")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    while True:
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Check for exit commands
        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        # -----------------------------------------------------------------
        # Handle slash commands
        # -----------------------------------------------------------------
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if command == "/help":
                print_help()

            elif command == "/filter":
                if arg.lower() in ("off", "none", "all", ""):
                    category_filter = None
                    print("  Filter disabled -- searching all categories.")
                elif arg.lower() in VALID_CATEGORIES:
                    category_filter = arg.lower()
                    print(f"  Filter set to: {category_filter}")
                else:
                    print(f"  Unknown category: '{arg}'")
                    print(f"  Valid: {', '.join(sorted(VALID_CATEGORIES))}")

            elif command == "/sources":
                if last_sources:
                    print("\n  Detailed sources from last answer:")
                    print("  " + "-" * 50)
                    for i, src in enumerate(last_sources, 1):
                        print(f"    {i}. [{src['category']}] {src['filename']}")
                        print(f"       Topic    : {src['topic']}")
                        print(f"       Relevance: {src['relevance']}")
                        print(f"       Distance : {src['distance']}")
                        preview = src['content'][:120].replace('\n', ' ').strip()
                        print(f"       Preview  : \"{preview}...\"")
                        print()
                else:
                    print("  No sources yet -- ask a question first!")

            elif command == "/rebuild":
                print("  Rebuilding knowledge base...")
                try:
                    from ex3_kb_builder import build_knowledge_base
                    collection, embedding_model = build_knowledge_base(
                        force_rebuild=True,
                    )
                    print("  Knowledge base rebuilt successfully!")
                except ImportError:
                    print("  ERROR: Cannot import ex3_kb_builder.")
                    print("  Run ex3 manually first.")

            else:
                print(f"  Unknown command: {command}")
                print("  Type /help for available commands.")

            continue

        # -----------------------------------------------------------------
        # Regular question -- run the RAG pipeline
        # -----------------------------------------------------------------
        filter_label = f" [filter: {category_filter}]" if category_filter else ""
        print(f"\n  Searching knowledge base{filter_label}...")

        start = time.time()

        answer, last_sources = ask_knowledge_base(
            question=user_input,
            collection=collection,
            embedding_model=embedding_model,
            category_filter=category_filter,
            show_sources=True,
        )

        elapsed = time.time() - start

        print(f"\n  Assistant ({elapsed:.1f}s):")
        print("  " + "-" * 56)

        # Print the answer with indentation for readability
        for line in answer.split("\n"):
            print(f"  {line}")

        print("  " + "-" * 56)


# #############################################################################
#
#   DEMO / TEST
#
# #############################################################################

if __name__ == "__main__":
    """
    Demo: Connect to the knowledge base and run the interactive chatbot.
    """

    print("\n" + "=" * 60)
    print("  Exercise 4 -- Knowledge Base Query System")
    print("=" * 60)

    # =========================================================================
    # Step 1: Connect to the knowledge base
    # =========================================================================
    try:
        collection, embedding_model = connect_to_knowledge_base(
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("\n  The knowledge base has not been built yet.")
        print("  Building it now (this may take a minute)...\n")

        # Auto-build if not found
        try:
            from ex3_kb_builder import build_knowledge_base
            collection, embedding_model = build_knowledge_base(
                force_rebuild=True,
            )
        except ImportError:
            # Build inline as last resort
            print("  Building inline (ex3_kb_builder not importable)...")
            embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
            client = chromadb.PersistentClient(
                path=str(CHROMA_PERSIST_DIR),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

            collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "DevOps knowledge base"},
            )

            documents = []
            for md_file in sorted(KNOWLEDGE_BASE_DIR.rglob("*.md")):
                content = md_file.read_text(encoding="utf-8")
                relative = md_file.relative_to(KNOWLEDGE_BASE_DIR)
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

            texts = [d["content"] for d in documents]
            embeddings = embedding_model.encode(texts).tolist()

            collection.add(
                ids=[f"doc_{i}" for i in range(len(documents))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[{
                    "category": d["metadata"]["category"],
                    "topic": d["metadata"]["topic"],
                    "filename": d["metadata"]["filename"],
                    "char_count": d["metadata"]["char_count"],
                } for d in documents],
            )
            print(f"  Built index with {collection.count()} documents.\n")

    # =========================================================================
    # Step 2: Quick demo (non-interactive test)
    # =========================================================================
    print("\n  " + "=" * 56)
    print("    Quick Demo: Automated Test Queries")
    print("  " + "=" * 56)

    demo_queries = [
        ("My Kubernetes pod is in CrashLoopBackOff", None),
        ("Terraform state lock error", "terraform"),
        ("Docker build COPY file not found", "docker"),
    ]

    for query, cat_filter in demo_queries:
        filter_label = f" [filter: {cat_filter}]" if cat_filter else ""
        print(f"\n  Question: \"{query}\"{filter_label}")

        context, sources = retrieve_context(
            query=query,
            collection=collection,
            embedding_model=embedding_model,
            n_results=3,
            category_filter=cat_filter,
        )

        print(f"  Retrieved {len(sources)} sources:")
        for i, src in enumerate(sources[:3], 1):
            print(f"    {i}. [{src['category']}] {src['filename']}"
                  f"  (relevance: {src['relevance']})")

        if context:
            print(f"  Context length: {len(context):,} chars")
        else:
            print("  No context retrieved.")

    # =========================================================================
    # Step 3: Interactive chat
    # =========================================================================
    print("\n  " + "=" * 56)
    print("    Starting Interactive Chat")
    print("  " + "=" * 56)

    run_interactive_chat(collection, embedding_model)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("  EXERCISE 4 COMPLETE")
    print("=" * 60)
    print("""
  KEY TAKEAWAYS:
  --------------
  1. RAG = Retrieve + Augment + Generate.  The query pipeline embeds
     the question, searches the vector DB, injects context into the
     prompt, and generates an answer with an LLM.

  2. Source tracking is essential.  Always show the user WHERE the
     answer came from so they can verify and dig deeper.

  3. Category filtering at query time narrows the search to the
     relevant domain, dramatically improving result precision.

  4. Both OpenAI (cloud, paid) and Ollama (local, free) work as
     LLM backends.  The same prompt works with either backend.

  5. Prompt engineering matters: the RAG prompt template tells the
     LLM to use the provided context and be honest when the context
     does not cover the question.

  6. Always handle errors gracefully.  Network failures, missing
     API keys, empty results -- your chatbot should handle them all
     with helpful error messages.

  MODULE 4 COMPLETE:
  ------------------
  You have now built the entire knowledge base pipeline:

    Exercise 1: Load documents + extract metadata
    Exercise 2: Metadata filtering in ChromaDB
    Exercise 3: Full build pipeline (load -> chunk -> embed -> store)
    Exercise 4: RAG query system (retrieve -> prompt -> generate)

  NEXT MODULE:
  ------------
  Module 5 (Vector Search) dives deeper into embeddings, similarity
  metrics, and advanced retrieval techniques.
""")

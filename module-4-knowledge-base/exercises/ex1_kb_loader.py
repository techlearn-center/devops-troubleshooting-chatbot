#!/usr/bin/env python3
"""
Exercise 1: Knowledge Base Loader
===================================

GOAL:
-----
Learn how to load documents from a structured directory (our DevOps knowledge
base) into Python, automatically detect each document's category from its
folder name, attach useful metadata, and display statistics about everything
we loaded.

WHY THIS MATTERS:
-----------------
A RAG (Retrieval-Augmented Generation) chatbot is only as good as its
knowledge base.  Before we can search documents, embed them, or feed them
to an LLM, we first need to **load** them into memory with clean metadata.

Think of this exercise as building the "intake pipeline" -- the very first
step in a RAG system:

    +------------------------------------------------------------------+
    |                    RAG PIPELINE OVERVIEW                          |
    +------------------------------------------------------------------+
    |                                                                    |
    |  >>> YOU ARE HERE <<<                                              |
    |  [1. LOAD DOCS]  -->  2. Chunk  -->  3. Embed  -->  4. Store     |
    |       |                                                            |
    |       v                                                            |
    |  Read files from disk                                              |
    |  Detect category (terraform, kubernetes, docker, cicd)             |
    |  Attach metadata (category, topic, filename)                       |
    |  Return structured list of documents                               |
    |                                                                    |
    +------------------------------------------------------------------+

KNOWLEDGE BASE STRUCTURE:
-------------------------
Our knowledge base lives in the `knowledge-base/` folder at the project
root.  It is organized by DevOps tool category:

    knowledge-base/
    |
    +-- terraform/              <-- category = "terraform"
    |   +-- errors.md           <-- topic = "errors"
    |   +-- state.md            <-- topic = "state"
    |   +-- best-practices.md   <-- topic = "best-practices"
    |
    +-- kubernetes/             <-- category = "kubernetes"
    |   +-- pod-errors.md       <-- topic = "pod-errors"
    |   +-- networking.md       <-- topic = "networking"
    |   +-- debugging.md        <-- topic = "debugging"
    |
    +-- docker/                 <-- category = "docker"
    |   +-- build-errors.md     <-- topic = "build-errors"
    |   +-- runtime.md          <-- topic = "runtime"
    |   +-- compose.md          <-- topic = "compose"
    |
    +-- cicd/                   <-- category = "cicd"
        +-- github-actions.md   <-- topic = "github-actions"
        +-- jenkins.md          <-- topic = "jenkins"
        +-- gitlab-ci.md        <-- topic = "gitlab-ci"

    Total: 4 categories, 12 documents

WHAT YOU WILL LEARN:
--------------------
1. How to walk a directory tree with pathlib
2. How to read Markdown files into Python
3. How to extract metadata from the file path
4. How to structure loaded documents for downstream use
5. How to display helpful statistics about the corpus

CONCEPTS INTRODUCED:
--------------------
- pathlib.Path  -- modern, cross-platform file-system paths
- .rglob()      -- recursive globbing (find files in nested folders)
- .relative_to()-- compute a path relative to a base directory
- Metadata      -- extra data *about* each document (category, topic, etc.)
- Document dict -- a simple {content, metadata} structure used everywhere

PREREQUISITES:
--------------
- Module 0 (Setup) completed
- Python 3.9+
- No API keys needed -- this exercise is entirely local

RUNNING THIS EXERCISE:
----------------------
    cd devops-troubleshooting-chatbot
    python module-4-knowledge-base/exercises/ex1_kb_loader.py

EXPECTED OUTPUT (approximate):
-------------------------------
    ============================================================
      Exercise 1 -- Knowledge Base Loader
    ============================================================

    Loading documents from: ...knowledge-base

    Loaded documents:
      [terraform ] errors.md            -- 4,312 chars
      [terraform ] state.md             -- 3,187 chars
      ...
      [cicd      ] gitlab-ci.md         -- 2,945 chars

    -------- Knowledge Base Statistics --------
    Total documents : 12
    Total characters: 45,210
    Categories      : cicd, docker, kubernetes, terraform

    Documents per category:
      terraform  : 3
      kubernetes : 3
      docker     : 3
      cicd       : 3

    Largest document : kubernetes/pod-errors.md (6,102 chars)
    Smallest document: cicd/jenkins.md (1,830 chars)
    Average size     : 3,767 chars
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# PATH SETUP
# =============================================================================
# We need to add the project root to Python's module search path so that:
#   1. We can find the knowledge-base/ directory reliably
#   2. Later exercises can import shared utilities
#
# Path(__file__)          -> this file
# .parent                 -> exercises/
# .parent                 -> module-4-knowledge-base/
# .parent                 -> project root (devops-troubleshooting-chatbot/)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOAD ENVIRONMENT VARIABLES
# =============================================================================
# Even though this exercise doesn't need API keys, loading .env is a good
# habit.  It keeps our code consistent across all exercises.
# =============================================================================
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# =============================================================================
# CONSTANTS
# =============================================================================
# Where the knowledge base lives, relative to the project root.
# We resolve() to get an absolute path -- this avoids confusion when the
# script is run from a different working directory.
# =============================================================================
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"


# #############################################################################
#
#   CORE FUNCTIONS
#
# #############################################################################


def load_document(file_path: Path, base_dir: Path) -> Dict[str, Any]:
    """
    Load a single document from disk and attach metadata.

    This function does THREE things:
      1. Read the file content (the actual Markdown text)
      2. Figure out the category from the folder name
      3. Figure out the topic from the filename (without extension)

    How metadata extraction works -- visual walkthrough:

        file_path = knowledge-base/kubernetes/pod-errors.md
        base_dir  = knowledge-base/

        relative  = file_path.relative_to(base_dir)
                  = kubernetes/pod-errors.md

        parts     = relative.parts
                  = ("kubernetes", "pod-errors.md")

        category  = parts[0]          = "kubernetes"
        filename  = parts[-1]         = "pod-errors.md"
        topic     = stem of filename  = "pod-errors"

    Args:
        file_path: Absolute or relative Path to the .md file
        base_dir:  The knowledge-base/ root directory so we can compute
                   relative paths for metadata

    Returns:
        A dictionary with the structure:
        {
            "content":  "<full text of the file>",
            "metadata": {
                "source":   "<full file path as string>",
                "filename": "<just the file name>",
                "category": "<folder name, e.g. terraform>",
                "topic":    "<file stem, e.g. errors>",
                "char_count": <integer length of content>
            }
        }

    Raises:
        FileNotFoundError: if file_path does not exist
        UnicodeDecodeError: if the file is not valid UTF-8

    Example:
        >>> doc = load_document(
        ...     Path("knowledge-base/terraform/errors.md"),
        ...     Path("knowledge-base")
        ... )
        >>> doc["metadata"]["category"]
        'terraform'
        >>> doc["metadata"]["topic"]
        'errors'
    """
    # -------------------------------------------------------------------------
    # Step 1: Read the file content
    # -------------------------------------------------------------------------
    # We open with encoding="utf-8" for cross-platform safety.
    # Markdown files are plain text, so this is straightforward.
    # -------------------------------------------------------------------------
    content = file_path.read_text(encoding="utf-8")

    # -------------------------------------------------------------------------
    # Step 2: Compute the relative path from the knowledge-base root
    # -------------------------------------------------------------------------
    # Example: if file_path  = /home/user/project/knowledge-base/docker/runtime.md
    #          and base_dir  = /home/user/project/knowledge-base
    #          then relative = docker/runtime.md
    # -------------------------------------------------------------------------
    relative = file_path.relative_to(base_dir)

    # -------------------------------------------------------------------------
    # Step 3: Extract metadata from the relative path
    # -------------------------------------------------------------------------
    # relative.parts gives us a tuple of path components.
    # For "docker/runtime.md" that is ("docker", "runtime.md").
    #
    # category = first part  -> the folder name  -> "docker"
    # filename = last part   -> the file name     -> "runtime.md"
    # topic    = stem        -> name without ext  -> "runtime"
    # -------------------------------------------------------------------------
    parts = relative.parts

    # Safely extract category -- if the file is directly in base_dir (no
    # subfolder), we fall back to "general".
    category = parts[0] if len(parts) > 1 else "general"

    filename = file_path.name          # e.g. "runtime.md"
    topic = file_path.stem             # e.g. "runtime"  (no .md extension)

    # -------------------------------------------------------------------------
    # Step 4: Build and return the document dictionary
    # -------------------------------------------------------------------------
    # This structure is used throughout the project.  Having a standard format
    # makes it easy to pass documents between functions and modules.
    # -------------------------------------------------------------------------
    return {
        "content": content,
        "metadata": {
            "source": str(file_path),   # Full path as string
            "filename": filename,       # Just the file name
            "category": category,       # Folder name (terraform, k8s, ...)
            "topic": topic,             # File stem (errors, runtime, ...)
            "char_count": len(content), # How many characters
        }
    }


def load_knowledge_base(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> List[Dict[str, Any]]:
    """
    Load ALL documents from the knowledge base directory.

    This function:
      1. Scans the kb_dir recursively for .md files
      2. Loads each one with load_document()
      3. Returns a list sorted by category then filename

    How .rglob("*.md") works:

        kb_dir.rglob("*.md")

        rglob = "recursive glob"
        It searches the directory AND all subdirectories for files that
        match the pattern "*.md" (any file ending in .md).

        +-- knowledge-base/
            +-- terraform/
            |   +-- errors.md        <-- matched!
            |   +-- state.md         <-- matched!
            |   +-- best-practices.md<-- matched!
            +-- kubernetes/
            |   +-- pod-errors.md    <-- matched!
            |   +-- ...
            +-- docker/
            |   +-- build-errors.md  <-- matched!
            |   +-- ...
            +-- cicd/
                +-- github-actions.md<-- matched!
                +-- ...

    Args:
        kb_dir: Path to the knowledge-base directory.
                Defaults to KNOWLEDGE_BASE_DIR (project root / knowledge-base).

    Returns:
        List of document dictionaries, each with "content" and "metadata".
        Sorted alphabetically by (category, filename).

    Raises:
        FileNotFoundError: If kb_dir does not exist.
    """
    # -------------------------------------------------------------------------
    # Validate that the directory exists
    # -------------------------------------------------------------------------
    if not kb_dir.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {kb_dir}\n"
            f"Make sure you're running from the project root, or that\n"
            f"the knowledge-base/ folder has been created."
        )

    if not kb_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but got a file: {kb_dir}"
        )

    # -------------------------------------------------------------------------
    # Find all Markdown files recursively
    # -------------------------------------------------------------------------
    # .rglob("*.md") yields Path objects for every .md file in the tree.
    # We convert to a sorted list so the output is deterministic (same order
    # every time, regardless of operating system).
    # -------------------------------------------------------------------------
    md_files = sorted(kb_dir.rglob("*.md"))

    if not md_files:
        print(f"  WARNING: No .md files found in {kb_dir}")
        return []

    # -------------------------------------------------------------------------
    # Load each file
    # -------------------------------------------------------------------------
    documents = []
    for file_path in md_files:
        try:
            doc = load_document(file_path, kb_dir)
            documents.append(doc)
        except Exception as e:
            # Don't let one bad file stop us from loading the rest.
            # In production you might use logging instead of print.
            print(f"  WARNING: Could not load {file_path.name}: {e}")

    # -------------------------------------------------------------------------
    # Sort by category then filename for consistent ordering
    # -------------------------------------------------------------------------
    documents.sort(
        key=lambda d: (d["metadata"]["category"], d["metadata"]["filename"])
    )

    return documents


def print_document_table(documents: List[Dict[str, Any]]) -> None:
    """
    Print a nicely formatted table of all loaded documents.

    Example output:
      [terraform ] errors.md            -- 4,312 chars
      [terraform ] state.md             -- 3,187 chars
      [kubernetes] pod-errors.md        -- 6,102 chars

    Args:
        documents: List of document dicts from load_knowledge_base()
    """
    print("\n  Loaded documents:")
    print("  " + "-" * 56)

    for doc in documents:
        meta = doc["metadata"]
        # Format the category in a fixed-width field for alignment
        category = meta["category"].ljust(10)
        # Format the filename in a fixed-width field
        filename = meta["filename"].ljust(22)
        # Format the character count with thousands separator
        chars = f"{meta['char_count']:,}"
        print(f"    [{category}] {filename} -- {chars} chars")

    print("  " + "-" * 56)


def print_statistics(documents: List[Dict[str, Any]]) -> None:
    """
    Print statistics about the loaded knowledge base.

    This gives you a quick overview of your corpus:
    - How many documents per category
    - Total size in characters
    - Largest / smallest / average document size

    WHY STATISTICS MATTER:
    ----------------------
    Understanding your corpus helps you make better decisions about:
    - Chunk size  (if docs are very long, you need chunking)
    - Balance     (if one category dominates, retrieval may be biased)
    - Coverage    (are there topics missing?)

    Args:
        documents: List of document dicts from load_knowledge_base()
    """
    if not documents:
        print("\n  No documents loaded -- nothing to show.")
        return

    # -------------------------------------------------------------------------
    # Compute basic statistics
    # -------------------------------------------------------------------------
    total_docs = len(documents)
    total_chars = sum(d["metadata"]["char_count"] for d in documents)

    # Collect all unique categories (sorted alphabetically)
    categories = sorted(set(d["metadata"]["category"] for d in documents))

    # Count documents per category using a dictionary comprehension
    #   For each category, count how many docs have that category.
    docs_per_category = {
        cat: sum(1 for d in documents if d["metadata"]["category"] == cat)
        for cat in categories
    }

    # Find the largest and smallest documents by character count
    largest = max(documents, key=lambda d: d["metadata"]["char_count"])
    smallest = min(documents, key=lambda d: d["metadata"]["char_count"])
    avg_chars = total_chars // total_docs  # integer division

    # -------------------------------------------------------------------------
    # Print the statistics
    # -------------------------------------------------------------------------
    print("\n  " + "=" * 46)
    print("    Knowledge Base Statistics")
    print("  " + "=" * 46)

    print(f"\n    Total documents : {total_docs}")
    print(f"    Total characters: {total_chars:,}")
    print(f"    Categories      : {', '.join(categories)}")

    print(f"\n    Documents per category:")
    for cat in categories:
        count = docs_per_category[cat]
        bar = "#" * count  # Simple ASCII bar chart
        print(f"      {cat.ljust(12)}: {count}  {bar}")

    largest_name = f"{largest['metadata']['category']}/{largest['metadata']['filename']}"
    smallest_name = f"{smallest['metadata']['category']}/{smallest['metadata']['filename']}"

    print(f"\n    Largest document : {largest_name} ({largest['metadata']['char_count']:,} chars)")
    print(f"    Smallest document: {smallest_name} ({smallest['metadata']['char_count']:,} chars)")
    print(f"    Average size     : {avg_chars:,} chars")

    # -------------------------------------------------------------------------
    # Content preview
    # -------------------------------------------------------------------------
    # Show the first 120 characters of each document so the learner can
    # visually verify the content looks right.
    # -------------------------------------------------------------------------
    print("\n  " + "-" * 46)
    print("    Content Previews (first 120 chars):")
    print("  " + "-" * 46)
    for doc in documents:
        meta = doc["metadata"]
        preview = doc["content"][:120].replace("\n", " ").strip()
        label = f"{meta['category']}/{meta['filename']}"
        print(f"    {label}:")
        print(f"      \"{preview}...\"")
        print()


# #############################################################################
#
#   DEMO / TEST
#
# #############################################################################

if __name__ == "__main__":
    """
    Demo: Load the knowledge base and display stats.

    This block runs only when you execute the script directly:
        python ex1_kb_loader.py

    It does NOT run when the file is imported by another module:
        from ex1_kb_loader import load_knowledge_base   # this skips __main__
    """

    # =========================================================================
    # Header
    # =========================================================================
    print("\n" + "=" * 60)
    print("  Exercise 1 -- Knowledge Base Loader")
    print("=" * 60)

    # =========================================================================
    # Step 1: Locate the knowledge base
    # =========================================================================
    print(f"\n  Knowledge base path: {KNOWLEDGE_BASE_DIR}")
    print(f"  Path exists: {KNOWLEDGE_BASE_DIR.exists()}")

    if not KNOWLEDGE_BASE_DIR.exists():
        print("\n  ERROR: knowledge-base/ directory not found!")
        print("  Make sure you run this from the project root:")
        print("    cd devops-troubleshooting-chatbot")
        print("    python module-4-knowledge-base/exercises/ex1_kb_loader.py")
        sys.exit(1)

    # =========================================================================
    # Step 2: Load all documents
    # =========================================================================
    print(f"\n  Loading documents from: {KNOWLEDGE_BASE_DIR}")
    documents = load_knowledge_base(KNOWLEDGE_BASE_DIR)

    if not documents:
        print("\n  No documents were loaded.  Check that knowledge-base/ ")
        print("  contains .md files in category subfolders.")
        sys.exit(1)

    # =========================================================================
    # Step 3: Display the loaded documents
    # =========================================================================
    print_document_table(documents)

    # =========================================================================
    # Step 4: Display statistics
    # =========================================================================
    print_statistics(documents)

    # =========================================================================
    # Step 5: Quick sanity checks
    # =========================================================================
    print("\n  " + "=" * 46)
    print("    Sanity Checks")
    print("  " + "=" * 46)

    # Check that every document has the expected metadata keys
    expected_keys = {"source", "filename", "category", "topic", "char_count"}
    all_have_keys = all(
        expected_keys.issubset(d["metadata"].keys()) for d in documents
    )
    status = "PASS" if all_have_keys else "FAIL"
    print(f"    All docs have required metadata keys : {status}")

    # Check that we have at least one document per expected category
    expected_categories = {"terraform", "kubernetes", "docker", "cicd"}
    found_categories = set(d["metadata"]["category"] for d in documents)
    all_cats = expected_categories.issubset(found_categories)
    status = "PASS" if all_cats else "FAIL"
    print(f"    All expected categories present      : {status}")
    if not all_cats:
        missing = expected_categories - found_categories
        print(f"      Missing: {missing}")

    # Check that no document has empty content
    none_empty = all(len(d["content"]) > 0 for d in documents)
    status = "PASS" if none_empty else "FAIL"
    print(f"    No empty documents                   : {status}")

    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 60)
    print("  EXERCISE 1 COMPLETE")
    print("=" * 60)
    print("""
  KEY TAKEAWAYS:
  --------------
  1. pathlib.Path is the modern way to work with file paths in Python.
     It is cross-platform (Windows, macOS, Linux) and intuitive.

  2. .rglob("*.md") recursively finds all matching files in a directory
     tree -- perfect for scanning a knowledge base.

  3. Metadata (category, topic, filename) is extracted from the directory
     structure.  Good folder organization = free metadata!

  4. Always validate your loaded data: check for empty files, missing
     categories, and expected metadata fields.

  5. A standard document structure {content, metadata} makes it easy to
     pass documents between functions and modules.

  NEXT STEP:
  ----------
  Proceed to Exercise 2 (ex2_metadata_filtering.py) to learn how
  metadata can be used to filter search results in a vector database.
""")

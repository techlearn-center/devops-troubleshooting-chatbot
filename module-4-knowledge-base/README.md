# Module 4: Building the DevOps Knowledge Base

**Time Required: 2-3 hours**

A RAG system is only as good as the documents you feed it. In this module, we'll build a comprehensive, well-structured DevOps documentation collection and learn how to load, organize, and index it for our chatbot.

---

## Learning Objectives

By the end of this module, you will:
- Understand what a knowledge base is and why structure matters
- Load documents from files and directories
- Add metadata for filtering and organization
- Use smart chunking for technical documentation
- Build a complete, indexed knowledge base

---

## Prerequisites

```bash
# Install required packages
pip install langchain langchain-community chromadb
pip install sentence-transformers   # Free local embeddings
pip install langchain-openai        # If using OpenAI
pip install langchain-ollama        # If using Ollama (free)
```

---

## What is a Knowledge Base?

Before we build one, let's understand what a knowledge base is:

```
WHAT IS A KNOWLEDGE BASE?
══════════════════════════

Think of it like a well-organized reference library for your AI:

┌──────────────────────────────────────────────────────────────────────┐
│                                                                       │
│  WITHOUT a Knowledge Base:                                           │
│  ┌──────────┐    "How do I fix         ┌──────────┐                 │
│  │  User    │ ──  state lock?" ──────► │   LLM    │ ── Generic,    │
│  └──────────┘                          └──────────┘    maybe wrong  │
│                                                                       │
│  WITH a Knowledge Base:                                              │
│  ┌──────────┐    "How do I fix         ┌──────────┐                 │
│  │  User    │ ──  state lock?" ──────► │  Search  │                 │
│  └──────────┘                          │  KB docs │                 │
│                                         └────┬─────┘                 │
│                                              │ Found!                │
│                                              ▼                       │
│                                     ┌──────────────┐                │
│                                     │ LLM + Context│ ── Accurate,  │
│                                     │ from YOUR    │    specific    │
│                                     │ documentation│    answer!     │
│                                     └──────────────┘                │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Real-World Analogy: The Filing Cabinet

```
A KNOWLEDGE BASE IS LIKE A SMART FILING CABINET:
═════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────┐
    │  📁 Terraform/                                      │
    │  │  📄 errors.md           Common errors & fixes    │
    │  │  📄 state.md            State management guide   │
    │  │  📄 best-practices.md   Do's and don'ts          │
    │  │                                                   │
    │  📁 Kubernetes/                                      │
    │  │  📄 pod-errors.md       Pod troubleshooting      │
    │  │  📄 networking.md       Service & ingress issues  │
    │  │  📄 debugging.md        kubectl debug commands    │
    │  │                                                   │
    │  📁 Docker/                                          │
    │  │  📄 build-errors.md     Dockerfile issues        │
    │  │  📄 runtime.md          Container runtime errors  │
    │  │  📄 compose.md          Docker Compose issues     │
    │  │                                                   │
    │  📁 CI-CD/                                           │
    │     📄 github-actions.md   GitHub Actions errors     │
    │     📄 jenkins.md          Jenkins pipeline issues   │
    │     📄 gitlab-ci.md        GitLab CI problems        │
    └─────────────────────────────────────────────────────┘

    Organization by CATEGORY makes retrieval faster & more accurate!
```

---

## Why Structure Matters

```
STRUCTURED vs UNSTRUCTURED DOCUMENTATION:
═════════════════════════════════════════

❌ BAD: One giant file with everything
┌──────────────────────────────────────────────────────────────────┐
│ all_docs.md (50,000 words)                                       │
│ Terraform stuff... Kubernetes stuff... Docker stuff...           │
│ Everything mixed together, hard to find anything!                │
└──────────────────────────────────────────────────────────────────┘

✅ GOOD: Organized by category and topic
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ terraform/  │  │ kubernetes/ │  │ docker/     │
│  errors.md  │  │  pods.md    │  │  builds.md  │
│  state.md   │  │  network.md │  │  runtime.md │
└─────────────┘  └─────────────┘  └─────────────┘

WHY?
1. Category-based FILTERING: "Only search Terraform docs"
2. Better CHUNKING: Each file focuses on one topic
3. Easier UPDATES: Change one file without affecting others
4. Source TRACKING: Know which doc provided the answer
```

---

## The Knowledge Base Pipeline

```
FROM RAW DOCS TO SEARCHABLE KNOWLEDGE:
═══════════════════════════════════════

Step 1: ORGANIZE                Step 2: LOAD
┌─────────────────┐            ┌─────────────────┐
│ Create folder    │            │ Read files with  │
│ structure with   │  ───────►  │ TextLoader or    │
│ categories       │            │ DirectoryLoader  │
└─────────────────┘            └────────┬────────┘
                                        │
                                        ▼
Step 3: ENRICH                  Step 4: CHUNK
┌─────────────────┐            ┌─────────────────┐
│ Add metadata:    │            │ Split into       │
│ - category       │  ◄───────  │ 400-600 char     │
│ - topic          │            │ pieces           │
│ - source file    │            └─────────────────┘
└────────┬────────┘
         │
         ▼
Step 5: EMBED                   Step 6: STORE
┌─────────────────┐            ┌─────────────────┐
│ Convert chunks   │            │ Save vectors in  │
│ to vectors       │  ───────►  │ ChromaDB for     │
│ [0.2, 0.8, ...]  │            │ fast search      │
└─────────────────┘            └─────────────────┘
```

---

## Document Loading Strategies

### Loading a Single File

```python
from langchain_community.document_loaders import TextLoader

# Load one markdown file
loader = TextLoader("knowledge-base/terraform/errors.md", encoding='utf-8')
documents = loader.load()

# Each document has content and metadata
doc = documents[0]
print(f"Content: {doc.page_content[:100]}...")
print(f"Source: {doc.metadata['source']}")
```

### Loading an Entire Directory

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load ALL .md files from knowledge-base/ and subdirectories
loader = DirectoryLoader(
    "knowledge-base/",
    glob="**/*.md",              # Pattern: all .md files at any depth
    loader_cls=TextLoader,       # Use TextLoader for each file
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
for doc in documents:
    print(f"  - {doc.metadata['source']}")
```

### Adding Custom Metadata

Metadata lets you filter searches by category:

```python
import os
from pathlib import Path

def load_with_metadata(kb_path: str) -> list:
    """
    Load documents with automatic category detection.

    The category is determined by the parent folder name:
        knowledge-base/terraform/errors.md → category: "terraform"
        knowledge-base/kubernetes/pods.md  → category: "kubernetes"
    """
    documents = []

    for md_file in Path(kb_path).rglob("*.md"):
        # Load the file
        loader = TextLoader(str(md_file), encoding='utf-8')
        docs = loader.load()

        # Determine category from parent folder
        category = md_file.parent.name  # "terraform", "kubernetes", etc.

        # Add metadata
        for doc in docs:
            doc.metadata["category"] = category
            doc.metadata["topic"] = md_file.stem      # filename without .md
            doc.metadata["filename"] = md_file.name

        documents.extend(docs)

    return documents
```

---

## Smart Chunking for Technical Docs

### Why Standard Chunking Fails for DevOps Docs

```
PROBLEM: Standard chunking might split an error + solution apart!

Original:
┌──────────────────────────────────────────────────────────────────┐
│ ## Error: State Lock                                              │
│ This error means another process holds the lock.                 │
│                                                                   │
│ ## Solution                                                       │ ← SPLIT HERE?
│ Run: terraform force-unlock LOCK_ID                              │
└──────────────────────────────────────────────────────────────────┘

If we split between "Error" and "Solution", the RAG system might
retrieve the error description WITHOUT the solution!

SOLUTION: Use markdown-aware splitting that keeps sections together.
```

### Markdown Header Splitting

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# Step 1: Split by markdown headers (keeps sections intact)
headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Step 2: Further split large sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def smart_chunk(document_text: str) -> list:
    """Two-stage chunking: headers first, then size."""

    # Split by headers
    header_chunks = md_splitter.split_text(document_text)

    # Further split any chunks that are too large
    final_chunks = []
    for chunk in header_chunks:
        if len(chunk.page_content) > 600:
            sub_chunks = text_splitter.split_text(chunk.page_content)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk.page_content)

    return final_chunks
```

---

## Exercises

### Exercise 1: Knowledge Base Loader
Build a loader that reads all docs with metadata.
→ `exercises/ex1_kb_loader.py`

### Exercise 2: Metadata Filtering
Implement category-based search filtering.
→ `exercises/ex2_metadata_filtering.py`

### Exercise 3: Knowledge Base Builder
Complete pipeline: load → chunk → embed → store.
→ `exercises/ex3_kb_builder.py`

### Exercise 4: Knowledge Base Query
Build a RAG chatbot powered by the knowledge base.
→ `exercises/ex4_kb_query.py`

---

## Key Takeaways

```
┌──────────────────────────────────────────────────────────────────────┐
│                         KEY TAKEAWAYS                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. STRUCTURE your docs by category for better retrieval             │
│                                                                       │
│  2. METADATA enables filtered search (search only K8s docs, etc.)   │
│                                                                       │
│  3. SMART CHUNKING keeps errors + solutions together                │
│                                                                       │
│  4. QUALITY over quantity - well-written docs beat large volumes     │
│                                                                       │
│  5. The knowledge base is the FOUNDATION of your RAG system         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What's Next?

In **Module 5**, we'll deep-dive into vector search and advanced retrieval:
- Comparing embedding models
- Similarity metrics (cosine, euclidean, dot product)
- Tuning retrieval parameters for better results
- Measuring retrieval quality

```bash
cd ../module-5-vector-search
```

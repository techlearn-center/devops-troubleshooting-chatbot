# Module 4: Building the DevOps Knowledge Base

**Time Required: 2 hours**

A RAG system is only as good as its knowledge base. In this module, we'll build a comprehensive DevOps documentation collection.

---

## Learning Objectives

By the end of this module, you will:
- Understand document loading strategies
- Master text chunking for optimal retrieval
- Learn metadata tagging and filtering
- Build a complete DevOps knowledge base

---

## Knowledge Base Architecture

```
+------------------------------------------------------------------+
|                    DEVOPS KNOWLEDGE BASE                          |
+------------------------------------------------------------------+
|                                                                   |
|  knowledge-base/                                                  |
|  |                                                                |
|  +-- terraform/                                                   |
|  |   |-- errors.md          (Common Terraform errors)            |
|  |   |-- state.md           (State management)                   |
|  |   |-- best-practices.md  (Best practices)                     |
|  |                                                                |
|  +-- kubernetes/                                                  |
|  |   |-- pod-errors.md      (Pod troubleshooting)                |
|  |   |-- networking.md      (Network issues)                     |
|  |   |-- debugging.md       (Debug commands)                     |
|  |                                                                |
|  +-- docker/                                                      |
|  |   |-- build-errors.md    (Build issues)                       |
|  |   |-- runtime.md         (Container runtime)                  |
|  |   |-- compose.md         (Docker Compose)                     |
|  |                                                                |
|  +-- cicd/                                                        |
|      |-- github-actions.md  (GitHub Actions)                     |
|      |-- jenkins.md         (Jenkins pipelines)                  |
|      |-- gitlab-ci.md       (GitLab CI)                          |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Document Loading Strategies

### Loading Different File Types

```python
from langchain_community.document_loaders import (
    TextLoader,           # .txt files
    UnstructuredMarkdownLoader,  # .md files
    PyPDFLoader,          # .pdf files
    DirectoryLoader,      # Multiple files
)

# Single markdown file
loader = UnstructuredMarkdownLoader("docs/terraform.md")
docs = loader.load()

# All markdown files in directory
loader = DirectoryLoader(
    "knowledge-base/",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader
)
docs = loader.load()

# With metadata
for doc in docs:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:100]}...")
```

### Adding Custom Metadata

Metadata enables filtering during retrieval:

```python
def load_with_metadata(directory: str) -> list:
    """Load documents with category metadata."""
    documents = []

    category_map = {
        "terraform": ["terraform", "tf", "hcl"],
        "kubernetes": ["kubernetes", "k8s", "kubectl"],
        "docker": ["docker", "container", "dockerfile"],
        "cicd": ["github", "jenkins", "gitlab"]
    }

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                path = os.path.join(root, file)

                # Determine category from path
                category = "general"
                for cat, keywords in category_map.items():
                    if any(kw in path.lower() for kw in keywords):
                        category = cat
                        break

                # Load with metadata
                loader = UnstructuredMarkdownLoader(path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["filename"] = file
                    documents.append(doc)

    return documents
```

---

## Optimal Chunking Strategies

### The Chunking Dilemma

```
TOO SMALL (100 chars):                TOO LARGE (2000 chars):
+------------------+                  +----------------------------------+
| "Error: resource |                  | # Terraform State Management     |
| already exists"  |                  |                                  |
+------------------+                  | ## Overview                      |
Missing context!                      | State is how Terraform maps...   |
Can't understand                      |                                  |
the full error.                       | ## Common Errors                 |
                                      | ### Error 1: Resource exists     |
                                      | This happens when...             |
                                      |                                  |
                                      | ### Error 2: State lock          |
                                      | This happens when...             |
                                      |                                  |
                                      | ## Best Practices                |
                                      | Always backup state...           |
                                      +----------------------------------+
                                      Too much irrelevant info!
                                      Dilutes the specific answer.

JUST RIGHT (400-600 chars):
+----------------------------------+
| ## Error: Resource Already Exists |
|                                  |
| This error occurs when Terraform |
| tries to create a resource that  |
| already exists.                  |
|                                  |
| **Causes:**                      |
| 1. Manual creation               |
| 2. Deleted state file            |
|                                  |
| **Solution:**                    |
| terraform import <resource> <id> |
+----------------------------------+
Complete, focused context!
```

### Chunking Best Practices

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

# Method 1: Recursive Character Splitting (General purpose)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "]
)

# Method 2: Markdown Header Splitting (For structured docs)
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

# Combined approach for DevOps docs
def smart_chunk(documents: list) -> list:
    """Split documents intelligently based on structure."""

    # First split by headers
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )

    # Then split large sections further
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = []
    for doc in documents:
        # Split by headers first
        header_chunks = md_splitter.split_text(doc.page_content)

        # Further split if needed
        for chunk in header_chunks:
            if len(chunk.page_content) > 600:
                sub_chunks = text_splitter.split_text(chunk.page_content)
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(chunk)

    return all_chunks
```

---

## Exercise 1: Build the Knowledge Base Loader

Create `exercises/ex1_kb_loader.py`:

```python
"""
Exercise 1: Knowledge Base Loader
=================================
Goal: Build a robust document loader for DevOps docs
"""

import os
from pathlib import Path
from typing import List, Dict
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DevOpsKnowledgeBase:
    """Manages the DevOps documentation knowledge base."""

    def __init__(self, kb_path: str = "knowledge-base"):
        self.kb_path = Path(kb_path)
        self.documents: List[Document] = []
        self.chunks: List[Document] = []

    def load_documents(self) -> List[Document]:
        """
        Load all documents from the knowledge base.

        Returns:
            List of Document objects with metadata
        """
        # TODO: Implement document loading
        # 1. Walk through the knowledge-base directory
        # 2. Load each .md file
        # 3. Add metadata (category, filename, path)
        # 4. Return list of documents
        pass

    def chunk_documents(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        Split documents into chunks for embedding.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of chunked documents
        """
        # TODO: Implement chunking
        # 1. Use RecursiveCharacterTextSplitter
        # 2. Preserve metadata in chunks
        # 3. Return list of chunks
        pass

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": self._count_categories(),
            "avg_chunk_size": self._avg_chunk_size()
        }

    def _count_categories(self) -> Dict[str, int]:
        """Count documents per category."""
        counts = {}
        for doc in self.documents:
            cat = doc.metadata.get("category", "unknown")
            counts[cat] = counts.get(cat, 0) + 1
        return counts

    def _avg_chunk_size(self) -> float:
        """Calculate average chunk size."""
        if not self.chunks:
            return 0
        return sum(len(c.page_content) for c in self.chunks) / len(self.chunks)


if __name__ == "__main__":
    kb = DevOpsKnowledgeBase()

    print("Loading documents...")
    kb.load_documents()

    print("Chunking documents...")
    kb.chunk_documents()

    print("\nKnowledge Base Stats:")
    stats = kb.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

---

## Exercise 2: Metadata Filtering

Enable category-based retrieval:

```python
"""
Exercise 2: Metadata Filtering
==============================
Goal: Implement filtered search by category
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class FilterableKnowledgeBase:
    """Knowledge base with category filtering."""

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = None

    def index_documents(self, chunks: list):
        """Create searchable index from chunks."""
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        category: str = None
    ) -> list:
        """
        Search for relevant documents.

        Args:
            query: Search query
            k: Number of results
            category: Optional category filter (terraform, kubernetes, etc.)

        Returns:
            List of relevant documents
        """
        # TODO: Implement filtered search
        # Hint: Use vectorstore.similarity_search with filter parameter
        # filter={"category": category} if category else None
        pass


if __name__ == "__main__":
    kb = FilterableKnowledgeBase()

    # Test filtered search
    print("All results:")
    results = kb.search("How to fix state errors", k=3)

    print("\nTerraform only:")
    results = kb.search("How to fix state errors", k=3, category="terraform")

    print("\nKubernetes only:")
    results = kb.search("How to fix pod errors", k=3, category="kubernetes")
```

---

## The Knowledge Base Content

We provide pre-built documentation in `knowledge-base/`. Here's a sample:

### Terraform Errors (terraform/errors.md)

```markdown
# Terraform Common Errors and Solutions

## Error: Resource Already Exists

**Error Message:**
```
Error: A resource with the ID "xxx" already exists
```

**What It Means:**
Terraform is trying to create a resource that already exists in your
cloud provider, but Terraform doesn't know about it (not in state).

**Common Causes:**
1. Resource was created manually via console
2. State file was deleted or corrupted
3. Terraform import was incomplete
4. Multiple Terraform workspaces conflict

**Solutions:**

1. **Import the existing resource:**
   ```bash
   terraform import aws_instance.example i-1234567890abcdef0
   ```

2. **Remove from state (if duplicate):**
   ```bash
   terraform state rm aws_instance.example
   ```

3. **Check for conflicts:**
   ```bash
   terraform state list
   terraform show
   ```

**Prevention:**
- Always use remote state with locking
- Never modify infrastructure outside Terraform
- Use workspaces carefully
```

---

## Key Takeaways

1. **Structure matters** - Organize docs by category for better retrieval
2. **Metadata enables filtering** - Tag documents for targeted search
3. **Chunk size is critical** - 400-600 chars works well for technical docs
4. **Preserve context** - Use header-aware splitting for structured docs
5. **Quality over quantity** - Well-written docs beat large volumes

---

## What's Next?

In **Module 5**, we'll dive deep into vector search and embeddings:
- How embeddings represent meaning
- Similarity metrics (cosine, euclidean)
- Tuning retrieval for better results

```bash
cd ../module-5-vector-search
```

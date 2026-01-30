#!/usr/bin/env python3
"""
Exercise 2: Document Loading
=============================

GOAL: Learn how to load different types of documents for RAG.

WHAT YOU'LL LEARN:
- Loading text and markdown files
- Loading multiple files from directories
- Loading PDF files (optional)
- Loading from web URLs
- Understanding document metadata

WHY DOCUMENT LOADING MATTERS:
----------------------------
Your RAG system is only as good as the documents you feed it!

┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT SOURCES                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  📄 Text/Markdown Files    (.txt, .md)                              │
│     - README files                                                   │
│     - Documentation                                                  │
│     - Runbooks                                                       │
│                                                                      │
│  📁 Directories            (multiple files)                          │
│     - Load all docs from docs/ folder                               │
│     - Recursive loading with glob patterns                          │
│                                                                      │
│  📑 PDF Files              (.pdf) - requires extra packages          │
│     - Technical manuals                                              │
│     - Reports                                                        │
│                                                                      │
│  🌐 Web Pages              (URLs)                                    │
│     - Online documentation                                           │
│     - Wiki pages                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install langchain langchain-community

    # For PDF support (optional):
    pip install pypdf

    # For web loading (optional):
    pip install beautifulsoup4

RUN THIS:
    python module-3-rag-fundamentals/exercises/ex2_document_loading.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LANGCHAIN DOCUMENT LOADERS
# =============================================================================
# LangChain provides many loaders for different file types
from langchain_community.document_loaders import (
    TextLoader,           # Single text/markdown file
    DirectoryLoader,      # Multiple files from a directory
    # PyPDFLoader,        # PDF files (requires pypdf)
    # WebBaseLoader,      # Web pages (requires beautifulsoup4)
)


def create_sample_docs():
    """
    Create sample documentation files for testing.

    Creates a directory structure:
        temp_docs/
        ├── terraform/
        │   ├── state_management.md
        │   └── common_errors.md
        ├── kubernetes/
        │   ├── pod_errors.md
        │   └── debugging.md
        └── README.txt
    """
    base_dir = Path(__file__).parent / "temp_docs"

    # Create directory structure
    (base_dir / "terraform").mkdir(parents=True, exist_ok=True)
    (base_dir / "kubernetes").mkdir(parents=True, exist_ok=True)

    # Terraform docs
    (base_dir / "terraform" / "state_management.md").write_text("""# Terraform State Management

## What is Terraform State?
Terraform state is a JSON file that tracks the resources Terraform manages.
It maps your configuration to real-world resources.

## State File Location
By default, state is stored in `terraform.tfstate` in your working directory.

## Remote State
For teams, use remote state backends:
- AWS S3 + DynamoDB
- Azure Blob Storage
- Google Cloud Storage
- Terraform Cloud

```hcl
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-west-2"
  }
}
```
""")

    (base_dir / "terraform" / "common_errors.md").write_text("""# Terraform Common Errors

## Error: Resource Already Exists
The resource exists in cloud but not in state.
Solution: Use `terraform import` to add it to state.

## Error: State Lock
Someone else is running Terraform or a previous run crashed.
Solution: Wait or use `terraform force-unlock LOCK_ID`

## Error: Provider Not Configured
Missing provider block in configuration.
Solution: Add the provider block:
```hcl
provider "aws" {
  region = "us-west-2"
}
```
""")

    # Kubernetes docs
    (base_dir / "kubernetes" / "pod_errors.md").write_text("""# Kubernetes Pod Errors

## CrashLoopBackOff
The container keeps crashing and Kubernetes keeps restarting it.

### Common Causes
1. Application error on startup
2. Missing environment variables
3. Failed health checks
4. Out of memory (OOMKilled)

### Debugging Steps
```bash
# Check pod status
kubectl get pods

# View pod logs
kubectl logs <pod-name>

# View previous container logs (if restarting)
kubectl logs <pod-name> --previous

# Describe pod for events
kubectl describe pod <pod-name>
```

## ImagePullBackOff
Kubernetes cannot pull the container image.

### Common Causes
1. Image doesn't exist
2. Wrong image tag
3. Private registry without credentials
4. Network issues

### Solutions
```bash
# Check if image exists
docker pull <image-name>

# Create registry secret
kubectl create secret docker-registry regcred \\
  --docker-server=<registry> \\
  --docker-username=<user> \\
  --docker-password=<password>
```
""")

    (base_dir / "kubernetes" / "debugging.md").write_text("""# Kubernetes Debugging Guide

## Essential Commands

### Pod Debugging
```bash
# Get all pods with status
kubectl get pods -o wide

# Get pods in all namespaces
kubectl get pods -A

# Watch pods in real-time
kubectl get pods -w
```

### Logs
```bash
# View logs
kubectl logs <pod-name>

# Follow logs (like tail -f)
kubectl logs -f <pod-name>

# Logs from specific container
kubectl logs <pod-name> -c <container-name>
```

### Shell Access
```bash
# Get shell in running container
kubectl exec -it <pod-name> -- /bin/bash

# Run a command in container
kubectl exec <pod-name> -- cat /etc/config
```

## Common Issues Checklist
1. Is the pod running? (`kubectl get pods`)
2. What do the logs say? (`kubectl logs`)
3. What events occurred? (`kubectl describe pod`)
4. Are resources available? (`kubectl describe node`)
""")

    # Root README
    (base_dir / "README.txt").write_text("""DevOps Documentation
====================

This directory contains troubleshooting guides for:
- Terraform infrastructure as code
- Kubernetes container orchestration

See subdirectories for specific topics.
""")

    print(f"✅ Created sample docs in: {base_dir}")
    return str(base_dir)


# =============================================================================
# LOADING METHODS
# =============================================================================

def demo_single_file_loading(docs_dir: str):
    """
    Demonstrate loading a single file.

    TextLoader is the simplest loader - reads one file.
    """
    print("\n" + "=" * 60)
    print("LOADING A SINGLE FILE")
    print("=" * 60)

    file_path = Path(docs_dir) / "terraform" / "common_errors.md"

    # Create the loader
    loader = TextLoader(str(file_path), encoding='utf-8')

    # Load the document
    documents = loader.load()

    print(f"\n📄 File: {file_path.name}")
    print(f"   Documents returned: {len(documents)}")

    # Each document has content and metadata
    doc = documents[0]
    print(f"\n   Content length: {len(doc.page_content)} characters")
    print(f"   Metadata: {doc.metadata}")
    print(f"\n   Preview:\n   {doc.page_content[:200]}...")

    return documents


def demo_directory_loading(docs_dir: str):
    """
    Demonstrate loading all files from a directory.

    DirectoryLoader can load multiple files with glob patterns.
    """
    print("\n" + "=" * 60)
    print("LOADING FROM A DIRECTORY")
    print("=" * 60)

    # Load all .md files from docs_dir and subdirectories
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",           # Pattern: any .md file, any depth
        loader_cls=TextLoader,    # Use TextLoader for each file
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    print(f"\n📁 Directory: {docs_dir}")
    print(f"   Pattern: **/*.md")
    print(f"   Documents loaded: {len(documents)}")

    print("\n   Files loaded:")
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        chars = len(doc.page_content)
        print(f"   - {Path(source).name}: {chars} chars")

    return documents


def demo_multiple_patterns(docs_dir: str):
    """
    Demonstrate loading with different file patterns.
    """
    print("\n" + "=" * 60)
    print("LOADING WITH DIFFERENT PATTERNS")
    print("=" * 60)

    patterns = [
        ("**/*.md", "All markdown files"),
        ("**/*.txt", "All text files"),
        ("terraform/*.md", "Only terraform markdown"),
        ("kubernetes/*.md", "Only kubernetes markdown"),
    ]

    for pattern, description in patterns:
        loader = DirectoryLoader(
            docs_dir,
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True  # Don't crash if no matches
        )

        try:
            docs = loader.load()
            print(f"\n   Pattern: {pattern}")
            print(f"   ({description})")
            print(f"   Loaded: {len(docs)} documents")
        except Exception as e:
            print(f"\n   Pattern: {pattern} - Error: {e}")


def demo_document_metadata():
    """
    Explain document metadata and why it matters.
    """
    print("\n" + "=" * 60)
    print("UNDERSTANDING DOCUMENT METADATA")
    print("=" * 60)

    print("""
    When you load a document, you get:

    ┌─────────────────────────────────────────────────────────────────┐
    │ Document Object                                                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  page_content: str                                               │
    │    The actual text content of the document                       │
    │                                                                  │
    │  metadata: dict                                                  │
    │    Information ABOUT the document:                               │
    │    - source: file path or URL                                    │
    │    - Custom fields you can add                                   │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

    Why metadata matters:

    1. SOURCE TRACKING
       When RAG returns a result, metadata tells you WHERE it came from.
       User: "How do I fix state lock?"
       RAG: "Based on terraform/common_errors.md, you should..."

    2. FILTERING
       You can filter searches by metadata:
       "Only search Kubernetes docs" → filter by source path

    3. VERSIONING
       Add version info: {"source": "...", "version": "1.2.0"}
    """)


def demo_custom_metadata(docs_dir: str):
    """
    Show how to add custom metadata to documents.
    """
    print("\n" + "=" * 60)
    print("ADDING CUSTOM METADATA")
    print("=" * 60)

    # Load a document
    file_path = Path(docs_dir) / "kubernetes" / "pod_errors.md"
    loader = TextLoader(str(file_path), encoding='utf-8')
    documents = loader.load()

    # Add custom metadata
    for doc in documents:
        doc.metadata["category"] = "kubernetes"
        doc.metadata["topic"] = "troubleshooting"
        doc.metadata["version"] = "1.0"
        doc.metadata["author"] = "DevOps Team"

    print(f"\n   Original metadata: {{'source': '...pod_errors.md'}}")
    print(f"\n   Enhanced metadata:")
    for key, value in documents[0].metadata.items():
        if key != 'source':
            print(f"      {key}: {value}")

    return documents


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 2: Document Loading")
    print("=" * 60)

    # Create sample documentation
    docs_dir = create_sample_docs()

    # Demo different loading methods
    demo_single_file_loading(docs_dir)
    demo_directory_loading(docs_dir)
    demo_multiple_patterns(docs_dir)
    demo_document_metadata()
    demo_custom_metadata(docs_dir)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    LOADERS CHEAT SHEET:
    ───────────────────────────────────────────────────────────

    TextLoader("file.md")
        → Load a single text/markdown file

    DirectoryLoader("docs/", glob="**/*.md")
        → Load all matching files from directory

    PyPDFLoader("document.pdf")
        → Load PDF file (requires: pip install pypdf)

    WebBaseLoader("https://docs.example.com")
        → Load web page (requires: pip install beautifulsoup4)

    ───────────────────────────────────────────────────────────

    GLOB PATTERNS:
    ───────────────────────────────────────────────────────────

    "*.md"        → All .md files in current directory
    "**/*.md"     → All .md files in all subdirectories
    "terraform/*" → All files in terraform/ directory
    "**/*.{md,txt}" → All .md and .txt files everywhere

    ───────────────────────────────────────────────────────────
    """)

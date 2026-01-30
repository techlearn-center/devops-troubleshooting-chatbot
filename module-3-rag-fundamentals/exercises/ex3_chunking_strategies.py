#!/usr/bin/env python3
"""
Exercise 3: Chunking Strategies
================================

GOAL: Learn how to split documents into chunks effectively.

WHAT YOU'LL LEARN:
- Why chunking matters for RAG
- Different chunking strategies
- How chunk size affects retrieval
- Finding the optimal chunk size

WHY CHUNKING MATTERS:
--------------------
Documents can be thousands of words. But for RAG:
- We want to retrieve SPECIFIC relevant parts
- LLMs have context limits
- Smaller chunks = more precise retrieval

┌──────────────────────────────────────────────────────────────────────┐
│                      CHUNKING TRADE-OFFS                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  TOO LARGE (2000+ chars):                                            │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Relevant info      │ Irrelevant info │ More irrelevant │     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ❌ Retrieves too much unrelated content                             │
│  ❌ Uses up context window with noise                                │
│                                                                       │
│  TOO SMALL (100 chars):                                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                       │
│  │ Part │ │  of  │ │  a   │ │sent- │ │ence  │                       │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘                       │
│  ❌ Loses context                                                     │
│  ❌ Incomplete information                                            │
│                                                                       │
│  JUST RIGHT (300-500 chars):                                         │
│  ┌─────────────────────────┐ ┌─────────────────────────┐            │
│  │ Complete concept 1      │ │ Complete concept 2      │            │
│  │ with full context       │ │ with full context       │            │
│  └─────────────────────────┘ └─────────────────────────┘            │
│  ✅ Each chunk has complete, focused information                     │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install langchain langchain-community

RUN THIS:
    python module-3-rag-fundamentals/exercises/ex3_chunking_strategies.py

    # With visualization demo
    python module-3-rag-fundamentals/exercises/ex3_chunking_strategies.py --visual
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# TEXT SPLITTERS FROM LANGCHAIN
# =============================================================================
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,  # Most common, respects structure
    CharacterTextSplitter,           # Simple character-based splitting
    MarkdownTextSplitter,            # Markdown-aware splitting
)


# Sample documentation for testing
SAMPLE_DOCUMENT = """# Kubernetes Pod Troubleshooting Guide

## Introduction

This guide covers common Kubernetes pod issues and how to resolve them.
Understanding pod states and error messages is crucial for effective debugging.

## Common Pod States

### Running
The pod is executing normally. All containers are running.

### Pending
The pod is waiting to be scheduled. Common causes:
- Insufficient resources (CPU, memory)
- Node selector doesn't match any nodes
- PersistentVolumeClaim not bound

### CrashLoopBackOff
The container keeps crashing and Kubernetes keeps restarting it.

#### Causes
1. Application error on startup
2. Missing environment variables or config
3. Failed liveness/readiness probes
4. Out of memory (OOMKilled)

#### Debugging Steps
```bash
# Check pod events
kubectl describe pod <pod-name>

# View container logs
kubectl logs <pod-name>

# View logs from previous crashed container
kubectl logs <pod-name> --previous

# Get into the container for debugging
kubectl exec -it <pod-name> -- /bin/sh
```

### ImagePullBackOff
Kubernetes cannot pull the container image.

#### Common Causes
1. Image name or tag is incorrect
2. Image doesn't exist in registry
3. Private registry requires authentication
4. Network issues reaching the registry

#### Solutions
1. Verify image exists: `docker pull <image>`
2. Check image name spelling
3. Create imagePullSecret for private registries
4. Check network connectivity

## Resource Issues

### OOMKilled (Exit Code 137)
The container was killed because it exceeded its memory limit.

#### Solution
Increase the memory limit in your pod spec:
```yaml
resources:
  limits:
    memory: "512Mi"
  requests:
    memory: "256Mi"
```

### CPU Throttling
The container is being throttled due to CPU limits.

#### Solution
Adjust CPU limits or requests:
```yaml
resources:
  limits:
    cpu: "1000m"
  requests:
    cpu: "500m"
```

## Network Issues

### Service Not Reachable
The pod cannot reach other services.

#### Debugging Steps
1. Check if pod has correct network policy
2. Verify DNS resolution: `kubectl exec <pod> -- nslookup <service>`
3. Check service exists: `kubectl get svc`
4. Test connectivity: `kubectl exec <pod> -- curl <service>:<port>`

## Summary

When debugging pods:
1. Start with `kubectl get pods` to see status
2. Use `kubectl describe pod` for events
3. Check logs with `kubectl logs`
4. Verify resource limits and requests
5. Check network policies and services
"""


def demo_character_splitter():
    """
    Demonstrate basic character-based splitting.

    This is the simplest splitter - just splits at a character count.
    Problem: Might split in the middle of words or sentences!
    """
    print("\n" + "=" * 60)
    print("CHARACTER TEXT SPLITTER")
    print("=" * 60)

    splitter = CharacterTextSplitter(
        separator="\n",      # Try to split at newlines
        chunk_size=200,      # Target chunk size
        chunk_overlap=20,    # Overlap between chunks
    )

    chunks = splitter.split_text(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=200, overlap=20")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


def demo_recursive_splitter():
    """
    Demonstrate the RecursiveCharacterTextSplitter.

    This is the RECOMMENDED splitter. It:
    - Tries to split at natural boundaries (paragraphs, sentences)
    - Falls back to smaller separators if needed
    - Maintains semantic coherence
    """
    print("\n" + "=" * 60)
    print("RECURSIVE CHARACTER TEXT SPLITTER (Recommended)")
    print("=" * 60)

    # The separators list defines split priorities
    # It tries the first separator, then the next if chunks are still too big
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=[
            "\n\n",  # First try: split at blank lines (paragraphs)
            "\n",    # Then: split at newlines
            ". ",    # Then: split at sentences
            " ",     # Then: split at words
            ""       # Last resort: split at characters
        ]
    )

    chunks = splitter.split_text(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=500, overlap=50")
    print(f"Separators: ['\\n\\n', '\\n', '. ', ' ', '']")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        lines = chunk.split('\n')[:5]
        print('\n'.join(lines))
        if len(chunk.split('\n')) > 5:
            print("...")


def demo_markdown_splitter():
    """
    Demonstrate Markdown-aware splitting.

    This splitter understands Markdown structure and splits at:
    - Headers (# ## ###)
    - Code blocks
    - Lists
    """
    print("\n" + "=" * 60)
    print("MARKDOWN TEXT SPLITTER")
    print("=" * 60)

    splitter = MarkdownTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=500, overlap=50")
    print(f"Markdown-aware: Splits at headers, code blocks, lists")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        # Show first few lines
        lines = chunk.split('\n')[:6]
        print('\n'.join(lines))
        if len(chunk.split('\n')) > 6:
            print("...")


def compare_chunk_sizes():
    """
    Compare different chunk sizes to see the impact.
    """
    print("\n" + "=" * 60)
    print("COMPARING CHUNK SIZES")
    print("=" * 60)

    sizes = [200, 500, 1000, 2000]

    print(f"\nDocument length: {len(SAMPLE_DOCUMENT)} characters")
    print(f"\n{'Size':<10} {'Chunks':<10} {'Avg Size':<12} {'Assessment'}")
    print("-" * 50)

    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=int(size * 0.1)  # 10% overlap
        )
        chunks = splitter.split_text(SAMPLE_DOCUMENT)
        avg_size = sum(len(c) for c in chunks) // len(chunks)

        # Assessment
        if size < 300:
            assessment = "Too small - lost context"
        elif size > 1500:
            assessment = "Too large - noisy retrieval"
        else:
            assessment = "Good for RAG ✓"

        print(f"{size:<10} {len(chunks):<10} {avg_size:<12} {assessment}")


def visualize_chunking():
    """
    Visual demonstration of how text gets chunked.
    """
    print("\n" + "=" * 60)
    print("VISUALIZING CHUNKING")
    print("=" * 60)

    # Simple example text
    text = """## Error: CrashLoopBackOff

The container keeps crashing and restarting.

### Causes
1. Application error on startup
2. Missing environment variables
3. Failed health checks

### Solutions
Check the logs with kubectl logs."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(text)

    print("\nOriginal text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

    print(f"\nChunk size: 150, Overlap: 20")
    print(f"Result: {len(chunks)} chunks\n")

    # Show chunks with visual boundaries
    for i, chunk in enumerate(chunks, 1):
        print(f"┌─ Chunk {i} ({len(chunk)} chars) " + "─" * (40 - len(str(len(chunk)))))
        for line in chunk.split('\n'):
            print(f"│ {line}")
        print(f"└" + "─" * 50)
        print()


def demo_overlap_importance():
    """
    Show why chunk overlap matters.
    """
    print("\n" + "=" * 60)
    print("WHY CHUNK OVERLAP MATTERS")
    print("=" * 60)

    text = """The solution to CrashLoopBackOff is to check the container logs.
Use kubectl logs to see what error is causing the crash.
Then fix the underlying issue in your application code."""

    print("\nOriginal text:")
    print(f"  \"{text}\"\n")

    # Without overlap
    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=0
    )
    chunks_no = splitter_no_overlap.split_text(text)

    print("WITHOUT OVERLAP (chunk_overlap=0):")
    print("-" * 50)
    for i, chunk in enumerate(chunks_no, 1):
        print(f"  Chunk {i}: \"{chunk}\"")

    print("\n  ⚠️  Notice: Context is lost between chunks!")
    print("      'kubectl logs' is split from 'CrashLoopBackOff'")

    # With overlap
    splitter_overlap = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=30
    )
    chunks_yes = splitter_overlap.split_text(text)

    print("\n\nWITH OVERLAP (chunk_overlap=30):")
    print("-" * 50)
    for i, chunk in enumerate(chunks_yes, 1):
        print(f"  Chunk {i}: \"{chunk}\"")

    print("\n  ✓ Notice: Overlapping text provides context continuity!")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 3: Chunking Strategies")
    print("=" * 60)

    if len(sys.argv) > 1 and sys.argv[1] == "--visual":
        visualize_chunking()
        demo_overlap_importance()
    else:
        demo_character_splitter()
        demo_recursive_splitter()
        demo_markdown_splitter()
        compare_chunk_sizes()
        demo_overlap_importance()

    print("\n" + "=" * 60)
    print("CHUNKING BEST PRACTICES")
    print("=" * 60)
    print("""
    ┌────────────────────────────────────────────────────────────────┐
    │                  CHUNKING RECOMMENDATIONS                       │
    ├────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. USE RecursiveCharacterTextSplitter                         │
    │     It's the best general-purpose splitter                      │
    │                                                                 │
    │  2. CHUNK SIZE: 300-500 characters                             │
    │     Good balance of context and precision                       │
    │                                                                 │
    │  3. OVERLAP: 10-20% of chunk size                              │
    │     Maintains context across chunk boundaries                   │
    │                                                                 │
    │  4. FOR MARKDOWN: Use MarkdownTextSplitter                     │
    │     Respects document structure                                 │
    │                                                                 │
    │  5. TEST YOUR CHUNKS                                            │
    │     Query your RAG system and check if retrieved                │
    │     chunks make sense                                           │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘

    TYPICAL SETTINGS FOR DEVOPS DOCS:
    ─────────────────────────────────────────────────────────────────

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\\n\\n", "\\n", ". ", " ", ""]
    )
    """)

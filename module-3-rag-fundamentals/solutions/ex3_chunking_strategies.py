#!/usr/bin/env python3
"""
SOLUTION: Exercise 3 - Chunking Strategies
===========================================

Complete chunking solution with analysis tools.

RUN THIS:
    python module-3-rag-fundamentals/solutions/ex3_chunking_strategies.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
)


class ChunkAnalyzer:
    """
    Analyze and optimize chunking strategies.
    """

    def __init__(self, text: str):
        self.text = text

    def analyze_chunk_size(self, sizes: list = [200, 300, 500, 800, 1000]):
        """Analyze different chunk sizes."""
        print("\n📊 Chunk Size Analysis")
        print("-" * 60)
        print(f"{'Size':<10} {'Chunks':<10} {'Avg':<10} {'Min':<10} {'Max':<10}")
        print("-" * 60)

        for size in sizes:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=size,
                chunk_overlap=int(size * 0.1)
            )
            chunks = splitter.split_text(self.text)
            lengths = [len(c) for c in chunks]

            print(f"{size:<10} {len(chunks):<10} {sum(lengths)//len(lengths):<10} {min(lengths):<10} {max(lengths):<10}")

    def optimal_chunk_size(self, target_chunks: int = 10) -> int:
        """Find chunk size to achieve target number of chunks."""
        # Binary search for optimal size
        low, high = 100, 2000

        while low < high:
            mid = (low + high) // 2
            splitter = RecursiveCharacterTextSplitter(chunk_size=mid, chunk_overlap=0)
            chunks = splitter.split_text(self.text)

            if len(chunks) > target_chunks:
                low = mid + 1
            else:
                high = mid

        return low

    def split_optimized(self, chunk_size: int = 500) -> list:
        """Split with optimized settings for RAG."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(self.text)


# Sample document for testing
SAMPLE_DOC = """# Kubernetes Troubleshooting Guide

## CrashLoopBackOff

The CrashLoopBackOff error indicates that a container keeps crashing.

### Debugging Steps
1. Check logs: kubectl logs <pod>
2. Check previous logs: kubectl logs <pod> --previous
3. Describe pod: kubectl describe pod <pod>

### Common Causes
- Application error on startup
- Missing environment variables
- Out of memory (OOMKilled)

## ImagePullBackOff

Cannot pull the container image.

### Solutions
- Verify image name and tag
- Check registry credentials
- Test with docker pull
"""


if __name__ == "__main__":
    analyzer = ChunkAnalyzer(SAMPLE_DOC)

    # Analyze different sizes
    analyzer.analyze_chunk_size()

    # Find optimal size for ~5 chunks
    optimal = analyzer.optimal_chunk_size(target_chunks=5)
    print(f"\n✅ Optimal chunk size for ~5 chunks: {optimal}")

    # Show optimized chunks
    chunks = analyzer.split_optimized(chunk_size=400)
    print(f"\n📄 Optimized Chunks ({len(chunks)} total):")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

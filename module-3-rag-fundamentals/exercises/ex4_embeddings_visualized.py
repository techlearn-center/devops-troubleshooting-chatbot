#!/usr/bin/env python3
"""
Exercise 4: Embeddings Visualized
==================================

GOAL: Understand how text becomes vectors and why similar texts
      have similar vectors.

WHAT YOU'LL LEARN:
- How embedding models work
- Why similar texts produce similar vectors
- How to measure similarity between vectors
- Visualizing embeddings (conceptually)

WHAT ARE EMBEDDINGS?
-------------------
Embeddings convert text into numbers (vectors) that capture MEANING.

    ┌──────────────────────┐          ┌──────────────────────────────┐
    │                      │          │                              │
    │  "pod crash error"   │  ──────► │  [0.82, -0.15, 0.43, ...]   │
    │                      │          │                              │
    │      (text)          │          │        (384 numbers)         │
    └──────────────────────┘          └──────────────────────────────┘

THE MAGIC: Similar meanings = Similar numbers!

    "pod crash error"      → [0.82, -0.15, 0.43, ...]  ─┐
                                                         ├─ Similar vectors!
    "container restarting" → [0.79, -0.12, 0.41, ...]  ─┘

    "terraform modules"    → [0.11, 0.65, -0.33, ...]  ← Different topic = different vector

REQUIREMENTS:
    pip install sentence-transformers langchain langchain-community numpy

RUN THIS:
    python module-3-rag-fundamentals/exercises/ex4_embeddings_visualized.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# NumPy for vector math
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  NumPy not installed. Run: pip install numpy")

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Calculate cosine similarity between two vectors.

    WHAT IS COSINE SIMILARITY?
    ─────────────────────────
    It measures how similar two vectors are based on their angle.
    - 1.0 = identical direction (very similar)
    - 0.0 = perpendicular (unrelated)
    - -1.0 = opposite direction (opposite meaning)

    For embeddings, similar text = similar direction = high cosine similarity

    VISUAL EXPLANATION:
    ──────────────────
            ▲ Vector A          ▲ Vector B
           /│                    │
          / │  angle θ         / │
         /  │      ▲          /  │
        /   │     /          /   │
       /θ   │    / Vector C     │
      ────────►         ────────►

    A & B: Small angle = High similarity (close to 1.0)
    A & C: Large angle = Low similarity (close to 0.0)

    Args:
        vec1: First vector (list of numbers)
        vec2: Second vector (list of numbers)

    Returns:
        Similarity score between -1 and 1
    """
    if HAS_NUMPY:
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    else:
        # Manual calculation without NumPy
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude_a = sum(a * a for a in vec1) ** 0.5
        magnitude_b = sum(b * b for b in vec2) ** 0.5
        return dot_product / (magnitude_a * magnitude_b)


def demo_basic_embeddings():
    """
    Demonstrate basic embedding creation.
    """
    print("\n" + "=" * 60)
    print("CREATING EMBEDDINGS")
    print("=" * 60)

    # Create embedding model (runs locally, free!)
    print("\n📦 Loading embedding model...")
    print("   Model: all-MiniLM-L6-v2 (free, local)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Sample text
    text = "Kubernetes pod CrashLoopBackOff error"

    print(f"\n📝 Text: \"{text}\"")

    # Create embedding
    print("\n🔄 Converting text to embedding...")
    vector = embeddings.embed_query(text)

    print(f"\n📊 Result:")
    print(f"   Vector dimensions: {len(vector)}")
    print(f"   First 10 values: {[round(v, 4) for v in vector[:10]]}")
    print(f"   Last 10 values:  {[round(v, 4) for v in vector[-10:]]}")

    return embeddings


def demo_similarity_comparison(embeddings):
    """
    Compare similarity between different texts.
    """
    print("\n" + "=" * 60)
    print("COMPARING TEXT SIMILARITY")
    print("=" * 60)

    # Test texts - some similar, some different
    texts = [
        "Kubernetes pod CrashLoopBackOff error",      # Reference
        "Container keeps crashing and restarting",    # Similar meaning
        "Pod fails to start in K8s cluster",         # Similar topic
        "How to fix OOMKilled in Kubernetes",        # Related topic
        "Terraform state lock error",                # Different topic
        "How to bake chocolate chip cookies",        # Completely different
    ]

    print("\n📝 Reference text: \"Kubernetes pod CrashLoopBackOff error\"")
    print("\n   Comparing with other texts...\n")

    # Get embedding for reference text
    ref_vector = embeddings.embed_query(texts[0])

    # Compare with all other texts
    print(f"   {'Text':<50} {'Similarity':<12} {'Assessment'}")
    print("   " + "-" * 75)

    for text in texts[1:]:
        # Get embedding for this text
        vector = embeddings.embed_query(text)

        # Calculate similarity
        similarity = cosine_similarity(ref_vector, vector)

        # Assessment based on similarity score
        if similarity > 0.7:
            assessment = "✅ Very similar"
            bar = "█" * int(similarity * 20)
        elif similarity > 0.5:
            assessment = "📗 Somewhat similar"
            bar = "█" * int(similarity * 20)
        elif similarity > 0.3:
            assessment = "📙 Slightly related"
            bar = "█" * int(similarity * 20)
        else:
            assessment = "❌ Different topic"
            bar = "█" * int(similarity * 20)

        # Truncate text for display
        display_text = text[:45] + "..." if len(text) > 45 else text
        print(f"   {display_text:<50} {similarity:.4f}  {bar:<20} {assessment}")


def demo_semantic_vs_keyword():
    """
    Show that embeddings capture meaning, not just keywords.
    """
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH vs KEYWORD SEARCH")
    print("=" * 60)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("""
    KEYWORD SEARCH only finds exact word matches:
    ─────────────────────────────────────────────
    Query: "container restart"
    ✅ "How to restart a container"  (has "container" and "restart")
    ❌ "Pod keeps crashing"          (no matching words!)

    SEMANTIC SEARCH (embeddings) finds meaning:
    ─────────────────────────────────────────────
    Query: "container restart"
    ✅ "How to restart a container"  (same topic)
    ✅ "Pod keeps crashing"          (same concept - containers crashing!)
    """)

    # Demonstrate
    query = "container restart"
    texts = [
        "How to restart a container",      # Keyword match
        "Pod keeps crashing and restarting",  # Semantic match (no exact keywords)
        "CrashLoopBackOff troubleshooting",   # Semantic match
        "Terraform provider configuration",   # Unrelated
    ]

    query_vector = embeddings.embed_query(query)

    print(f"   Query: \"{query}\"")
    print(f"\n   {'Document':<45} {'Has Keywords?':<15} {'Semantic Score'}")
    print("   " + "-" * 75)

    for text in texts:
        # Check keyword presence
        has_keywords = "container" in text.lower() or "restart" in text.lower()
        keyword_str = "✅ Yes" if has_keywords else "❌ No"

        # Get semantic similarity
        text_vector = embeddings.embed_query(text)
        similarity = cosine_similarity(query_vector, text_vector)

        bar = "█" * int(similarity * 15)
        print(f"   {text:<45} {keyword_str:<15} {similarity:.3f} {bar}")

    print("""
    NOTICE: Semantic search found "Pod keeps crashing" and
            "CrashLoopBackOff" as relevant, even though they
            don't contain "container" or "restart"!

    This is the power of embeddings - they understand MEANING!
    """)


def demo_embedding_dimensions():
    """
    Explain what the vector dimensions represent.
    """
    print("\n" + "=" * 60)
    print("UNDERSTANDING VECTOR DIMENSIONS")
    print("=" * 60)

    print("""
    WHAT DO THE 384 NUMBERS MEAN?
    ────────────────────────────

    Each number represents a "feature" or "aspect" of the text.

    Think of it like describing a person with numbers:
    ┌──────────────────────────────────────────────────────────────┐
    │ Person Description Vector:                                    │
    │   [height, weight, age, friendliness, intelligence, ...]     │
    │   [1.75,   70,     30,  0.8,         0.9,           ...]     │
    └──────────────────────────────────────────────────────────────┘

    Text embeddings are similar, but with 384 abstract features:
    ┌──────────────────────────────────────────────────────────────┐
    │ Text Embedding Vector:                                        │
    │   [feature_1, feature_2, ..., feature_384]                   │
    │   [0.23,      -0.45,    ...,  0.12      ]                    │
    │                                                               │
    │ These features might represent things like:                   │
    │   - "Technical-ness"                                          │
    │   - "Error-related-ness"                                      │
    │   - "Kubernetes-related-ness"                                 │
    │   - ... and 381 other abstract concepts!                     │
    └──────────────────────────────────────────────────────────────┘

    We can't easily interpret individual dimensions, but the MODEL
    learned which combinations of features capture meaning!

    DIFFERENT MODELS = DIFFERENT DIMENSIONS:
    ───────────────────────────────────────
    all-MiniLM-L6-v2:        384 dimensions (fast, good quality)
    all-mpnet-base-v2:       768 dimensions (higher quality, slower)
    OpenAI text-embedding-3: 1536 dimensions (highest quality, paid)
    """)


def demo_visualize_concept():
    """
    Conceptual visualization of embeddings in 2D.
    """
    print("\n" + "=" * 60)
    print("CONCEPTUAL VISUALIZATION")
    print("=" * 60)

    print("""
    If we could reduce 384 dimensions to 2D, embeddings might look like:

                              Kubernetes Topics
                                     ▲
                                     │
                        ┌────────────┼────────────┐
                        │            │            │
                        │   •pod     │            │
                        │    crash   │            │
                        │   •crash   │            │
                        │    loop    │            │
                        │            │•k8s        │
                        │            │ networking │
    Error Topics ◄──────┼────────────┼────────────┼──────► Config Topics
                        │            │            │
                        │•terraform  │            │•terraform
                        │ state lock │            │ modules
                        │            │            │
                        │            │•docker     │
                        │            │ compose    │
                        └────────────┼────────────┘
                                     │
                                     ▼
                              Docker Topics

    Similar texts cluster together in this high-dimensional space!

    When you search for "pod crash error":
    1. Your query becomes a point in this space
    2. We find the closest document points
    3. Those are the most relevant documents!
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 4: Embeddings Visualized")
    print("=" * 60)

    # Load embedding model once
    print("\n📦 Loading embedding model (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("   ✅ Model loaded!")

    # Run demos
    demo_basic_embeddings()
    demo_similarity_comparison(embeddings)
    demo_semantic_vs_keyword()
    demo_embedding_dimensions()
    demo_visualize_concept()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  1. EMBEDDINGS convert text to vectors (lists of numbers)      │
    │                                                                 │
    │  2. SIMILAR MEANINGS produce SIMILAR VECTORS                   │
    │     - "pod crash" ≈ "container restart" (high similarity)      │
    │     - "pod crash" ≠ "terraform modules" (low similarity)       │
    │                                                                 │
    │  3. COSINE SIMILARITY measures vector similarity               │
    │     - 1.0 = identical meaning                                   │
    │     - 0.0 = unrelated                                           │
    │                                                                 │
    │  4. SEMANTIC SEARCH beats keyword search                       │
    │     - Finds documents by MEANING, not just word matching       │
    │                                                                 │
    │  5. EMBEDDING MODELS vary in quality and speed                 │
    │     - all-MiniLM-L6-v2: Fast, free, 384 dimensions             │
    │     - OpenAI: Highest quality, paid, 1536 dimensions           │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘
    """)

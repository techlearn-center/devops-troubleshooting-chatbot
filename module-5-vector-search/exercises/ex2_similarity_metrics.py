#!/usr/bin/env python3
"""
Exercise 2: Similarity Metrics Hands-On
=========================================

GOAL: Implement cosine similarity, Euclidean distance, and dot product
      FROM SCRATCH, understand the math behind each one, and learn
      when to use which metric for your vector search.

WHAT YOU'LL LEARN:
- The math behind three core similarity/distance metrics
- How to implement each one with just NumPy (no libraries hiding the magic)
- When each metric gives DIFFERENT results (and why that matters)
- How to pick the right metric for your DevOps chatbot's vector DB

WHY DO SIMILARITY METRICS MATTER?
──────────────────────────────────
When your user asks "fix pod crash", the vector database needs to find
the most similar documents. But "most similar" depends on HOW you
measure similarity!

Different metrics can rank documents differently:

    Query: "fix pod crash"  ->  vector: [0.8, 0.6]

    Doc A vector: [0.4, 0.3]  (same direction, shorter)
    Doc B vector: [0.7, 0.8]  (slightly different direction, similar length)

    Cosine similarity:   Doc A wins! (same direction)
    Euclidean distance:  Doc B wins! (closer in space)
    Dot product:         Doc B wins! (larger projection)

THE THREE METRICS VISUALIZED:
─────────────────────────────

    Cosine Similarity (measures ANGLE):
    ────────────────────────────────────
                ▲ Doc A
               /
              / ) angle = small  ->  HIGH similarity
             /  )
            /───────────► Query

    Does NOT care about vector length, ONLY direction.
    Best when: document length varies a lot.


    Euclidean Distance (measures STRAIGHT-LINE distance):
    ──────────────────────────────────────────────────────
            Doc A •
                  |  distance
                  |  = short  ->  HIGH similarity
            Query •

    Cares about BOTH direction AND magnitude.
    Best when: vector magnitudes carry meaningful information.


    Dot Product (measures PROJECTION):
    ───────────────────────────────────
            ──────────► Doc A
            ────────►   Query
            │projection│

    Combines direction AND magnitude.
    Best when: you want to reward both similarity AND "strength" of match.


COMPARISON TABLE:
─────────────────

    ┌──────────────────┬────────────────┬────────────────┬────────────────┐
    │ Property         │ Cosine         │ Euclidean      │ Dot Product    │
    ├──────────────────┼────────────────┼────────────────┼────────────────┤
    │ Measures         │ Angle          │ Distance       │ Projection     │
    │ Range            │ [-1, 1]        │ [0, inf)       │ (-inf, inf)    │
    │ "Similar" means  │ Close to 1     │ Close to 0     │ Large positive │
    │ Normalized vecs? │ Recommended    │ Optional       │ Required       │
    │ Ignores length?  │ Yes            │ No             │ No             │
    │ ChromaDB name    │ "cosine"       │ "l2"           │ "ip"           │
    │ Default in most  │ Yes            │ Sometimes      │ Rarely         │
    │ vector DBs       │                │                │                │
    └──────────────────┴────────────────┴────────────────┴────────────────┘

REQUIREMENTS:
    pip install numpy sentence-transformers langchain-community

RUN THIS:
    python module-5-vector-search/exercises/ex2_similarity_metrics.py
"""

import os
import sys
import math
from pathlib import Path

# =============================================================================
# PROJECT SETUP
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# NumPy for numerical operations on vectors (arrays of numbers)
import numpy as np

# We use the embedding model to generate real vectors for our tests
from sentence_transformers import SentenceTransformer


# =============================================================================
# METRIC 1: COSINE SIMILARITY (from scratch)
# =============================================================================

def cosine_similarity_scratch(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors FROM SCRATCH.

    THE MATH:
    ─────────
    Cosine similarity measures the cosine of the angle between two vectors.

    Formula:
                        A . B              sum(a_i * b_i)
        cos(theta) = ───────────── = ─────────────────────────────────
                     ||A|| * ||B||   sqrt(sum(a_i^2)) * sqrt(sum(b_i^2))

    Let's break this down step by step:

    STEP 1: Dot Product (A . B)
    ───────────────────────────
    Multiply corresponding elements and sum them up:

        A = [1, 2, 3]
        B = [4, 5, 6]
        A . B = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32

    STEP 2: Magnitudes (||A|| and ||B||)
    ─────────────────────────────────────
    The "length" of each vector (Euclidean norm):

        ||A|| = sqrt(1^2 + 2^2 + 3^2) = sqrt(1 + 4 + 9) = sqrt(14) = 3.742
        ||B|| = sqrt(4^2 + 5^2 + 6^2) = sqrt(16 + 25 + 36) = sqrt(77) = 8.775

    STEP 3: Divide
    ───────────────
        cos(theta) = 32 / (3.742 * 8.775) = 32 / 32.833 = 0.9746

    Result: 0.9746 -- these vectors are very similar!

    VISUAL:
    ───────
        Cosine = 1.0          Cosine = 0.0          Cosine = -1.0
        (identical)            (perpendicular)        (opposite)

           B ▲                    B ▲                     ▲ A
            /                      │                     │
           /                       │                     │
          /  A                     ──────► A             │
         ──────►                                         ▼ B

    Args:
        vec_a: First vector (list or numpy array)
        vec_b: Second vector (list or numpy array)

    Returns:
        Float between -1.0 and 1.0 (higher = more similar)
    """
    # Convert to numpy arrays if they aren't already.
    # numpy arrays support element-wise operations which makes the math clean.
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    # STEP 1: Calculate the dot product
    # This is the "numerator" of our formula.
    # np.dot multiplies element-wise and sums: a[0]*b[0] + a[1]*b[1] + ...
    dot_product = np.dot(a, b)

    # STEP 2: Calculate the magnitudes (norms)
    # np.linalg.norm computes sqrt(sum(x_i^2)) -- the Euclidean length
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # STEP 3: Guard against division by zero
    # A zero vector has no direction, so cosine similarity is undefined.
    # In practice, this only happens with degenerate inputs.
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0

    # STEP 4: Divide dot product by product of magnitudes
    similarity = dot_product / (magnitude_a * magnitude_b)

    # Clamp to [-1, 1] to handle floating-point rounding errors
    # (e.g., 1.0000000002 due to float math)
    return float(np.clip(similarity, -1.0, 1.0))


# =============================================================================
# METRIC 2: EUCLIDEAN DISTANCE (from scratch)
# =============================================================================

def euclidean_distance_scratch(vec_a, vec_b):
    """
    Calculate Euclidean distance between two vectors FROM SCRATCH.

    THE MATH:
    ─────────
    Euclidean distance is the straight-line distance between two points.
    It's the same formula you'd use to measure distance on a map!

    Formula (Pythagorean theorem extended to N dimensions):

        d(A, B) = sqrt( sum( (a_i - b_i)^2 ) )

    Let's break this down:

    STEP 1: Element-wise Difference
    ────────────────────────────────
        A = [1, 2, 3]
        B = [4, 5, 6]
        A - B = [-3, -3, -3]

    STEP 2: Square Each Difference
    ──────────────────────────────
        (-3)^2 = 9, (-3)^2 = 9, (-3)^2 = 9

    STEP 3: Sum the Squares
    ───────────────────────
        9 + 9 + 9 = 27

    STEP 4: Square Root
    ───────────────────
        sqrt(27) = 5.196

    IN 2D (easy to visualize):
    ──────────────────────────

        A = (1, 2),  B = (4, 6)

             B(4,6)
             *
            /|
           / |  distance = sqrt((4-1)^2 + (6-2)^2)
          /  |           = sqrt(9 + 16)
         /   |           = sqrt(25)
        *────┘           = 5.0
      A(1,2)

    IMPORTANT DIFFERENCE FROM COSINE:
    ──────────────────────────────────
    - Euclidean distance = 0 means IDENTICAL (lower is better)
    - Cosine similarity = 1 means IDENTICAL (higher is better)

    They point in "opposite directions"! This trips up beginners.

    Args:
        vec_a: First vector (list or numpy array)
        vec_b: Second vector (list or numpy array)

    Returns:
        Float >= 0 (lower = more similar, 0 = identical)
    """
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    # STEP 1: Element-wise difference
    diff = a - b

    # STEP 2: Square each difference
    squared_diff = diff ** 2

    # STEP 3: Sum all squared differences
    sum_squared = np.sum(squared_diff)

    # STEP 4: Take the square root
    distance = np.sqrt(sum_squared)

    return float(distance)


def euclidean_to_similarity(distance):
    """
    Convert Euclidean distance to a similarity score between 0 and 1.

    WHY WE NEED THIS:
    ──────────────────
    Euclidean distance gives us "dissimilarity" (lower = more similar).
    But for comparison with cosine similarity, we want a number where
    higher = more similar.

    Common conversion formula:
        similarity = 1 / (1 + distance)

    This maps:
        distance = 0    ->  similarity = 1.0  (identical)
        distance = 1    ->  similarity = 0.5
        distance = inf  ->  similarity = 0.0  (completely different)

    Args:
        distance: Euclidean distance (>= 0)

    Returns:
        Similarity score between 0 and 1
    """
    return 1.0 / (1.0 + distance)


# =============================================================================
# METRIC 3: DOT PRODUCT (from scratch)
# =============================================================================

def dot_product_scratch(vec_a, vec_b):
    """
    Calculate dot product (inner product) between two vectors FROM SCRATCH.

    THE MATH:
    ─────────
    The dot product multiplies corresponding elements and sums them.

    Formula:
        A . B = sum(a_i * b_i) = a_1*b_1 + a_2*b_2 + ... + a_n*b_n

    Example:
        A = [1, 2, 3]
        B = [4, 5, 6]
        A . B = (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32

    GEOMETRIC MEANING:
    ──────────────────
    The dot product measures how much one vector "projects onto" another.

        A . B = ||A|| * ||B|| * cos(theta)

    This means it combines:
      - The MAGNITUDE (length) of both vectors
      - The ANGLE between them (via cosine)

    KEY INSIGHT:
    ────────────
    When vectors are NORMALIZED (length = 1):
        dot product = cosine similarity

    Because:
        A . B = ||A|| * ||B|| * cos(theta)
              = 1 * 1 * cos(theta)
              = cos(theta)

    This is why many vector databases normalize embeddings!
    It lets them use the faster dot product operation to get
    cosine similarity results.

    ┌────────────────────────────────────────────────────────────────┐
    │  PRACTICAL TIP:                                                │
    │  If you normalize your embeddings, dot product = cosine.      │
    │  Dot product is computationally cheaper (no norm calculation), │
    │  so normalized vectors + dot product is the fastest approach!  │
    └────────────────────────────────────────────────────────────────┘

    WHEN DOT PRODUCT DIFFERS FROM COSINE:
    ──────────────────────────────────────
    With UN-normalized vectors, dot product favors LONGER vectors.

        A = [1, 0]   (short vector pointing right)
        B = [10, 1]  (long vector pointing mostly right)
        C = [1, 0.1] (short vector pointing right)

        dot(A, B) = 10  (high! because B is long)
        dot(A, C) = 1   (low, even though C points in nearly same direction)

        cosine(A, B) = 0.995  (high, similar direction)
        cosine(A, C) = 0.995  (high, similar direction, SAME as above!)

    Dot product gives B a big advantage just because it's "louder".
    This can be useful when longer documents should be prioritized.

    Args:
        vec_a: First vector (list or numpy array)
        vec_b: Second vector (list or numpy array)

    Returns:
        Float (larger = more similar for same-direction vectors)
    """
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    # Element-wise multiply and sum -- that's it!
    # The dot product is the simplest of the three metrics.
    product = np.dot(a, b)

    return float(product)


# =============================================================================
# HELPER: TERMINAL BAR CHART
# =============================================================================

def print_comparison_bars(label_a, value_a, label_b, value_b, label_c, value_c,
                          width=35, header=""):
    """
    Print a grouped bar chart comparing three metrics in the terminal.

    Creates output like:

        Pair 1: pod crash vs container restart
        Cosine:    [###########################....] 0.8742
        Euclidean: [#######################........] 0.7531
        Dot Prod:  [########################.......] 0.7912

    Args:
        label_a/b/c: Labels for each metric
        value_a/b/c: Values (all normalized to 0-1 range)
        width: Bar width in characters
        header: Optional header text
    """
    if header:
        print(f"\n    {header}")

    for label, value in [(label_a, value_a), (label_b, value_b), (label_c, value_c)]:
        # Clamp value to [0, 1] for display
        clamped = max(0.0, min(1.0, value))
        filled = int(clamped * width)
        empty = width - filled

        bar = "#" * filled + "." * empty
        print(f"    {label:<12} [{bar}] {value:.4f}")


# =============================================================================
# DEMO 1: BASIC METRIC CALCULATIONS
# =============================================================================

def demo_basic_calculations():
    """
    Demonstrate each metric on simple 2D/3D vectors before moving
    to real embeddings. Small vectors are easier to reason about!
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: Basic Metric Calculations with Simple Vectors")
    print("=" * 70)

    print("""
    Let's start with simple 3D vectors to build intuition.
    These are small enough that you can verify the math by hand!
    """)

    # ─── Test Case 1: Identical Vectors ────────────────────────────────
    print("  TEST 1: Identical Vectors")
    print("  " + "-" * 60)
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    print(f"    A = {a}")
    print(f"    B = {b}")
    print(f"    Cosine Similarity:  {cosine_similarity_scratch(a, b):.4f}  (expected: 1.0)")
    print(f"    Euclidean Distance: {euclidean_distance_scratch(a, b):.4f}  (expected: 0.0)")
    print(f"    Dot Product:        {dot_product_scratch(a, b):.4f}  (expected: 14.0)")
    print(f"    (1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14)")

    # ─── Test Case 2: Same Direction, Different Length ─────────────────
    print(f"\n  TEST 2: Same Direction, Different Magnitude")
    print("  " + "-" * 60)
    a = [1.0, 0.0, 0.0]
    b = [5.0, 0.0, 0.0]
    print(f"    A = {a}  (short vector)")
    print(f"    B = {b}  (5x longer, SAME direction)")
    cos = cosine_similarity_scratch(a, b)
    euc = euclidean_distance_scratch(a, b)
    dot = dot_product_scratch(a, b)
    print(f"    Cosine Similarity:  {cos:.4f}  -> Cosine says: IDENTICAL direction!")
    print(f"    Euclidean Distance: {euc:.4f}  -> Euclidean says: far apart!")
    print(f"    Dot Product:        {dot:.4f}  -> Dot product: boosted by B's length!")
    print("""
    KEY INSIGHT: Cosine IGNORES magnitude. It only cares about direction.
    Euclidean and dot product are both affected by magnitude.""")

    # ─── Test Case 3: Perpendicular (Orthogonal) Vectors ──────────────
    print(f"\n  TEST 3: Perpendicular (Orthogonal) Vectors")
    print("  " + "-" * 60)
    a = [1.0, 0.0, 0.0]  # Points along X axis
    b = [0.0, 1.0, 0.0]  # Points along Y axis
    print(f"    A = {a}  (points along X axis)")
    print(f"    B = {b}  (points along Y axis)")
    cos = cosine_similarity_scratch(a, b)
    euc = euclidean_distance_scratch(a, b)
    dot = dot_product_scratch(a, b)
    print(f"    Cosine Similarity:  {cos:.4f}  -> Zero! Completely unrelated directions.")
    print(f"    Euclidean Distance: {euc:.4f}  -> sqrt(2) apart.")
    print(f"    Dot Product:        {dot:.4f}  -> Zero! No projection of one onto the other.")

    # ─── Test Case 4: Opposite Vectors ─────────────────────────────────
    print(f"\n  TEST 4: Opposite Vectors")
    print("  " + "-" * 60)
    a = [1.0, 2.0, 3.0]
    b = [-1.0, -2.0, -3.0]
    print(f"    A = {a}")
    print(f"    B = {b}  (exact opposite)")
    cos = cosine_similarity_scratch(a, b)
    euc = euclidean_distance_scratch(a, b)
    dot = dot_product_scratch(a, b)
    print(f"    Cosine Similarity:  {cos:.4f}  -> -1! Completely opposite meaning.")
    print(f"    Euclidean Distance: {euc:.4f}  -> Far apart (twice the magnitude).")
    print(f"    Dot Product:        {dot:.4f} -> Negative! Anti-correlated.")


# =============================================================================
# DEMO 2: COMPARING METRICS ON REAL EMBEDDINGS
# =============================================================================

def demo_real_embeddings():
    """
    Compare all three metrics on real DevOps text embeddings.

    This is where it gets practical -- we'll see how the metrics
    behave on actual text that your chatbot would process.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Comparing Metrics on Real DevOps Embeddings")
    print("=" * 70)

    # Load the embedding model
    print("\n  Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded!")

    # Define text pairs to compare
    text_pairs = [
        {
            "text_a": "Kubernetes pod CrashLoopBackOff error",
            "text_b": "Container keeps crashing and restarting in K8s",
            "label": "SIMILAR: Same K8s crash issue"
        },
        {
            "text_a": "Kubernetes pod CrashLoopBackOff error",
            "text_b": "Docker image build optimization with multi-stage",
            "label": "RELATED: Both container tech, different topics"
        },
        {
            "text_a": "Kubernetes pod CrashLoopBackOff error",
            "text_b": "Terraform state lock error resolution",
            "label": "DIFFERENT: K8s error vs Terraform error"
        },
        {
            "text_a": "Kubernetes pod CrashLoopBackOff error",
            "text_b": "How to bake chocolate chip cookies at home",
            "label": "UNRELATED: DevOps vs cooking"
        },
    ]

    # Reference query for display
    print(f"\n  Reference: \"{text_pairs[0]['text_a']}\"")
    print(f"  Comparing against {len(text_pairs)} texts using all 3 metrics:\n")

    for pair in text_pairs:
        # Generate embeddings
        vec_a = model.encode(pair["text_a"])
        vec_b = model.encode(pair["text_b"])

        # Calculate all three metrics
        cos = cosine_similarity_scratch(vec_a, vec_b)
        euc = euclidean_distance_scratch(vec_a, vec_b)
        euc_sim = euclidean_to_similarity(euc)  # Convert to 0-1 for comparison
        dot = dot_product_scratch(vec_a, vec_b)

        # For bar display, normalize dot product to 0-1 range
        # Dot product with normalized embeddings is same as cosine
        # but MiniLM doesn't normalize by default, so we scale
        dot_normalized = (dot + 1) / 2  # Rough normalization for display

        print(f"  {pair['label']}")
        print(f"    vs \"{pair['text_b'][:55]}...\"" if len(pair['text_b']) > 55
              else f"    vs \"{pair['text_b']}\"")
        print_comparison_bars(
            "Cosine", cos,
            "Eucl(sim)", euc_sim,
            "Dot(norm)", dot_normalized,
        )
        print(f"    Raw values -> Cosine: {cos:.4f} | Euclidean dist: {euc:.4f} | Dot product: {dot:.4f}")
        print()


# =============================================================================
# DEMO 3: WHEN METRICS DISAGREE
# =============================================================================

def demo_metrics_disagree():
    """
    Demonstrate scenarios where different metrics rank documents differently.

    This is the critical lesson: your choice of metric CAN change which
    documents are returned as "most similar"!
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: When Metrics Give Different Rankings")
    print("=" * 70)

    print("""
    The most important question: does the choice of metric change
    which documents get returned? YES, it can!

    We'll construct a scenario where this happens.
    """)

    # ─── Constructed Example with Simple Vectors ───────────────────────
    print("  SCENARIO: Constructed 2D example")
    print("  " + "-" * 60)

    query = np.array([1.0, 0.0])

    # Doc A: same direction as query, but short
    doc_a = np.array([0.3, 0.0])

    # Doc B: slightly different direction, but longer
    doc_b = np.array([0.8, 0.5])

    # Doc C: moderately different direction, very long
    doc_c = np.array([3.0, 2.5])

    docs = {"Doc A (short, same dir)": doc_a,
            "Doc B (medium, similar dir)": doc_b,
            "Doc C (long, diff dir)": doc_c}

    print(f"""
    Query vector:   {query.tolist()}  (points directly along X axis)
    Doc A vector:   {doc_a.tolist()}  (same direction, short)
    Doc B vector:   {doc_b.tolist()}  (slightly off, medium length)
    Doc C vector:   {doc_c.tolist()}  (more off, very long)

    2D VISUALIZATION:
    ─────────────────
         Y
         ^
     3.0 |               * Doc C (3.0, 2.5)
         |              .
     2.5 |            .
         |          .
     2.0 |        .
         |      .
     1.5 |    .
         |  .
     1.0 | .
         |.
     0.5 |  * Doc B (0.8, 0.5)
         |
     0.0 *──*──────────────────────────────────> X
       Query Doc A
      (1,0) (0.3,0)
    """)

    # Calculate all metrics for each document
    results = {}
    for name, doc_vec in docs.items():
        cos = cosine_similarity_scratch(query, doc_vec)
        euc = euclidean_distance_scratch(query, doc_vec)
        dot = dot_product_scratch(query, doc_vec)
        results[name] = {"cosine": cos, "euclidean": euc, "dot": dot}

    # Display results
    print("  METRIC SCORES:")
    print("  " + "-" * 60)
    print(f"    {'Document':<30} {'Cosine':>8} {'Eucl Dist':>10} {'Dot Prod':>10}")
    print(f"    {'':─<30} {'':─>8} {'':─>10} {'':─>10}")

    for name, scores in results.items():
        print(f"    {name:<30} {scores['cosine']:>8.4f} {scores['euclidean']:>10.4f} {scores['dot']:>10.4f}")

    # Determine winners for each metric
    cos_winner = max(results.items(), key=lambda x: x[1]["cosine"])[0]
    euc_winner = min(results.items(), key=lambda x: x[1]["euclidean"])[0]
    dot_winner = max(results.items(), key=lambda x: x[1]["dot"])[0]

    print(f"""
    RANKINGS (which doc is "most similar" to the query?):
    ─────────────────────────────────────────────────────
    Cosine similarity winner:    {cos_winner}
    Euclidean distance winner:   {euc_winner}
    Dot product winner:          {dot_winner}
    """)

    if cos_winner != euc_winner or euc_winner != dot_winner:
        print("""    The metrics DISAGREE on which document is most similar!

    WHY?
    ────
    - COSINE picks Doc A because it points in the EXACT same direction
      as the query, regardless of length.
    - EUCLIDEAN picks the doc whose vector endpoint is closest to the
      query's endpoint in space -- magnitude matters.
    - DOT PRODUCT picks the doc with the largest "projection" onto the
      query -- it rewards BOTH similar direction AND large magnitude.
    """)
    else:
        print("    All metrics agree on this example (which can happen too!)")

    # ─── Real Embedding Example ────────────────────────────────────────
    print("\n  SCENARIO: Real DevOps embeddings")
    print("  " + "-" * 60)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_text = "fix kubernetes pod crash"
    doc_texts = [
        "CrashLoopBackOff troubleshooting steps for Kubernetes pods",
        "Complete guide to Kubernetes including deployment, services, and pod management",
        "Quick fix: restart the crashing container with kubectl delete pod",
    ]

    query_vec = model.encode(query_text)
    doc_vecs = [model.encode(t) for t in doc_texts]

    print(f"    Query: \"{query_text}\"")
    print()

    # Calculate and display rankings for each metric
    for metric_name, metric_fn, reverse in [
        ("COSINE SIMILARITY", cosine_similarity_scratch, True),
        ("EUCLIDEAN DISTANCE", euclidean_distance_scratch, False),
        ("DOT PRODUCT", dot_product_scratch, True),
    ]:
        scores = [(text, metric_fn(query_vec, dv)) for text, dv in zip(doc_texts, doc_vecs)]
        scores.sort(key=lambda x: x[1], reverse=reverse)

        better = "higher" if reverse else "lower"
        print(f"    {metric_name} ranking ({better} = better):")
        for rank, (text, score) in enumerate(scores, 1):
            short_text = text[:55] + "..." if len(text) > 55 else text
            print(f"      {rank}. [{score:>8.4f}] {short_text}")
        print()


# =============================================================================
# DEMO 4: NORMALIZED VECTORS -- THE GREAT EQUALIZER
# =============================================================================

def demo_normalized_vectors():
    """
    Show that when vectors are normalized, cosine = dot product.

    This is a crucial insight for vector database configuration!
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: Normalized Vectors -- When Cosine Equals Dot Product")
    print("=" * 70)

    print("""
    WHAT IS NORMALIZATION?
    ──────────────────────
    Normalizing a vector means scaling it so its length (magnitude) = 1.
    You divide each element by the vector's total length.

        Original:   A = [3, 4]       ||A|| = sqrt(9+16) = 5
        Normalized: A = [3/5, 4/5] = [0.6, 0.8]    ||A|| = 1.0

    WHY NORMALIZE?
    ──────────────
    1. Cosine similarity becomes equivalent to dot product
    2. Dot product is FASTER to compute (no norm calculation needed)
    3. All vectors live on the "unit sphere" -- easier to reason about
    4. Euclidean distance becomes related to cosine: d^2 = 2(1 - cos)

    ┌────────────────────────────────────────────────────────────────────┐
    │  When normalized:                                                  │
    │    dot_product(A, B) = cosine_similarity(A, B)                    │
    │    euclidean_distance(A, B)^2 = 2 * (1 - cosine_similarity(A, B))│
    │                                                                    │
    │  All three metrics become EQUIVALENT rankings!                    │
    │  (They might give different numbers, but the same ORDER.)         │
    └────────────────────────────────────────────────────────────────────┘
    """)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [
        "Kubernetes pod CrashLoopBackOff",
        "Container keeps crashing in K8s",
        "Terraform state lock error",
    ]

    # Generate embeddings (MiniLM does NOT normalize by default)
    raw_vecs = [model.encode(t) for t in texts]

    # Manually normalize: divide each vector by its magnitude
    normalized_vecs = []
    for vec in raw_vecs:
        magnitude = np.linalg.norm(vec)
        normalized_vecs.append(vec / magnitude)

    # Verify normalization
    print("  Verifying normalization:")
    for i, (raw, norm) in enumerate(zip(raw_vecs, normalized_vecs)):
        print(f"    Text {i+1}: raw magnitude = {np.linalg.norm(raw):.4f} "
              f"-> normalized magnitude = {np.linalg.norm(norm):.4f}")

    # Compare metrics on RAW (un-normalized) vectors
    print(f"\n  UN-NORMALIZED vectors (raw from model):")
    print(f"  " + "-" * 60)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            cos = cosine_similarity_scratch(raw_vecs[i], raw_vecs[j])
            dot = dot_product_scratch(raw_vecs[i], raw_vecs[j])
            print(f"    [{i+1}] vs [{j+1}]:  Cosine = {cos:.6f}  |  Dot = {dot:.6f}  |  Difference = {abs(cos-dot):.6f}")

    # Compare metrics on NORMALIZED vectors
    print(f"\n  NORMALIZED vectors (length = 1):")
    print(f"  " + "-" * 60)
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            cos = cosine_similarity_scratch(normalized_vecs[i], normalized_vecs[j])
            dot = dot_product_scratch(normalized_vecs[i], normalized_vecs[j])
            print(f"    [{i+1}] vs [{j+1}]:  Cosine = {cos:.6f}  |  Dot = {dot:.6f}  |  Difference = {abs(cos-dot):.6f}")

    print("""
    NOTICE: With normalized vectors, cosine and dot product give
    IDENTICAL results! The difference is essentially zero (just
    floating-point rounding).

    PRACTICAL TAKEAWAY:
    ───────────────────
    When configuring your vector database:

      # If you normalize embeddings, use "ip" (inner product / dot product)
      # for fastest search -- it's equivalent to cosine on normalized vectors.
      embeddings = HuggingFaceEmbeddings(
          model_name="all-MiniLM-L6-v2",
          encode_kwargs={"normalize_embeddings": True}  # <-- Key setting!
      )
    """)


# =============================================================================
# DEMO 5: FULL TERMINAL VISUALIZATION
# =============================================================================

def demo_visual_comparison():
    """
    Create a comprehensive visual comparison of all three metrics
    across multiple DevOps text pairs.
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: Full Visual Comparison Dashboard")
    print("=" * 70)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query = "How to debug a crashing Kubernetes pod"
    documents = [
        "CrashLoopBackOff error: pod keeps restarting due to application failure",
        "kubectl logs command to view container output and debug issues",
        "Kubernetes pod resource limits and OOMKilled exit code 137",
        "Docker build multi-stage optimization for smaller images",
        "Terraform state management and remote backend configuration",
        "How to make sourdough bread from scratch at home",
    ]

    query_vec = model.encode(query)
    doc_vecs = [model.encode(d) for d in documents]

    print(f"\n  Query: \"{query}\"")
    print(f"\n  Comparing against {len(documents)} documents:\n")

    # Header
    print(f"    {'#':<4} {'Cosine':^10} {'Eucl(sim)':^10} {'Dot Prod':^10} {'Document'}")
    print(f"    {'─'*4} {'─'*10} {'─'*10} {'─'*10} {'─'*45}")

    # Calculate all metrics for sorting comparison later
    all_scores = []

    for i, (doc, doc_vec) in enumerate(zip(documents, doc_vecs), 1):
        cos = cosine_similarity_scratch(query_vec, doc_vec)
        euc = euclidean_distance_scratch(query_vec, doc_vec)
        euc_sim = euclidean_to_similarity(euc)
        dot = dot_product_scratch(query_vec, doc_vec)

        # Truncate document for display
        short_doc = doc[:42] + "..." if len(doc) > 42 else doc

        print(f"    {i:<4} {cos:^10.4f} {euc_sim:^10.4f} {dot:^10.4f} {short_doc}")

        all_scores.append({
            "doc": doc, "idx": i,
            "cosine": cos, "euclidean": euc, "euc_sim": euc_sim, "dot": dot
        })

    # Show rankings comparison
    print(f"\n  RANKING COMPARISON (top 3 per metric):")
    print(f"  " + "-" * 60)

    cos_ranked = sorted(all_scores, key=lambda x: x["cosine"], reverse=True)
    euc_ranked = sorted(all_scores, key=lambda x: x["euclidean"])  # lower = better
    dot_ranked = sorted(all_scores, key=lambda x: x["dot"], reverse=True)

    print(f"\n    {'Rank':<6} {'Cosine':<20} {'Euclidean':<20} {'Dot Product':<20}")
    print(f"    {'─'*6} {'─'*20} {'─'*20} {'─'*20}")

    for rank in range(3):
        cos_doc = f"Doc {cos_ranked[rank]['idx']}"
        euc_doc = f"Doc {euc_ranked[rank]['idx']}"
        dot_doc = f"Doc {dot_ranked[rank]['idx']}"
        print(f"    #{rank+1:<5} {cos_doc:<20} {euc_doc:<20} {dot_doc:<20}")

    # Check if rankings agree
    cos_top3 = set(s["idx"] for s in cos_ranked[:3])
    euc_top3 = set(s["idx"] for s in euc_ranked[:3])
    dot_top3 = set(s["idx"] for s in dot_ranked[:3])

    if cos_top3 == euc_top3 == dot_top3:
        print("\n    All three metrics agree on the top 3 documents!")
    else:
        print("\n    The metrics DISAGREE on some rankings!")
        diff_docs = (cos_top3 | euc_top3 | dot_top3) - (cos_top3 & euc_top3 & dot_top3)
        if diff_docs:
            print(f"    Documents with different rankings: {diff_docs}")

    print("""
    VISUAL SUMMARY OF METRIC BEHAVIOR:
    ───────────────────────────────────

      High Similarity                           Low Similarity
      ◄──────────────────────────────────────────────────────►

      Cosine:     [1.0 ════════════════════════════ 0.0] -1.0
                   same        related       unrelated  opposite
                   topic       topics        topics     meaning

      Euclidean:  [0.0 ════════════════════════════ inf]
                   identical   close         far apart
                   (convert: sim = 1/(1+dist) for 0-1 range)

      Dot Product: [large + ═══════════════════ 0 ═══ large -]
                    aligned   somewhat   perpendicular  opposite
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 5, EXERCISE 2: Similarity Metrics Hands-On")
    print("  Cosine Similarity | Euclidean Distance | Dot Product")
    print("=" * 70)

    # Demo 1: Basic math with simple vectors
    demo_basic_calculations()

    # Demo 2: Real embeddings comparison
    demo_real_embeddings()

    # Demo 3: When metrics disagree
    demo_metrics_disagree()

    # Demo 4: Normalized vectors
    demo_normalized_vectors()

    # Demo 5: Full visual comparison
    demo_visual_comparison()

    # ─── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  1. COSINE SIMILARITY is the DEFAULT for most vector search            │
    │     - Measures angle between vectors (ignores magnitude)               │
    │     - Range: [-1, 1] where 1 = identical                              │
    │     - Best when document lengths vary                                  │
    │     - ChromaDB setting: distance_function="cosine"                    │
    │                                                                        │
    │  2. EUCLIDEAN DISTANCE is intuitive but INVERTED                       │
    │     - Lower = more similar (opposite of cosine!)                      │
    │     - Sensitive to vector magnitude                                    │
    │     - Convert to similarity: sim = 1 / (1 + distance)                 │
    │     - ChromaDB setting: distance_function="l2"                        │
    │                                                                        │
    │  3. DOT PRODUCT equals cosine when vectors are NORMALIZED              │
    │     - Fastest to compute (no norm calculation)                         │
    │     - Use with normalize_embeddings=True for best performance         │
    │     - ChromaDB setting: distance_function="ip"                        │
    │                                                                        │
    │  4. FOR YOUR DEVOPS CHATBOT, USE COSINE (or normalized + dot product) │
    │     - Cosine is the safest default                                     │
    │     - It handles varying document sizes gracefully                     │
    │     - For speed optimization: normalize embeddings + use dot product  │
    │                                                                        │
    │  5. THE METRIC CHOICE CAN CHANGE YOUR SEARCH RESULTS                  │
    │     - Different metrics can rank documents differently                 │
    │     - Always test with your actual data to verify                      │
    │     - When in doubt, stick with cosine                                │
    │                                                                        │
    │  NEXT: Exercise 3 shows how to use ChromaDB's native API directly,    │
    │        including configuring distance functions per collection.        │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """)

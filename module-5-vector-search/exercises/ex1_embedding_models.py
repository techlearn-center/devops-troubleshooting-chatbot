#!/usr/bin/env python3
"""
Exercise 1: Embedding Model Comparison
=======================================

GOAL: Understand how different embedding models affect your vector search
      quality, speed, and memory usage -- and learn to pick the right one
      for your DevOps chatbot.

WHAT YOU'LL LEARN:
- What embedding models are and why the choice matters
- How to compare all-MiniLM-L6-v2 vs all-mpnet-base-v2
- Dimension differences (384 vs 768) and what they mean in practice
- How to benchmark speed vs quality trade-offs
- Practical guidance for choosing a model in production

WHY DOES THE EMBEDDING MODEL MATTER?
-------------------------------------
The embedding model is the "translator" that converts your text into numbers.
A better translator captures more subtle meaning. A faster translator lets
you process more documents per second.

    Think of it like cameras:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                                                                      │
    │  Smartphone Camera (MiniLM-L6-v2)      DSLR Camera (mpnet-base-v2) │
    │  ─────────────────────────────────      ──────────────────────────── │
    │  - 384 "pixels" (dimensions)           - 768 "pixels" (dimensions) │
    │  - Fast to take a photo                - Slower to take a photo    │
    │  - Good enough for most tasks          - More detail & nuance      │
    │  - Uses less memory/storage            - Uses more memory/storage  │
    │  - Great for prototypes                - Great for production      │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘

    Both take "photos" of your text (embeddings), but with different
    levels of detail!

MODEL COMPARISON AT A GLANCE:
─────────────────────────────
    ┌─────────────────────┬────────────────────┬──────────────────────────┐
    │ Property            │ all-MiniLM-L6-v2   │ all-mpnet-base-v2        │
    ├─────────────────────┼────────────────────┼──────────────────────────┤
    │ Dimensions          │ 384                │ 768                      │
    │ Parameters          │ ~22M               │ ~109M                    │
    │ Max Sequence Length  │ 256 tokens         │ 384 tokens               │
    │ Speed (relative)    │ ~5x faster         │ 1x (baseline)            │
    │ Quality (STS bench) │ Good (78.9)        │ Best (83.4)              │
    │ Memory Usage        │ ~90 MB             │ ~420 MB                  │
    │ Best For            │ Prototyping, real-  │ Production when quality  │
    │                     │ time apps, large    │ matters most, smaller    │
    │                     │ doc collections     │ doc collections          │
    └─────────────────────┴────────────────────┴──────────────────────────┘

ARCHITECTURE: How Embedding Models Work Under the Hood
──────────────────────────────────────────────────────

    Input Text: "Kubernetes pod CrashLoopBackOff"
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │  TOKENIZER                                                         │
    │  Splits text into sub-word tokens:                                 │
    │  ["kubernetes", "pod", "crash", "##loop", "##back", "##off"]      │
    └────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │  TRANSFORMER LAYERS                                                │
    │                                                                    │
    │  MiniLM-L6-v2:  6 layers  ──► Less processing, faster             │
    │  mpnet-base-v2: 12 layers ──► More processing, better quality     │
    │                                                                    │
    │  Each layer refines the understanding of meaning, context,         │
    │  and relationships between tokens.                                 │
    └────────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │  POOLING LAYER                                                     │
    │  Combines all token vectors into ONE sentence vector:              │
    │                                                                    │
    │  MiniLM:  [0.23, -0.45, 0.12, ..., 0.67]   (384 numbers)         │
    │  mpnet:   [0.18, -0.39, 0.08, ..., 0.54]   (768 numbers)         │
    └────────────────────────────────────────────────────────────────────┘

REQUIREMENTS:
    pip install sentence-transformers langchain langchain-community numpy

RUN THIS:
    python module-5-vector-search/exercises/ex1_embedding_models.py
"""

import os
import sys
import time
from pathlib import Path

# =============================================================================
# PROJECT SETUP
# =============================================================================
# Add the project root to Python's module search path so we can import
# shared utilities from the project. Path(__file__) is THIS file's location,
# and .parent.parent.parent goes up three directories to the project root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file (API keys, config, etc.)
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# IMPORTS
# =============================================================================
# NumPy: The foundational library for numerical computing in Python.
# We use it for fast vector math (dot products, norms, etc.)
import numpy as np

# SentenceTransformer: The underlying library that HuggingFaceEmbeddings wraps.
# We import it directly here so we can compare models at a lower level and
# access properties like dimension size that LangChain abstracts away.
from sentence_transformers import SentenceTransformer

# HuggingFaceEmbeddings: LangChain's wrapper around SentenceTransformers.
# In your RAG pipeline you'll use this wrapper, but here we also use the
# raw SentenceTransformer to show what's happening under the hood.
from langchain_community.embeddings import HuggingFaceEmbeddings


# =============================================================================
# DEVOPS TEXT PAIRS FOR TESTING
# =============================================================================
# These pairs are carefully chosen to test different aspects of semantic
# understanding. Each pair has a "query" (what a user might type) and a
# "document" (what might be stored in your knowledge base).
#
# We also assign an "expected_similarity" label so we can verify our
# models are capturing meaning correctly.

DEVOPS_TEST_PAIRS = [
    # ─── HIGH SIMILARITY PAIRS ─────────────────────────────────────────
    # These pairs mean the same thing but use different words.
    # A good embedding model should give these HIGH similarity scores.
    {
        "query": "Kubernetes pod keeps crashing and restarting",
        "document": "CrashLoopBackOff error in K8s container",
        "expected": "high",
        "explanation": "Same concept (pod crash), different terminology"
    },
    {
        "query": "How to check container resource usage",
        "document": "Monitor CPU and memory consumption in Docker",
        "expected": "high",
        "explanation": "Same intent (resource monitoring), different phrasing"
    },
    {
        "query": "Terraform state file is locked by another process",
        "document": "Error acquiring the state lock in Terraform",
        "expected": "high",
        "explanation": "Same error described from user vs system perspective"
    },

    # ─── MEDIUM SIMILARITY PAIRS ───────────────────────────────────────
    # These pairs are related but not about the exact same thing.
    # A good model should give these MODERATE similarity scores.
    {
        "query": "How to deploy to Kubernetes",
        "document": "Creating Docker images for production",
        "expected": "medium",
        "explanation": "Related (deployment pipeline) but different stages"
    },
    {
        "query": "Ansible playbook for server configuration",
        "document": "Terraform module for infrastructure provisioning",
        "expected": "medium",
        "explanation": "Both are IaC tools but different paradigms"
    },

    # ─── LOW SIMILARITY PAIRS ──────────────────────────────────────────
    # These pairs are about completely different topics.
    # A good model should give these LOW similarity scores.
    {
        "query": "Kubernetes pod CrashLoopBackOff error",
        "document": "How to make chocolate chip cookies at home",
        "expected": "low",
        "explanation": "Completely unrelated topics"
    },
    {
        "query": "Jenkins pipeline syntax for CI/CD",
        "document": "React component lifecycle hooks",
        "expected": "low",
        "explanation": "Both tech but entirely different domains"
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cosine_similarity(vec_a, vec_b):
    """
    Calculate cosine similarity between two vectors.

    MATH EXPLANATION:
    ─────────────────
    Cosine similarity measures the angle between two vectors.

                 A . B           (dot product of A and B)
    cos(theta) = ─────── = ──────────────────────────────────
                ||A|| ||B||    (magnitude of A) * (magnitude of B)

    Result ranges from -1 to 1:
        1.0  = vectors point same direction  (identical meaning)
        0.0  = vectors are perpendicular     (unrelated)
       -1.0  = vectors point opposite        (opposite meaning)

    WHY COSINE SIMILARITY FOR EMBEDDINGS?
    ──────────────────────────────────────
    - It ignores vector magnitude (length), focusing only on direction
    - Two texts about the same topic will "point" in similar directions
    - It works well in high-dimensional spaces (384 or 768 dimensions)

    Args:
        vec_a: First embedding vector (numpy array)
        vec_b: Second embedding vector (numpy array)

    Returns:
        Float between -1.0 and 1.0 (higher = more similar)
    """
    # np.dot computes the dot product: sum of element-wise products
    dot_product = np.dot(vec_a, vec_b)

    # np.linalg.norm computes the Euclidean norm (magnitude/length)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)

    # Avoid division by zero (shouldn't happen with real embeddings)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return float(dot_product / (magnitude_a * magnitude_b))


def print_bar(value, max_width=30, label=""):
    """
    Print a horizontal bar chart in the terminal.

    This creates a simple ASCII visualization like:
        Quality: ████████████████████░░░░░░░░░░ 0.67

    Args:
        value: Float between 0 and 1
        max_width: Maximum bar width in characters
        label: Optional label to show before the bar
    """
    # Calculate how many filled vs empty blocks to show
    filled = int(value * max_width)
    empty = max_width - filled

    # Build the bar string
    bar = "#" * filled + "." * empty

    # Print with label if provided
    if label:
        print(f"   {label}: [{bar}] {value:.4f}")
    else:
        print(f"   [{bar}] {value:.4f}")


# =============================================================================
# DEMO 1: LOAD AND INSPECT MODELS
# =============================================================================

def demo_load_and_inspect():
    """
    Load both embedding models and inspect their properties.

    This demo shows you the key differences between the two models
    before we start comparing their output quality.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: Loading and Inspecting Embedding Models")
    print("=" * 70)

    print("""
    We are loading two popular open-source embedding models from
    HuggingFace. Both run locally on your machine -- no API key needed!

    Loading models for the first time will download them (~50-400 MB).
    Subsequent runs use the cached version.
    """)

    # ─── Load Model 1: MiniLM ──────────────────────────────────────────
    print("  [1/2] Loading all-MiniLM-L6-v2 ...")
    start_time = time.time()

    # SentenceTransformer is the direct library. We use it here to
    # access model properties. In your RAG pipeline, you'd normally
    # use HuggingFaceEmbeddings (which wraps this under the hood).
    model_mini = SentenceTransformer("all-MiniLM-L6-v2")

    mini_load_time = time.time() - start_time
    print(f"         Loaded in {mini_load_time:.2f}s")

    # ─── Load Model 2: MPNet ───────────────────────────────────────────
    print("  [2/2] Loading all-mpnet-base-v2 ...")
    start_time = time.time()

    model_mpnet = SentenceTransformer("all-mpnet-base-v2")

    mpnet_load_time = time.time() - start_time
    print(f"         Loaded in {mpnet_load_time:.2f}s")

    # ─── Display Properties ────────────────────────────────────────────
    print("\n  " + "-" * 66)
    print("  MODEL PROPERTIES COMPARISON:")
    print("  " + "-" * 66)

    # Get the embedding dimension by encoding a test sentence
    test_embedding_mini = model_mini.encode("test")
    test_embedding_mpnet = model_mpnet.encode("test")

    print(f"""
    ┌──────────────────────────┬──────────────────┬──────────────────────┐
    │ Property                 │ MiniLM-L6-v2     │ mpnet-base-v2        │
    ├──────────────────────────┼──────────────────┼──────────────────────┤
    │ Embedding Dimensions     │ {len(test_embedding_mini):<16} │ {len(test_embedding_mpnet):<20} │
    │ Max Sequence Length      │ {model_mini.max_seq_length:<16} │ {model_mpnet.max_seq_length:<20} │
    │ Load Time                │ {mini_load_time:<16.2f} │ {mpnet_load_time:<20.2f} │
    └──────────────────────────┴──────────────────┴──────────────────────┘

    WHAT DO DIMENSIONS MEAN?
    ────────────────────────
    - {len(test_embedding_mini)} dimensions = each text becomes a list of {len(test_embedding_mini)} numbers
    - {len(test_embedding_mpnet)} dimensions = each text becomes a list of {len(test_embedding_mpnet)} numbers

    More dimensions can capture more nuance in meaning, but:
    - They use more memory (768 floats vs 384 floats per document)
    - They take longer to compare (more numbers to multiply)
    - They need more storage in your vector database

    WHAT IS MAX SEQUENCE LENGTH?
    ────────────────────────────
    The maximum number of tokens (roughly words) the model processes.
    Text beyond this limit is SILENTLY TRUNCATED -- the model ignores it!

    If your DevOps docs have long paragraphs, you need good chunking
    (see Module 3, Exercise 3) to stay within this limit.
    """)

    return model_mini, model_mpnet


# =============================================================================
# DEMO 2: EMBEDDING OUTPUT COMPARISON
# =============================================================================

def demo_embedding_output(model_mini, model_mpnet):
    """
    Compare the actual embedding vectors produced by each model.

    This shows you what the raw output looks like and helps build
    intuition about what these "384 numbers" actually are.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Comparing Embedding Outputs")
    print("=" * 70)

    text = "Kubernetes pod CrashLoopBackOff error"
    print(f"\n  Input text: \"{text}\"")

    # ─── Generate Embeddings ───────────────────────────────────────────
    # .encode() converts the text to a numpy array of floats
    vec_mini = model_mini.encode(text)
    vec_mpnet = model_mpnet.encode(text)

    # ─── Display Raw Vectors ───────────────────────────────────────────
    print(f"""
    MiniLM-L6-v2 output:
    ─────────────────────
      Shape:        {vec_mini.shape}
      First 8 vals: {[f'{v:.4f}' for v in vec_mini[:8]]}
      Last 8 vals:  {[f'{v:.4f}' for v in vec_mini[-8:]]}
      Min value:    {vec_mini.min():.4f}
      Max value:    {vec_mini.max():.4f}
      Mean value:   {vec_mini.mean():.4f}

    mpnet-base-v2 output:
    ─────────────────────
      Shape:        {vec_mpnet.shape}
      First 8 vals: {[f'{v:.4f}' for v in vec_mpnet[:8]]}
      Last 8 vals:  {[f'{v:.4f}' for v in vec_mpnet[-8:]]}
      Min value:    {vec_mpnet.min():.4f}
      Max value:    {vec_mpnet.max():.4f}
      Mean value:   {vec_mpnet.mean():.4f}
    """)

    print("""    WHAT TO NOTICE:
    ─────────────────
    1. Both models produce normalized vectors (values roughly between -1 and 1)
    2. The numbers look random, but they encode MEANING
    3. mpnet has TWICE as many numbers (768 vs 384) -- more detail
    4. You CANNOT compare vectors from different models!
       A MiniLM vector and an mpnet vector are in completely different
       "coordinate systems" -- like comparing Fahrenheit to Celsius directly.

    ┌────────────────────────────────────────────────────────────────────┐
    │  IMPORTANT: Never mix embeddings from different models in the     │
    │  same vector database! All documents AND queries must use the     │
    │  SAME model. If you switch models, you must re-embed everything. │
    └────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# DEMO 3: QUALITY COMPARISON ON DEVOPS TEXT PAIRS
# =============================================================================

def demo_quality_comparison(model_mini, model_mpnet):
    """
    Compare how well each model captures semantic similarity
    on our DevOps-specific text pairs.

    This is the most important test: which model better understands
    the meaning of your DevOps documentation?
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Quality Comparison on DevOps Text Pairs")
    print("=" * 70)

    print("""
    We test both models on carefully chosen DevOps text pairs.
    Each pair has an "expected" similarity level (high, medium, low).
    A good model should match our expectations.

    SCORING GUIDE:
    ──────────────
      > 0.70 = HIGH   (texts mean roughly the same thing)
      0.40 - 0.70 = MEDIUM (texts are related but different)
      < 0.40 = LOW    (texts are about different topics)
    """)

    # Track scores for summary statistics
    mini_scores = []
    mpnet_scores = []
    mini_correct = 0
    mpnet_correct = 0

    for i, pair in enumerate(DEVOPS_TEST_PAIRS, 1):
        print(f"\n  --- Pair {i}: {pair['explanation']} ---")
        print(f"  Query:    \"{pair['query']}\"")
        print(f"  Document: \"{pair['document']}\"")
        print(f"  Expected: {pair['expected'].upper()}")

        # Generate embeddings with each model
        q_mini = model_mini.encode(pair["query"])
        d_mini = model_mini.encode(pair["document"])
        q_mpnet = model_mpnet.encode(pair["query"])
        d_mpnet = model_mpnet.encode(pair["document"])

        # Calculate cosine similarity
        sim_mini = cosine_similarity(q_mini, d_mini)
        sim_mpnet = cosine_similarity(q_mpnet, d_mpnet)

        mini_scores.append(sim_mini)
        mpnet_scores.append(sim_mpnet)

        # Display results with visual bars
        print_bar(max(0, sim_mini), label="MiniLM  ")
        print_bar(max(0, sim_mpnet), label="mpnet   ")

        # Check if the model's score matches our expected category
        def categorize(score):
            if score > 0.70:
                return "high"
            elif score > 0.40:
                return "medium"
            else:
                return "low"

        mini_cat = categorize(sim_mini)
        mpnet_cat = categorize(sim_mpnet)

        mini_match = "CORRECT" if mini_cat == pair["expected"] else f"WRONG (got {mini_cat})"
        mpnet_match = "CORRECT" if mpnet_cat == pair["expected"] else f"WRONG (got {mpnet_cat})"

        if mini_cat == pair["expected"]:
            mini_correct += 1
        if mpnet_cat == pair["expected"]:
            mpnet_correct += 1

        print(f"   MiniLM category:  {mini_match}")
        print(f"   mpnet category:   {mpnet_match}")

    # ─── Summary Statistics ────────────────────────────────────────────
    total = len(DEVOPS_TEST_PAIRS)
    print(f"""
    {'=' * 66}
    QUALITY SUMMARY:
    {'=' * 66}

    Category Accuracy (did the model agree with our expected labels?):
      MiniLM-L6-v2:   {mini_correct}/{total} correct ({mini_correct/total*100:.0f}%)
      mpnet-base-v2:  {mpnet_correct}/{total} correct ({mpnet_correct/total*100:.0f}%)

    Average Similarity Scores:
      MiniLM-L6-v2:   {np.mean(mini_scores):.4f}
      mpnet-base-v2:  {np.mean(mpnet_scores):.4f}

    Score Spread (std dev -- higher = more discriminating):
      MiniLM-L6-v2:   {np.std(mini_scores):.4f}
      mpnet-base-v2:  {np.std(mpnet_scores):.4f}

    INTERPRETATION:
    ───────────────
    - Higher accuracy = better at distinguishing similar vs different text
    - Higher std dev = better at separating high/medium/low similarity
    - A model that gives everything ~0.5 is WORSE than one that clearly
      separates relevant (>0.7) from irrelevant (<0.3) documents
    """)

    return mini_scores, mpnet_scores


# =============================================================================
# DEMO 4: SPEED BENCHMARKING
# =============================================================================

def demo_speed_benchmark(model_mini, model_mpnet):
    """
    Benchmark the encoding speed of both models.

    Speed matters when you're:
    - Ingesting thousands of documents into your vector DB
    - Handling real-time user queries (latency sensitive)
    - Running on limited hardware (CI/CD runners, small VMs)
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: Speed Benchmarking")
    print("=" * 70)

    # A collection of DevOps-related texts to benchmark with.
    # We use realistic content to get accurate timing numbers.
    benchmark_texts = [
        "Kubernetes pod CrashLoopBackOff error troubleshooting guide",
        "Docker container fails to start due to missing entrypoint",
        "Terraform state lock error when running plan in CI/CD",
        "Ansible playbook for configuring nginx reverse proxy",
        "Jenkins pipeline fails at build stage with exit code 1",
        "Prometheus alerting rules for high CPU usage on nodes",
        "Grafana dashboard showing 5xx errors on API gateway",
        "Helm chart values override for production deployment",
        "ArgoCD sync failed due to schema validation error",
        "Vault secrets injection into Kubernetes pod environment",
        "GitHub Actions workflow for automated testing and deployment",
        "ELK stack configuration for centralized logging",
        "Istio service mesh traffic routing and canary deploys",
        "AWS IAM role assumption for cross-account access",
        "Terraform module for VPC with public and private subnets",
        "Kubernetes NetworkPolicy to restrict pod-to-pod traffic",
        "Docker multi-stage build optimization for Node.js apps",
        "Nginx ingress controller TLS termination setup",
        "Redis cluster configuration for session management",
        "PostgreSQL replication setup for high availability",
    ]

    num_texts = len(benchmark_texts)
    # We run multiple iterations to get stable timing
    num_iterations = 3

    print(f"\n  Benchmark setup:")
    print(f"    Texts to encode: {num_texts}")
    print(f"    Iterations: {num_iterations}")
    print(f"    Total encodes per model: {num_texts * num_iterations}")
    print(f"\n  Running benchmark (this may take a moment)...\n")

    # ─── Benchmark MiniLM ──────────────────────────────────────────────
    mini_times = []
    for iteration in range(num_iterations):
        start = time.time()
        # encode() can take a list of strings for batch processing
        # This is MUCH faster than encoding one at a time!
        model_mini.encode(benchmark_texts)
        elapsed = time.time() - start
        mini_times.append(elapsed)

    mini_avg = np.mean(mini_times)
    mini_per_text = mini_avg / num_texts

    # ─── Benchmark mpnet ───────────────────────────────────────────────
    mpnet_times = []
    for iteration in range(num_iterations):
        start = time.time()
        model_mpnet.encode(benchmark_texts)
        elapsed = time.time() - start
        mpnet_times.append(elapsed)

    mpnet_avg = np.mean(mpnet_times)
    mpnet_per_text = mpnet_avg / num_texts

    # ─── Display Results ───────────────────────────────────────────────
    speedup = mpnet_avg / mini_avg if mini_avg > 0 else float('inf')

    print(f"  SPEED RESULTS ({num_texts} texts, averaged over {num_iterations} runs):")
    print(f"  {'─' * 60}")
    print(f"""
    ┌──────────────────────────┬──────────────────┬──────────────────────┐
    │ Metric                   │ MiniLM-L6-v2     │ mpnet-base-v2        │
    ├──────────────────────────┼──────────────────┼──────────────────────┤
    │ Batch time ({num_texts} texts)   │ {mini_avg*1000:>10.1f} ms    │ {mpnet_avg*1000:>14.1f} ms    │
    │ Per-text time            │ {mini_per_text*1000:>10.2f} ms    │ {mpnet_per_text*1000:>14.2f} ms    │
    │ Texts per second         │ {num_texts/mini_avg:>10.0f}        │ {num_texts/mpnet_avg:>14.0f}        │
    └──────────────────────────┴──────────────────┴──────────────────────┘

    MiniLM is approximately {speedup:.1f}x FASTER than mpnet.
    """)

    # ─── Visual Comparison ─────────────────────────────────────────────
    # Normalize to the slower model for bar display
    max_time = max(mini_avg, mpnet_avg)
    print("  Speed comparison (shorter bar = faster = better):")
    print_bar(mini_avg / max_time, label="MiniLM  ")
    print_bar(mpnet_avg / max_time, label="mpnet   ")

    print(f"""
    PRACTICAL IMPLICATIONS:
    ───────────────────────
    Scenario: Ingesting 10,000 DevOps documentation pages

      MiniLM-L6-v2:  ~{10000 * mini_per_text:.0f} seconds ({10000 * mini_per_text / 60:.1f} minutes)
      mpnet-base-v2: ~{10000 * mpnet_per_text:.0f} seconds ({10000 * mpnet_per_text / 60:.1f} minutes)

    Scenario: Real-time query (user waiting for response)

      MiniLM-L6-v2:  {mini_per_text * 1000:.1f} ms per query embedding
      mpnet-base-v2: {mpnet_per_text * 1000:.1f} ms per query embedding

      Both are fast enough for real-time use. The bottleneck is usually
      the LLM generation step (hundreds of milliseconds), not the
      embedding step.
    """)


# =============================================================================
# DEMO 5: LANGCHAIN INTEGRATION
# =============================================================================

def demo_langchain_integration():
    """
    Show how to use both models through LangChain's HuggingFaceEmbeddings.

    In your actual RAG pipeline (Module 3 and beyond), you'll use
    HuggingFaceEmbeddings, not SentenceTransformer directly.
    This demo shows that transition.
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: Using Models in LangChain")
    print("=" * 70)

    print("""
    In your RAG pipeline, you use LangChain's HuggingFaceEmbeddings
    wrapper. Here's how to configure each model:
    """)

    # ─── MiniLM through LangChain ─────────────────────────────────────
    print("  Loading MiniLM through LangChain...")
    lc_mini = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        # model_kwargs control the underlying SentenceTransformer
        model_kwargs={
            "device": "cpu"  # Use "cuda" if you have a GPU
        },
        # encode_kwargs control the .encode() call
        encode_kwargs={
            "normalize_embeddings": True  # Normalize for cosine similarity
        }
    )

    # ─── mpnet through LangChain ───────────────────────────────────────
    print("  Loading mpnet through LangChain...")
    lc_mpnet = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    # ─── Test Both ─────────────────────────────────────────────────────
    test_query = "How to fix OOMKilled error in Kubernetes"

    print(f"\n  Test query: \"{test_query}\"")

    # embed_query is LangChain's method for single text embedding
    # (embed_documents is for multiple documents -- batched)
    vec_mini = lc_mini.embed_query(test_query)
    vec_mpnet = lc_mpnet.embed_query(test_query)

    print(f"""
    LangChain HuggingFaceEmbeddings results:
    ─────────────────────────────────────────
      MiniLM dimensions:  {len(vec_mini)}
      mpnet dimensions:   {len(vec_mpnet)}

    CODE SNIPPETS FOR YOUR RAG PIPELINE:
    ─────────────────────────────────────

    # Option A: Fast & lightweight (recommended for prototyping)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={{"normalize_embeddings": True}}
    )

    # Option B: Higher quality (recommended for production)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        encode_kwargs={{"normalize_embeddings": True}}
    )

    # Then use with ChromaDB (same API for either model):
    from langchain_community.vectorstores import Chroma

    vectorstore = Chroma.from_documents(
        documents=your_docs,
        embedding=embeddings,       # <-- swap model here
        collection_name="devops_kb"
    )
    """)


# =============================================================================
# DEMO 6: EDGE CASES AND GOTCHAS
# =============================================================================

def demo_edge_cases(model_mini, model_mpnet):
    """
    Explore edge cases that trip up beginners.

    Understanding these will save you hours of debugging when your
    RAG pipeline returns unexpected results.
    """
    print("\n" + "=" * 70)
    print("  DEMO 6: Edge Cases and Gotchas")
    print("=" * 70)

    # ─── Gotcha 1: Case Sensitivity ───────────────────────────────────
    print("\n  GOTCHA 1: Are embeddings case-sensitive?")
    print("  " + "-" * 60)

    texts_case = [
        "kubernetes crashloopbackoff",
        "Kubernetes CrashLoopBackOff",
        "KUBERNETES CRASHLOOPBACKOFF",
    ]

    vecs = [model_mini.encode(t) for t in texts_case]

    print(f"    \"{texts_case[0]}\"")
    print(f"    vs \"{texts_case[1]}\"")
    sim_1 = cosine_similarity(vecs[0], vecs[1])
    print(f"    Similarity: {sim_1:.4f}")

    print(f"\n    \"{texts_case[1]}\"")
    print(f"    vs \"{texts_case[2]}\"")
    sim_2 = cosine_similarity(vecs[1], vecs[2])
    print(f"    Similarity: {sim_2:.4f}")

    print(f"""
    Verdict: Similarity is {sim_1:.2f} and {sim_2:.2f} -- very close to 1.0!
    These models are mostly case-INSENSITIVE. Small differences exist
    because capitalization carries some semantic signal (e.g., proper
    nouns), but for DevOps terms it hardly matters.
    """)

    # ─── Gotcha 2: Truncation ─────────────────────────────────────────
    print("  GOTCHA 2: What happens with very long text?")
    print("  " + "-" * 60)

    short_text = "Fix Kubernetes pod crash"
    long_text = "Fix Kubernetes pod crash. " * 100  # Way beyond token limit

    vec_short = model_mini.encode(short_text)
    vec_long = model_mini.encode(long_text)

    sim = cosine_similarity(vec_short, vec_long)
    print(f"    Short text: \"{short_text}\" ({len(short_text)} chars)")
    print(f"    Long text:  Same sentence repeated 100x ({len(long_text)} chars)")
    print(f"    Similarity: {sim:.4f}")
    print(f"""
    The similarity is high because the model TRUNCATES long text to its
    max sequence length ({model_mini.max_seq_length} tokens for MiniLM).
    Everything beyond that is silently ignored!

    LESSON: Always chunk your documents (Module 3, Exercise 3) so each
    chunk fits within the model's max sequence length.
    """)

    # ─── Gotcha 3: Empty / Meaningless Input ──────────────────────────
    print("  GOTCHA 3: Empty and meaningless inputs")
    print("  " + "-" * 60)

    meaningful = "Kubernetes pod troubleshooting guide"
    empty_ish = "the a an is"
    single_char = "x"

    vec_m = model_mini.encode(meaningful)
    vec_e = model_mini.encode(empty_ish)
    vec_s = model_mini.encode(single_char)

    sim_me = cosine_similarity(vec_m, vec_e)
    sim_ms = cosine_similarity(vec_m, vec_s)

    print(f"    Meaningful: \"{meaningful}\"")
    print(f"    Stop words: \"{empty_ish}\" -> Similarity: {sim_me:.4f}")
    print(f"    Single char: \"{single_char}\"  -> Similarity: {sim_ms:.4f}")
    print(f"""
    Stop-word-only text and single characters produce LOW similarity
    with meaningful text. The model knows these carry little meaning.

    LESSON: Filter out very short or empty chunks during document
    processing. They waste space in your vector DB and can pollute
    search results.
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 5, EXERCISE 1: Embedding Model Comparison")
    print("  Comparing all-MiniLM-L6-v2 vs all-mpnet-base-v2")
    print("=" * 70)

    # Demo 1: Load and inspect both models
    model_mini, model_mpnet = demo_load_and_inspect()

    # Demo 2: Compare raw embedding output
    demo_embedding_output(model_mini, model_mpnet)

    # Demo 3: Quality comparison on DevOps text pairs
    demo_quality_comparison(model_mini, model_mpnet)

    # Demo 4: Speed benchmarking
    demo_speed_benchmark(model_mini, model_mpnet)

    # Demo 5: LangChain integration (how you'll actually use these)
    demo_langchain_integration()

    # Demo 6: Edge cases and gotchas
    demo_edge_cases(model_mini, model_mpnet)

    # ─── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  1. CHOOSE THE RIGHT MODEL FOR YOUR USE CASE                          │
    │     - Prototyping / large collections: all-MiniLM-L6-v2 (fast)       │
    │     - Production / quality-critical:   all-mpnet-base-v2 (accurate)  │
    │                                                                        │
    │  2. DIMENSIONS MATTER                                                  │
    │     - 384 dims (MiniLM) = less memory, faster search                  │
    │     - 768 dims (mpnet) = more nuance, better quality                  │
    │     - More dimensions = more storage in your vector DB                │
    │                                                                        │
    │  3. NEVER MIX MODELS                                                   │
    │     - All documents and queries MUST use the same model               │
    │     - Switching models = re-embed your entire document collection     │
    │                                                                        │
    │  4. BOTH MODELS ARE CASE-INSENSITIVE (mostly)                         │
    │     - "CrashLoopBackOff" and "crashloopbackoff" produce similar       │
    │       embeddings. No need to normalize case.                          │
    │                                                                        │
    │  5. WATCH OUT FOR TRUNCATION                                           │
    │     - MiniLM truncates at 256 tokens, mpnet at 384 tokens             │
    │     - Always chunk documents to fit within the model's limit          │
    │                                                                        │
    │  6. NORMALIZE EMBEDDINGS FOR COSINE SIMILARITY                        │
    │     - Use encode_kwargs={"normalize_embeddings": True}                │
    │     - Normalized vectors make cosine similarity = dot product         │
    │     - This speeds up search in ChromaDB                               │
    │                                                                        │
    │  NEXT: Exercise 2 dives deep into similarity metrics (cosine,         │
    │        euclidean, dot product) and when each one shines.              │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """)

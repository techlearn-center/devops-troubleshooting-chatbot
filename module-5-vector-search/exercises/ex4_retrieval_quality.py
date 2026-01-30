#!/usr/bin/env python3
"""
Exercise 4: Retrieval Quality Analysis
========================================

GOAL: Measure and improve how well your vector search retrieves the
      RIGHT documents. A chatbot is only as good as the documents it
      finds -- garbage retrieval = garbage answers.

WHAT YOU'LL LEARN:
- How to define test queries with known expected results
- Calculating precision and recall (the two pillars of retrieval quality)
- Comparing different k values (1, 3, 5, 10)
- Standard search vs MMR (Maximum Marginal Relevance)
- Score threshold analysis -- when to cut off results
- Building a repeatable evaluation framework for your RAG system

WHY RETRIEVAL QUALITY MATTERS:
──────────────────────────────
Your RAG pipeline has two stages. If Stage 1 (retrieval) fails, Stage 2
(the LLM) gets bad context and produces wrong answers -- no matter
how smart the LLM is!

    ┌──────────────┐      ┌──────────────────┐      ┌──────────────┐
    │   User       │      │  RETRIEVAL       │      │   LLM        │
    │   Question   │─────►│  (Stage 1)       │─────►│   Answer     │
    └──────────────┘      │  Find relevant   │      │   (Stage 2)  │
                          │  documents       │      └──────────────┘
                          └──────────────────┘
                                 │
                          ┌──────┴──────┐
                          │  If wrong   │
                          │  docs are   │
                          │  retrieved, │
                          │  the LLM    │
                          │  generates  │
                          │  WRONG      │
                          │  answers!   │
                          └─────────────┘

    "A RAG system that retrieves the wrong documents is like
     a student studying the wrong textbook for an exam."

PRECISION vs RECALL EXPLAINED:
──────────────────────────────

    Imagine searching for "fix pod crash" and you have 3 relevant docs
    in your database (out of 20 total). You retrieve k=5 documents.

    Retrieved 5 docs:  [Doc1, Doc7, Doc3, Doc15, Doc9]
    Actually relevant:  Doc1, Doc3, Doc7

    PRECISION = relevant results / total results returned
              = 3 / 5 = 0.60 (60%)
              "Of the docs I found, how many were actually useful?"

    RECALL    = relevant results / total relevant that exist
              = 3 / 3 = 1.00 (100%)
              "Of all the useful docs, how many did I find?"

    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                    │
    │                    ALL DOCUMENTS IN DATABASE                       │
    │   ┌─────────────────────────────────────────────────────────┐     │
    │   │                                                         │     │
    │   │          RELEVANT           RETRIEVED                   │     │
    │   │         ┌───────┐          ┌───────────┐                │     │
    │   │         │       │          │           │                │     │
    │   │         │   ┌───┼──────────┼──┐        │                │     │
    │   │         │   │   │  TRUE    │  │ FALSE  │                │     │
    │   │         │   │   │POSITIVES │  │POSITIVES                │     │
    │   │         │   │   │ (good!)  │  │ (noise)│                │     │
    │   │         │   └───┼──────────┼──┘        │                │     │
    │   │         │ FALSE │          │           │                │     │
    │   │         │ NEG.  │          └───────────┘                │     │
    │   │         │(missed)                                       │     │
    │   │         └───────┘                                       │     │
    │   └─────────────────────────────────────────────────────────┘     │
    │                                                                    │
    │   Precision = True Positives / (True Positives + False Positives) │
    │   Recall    = True Positives / (True Positives + False Negatives) │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

    THE TRADE-OFF:
    ──────────────
    - Higher k = Better recall (find more relevant docs)
                 but worse precision (more noise mixed in)
    - Lower k = Better precision (mostly relevant results)
                but worse recall (might miss some relevant docs)

    k=1:  Precision: HIGH   Recall: LOW    "I found THE best doc"
    k=10: Precision: LOW    Recall: HIGH   "I found everything, plus junk"
    k=3-5: Good balance for most RAG systems

REQUIREMENTS:
    pip install chromadb sentence-transformers langchain langchain-community numpy

RUN THIS:
    python module-5-vector-search/exercises/ex4_retrieval_quality.py
"""

import os
import sys
from pathlib import Path

# =============================================================================
# PROJECT SETUP
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document


# =============================================================================
# KNOWLEDGE BASE: DEVOPS DOCUMENTS
# =============================================================================
# A curated set of DevOps documents. Each has a unique ID that we'll use
# in our test cases to define "expected" results.

DOCUMENTS = [
    # ─── Kubernetes: Pod Errors ────────────────────────────────────────
    {
        "id": "k8s_crash_001",
        "text": "CrashLoopBackOff: The container in your pod crashes immediately "
                "after starting, causing Kubernetes to restart it repeatedly. "
                "Debug with kubectl logs <pod> --previous to see crash output. "
                "Common causes: application errors, missing config, bad entrypoint.",
        "metadata": {"category": "kubernetes", "topic": "pod-errors", "severity": "critical"}
    },
    {
        "id": "k8s_crash_002",
        "text": "Pod stuck in CrashLoopBackOff with exit code 1 indicates an "
                "application-level error. Check your application logs, verify "
                "environment variables are set, and ensure all dependencies are "
                "available. Try running the container locally with docker run.",
        "metadata": {"category": "kubernetes", "topic": "pod-errors", "severity": "critical"}
    },
    {
        "id": "k8s_oom_001",
        "text": "OOMKilled (Out of Memory Killed, Exit Code 137): The container "
                "exceeded its memory limit. Kubernetes kills the container to protect "
                "the node. Fix: increase resources.limits.memory or optimize your "
                "application's memory usage. Profile with kubectl top pods.",
        "metadata": {"category": "kubernetes", "topic": "pod-errors", "severity": "critical"}
    },
    {
        "id": "k8s_image_001",
        "text": "ImagePullBackOff: Kubernetes cannot download the container image. "
                "Verify image name and tag exist. For private registries, create "
                "an imagePullSecret. Check network connectivity to registry.",
        "metadata": {"category": "kubernetes", "topic": "pod-errors", "severity": "high"}
    },

    # ─── Kubernetes: Networking ────────────────────────────────────────
    {
        "id": "k8s_net_001",
        "text": "Kubernetes Service types: ClusterIP (internal only), NodePort "
                "(external via node port), LoadBalancer (cloud load balancer), "
                "and ExternalName (DNS alias). Use ClusterIP for internal "
                "microservice communication.",
        "metadata": {"category": "kubernetes", "topic": "networking", "severity": "medium"}
    },
    {
        "id": "k8s_net_002",
        "text": "Kubernetes Ingress manages external HTTP/HTTPS access to services. "
                "Requires an Ingress Controller (nginx, traefik). Configure TLS "
                "termination, path-based routing, and virtual hosts.",
        "metadata": {"category": "kubernetes", "topic": "networking", "severity": "medium"}
    },

    # ─── Kubernetes: Scaling ───────────────────────────────────────────
    {
        "id": "k8s_scale_001",
        "text": "Horizontal Pod Autoscaler (HPA) automatically adjusts pod replicas "
                "based on CPU or memory utilization. Set up with kubectl autoscale "
                "deployment <name> --min=2 --max=10 --cpu-percent=80.",
        "metadata": {"category": "kubernetes", "topic": "scaling", "severity": "medium"}
    },

    # ─── Terraform ─────────────────────────────────────────────────────
    {
        "id": "tf_state_001",
        "text": "Terraform state lock error: another process holds the state lock. "
                "Wait for the other process or force-unlock with terraform "
                "force-unlock <LOCK_ID>. Only force-unlock when certain no other "
                "process is running Terraform.",
        "metadata": {"category": "terraform", "topic": "state", "severity": "high"}
    },
    {
        "id": "tf_state_002",
        "text": "Terraform remote state with S3 backend: configure backend block "
                "with bucket, key, region, and DynamoDB table for locking. "
                "Run terraform init to initialize the remote backend.",
        "metadata": {"category": "terraform", "topic": "state", "severity": "medium"}
    },
    {
        "id": "tf_import_001",
        "text": "Terraform import brings existing infrastructure under management. "
                "Usage: terraform import <resource>.<name> <id>. After importing, "
                "write the resource block and run terraform plan to verify.",
        "metadata": {"category": "terraform", "topic": "import", "severity": "medium"}
    },

    # ─── Docker ────────────────────────────────────────────────────────
    {
        "id": "docker_build_001",
        "text": "Docker multi-stage builds dramatically reduce image size. Use "
                "multiple FROM statements to separate build and runtime. Copy only "
                "necessary artifacts to the final stage.",
        "metadata": {"category": "docker", "topic": "optimization", "severity": "low"}
    },
    {
        "id": "docker_debug_001",
        "text": "Debug a failing Docker container: check logs with docker logs "
                "<container>, exec into running container with docker exec -it "
                "<container> /bin/sh, inspect with docker inspect <container>.",
        "metadata": {"category": "docker", "topic": "debugging", "severity": "high"}
    },

    # ─── CI/CD ─────────────────────────────────────────────────────────
    {
        "id": "cicd_gh_001",
        "text": "GitHub Actions workflow troubleshooting: check run logs in the "
                "Actions tab, verify secrets in repository settings, ensure "
                "runners have required tools, and check YAML syntax.",
        "metadata": {"category": "cicd", "topic": "github-actions", "severity": "high"}
    },
    {
        "id": "cicd_jenkins_001",
        "text": "Jenkins pipeline best practices: use declarative pipeline syntax, "
                "organize stages clearly, store Jenkinsfile in source control, "
                "use shared libraries, implement proper error handling with post blocks.",
        "metadata": {"category": "cicd", "topic": "jenkins", "severity": "low"}
    },

    # ─── Monitoring ────────────────────────────────────────────────────
    {
        "id": "mon_prom_001",
        "text": "Prometheus collects metrics using a pull model. Configure scrape "
                "targets in prometheus.yml. Use PromQL to query metrics and create "
                "alerts with Alertmanager for critical thresholds.",
        "metadata": {"category": "monitoring", "topic": "prometheus", "severity": "medium"}
    },
]


# =============================================================================
# TEST QUERIES WITH EXPECTED RESULTS
# =============================================================================
# This is the core evaluation dataset. For each query, we define which
# documents SHOULD be returned. This lets us calculate precision and recall.
#
# HOW TO BUILD YOUR OWN TEST SET:
# 1. Think of questions users actually ask
# 2. For each question, manually identify which docs answer it
# 3. Include easy cases (obvious matches) and hard cases (subtle matches)
# 4. Include negative cases (queries with no good matches)

TEST_QUERIES = [
    {
        "query": "my pod keeps crashing and restarting",
        "relevant_ids": ["k8s_crash_001", "k8s_crash_002"],
        "description": "Direct match: CrashLoopBackOff troubleshooting"
    },
    {
        "query": "container killed because of memory",
        "relevant_ids": ["k8s_oom_001"],
        "description": "Semantic match: OOMKilled with different wording"
    },
    {
        "query": "cannot pull docker image in kubernetes",
        "relevant_ids": ["k8s_image_001"],
        "description": "Cross-terminology: Docker image + Kubernetes"
    },
    {
        "query": "how to expose my application to the internet",
        "relevant_ids": ["k8s_net_001", "k8s_net_002"],
        "description": "Conceptual match: external access via Service/Ingress"
    },
    {
        "query": "terraform state is locked by someone else",
        "relevant_ids": ["tf_state_001"],
        "description": "Direct match: state lock error"
    },
    {
        "query": "set up monitoring and alerts for kubernetes cluster",
        "relevant_ids": ["mon_prom_001"],
        "description": "Topic match: monitoring and alerting"
    },
    {
        "query": "optimize docker image size for production",
        "relevant_ids": ["docker_build_001"],
        "description": "Specific topic: Docker image optimization"
    },
    {
        "query": "debug failing kubernetes pod",
        "relevant_ids": ["k8s_crash_001", "k8s_crash_002", "k8s_oom_001", "k8s_image_001"],
        "description": "Broad query: any pod error doc is relevant"
    },
]


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def calculate_precision(retrieved_ids, relevant_ids):
    """
    Calculate precision: what fraction of retrieved docs are relevant?

    FORMULA:
    ────────
                  |retrieved AND relevant|
    Precision = ───────────────────────────
                      |retrieved|

    EXAMPLE:
    ────────
    Retrieved: [k8s_crash_001, k8s_net_001, k8s_crash_002, docker_build_001, tf_state_001]
    Relevant:  [k8s_crash_001, k8s_crash_002]

    Precision = 2 / 5 = 0.40 (40%)
    Meaning: only 2 out of 5 retrieved docs were useful.

    Args:
        retrieved_ids: List of document IDs returned by search
        relevant_ids: List of document IDs that are actually relevant

    Returns:
        Float between 0.0 and 1.0 (higher = better)
    """
    if len(retrieved_ids) == 0:
        return 0.0

    # Convert to sets for intersection operation
    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    # True positives: documents that are both retrieved AND relevant
    true_positives = retrieved_set.intersection(relevant_set)

    return len(true_positives) / len(retrieved_set)


def calculate_recall(retrieved_ids, relevant_ids):
    """
    Calculate recall: what fraction of relevant docs were retrieved?

    FORMULA:
    ────────
                |retrieved AND relevant|
    Recall = ─────────────────────────────
                    |relevant|

    EXAMPLE:
    ────────
    Retrieved: [k8s_crash_001, k8s_net_001, k8s_crash_002]
    Relevant:  [k8s_crash_001, k8s_crash_002, k8s_oom_001]

    Recall = 2 / 3 = 0.67 (67%)
    Meaning: we found 2 out of 3 relevant docs. Missed k8s_oom_001.

    Args:
        retrieved_ids: List of document IDs returned by search
        relevant_ids: List of document IDs that are actually relevant

    Returns:
        Float between 0.0 and 1.0 (higher = better)
    """
    if len(relevant_ids) == 0:
        return 1.0  # If nothing is relevant, recall is trivially perfect

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    true_positives = retrieved_set.intersection(relevant_set)

    return len(true_positives) / len(relevant_set)


def calculate_f1(precision, recall):
    """
    Calculate F1 score: the harmonic mean of precision and recall.

    FORMULA:
    ────────
              2 * precision * recall
    F1 = ─────────────────────────────
              precision + recall

    WHY HARMONIC MEAN (not regular average)?
    ─────────────────────────────────────────
    The harmonic mean penalizes extreme imbalances.

    Example: precision=1.0, recall=0.0
      Regular average: (1.0 + 0.0) / 2 = 0.50 (misleadingly good!)
      F1 (harmonic):    2*1.0*0.0 / (1.0+0.0) = 0.00 (correctly bad!)

    F1 SAYS: "You need BOTH precision AND recall to be good."

    Args:
        precision: Precision score (0-1)
        recall: Recall score (0-1)

    Returns:
        F1 score between 0.0 and 1.0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =============================================================================
# SETUP: CREATE VECTOR STORE
# =============================================================================

def create_evaluation_vectorstore():
    """
    Create a ChromaDB vector store populated with our test documents.

    We use both the native ChromaDB API (for standard search) and
    LangChain's wrapper (for MMR search), showing how both approaches
    work for evaluation.
    """
    print("\n  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # ─── Native ChromaDB setup (for standard search) ──────────────────
    print("  Creating ChromaDB collection...")
    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection("eval_collection")
    except Exception:
        pass

    native_collection = chroma_client.create_collection(
        name="eval_collection",
        metadata={"hnsw:space": "cosine"}
    )

    # Add documents to native collection
    ids = [doc["id"] for doc in DOCUMENTS]
    texts = [doc["text"] for doc in DOCUMENTS]
    metas = [doc["metadata"] for doc in DOCUMENTS]
    embeddings = [model.encode(t).tolist() for t in texts]

    native_collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metas,
    )

    # ─── LangChain Chroma setup (for MMR search) ──────────────────────
    print("  Creating LangChain vectorstore (for MMR support)...")
    lc_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    lc_docs = [
        Document(
            page_content=doc["text"],
            metadata={**doc["metadata"], "doc_id": doc["id"]}
        )
        for doc in DOCUMENTS
    ]

    lc_vectorstore = Chroma.from_documents(
        documents=lc_docs,
        embedding=lc_embeddings,
        collection_name="eval_lc_collection"
    )

    print(f"  Created vectorstore with {native_collection.count()} documents")

    return native_collection, lc_vectorstore, model


# =============================================================================
# DEMO 1: BASIC RETRIEVAL EVALUATION
# =============================================================================

def demo_basic_evaluation(collection, model):
    """
    Run all test queries and calculate precision, recall, and F1
    at a fixed k value.
    """
    print("\n" + "=" * 70)
    print("  DEMO 1: Basic Retrieval Evaluation (k=3)")
    print("=" * 70)

    k = 3
    all_precisions = []
    all_recalls = []
    all_f1s = []

    print(f"\n  Evaluating {len(TEST_QUERIES)} test queries with k={k}:\n")
    print(f"  {'Query':<45} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Found/Rel':>10}")
    print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*6} {'─'*10}")

    for test in TEST_QUERIES:
        # Generate query embedding
        query_vec = model.encode(test["query"]).tolist()

        # Search
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["distances"]
        )

        retrieved_ids = results["ids"][0]
        relevant_ids = test["relevant_ids"]

        # Calculate metrics
        precision = calculate_precision(retrieved_ids, relevant_ids)
        recall = calculate_recall(retrieved_ids, relevant_ids)
        f1 = calculate_f1(precision, recall)

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # Count how many relevant docs were found
        found = len(set(retrieved_ids) & set(relevant_ids))
        total_rel = len(relevant_ids)

        short_query = test["query"][:42] + "..." if len(test["query"]) > 42 else test["query"]
        print(f"  {short_query:<45} {precision:>6.2f} {recall:>6.2f} {f1:>6.2f} {found:>4}/{total_rel:<4}")

    # ─── Summary ───────────────────────────────────────────────────────
    avg_p = np.mean(all_precisions)
    avg_r = np.mean(all_recalls)
    avg_f1 = np.mean(all_f1s)

    print(f"\n  {'AVERAGE':<45} {avg_p:>6.2f} {avg_r:>6.2f} {avg_f1:>6.2f}")

    print(f"""
    INTERPRETATION:
    ───────────────
    Average Precision: {avg_p:.2f} ({avg_p*100:.0f}%)
      -> Of the documents retrieved, {avg_p*100:.0f}% were actually relevant.
         {"Good!" if avg_p > 0.6 else "Could be improved. Too much noise in results."}

    Average Recall: {avg_r:.2f} ({avg_r*100:.0f}%)
      -> Of all relevant documents, we found {avg_r*100:.0f}%.
         {"Good!" if avg_r > 0.6 else "Missing too many relevant docs. Try increasing k."}

    Average F1: {avg_f1:.2f} ({avg_f1*100:.0f}%)
      -> Combined quality score.
         {"Excellent!" if avg_f1 > 0.7 else "Good!" if avg_f1 > 0.5 else "Needs improvement."}
    """)


# =============================================================================
# DEMO 2: K VALUE COMPARISON
# =============================================================================

def demo_k_comparison(collection, model):
    """
    Compare retrieval quality across different k values.

    This helps you find the optimal k for your specific use case.
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Comparing k Values (1, 3, 5, 10)")
    print("=" * 70)

    print("""
    THE PRECISION-RECALL TRADE-OFF:
    ───────────────────────────────

                  Precision                         Recall
         1.0 ─┐                              ┌─ 1.0
               │ \\                           / │
               │   \\                       /   │
               │     \\                   /     │
               │       \\     sweet    /       │
               │         \\   spot   /         │
               │           \\ _____ /           │
               │                               │
         0.0 ─┘                              └─ 0.0
               k=1   k=3   k=5   k=10

    - Small k: High precision, low recall (picky but misses docs)
    - Large k: Low precision, high recall (finds everything but adds noise)
    - Sweet spot: Usually k=3-5 for DevOps RAG
    """)

    k_values = [1, 3, 5, 10]
    results_by_k = {}

    for k in k_values:
        precisions = []
        recalls = []
        f1s = []

        for test in TEST_QUERIES:
            query_vec = model.encode(test["query"]).tolist()

            results = collection.query(
                query_embeddings=[query_vec],
                n_results=k,
                include=["distances"]
            )

            retrieved_ids = results["ids"][0]
            p = calculate_precision(retrieved_ids, test["relevant_ids"])
            r = calculate_recall(retrieved_ids, test["relevant_ids"])
            f = calculate_f1(p, r)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        results_by_k[k] = {
            "precision": np.mean(precisions),
            "recall": np.mean(recalls),
            "f1": np.mean(f1s),
        }

    # ─── Display Results Table ─────────────────────────────────────────
    print(f"  {'k':>4} {'Precision':>12} {'Recall':>12} {'F1 Score':>12}    {'Assessment'}")
    print(f"  {'─'*4} {'─'*12} {'─'*12} {'─'*12}    {'─'*20}")

    best_f1 = 0
    best_k = 1

    for k in k_values:
        r = results_by_k[k]
        # Simple assessment
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_k = k

        assessment = ""
        if k == 1:
            assessment = "Very picky"
        elif k == 3:
            assessment = "Balanced"
        elif k == 5:
            assessment = "Broader coverage"
        elif k == 10:
            assessment = "Maximum recall"

        print(f"  {k:>4} {r['precision']:>12.3f} {r['recall']:>12.3f} {r['f1']:>12.3f}    {assessment}")

    print(f"\n  Best F1 score: {best_f1:.3f} at k={best_k}")

    # ─── Visual Comparison ─────────────────────────────────────────────
    print(f"\n  VISUAL COMPARISON:")
    print(f"  " + "-" * 60)

    bar_width = 30
    for k in k_values:
        r = results_by_k[k]

        p_bar = "#" * int(r["precision"] * bar_width) + "." * (bar_width - int(r["precision"] * bar_width))
        r_bar = "#" * int(r["recall"] * bar_width) + "." * (bar_width - int(r["recall"] * bar_width))
        f_bar = "#" * int(r["f1"] * bar_width) + "." * (bar_width - int(r["f1"] * bar_width))

        marker = " <-- BEST" if k == best_k else ""
        print(f"\n  k={k}:{marker}")
        print(f"    Precision: [{p_bar}] {r['precision']:.3f}")
        print(f"    Recall:    [{r_bar}] {r['recall']:.3f}")
        print(f"    F1:        [{f_bar}] {r['f1']:.3f}")

    print(f"""
    RECOMMENDATION:
    ───────────────
    For this DevOps knowledge base, k={best_k} gives the best balance
    of precision and recall (F1={best_f1:.3f}).

    In production, consider:
    - k=3 for simple Q&A (user asks specific questions)
    - k=5 for troubleshooting (need broader context)
    - k=1-2 for auto-complete or quick suggestions
    """)


# =============================================================================
# DEMO 3: STANDARD SEARCH vs MMR
# =============================================================================

def demo_search_vs_mmr(lc_vectorstore):
    """
    Compare standard similarity search with MMR (Maximum Marginal Relevance).

    MMR diversifies results to avoid returning multiple near-duplicate documents.
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Standard Search vs MMR (Maximum Marginal Relevance)")
    print("=" * 70)

    print("""
    WHAT IS MMR?
    ────────────
    Standard search returns the k MOST SIMILAR documents.
    But similar documents might all say the same thing!

    MMR balances RELEVANCE and DIVERSITY:

    Standard Search (k=3):               MMR Search (k=3):
    ┌──────────────────────────┐         ┌──────────────────────────┐
    │ 1. CrashLoopBackOff guide│         │ 1. CrashLoopBackOff guide│
    │ 2. CrashLoop debug steps │ <same!  │ 2. OOMKilled debugging   │ diverse!
    │ 3. CrashLoop solutions   │ <same!  │ 3. ImagePullBackOff fix  │ diverse!
    └──────────────────────────┘         └──────────────────────────┘

    MMR ALGORITHM:
    ──────────────
    For each candidate document:
      score = lambda * similarity(query, doc)
              - (1 - lambda) * max_similarity(doc, already_selected)

    lambda = 1.0: Pure relevance (same as standard search)
    lambda = 0.0: Pure diversity (ignore relevance entirely)
    lambda = 0.5: Equal balance (typical default)
    """)

    query = "debug kubernetes pod errors and crashes"

    # ─── Standard Search ───────────────────────────────────────────────
    print(f"  Query: \"{query}\"")
    print(f"\n  STANDARD SEARCH (k=4):")
    print("  " + "-" * 60)

    standard_results = lc_vectorstore.similarity_search_with_score(query, k=4)

    for i, (doc, score) in enumerate(standard_results, 1):
        doc_id = doc.metadata.get("doc_id", "?")
        topic = doc.metadata.get("topic", "?")
        preview = doc.page_content[:55] + "..."
        print(f"    {i}. [{doc_id}] topic={topic} (dist={score:.4f})")
        print(f"       {preview}")

    # ─── MMR Search ────────────────────────────────────────────────────
    print(f"\n  MMR SEARCH (k=4, fetch_k=10):")
    print("  " + "-" * 60)

    # MMR fetches more candidates (fetch_k) then selects diverse subset (k)
    mmr_results = lc_vectorstore.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=10,    # Fetch top 10 candidates, then diversify to 4
        lambda_mult=0.5  # Balance between relevance (1.0) and diversity (0.0)
    )

    for i, doc in enumerate(mmr_results, 1):
        doc_id = doc.metadata.get("doc_id", "?")
        topic = doc.metadata.get("topic", "?")
        preview = doc.page_content[:55] + "..."
        print(f"    {i}. [{doc_id}] topic={topic}")
        print(f"       {preview}")

    # ─── Evaluate Both ─────────────────────────────────────────────────
    print(f"\n  EVALUATION on full test set:")
    print("  " + "-" * 60)

    k = 4
    std_precisions, std_recalls, std_f1s = [], [], []
    mmr_precisions, mmr_recalls, mmr_f1s = [], [], []

    for test in TEST_QUERIES:
        # Standard search
        std_results = lc_vectorstore.similarity_search(test["query"], k=k)
        std_ids = [doc.metadata.get("doc_id", "") for doc in std_results]

        std_p = calculate_precision(std_ids, test["relevant_ids"])
        std_r = calculate_recall(std_ids, test["relevant_ids"])
        std_f = calculate_f1(std_p, std_r)

        std_precisions.append(std_p)
        std_recalls.append(std_r)
        std_f1s.append(std_f)

        # MMR search
        mmr_results = lc_vectorstore.max_marginal_relevance_search(
            test["query"], k=k, fetch_k=10, lambda_mult=0.5
        )
        mmr_ids = [doc.metadata.get("doc_id", "") for doc in mmr_results]

        mmr_p = calculate_precision(mmr_ids, test["relevant_ids"])
        mmr_r = calculate_recall(mmr_ids, test["relevant_ids"])
        mmr_f = calculate_f1(mmr_p, mmr_r)

        mmr_precisions.append(mmr_p)
        mmr_recalls.append(mmr_r)
        mmr_f1s.append(mmr_f)

    print(f"  {'Method':<12} {'Precision':>12} {'Recall':>12} {'F1':>12}")
    print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Standard':<12} {np.mean(std_precisions):>12.3f} {np.mean(std_recalls):>12.3f} {np.mean(std_f1s):>12.3f}")
    print(f"  {'MMR':<12} {np.mean(mmr_precisions):>12.3f} {np.mean(mmr_recalls):>12.3f} {np.mean(mmr_f1s):>12.3f}")

    # Determine which is better
    std_avg_f1 = np.mean(std_f1s)
    mmr_avg_f1 = np.mean(mmr_f1s)

    if mmr_avg_f1 > std_avg_f1:
        winner = "MMR"
        diff = mmr_avg_f1 - std_avg_f1
    else:
        winner = "Standard"
        diff = std_avg_f1 - mmr_avg_f1

    print(f"""
    RESULT: {winner} search wins by {diff:.3f} F1 points.

    WHEN TO USE MMR:
    ────────────────
    - When your knowledge base has REDUNDANT content
      (multiple docs covering the same topic)
    - When you want the LLM to see DIVERSE context
    - When users ask broad questions that span multiple sub-topics

    WHEN TO USE STANDARD SEARCH:
    ────────────────────────────
    - When each document covers a unique topic
    - When you need the MOST relevant documents regardless of overlap
    - When precision matters more than diversity
    """)


# =============================================================================
# DEMO 4: SCORE THRESHOLD ANALYSIS
# =============================================================================

def demo_score_threshold(collection, model):
    """
    Analyze how score thresholds affect retrieval quality.

    A score threshold filters out low-confidence results, which can
    improve precision at the cost of recall.
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: Score Threshold Analysis")
    print("=" * 70)

    print("""
    WHAT IS A SCORE THRESHOLD?
    ──────────────────────────
    Instead of always returning k results, only return results
    ABOVE a minimum similarity score.

    Without threshold (k=5):    With threshold=0.65 (k=5):
    ┌───────────────────────┐   ┌───────────────────────┐
    │ Doc1: 0.92 (relevant) │   │ Doc1: 0.92 (relevant) │
    │ Doc2: 0.85 (relevant) │   │ Doc2: 0.85 (relevant) │
    │ Doc3: 0.71 (relevant) │   │ Doc3: 0.71 (relevant) │
    │ Doc4: 0.45 (noise!)   │   │ Doc4: FILTERED OUT    │
    │ Doc5: 0.32 (noise!)   │   │ Doc5: FILTERED OUT    │
    └───────────────────────┘   └───────────────────────┘

    Threshold removes noise but might filter relevant docs too!
    """)

    # ─── Analyze Score Distributions ───────────────────────────────────
    print("  SCORE DISTRIBUTION ANALYSIS:")
    print("  " + "-" * 60)

    # Collect all scores across all queries
    all_relevant_scores = []    # Scores for relevant documents
    all_irrelevant_scores = []  # Scores for irrelevant documents

    for test in TEST_QUERIES:
        query_vec = model.encode(test["query"]).tolist()

        # Fetch many results to see score distribution
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(15, len(DOCUMENTS)),
            include=["distances"]
        )

        for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
            # Convert ChromaDB cosine distance to similarity
            similarity = 1 - distance

            if doc_id in test["relevant_ids"]:
                all_relevant_scores.append(similarity)
            else:
                all_irrelevant_scores.append(similarity)

    # ─── Display Score Distributions ───────────────────────────────────
    if all_relevant_scores and all_irrelevant_scores:
        rel_mean = np.mean(all_relevant_scores)
        rel_min = np.min(all_relevant_scores)
        rel_max = np.max(all_relevant_scores)

        irr_mean = np.mean(all_irrelevant_scores)
        irr_min = np.min(all_irrelevant_scores)
        irr_max = np.max(all_irrelevant_scores)

        print(f"""
    Relevant documents (ones we WANT to find):
      Count:   {len(all_relevant_scores)}
      Mean:    {rel_mean:.4f}
      Range:   [{rel_min:.4f}, {rel_max:.4f}]

    Irrelevant documents (noise):
      Count:   {len(all_irrelevant_scores)}
      Mean:    {irr_mean:.4f}
      Range:   [{irr_min:.4f}, {irr_max:.4f}]

    Score Distribution Visualization:
    ─────────────────────────────────

    Similarity: 0.0        0.3        0.5        0.7        0.9    1.0
                │          │          │          │          │       │
    Irrelevant: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
    Relevant:                                   ██████████████████████

    The LESS overlap, the easier it is to find a good threshold.
    """)

    # ─── Test Different Thresholds ─────────────────────────────────────
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    k_max = 10  # Maximum docs to retrieve before filtering

    print(f"  THRESHOLD COMPARISON (retrieve up to k={k_max}, then filter):")
    print(f"  " + "-" * 60)
    print(f"  {'Threshold':>10} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Avg Results':>13}")
    print(f"  {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*13}")

    best_threshold_f1 = 0
    best_threshold = 0

    for threshold in thresholds:
        precisions = []
        recalls = []
        result_counts = []

        for test in TEST_QUERIES:
            query_vec = model.encode(test["query"]).tolist()

            results = collection.query(
                query_embeddings=[query_vec],
                n_results=k_max,
                include=["distances"]
            )

            # Apply threshold: only keep docs with similarity > threshold
            filtered_ids = []
            for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
                similarity = 1 - distance  # Convert distance to similarity
                if similarity >= threshold:
                    filtered_ids.append(doc_id)

            result_counts.append(len(filtered_ids))

            p = calculate_precision(filtered_ids, test["relevant_ids"])
            r = calculate_recall(filtered_ids, test["relevant_ids"])

            precisions.append(p)
            recalls.append(r)

        avg_p = np.mean(precisions)
        avg_r = np.mean(recalls)
        avg_f1 = calculate_f1(avg_p, avg_r)
        avg_count = np.mean(result_counts)

        if avg_f1 > best_threshold_f1:
            best_threshold_f1 = avg_f1
            best_threshold = threshold

        marker = " <-- BEST" if threshold == best_threshold and avg_f1 == best_threshold_f1 else ""
        print(f"  {threshold:>10.1f} {avg_p:>12.3f} {avg_r:>12.3f} {avg_f1:>12.3f} {avg_count:>13.1f}{marker}")

    print(f"""
    ANALYSIS:
    ─────────
    Best threshold: {best_threshold} (F1 = {best_threshold_f1:.3f})

    - Low threshold (0.3): Returns many results, high recall, low precision
    - High threshold (0.8): Returns few results, high precision, low recall
    - Optimal threshold depends on your use case:
      * Chatbot: moderate threshold (0.5-0.6) for balanced answers
      * Critical alerts: low threshold (0.3) to never miss relevant docs
      * Auto-suggestions: high threshold (0.7+) for confident results

    ┌────────────────────────────────────────────────────────────────────┐
    │  TIP: In production, use threshold + minimum k:                   │
    │                                                                    │
    │  results = search(query, k=10)                                    │
    │  filtered = [r for r in results if r.score >= threshold]          │
    │  if len(filtered) < min_k:                                        │
    │      filtered = results[:min_k]  # Always return at least min_k  │
    │                                                                    │
    │  This guarantees the LLM always gets SOME context, even if       │
    │  nothing scores above the threshold (e.g., out-of-domain query). │
    └────────────────────────────────────────────────────────────────────┘
    """)


# =============================================================================
# DEMO 5: PER-QUERY DETAILED ANALYSIS
# =============================================================================

def demo_detailed_per_query(collection, model):
    """
    Show a detailed breakdown for individual queries so you can
    diagnose exactly where retrieval is succeeding or failing.
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: Detailed Per-Query Analysis")
    print("=" * 70)

    print("""
    This is your DEBUGGING TOOL. When retrieval quality is poor,
    examine individual queries to understand WHY.
    """)

    k = 5

    # Pick a few interesting test queries
    detailed_tests = [TEST_QUERIES[0], TEST_QUERIES[3], TEST_QUERIES[7]]

    for test in detailed_tests:
        query_vec = model.encode(test["query"]).tolist()

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "distances", "metadatas"]
        )

        print(f"\n  Query: \"{test['query']}\"")
        print(f"  Description: {test['description']}")
        print(f"  Expected relevant: {test['relevant_ids']}")
        print(f"  " + "-" * 60)

        retrieved_ids = results["ids"][0]
        relevant_set = set(test["relevant_ids"])

        for i in range(len(retrieved_ids)):
            doc_id = retrieved_ids[i]
            distance = results["distances"][0][i]
            similarity = 1 - distance
            doc_text = results["documents"][0][i][:60] + "..."
            metadata = results["metadatas"][0][i]

            # Is this result relevant?
            is_relevant = doc_id in relevant_set
            status = "RELEVANT" if is_relevant else "noise"

            print(f"    {i+1}. [{doc_id}] sim={similarity:.4f} [{status}]")
            print(f"       cat={metadata['category']}, topic={metadata['topic']}")
            print(f"       {doc_text}")

        # Check for missed relevant docs
        missed = relevant_set - set(retrieved_ids)
        if missed:
            print(f"\n    MISSED relevant docs (not in top-{k}): {missed}")
            print(f"    -> Consider increasing k or improving document content")
        else:
            print(f"\n    All relevant docs found in top-{k}!")

        # Metrics for this query
        p = calculate_precision(retrieved_ids, test["relevant_ids"])
        r = calculate_recall(retrieved_ids, test["relevant_ids"])
        f = calculate_f1(p, r)
        print(f"    Precision={p:.2f}, Recall={r:.2f}, F1={f:.2f}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 5, EXERCISE 4: Retrieval Quality Analysis")
    print("  Measuring and Improving Vector Search Quality")
    print("=" * 70)

    # Setup
    print("\n  Setting up evaluation environment...")
    native_collection, lc_vectorstore, model = create_evaluation_vectorstore()

    # Demo 1: Basic evaluation
    demo_basic_evaluation(native_collection, model)

    # Demo 2: K value comparison
    demo_k_comparison(native_collection, model)

    # Demo 3: Standard vs MMR
    demo_search_vs_mmr(lc_vectorstore)

    # Demo 4: Score threshold analysis
    demo_score_threshold(native_collection, model)

    # Demo 5: Detailed per-query analysis
    demo_detailed_per_query(native_collection, model)

    # ─── Final Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  1. ALWAYS MEASURE RETRIEVAL QUALITY                                  │
    │     - Build a test set with queries and expected relevant docs        │
    │     - Calculate precision, recall, and F1 for every change you make  │
    │     - Without measurement, you're optimizing blindly                  │
    │                                                                        │
    │  2. PRECISION vs RECALL TRADE-OFF                                     │
    │     - Precision: "Are my results relevant?" (quality of results)     │
    │     - Recall: "Did I find everything?" (completeness of results)     │
    │     - F1: Harmonic mean -- balances both                              │
    │     - Your use case determines which matters more                     │
    │                                                                        │
    │  3. TUNE YOUR K VALUE                                                 │
    │     - k=1-2: Best single answer (high precision)                      │
    │     - k=3-5: Good balance (recommended for most RAG chatbots)        │
    │     - k=10+: Maximum recall (comprehensive but noisy)                 │
    │     - Always test with YOUR data to find the sweet spot              │
    │                                                                        │
    │  4. MMR REDUCES REDUNDANCY                                            │
    │     - Use when docs have overlapping content                          │
    │     - Gives the LLM more diverse context to work with               │
    │     - May slightly hurt precision for the top result                  │
    │     - But often improves overall answer quality                       │
    │                                                                        │
    │  5. SCORE THRESHOLDS FILTER NOISE                                     │
    │     - Prevent low-quality matches from polluting LLM context         │
    │     - Always have a fallback (minimum k) for out-of-domain queries   │
    │     - Analyze score distributions to find optimal threshold          │
    │                                                                        │
    │  6. DEBUG INDIVIDUAL QUERIES                                           │
    │     - When average metrics are bad, look at individual queries       │
    │     - Find which queries fail and WHY                                 │
    │     - Common fixes: improve document content, add synonyms,          │
    │       adjust chunking strategy, or add metadata filters              │
    │                                                                        │
    │  EVALUATION CHECKLIST FOR YOUR DEVOPS CHATBOT:                        │
    │  [ ] Create 20+ test queries covering common user questions          │
    │  [ ] Label expected relevant docs for each query                      │
    │  [ ] Run evaluation with k=3, k=5 (compare)                          │
    │  [ ] Test standard vs MMR search                                      │
    │  [ ] Find optimal score threshold                                     │
    │  [ ] Re-evaluate after any change to docs, chunking, or embeddings   │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """)

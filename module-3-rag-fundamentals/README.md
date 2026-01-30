# Module 3: RAG Fundamentals

**Time Required: 2-3 hours**

Our chatbot works, but it can only use knowledge from its training data. What if we want it to know about OUR specific documentation? Enter RAG - Retrieval-Augmented Generation.

---

## Learning Objectives

By the end of this module, you will:
- Understand why LLMs need external knowledge
- Know what vectors and embeddings are (explained simply!)
- Understand what RAG is and how it works
- Set up ChromaDB for vector storage
- Use LangChain to build a RAG pipeline
- Implement a working RAG system for DevOps docs

---

## Prerequisites

Before starting, make sure you have:
```bash
# Install required packages
pip install chromadb langchain langchain-community langchain-openai
pip install sentence-transformers  # For local embeddings (free!)

# If using Ollama (free, local)
pip install langchain-ollama
```

---

## The Problem with LLMs

### LLMs Have Limitations

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LLM KNOWLEDGE LIMITATIONS                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. KNOWLEDGE CUTOFF                                                  │
│     ┌─────────────────────────────────────────┐                      │
│     │ Training Data │ ? ? ? │    Future       │                      │
│     └─────────────────────────────────────────┘                      │
│               ▲                                                       │
│               │                                                       │
│         Cutoff Date                                                   │
│     (e.g., Jan 2024)                                                 │
│                                                                       │
│  2. NO ACCESS TO YOUR PRIVATE DOCS                                   │
│     - Your company's runbooks                                        │
│     - Internal documentation                                          │
│     - Custom configurations                                           │
│                                                                       │
│  3. HALLUCINATION (Making Things Up)                                 │
│     When the LLM doesn't know something, it might                    │
│     confidently give you a wrong answer!                             │
│                                                                       │
│  4. GENERIC ANSWERS                                                   │
│     Without your specific context, answers are                        │
│     general and may not apply to your situation.                     │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Real Example: Without vs With RAG

```
WITHOUT RAG:
┌─────────────────────────────────────────────────────────────────────┐
│ User: "How do I fix the database connection error in our            │
│        payment-service on Kubernetes?"                               │
│                                                                      │
│ LLM: "Generally, database connection errors can be caused by        │
│       incorrect credentials, network issues, or..."                  │
│                                                                      │
│       ^^^ Generic advice - doesn't know YOUR payment-service        │
└─────────────────────────────────────────────────────────────────────┘

WITH RAG:
┌─────────────────────────────────────────────────────────────────────┐
│ User: "How do I fix the database connection error in our            │
│        payment-service on Kubernetes?"                               │
│                                                                      │
│ [RAG retrieves YOUR documentation about payment-service]            │
│                                                                      │
│ LLM: "Based on our payment-service docs, this error usually         │
│       means the DB_SECRET isn't mounted. Check that the             │
│       'payment-db-credentials' secret exists in the 'prod'          │
│       namespace. Run: kubectl get secret payment-db-credentials..." │
│                                                                      │
│       ^^^ Specific to YOUR system, actionable, accurate             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Understanding Vectors and Embeddings

Before we dive into RAG, we need to understand two key concepts: **vectors** and **embeddings**. Don't worry - we'll explain them simply!

### What is a Vector?

**A vector is just a list of numbers.** That's it!

```
VECTOR EXAMPLES:
────────────────

Simple vector (2 numbers):  [3, 4]
                             ↑  ↑
                             x  y  (like coordinates on a map!)

Longer vector (4 numbers):  [0.2, -0.5, 0.8, 0.1]

Real embedding (384 numbers): [0.12, -0.45, 0.78, 0.33, ..., 0.21]
                               └─────────────────────────────────┘
                                    384 dimensions
```

**Think of it like GPS coordinates:**
- Your location can be described as [latitude, longitude]
- That's a 2-dimensional vector!
- Embeddings are like "concept coordinates" in many dimensions

### What is an Embedding?

**An embedding converts text into a vector (list of numbers) that captures its meaning.**

```
EMBEDDING = Translating Words to Numbers
════════════════════════════════════════

┌────────────────────┐         ┌─────────────────────────────┐
│                    │         │                             │
│  "Kubernetes pod"  │  ───►   │  [0.82, -0.15, 0.43, ...]  │
│                    │         │                             │
│      (text)        │         │        (vector)             │
└────────────────────┘         └─────────────────────────────┘
        │                                    │
        │                                    │
        ▼                                    ▼
   Human readable               Computer can do math with it!
```

### Why Do We Need Embeddings?

Computers can't understand "meaning" directly. But with embeddings:

```
MAGIC OF EMBEDDINGS: Similar Meanings = Similar Numbers
═══════════════════════════════════════════════════════

"container crash"      → [0.8, 0.2, 0.5, ...]  ─┐
                                                 │ Very similar vectors!
"pod keeps restarting" → [0.79, 0.21, 0.48, ...] ─┘  (same concept)


"container crash"      → [0.8, 0.2, 0.5, ...]  ─┐
                                                 │ Very different vectors
"terraform modules"    → [0.1, 0.9, -0.3, ...] ─┘  (different topics)


This lets us find RELATED content, not just exact keyword matches!
```

### Real-World Analogy: Library Books

```
TRADITIONAL SEARCH (Keywords):
─────────────────────────────
Search: "container restart"
Result: Only finds docs with those EXACT words

SEMANTIC SEARCH (Embeddings):
────────────────────────────
Search: "container restart"
Results:
  ✓ "Pod keeps crashing and restarting"     (related meaning)
  ✓ "CrashLoopBackOff troubleshooting"      (same problem!)
  ✓ "Container exit codes explained"         (related topic)

The AI understands these are all about the SAME PROBLEM!
```

---

## What is a Vector Database?

Now that we have vectors, we need somewhere to store them. Enter **vector databases** like ChromaDB!

### What is ChromaDB?

**ChromaDB is a database optimized for storing and searching vectors.**

```
TRADITIONAL DATABASE vs VECTOR DATABASE
═══════════════════════════════════════

Traditional Database (SQL):
┌────────────────────────────────────────┐
│ id │ title              │ content      │
├────────────────────────────────────────┤
│ 1  │ "Terraform Errors" │ "When you..." │
│ 2  │ "K8s Pods"         │ "A pod is..." │
└────────────────────────────────────────┘
Search: WHERE title LIKE '%error%'  (exact match only)


Vector Database (ChromaDB):
┌──────────────────────────────────────────────────────────┐
│ id │ content           │ vector                          │
├──────────────────────────────────────────────────────────┤
│ 1  │ "Terraform..."    │ [0.2, 0.8, -0.3, ..., 0.5]     │
│ 2  │ "K8s pods..."     │ [0.9, -0.1, 0.6, ..., -0.2]    │
└──────────────────────────────────────────────────────────┘
Search: Find vectors SIMILAR to [0.85, -0.05, 0.55, ...]
        (finds related content by MEANING!)
```

### How ChromaDB Works

```
STORING DOCUMENTS IN CHROMADB:
══════════════════════════════

Step 1: Your document
        ┌─────────────────────────────────────┐
        │ "To fix CrashLoopBackOff, first     │
        │  check the container logs using     │
        │  kubectl logs pod-name..."          │
        └─────────────────────────────────────┘
                        │
                        ▼
Step 2: Convert to embedding (via embedding model)
        ┌─────────────────────────────────────┐
        │ [0.23, -0.45, 0.67, ..., 0.12]     │
        └─────────────────────────────────────┘
                        │
                        ▼
Step 3: Store in ChromaDB
        ┌─────────────────────────────────────┐
        │ ChromaDB                            │
        │  ┌─────────────────────────────┐   │
        │  │ doc_1: [0.23, -0.45, ...]   │   │
        │  │ doc_2: [0.11, 0.89, ...]    │   │
        │  │ doc_3: [0.56, -0.23, ...]   │   │
        │  └─────────────────────────────┘   │
        └─────────────────────────────────────┘
```

---

## What is LangChain?

**LangChain is a framework that makes it easy to build LLM applications.**

Think of it like a toolkit that provides ready-made pieces for common tasks:

```
LANGCHAIN COMPONENTS WE'LL USE:
═══════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  Document Loaders    Load files (MD, TXT, PDF, etc.)                │
│  ───────────────     loader = TextLoader("docs/errors.md")          │
│                                                                      │
│  Text Splitters      Break documents into chunks                     │
│  ─────────────       splitter = RecursiveCharacterTextSplitter()    │
│                                                                      │
│  Embeddings          Convert text to vectors                         │
│  ──────────          embeddings = HuggingFaceEmbeddings()           │
│                                                                      │
│  Vector Stores       Store and search vectors                        │
│  ────────────        vectorstore = Chroma.from_documents()          │
│                                                                      │
│  LLMs                Connect to AI models                            │
│  ────                llm = ChatOpenAI() or ChatOllama()             │
│                                                                      │
│  Chains              Combine components into workflows               │
│  ──────              chain = RetrievalQA.from_chain_type()          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## What is RAG?

Now we can understand RAG!

**RAG = Retrieval-Augmented Generation**

```
RAG IN PLAIN ENGLISH:
═════════════════════

1. RETRIEVAL  = Find relevant documents from your knowledge base
2. AUGMENTED  = Add those documents to the AI's prompt
3. GENERATION = AI generates answer using YOUR documents as context

It's like giving the AI a "cheat sheet" of relevant info before asking!
```

### The Complete RAG Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                     THE RAG PIPELINE                                  │
└──────────────────────────────────────────────────────────────────────┘

PHASE 1: INDEXING (One-time setup - prepare your documents)
══════════════════════════════════════════════════════════

   Your Docs              Chunk              Embed              Store
┌─────────────┐      ┌───────────┐      ┌───────────┐      ┌──────────┐
│ terraform/  │      │ Split     │      │ Convert   │      │ ChromaDB │
│ kubernetes/ │ ───► │ into      │ ───► │ to        │ ───► │ Vector   │
│ docker/     │      │ pieces    │      │ vectors   │      │ Database │
└─────────────┘      └───────────┘      └───────────┘      └──────────┘
     │                    │                  │                   │
     │                    │                  │                   │
 "errors.md"         500-char            [0.2, 0.8,         Stored for
 "pods.md"           chunks               -0.3, ...]         fast search
 "docker.md"


PHASE 2: QUERYING (Every time user asks a question)
═══════════════════════════════════════════════════

User Question         Embed Query         Search DB          Top Results
"How to fix          ┌───────────┐      ┌───────────┐      ┌──────────┐
 pod crash?"    ───► │ [0.3,     │ ───► │ Find      │ ───► │ Doc 1    │
                     │  0.7, ...]│      │ similar   │      │ Doc 2    │
                     └───────────┘      │ vectors   │      │ Doc 3    │
                                        └───────────┘      └──────────┘
                                                                │
                                                                │
              ┌─────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        BUILD PROMPT                                  │
├─────────────────────────────────────────────────────────────────────┤
│ System: You are a DevOps expert. Use the following docs to answer.  │
│                                                                      │
│ Context:                                                             │
│   [Doc 1: To fix CrashLoopBackOff, check logs with kubectl logs...] │
│   [Doc 2: Common pod failures include OOMKilled, ImagePullBackOff...│
│   [Doc 3: Container exit codes: 0=success, 1=error, 137=OOMKilled...│
│                                                                      │
│ Question: How to fix pod crash?                                      │
└─────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           LLM                                        │
│                   (OpenAI or Ollama)                                 │
└─────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Answer: "To fix a crashing pod, first check the logs:               │
│         kubectl logs <pod-name>                                      │
│         Common causes include OOMKilled (exit code 137)..."         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why RAG Instead of Fine-Tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Setup time** | Hours | Days to weeks |
| **Cost** | Low (just storage) | High (GPU training) |
| **Update docs** | Easy (re-index) | Hard (retrain model) |
| **Accuracy** | High (cites sources) | Can still hallucinate |
| **Best for** | Facts, documentation | Style, behavior changes |
| **Requires** | Vector DB | Training infrastructure |

**For DevOps troubleshooting:** RAG is the clear winner!

---

## Building Your First RAG Pipeline

Let's build a complete RAG system step by step.

### Step 1: Document Loading

Load your documentation files:

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Load a single markdown file
loader = TextLoader("docs/terraform_errors.md")
documents = loader.load()

# Load ALL markdown files from a directory
loader = DirectoryLoader(
    "docs/",           # Directory path
    glob="**/*.md",    # Pattern: all .md files, including subdirectories
    loader_cls=TextLoader
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
```

### Step 2: Text Chunking

Split documents into smaller pieces (chunks):

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Target size in characters
    chunk_overlap=50,    # Overlap between chunks (maintains context)
    separators=["\n\n", "\n", ". ", " ", ""]  # Split priorities
)

# Split documents into chunks
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")
```

**Why Chunk?**

```
WHY WE NEED CHUNKING:
═════════════════════

Problem: Documents can be thousands of words long
         - LLMs have context limits
         - We want to retrieve SPECIFIC relevant parts

Solution: Split into small, focused chunks

Example:
┌─────────────────────────────────────────────────────────────────────┐
│ Original Document (2000 words):                                      │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ # Terraform Errors Guide                                        │ │
│ │ Introduction... (500 words)                                     │ │
│ │ ## Error: Resource Exists... (500 words)                        │ │
│ │ ## Error: State Lock... (500 words)                             │ │
│ │ ## Error: Provider Issues... (500 words)                        │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│ After Chunking (4 chunks of ~500 chars):                            │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐        │
│ │ Chunk 1    │ │ Chunk 2    │ │ Chunk 3    │ │ Chunk 4    │        │
│ │ Intro...   │ │ Resource   │ │ State      │ │ Provider   │        │
│ │            │ │ Exists...  │ │ Lock...    │ │ Issues...  │        │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘        │
│                                                                      │
│ Now when user asks about "state lock", we only retrieve Chunk 3!    │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 3: Creating Embeddings

Convert text chunks to vectors:

```python
# Option 1: HuggingFace Embeddings (FREE, runs locally)
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
)

# Option 2: OpenAI Embeddings (Paid, higher quality)
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # Cheap and good
)

# Test it out
test_vector = embeddings.embed_query("Kubernetes pod error")
print(f"Vector has {len(test_vector)} dimensions")
# Output: Vector has 384 dimensions
```

### Step 4: Vector Storage with ChromaDB

Store embeddings for fast retrieval:

```python
from langchain_community.vectorstores import Chroma

# Create vector store from documents
vectorstore = Chroma.from_documents(
    documents=chunks,              # Our chunked documents
    embedding=embeddings,          # Embedding model to use
    persist_directory="./chroma_db"  # Where to save (optional)
)

# Search for similar documents
results = vectorstore.similarity_search(
    "How to fix resource already exists error",
    k=3  # Return top 3 matches
)

for doc in results:
    print(f"Found: {doc.page_content[:100]}...")
```

### Step 5: Complete RAG Chain

Put it all together:

```python
# For OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# For Ollama (free, local)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama2", temperature=0.2)

# Create retriever from vector store
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 documents
)

# Create RAG chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # Show which docs were used
)

# Ask a question!
result = qa_chain.invoke({"query": "How do I fix Terraform state lock errors?"})

print("Answer:", result["result"])
print("\nSources used:")
for doc in result["source_documents"]:
    print(f"  - {doc.page_content[:50]}...")
```

---

## Understanding Similarity Search

How does the vector database find relevant documents?

```
SIMILARITY SEARCH EXPLAINED:
════════════════════════════

User Query: "How to fix Terraform state lock"

Step 1: Convert query to vector
        Query Vector: [0.2, 0.8, -0.3, ..., 0.5]

Step 2: Compare with all document vectors in database

        Document Vectors:
        ┌─────────────────────────────────────────────────────────────┐
        │ Doc 1 (Terraform state lock):  [0.21, 0.79, -0.28, ..., 0.48] │
        │ Doc 2 (K8s pod errors):        [0.9, -0.1, 0.6, ..., -0.2]    │
        │ Doc 3 (Terraform import):      [0.15, 0.65, -0.2, ..., 0.4]   │
        │ Doc 4 (Docker networking):     [-0.3, 0.2, 0.8, ..., 0.1]     │
        └─────────────────────────────────────────────────────────────┘

Step 3: Calculate similarity scores (cosine similarity)

        ┌──────────────────────────────────────────────────────────┐
        │ Doc 1: 0.95  ████████████████████  Very similar! ✓      │
        │ Doc 3: 0.72  ██████████████        Related              │
        │ Doc 2: 0.23  █████                 Not related          │
        │ Doc 4: 0.11  ██                    Not related          │
        └──────────────────────────────────────────────────────────┘

Step 4: Return top-k (e.g., top 3) most similar documents
        → Doc 1, Doc 3 are returned
```

---

## Exercises

### Exercise 1: Basic RAG Implementation
Build a complete RAG pipeline from scratch.
→ `exercises/ex1_basic_rag.py`

### Exercise 2: Document Loading
Learn to load different document formats.
→ `exercises/ex2_document_loading.py`

### Exercise 3: Chunking Strategies
Experiment with different chunk sizes and overlaps.
→ `exercises/ex3_chunking_strategies.py`

### Exercise 4: Embeddings Visualization
See how text becomes vectors and understand similarity.
→ `exercises/ex4_embeddings_visualized.py`

### Exercise 5: Similarity Search
Query the vector database and understand results.
→ `exercises/ex5_similarity_search.py`

---

## Key Takeaways

```
┌──────────────────────────────────────────────────────────────────────┐
│                         KEY TAKEAWAYS                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. VECTORS are just lists of numbers                                │
│     - Like coordinates that represent meaning                         │
│                                                                       │
│  2. EMBEDDINGS convert text to vectors                               │
│     - Similar meanings = similar vectors                              │
│                                                                       │
│  3. CHROMADB stores vectors for fast similarity search               │
│     - Finds documents by meaning, not just keywords                   │
│                                                                       │
│  4. RAG = Retrieval + Augmentation + Generation                      │
│     - Find relevant docs → Add to prompt → Generate answer           │
│                                                                       │
│  5. CHUNKING matters                                                  │
│     - Too big = irrelevant content included                          │
│     - Too small = lost context                                        │
│                                                                       │
│  6. RAG REDUCES HALLUCINATION                                        │
│     - Answers are grounded in your actual documents                   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## What's Next?

In **Module 4**, we'll build a comprehensive DevOps knowledge base with:
- Terraform documentation and error guides
- Kubernetes troubleshooting guides
- Docker error references
- CI/CD pipeline debugging tips

```bash
cd ../module-4-knowledge-base
```

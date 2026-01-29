# Module 3: RAG Fundamentals

**Time Required: 2 hours**

Our chatbot works, but it can only use knowledge from its training data. What if we want it to know about OUR specific documentation? Enter RAG - Retrieval-Augmented Generation.

---

## Learning Objectives

By the end of this module, you will:
- Understand why LLMs need external knowledge
- Know what RAG is and how it works
- Implement a basic RAG pipeline
- Understand when to use RAG vs fine-tuning

---

## The Problem with LLMs

### LLMs Have Limitations

```
+------------------------------------------------------------------+
|                    LLM KNOWLEDGE LIMITATIONS                      |
+------------------------------------------------------------------+
|                                                                   |
| 1. KNOWLEDGE CUTOFF                                               |
|    LLM was trained on data up to a certain date.                 |
|    It doesn't know about recent updates or versions.             |
|                                                                   |
| 2. NO ACCESS TO YOUR DOCS                                         |
|    LLM doesn't know your company's runbooks, internal docs,      |
|    or specific configurations.                                    |
|                                                                   |
| 3. HALLUCINATION                                                  |
|    When LLM doesn't know something, it might make things up      |
|    with high confidence.                                          |
|                                                                   |
| 4. GENERIC ANSWERS                                                |
|    Without specific context, answers are general and may not     |
|    apply to your exact situation.                                 |
|                                                                   |
+------------------------------------------------------------------+
```

### Example: Without RAG

```
User: "How do I fix the 'TF_PLUGIN_CACHE_DIR' error in Terraform 1.7?"

LLM (without RAG): "I'm not sure about Terraform 1.7 specifically,
     but generally you can set the plugin cache directory by..."

     ^^^ Vague, potentially outdated or wrong
```

### Example: With RAG

```
User: "How do I fix the 'TF_PLUGIN_CACHE_DIR' error in Terraform 1.7?"

[RAG retrieves relevant docs about Terraform 1.7 plugin cache]

LLM (with RAG): "In Terraform 1.7, the TF_PLUGIN_CACHE_DIR error
     typically occurs when... According to the Terraform 1.7
     release notes, you should..."

     ^^^ Accurate, specific, references actual documentation
```

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

It's a technique that:
1. **Retrieves** relevant documents based on the user's question
2. **Augments** the LLM prompt with this retrieved context
3. **Generates** a response using both the question AND the context

```
+------------------------------------------------------------------+
|                        THE RAG PIPELINE                           |
+------------------------------------------------------------------+

                     INDEXING PHASE (One-time setup)
+------------------------------------------------------------------+

  DevOps Docs          Chunking            Embedding         Vector DB
+-------------+      +-----------+      +------------+      +--------+
| Terraform   |      | Split     |      | Convert to |      | Store  |
| K8s errors  | ---> | into      | ---> | vectors    | ---> | in     |
| Docker tips |      | chunks    |      | [0.2, 0.8] |      | ChromaDB
+-------------+      +-----------+      +------------+      +--------+


                     QUERY PHASE (Every question)
+------------------------------------------------------------------+

User Question        Embed Query         Search           Top K Docs
"How to fix         [0.3, 0.7...]       Find similar     [Doc 1]
 pod crash?"   ---> (vector)      ---> vectors     ---> [Doc 2]
                                                         [Doc 3]
       |                                                     |
       |              +-------------------------+            |
       +----------->  |    Combined Prompt      | <----------+
                      | Question + Context Docs |
                      +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      |          LLM            |
                      |   Generates response    |
                      |   using the context     |
                      +------------+------------+
                                   |
                                   v
                      +-------------------------+
                      |   Informed Answer       |
                      |   "To fix pod crash,    |
                      |    based on the docs..."  |
                      +-------------------------+
```

---

## Why RAG Instead of Fine-Tuning?

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Setup time** | Hours | Days to weeks |
| **Cost** | Low (just storage) | High (GPU training) |
| **Update docs** | Easy (re-index) | Hard (retrain) |
| **Accuracy** | High (cites sources) | Can still hallucinate |
| **Best for** | Facts, documentation | Style, behavior |

**For DevOps troubleshooting:** RAG is the clear winner!

---

## Building Your First RAG Pipeline

### Step 1: Document Loading

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader

# Load a single file
loader = TextLoader("docs/terraform_errors.md")
documents = loader.load()

# Load all files from a directory
loader = DirectoryLoader("docs/", glob="**/*.md")
documents = loader.load()
```

### Step 2: Text Chunking

Why chunk? LLMs have context limits, and we want to retrieve specific, relevant pieces.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50,    # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]  # Split priorities
)

chunks = splitter.split_documents(documents)
```

**Chunking strategies:**

```
Original Document (1500 chars):
+------------------------------------------------------------------+
| # Terraform Error: Resource Already Exists                        |
|                                                                   |
| This error occurs when Terraform tries to create a resource       |
| that already exists in the cloud provider but is not in the       |
| Terraform state...                                                |
|                                                                   |
| ## Common Causes                                                  |
| 1. Resource was created manually                                  |
| 2. State file was deleted                                         |
| 3. Import was incomplete                                          |
|                                                                   |
| ## Solutions                                                      |
| To fix this error, you have several options...                    |
+------------------------------------------------------------------+

After Chunking (3 chunks of ~500 chars each):
+--------------------+  +--------------------+  +--------------------+
| Chunk 1            |  | Chunk 2            |  | Chunk 3            |
| # Terraform Error  |  | ## Common Causes   |  | ## Solutions       |
| This error...      |  | 1. Resource was... |  | To fix this...     |
+--------------------+  +--------------------+  +--------------------+
```

### Step 3: Creating Embeddings

Embeddings convert text to vectors for similarity search:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

# Free, local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Convert text to vector
vector = embeddings.embed_query("Terraform state error")
# Returns: [0.12, -0.45, 0.78, ..., 0.33] (384 dimensions)
```

### Step 4: Vector Storage

Store embeddings for fast retrieval:

```python
from langchain_community.vectorstores import Chroma

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Search for similar documents
results = vectorstore.similarity_search(
    "How to fix resource already exists error",
    k=3  # Return top 3 matches
)
```

### Step 5: RAG Chain

Put it all together:

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Create LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
result = qa_chain.invoke({"query": "How do I fix Terraform state errors?"})
print(result["result"])
```

---

## Exercise 1: Basic RAG Implementation

Create `exercises/ex1_basic_rag.py`:

```python
"""
Exercise 1: Build Your First RAG Pipeline
==========================================
Goal: Create a simple RAG system for DevOps documentation
"""

import os
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()


def create_sample_docs():
    """Create sample DevOps documentation for testing."""
    docs_content = """
# Terraform Common Errors

## Error: Resource Already Exists

This error occurs when Terraform tries to create a resource that already
exists in your cloud provider but is not tracked in Terraform state.

### Causes
1. Resource was created manually via console
2. State file was deleted or corrupted
3. Another Terraform workspace created the resource

### Solutions
1. Import the existing resource:
   ```
   terraform import aws_instance.example i-1234567890abcdef0
   ```
2. Remove from state if duplicate:
   ```
   terraform state rm aws_instance.example
   ```
3. Use data source instead of resource if read-only access needed

## Error: State Lock

This error occurs when another Terraform process holds the state lock.

### Causes
1. Previous terraform apply was interrupted
2. Another team member is running terraform
3. CI/CD pipeline is running

### Solutions
1. Wait for other process to complete
2. Force unlock (use with caution):
   ```
   terraform force-unlock LOCK_ID
   ```
"""
    # Save to file
    os.makedirs("temp_docs", exist_ok=True)
    with open("temp_docs/terraform_errors.md", "w") as f:
        f.write(docs_content)

    return "temp_docs/terraform_errors.md"


class SimpleRAG:
    """A simple RAG implementation."""

    def __init__(self, docs_path: str):
        """
        Initialize the RAG system.

        Args:
            docs_path: Path to documentation file or directory
        """
        # TODO: Implement initialization
        # 1. Load documents
        # 2. Split into chunks
        # 3. Create embeddings
        # 4. Create vector store
        pass

    def query(self, question: str) -> str:
        """
        Query the RAG system.

        Args:
            question: User's question

        Returns:
            Answer based on retrieved documents
        """
        # TODO: Implement query
        # 1. Retrieve relevant documents
        # 2. Build prompt with context
        # 3. Call LLM
        # 4. Return answer
        pass


if __name__ == "__main__":
    # Create sample docs
    docs_path = create_sample_docs()

    # Initialize RAG
    print("Initializing RAG system...")
    rag = SimpleRAG(docs_path)

    # Test queries
    questions = [
        "How do I fix 'resource already exists' error in Terraform?",
        "What causes state lock errors?",
        "How do I import an existing resource into Terraform?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer = rag.query(q)
        print(f"Answer: {answer}")
```

---

## Exercise 2: Custom Prompts for RAG

The prompt template affects response quality:

```python
"""
Exercise 2: Custom RAG Prompts
==============================
Goal: Improve response quality with better prompts
"""

from langchain.prompts import PromptTemplate

# Basic prompt (often produces generic answers)
BASIC_PROMPT = """Answer the question based on the context below.

Context: {context}

Question: {question}

Answer:"""

# DevOps-optimized prompt
DEVOPS_PROMPT = """You are an expert DevOps engineer helping troubleshoot issues.

Use the following documentation to answer the question. If the documentation
doesn't contain the answer, say "I don't have information about that in my
documentation" - don't make up answers.

Documentation:
{context}

User's Question: {question}

Provide a helpful answer that:
1. Directly addresses the question
2. Includes specific commands or code when applicable
3. Explains WHY the solution works
4. Mentions any caveats or warnings

Answer:"""

# TODO: Create your own improved prompt template
YOUR_PROMPT = """
"""

# Use with LangChain
prompt_template = PromptTemplate(
    template=DEVOPS_PROMPT,
    input_variables=["context", "question"]
)
```

---

## Understanding Similarity Search

How does the vector database find relevant documents?

```
Query: "How to fix Terraform state lock"

Query Vector: [0.2, 0.8, -0.3, ..., 0.5]

Document Vectors in DB:
  Doc 1 (Terraform state lock): [0.21, 0.79, -0.28, ..., 0.48]  <- Similar! ✓
  Doc 2 (K8s pod errors):       [0.9, -0.1, 0.6, ..., -0.2]    <- Different
  Doc 3 (Terraform import):     [0.15, 0.65, -0.2, ..., 0.4]   <- Somewhat similar

Similarity Scores (cosine similarity):
  Doc 1: 0.95 (very similar)
  Doc 3: 0.72 (related)
  Doc 2: 0.23 (not related)

Return: Doc 1 and Doc 3
```

---

## Key Takeaways

1. **RAG = Retrieval + Generation** - Combine document search with LLM
2. **Chunking matters** - Too big = irrelevant content; too small = lost context
3. **Embeddings enable semantic search** - Find meaning, not just keywords
4. **Prompts shape output** - Design prompts for your specific use case
5. **RAG reduces hallucination** - LLM answers are grounded in real documents

---

## What's Next?

In **Module 4**, we'll build a comprehensive DevOps knowledge base with:
- Terraform documentation
- Kubernetes troubleshooting guides
- Docker error references
- CI/CD pipeline debugging tips

```bash
cd ../module-4-knowledge-base
```

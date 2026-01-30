#!/usr/bin/env python3
"""
Exercise 1: Build Your First RAG Pipeline
==========================================

GOAL: Create a complete RAG (Retrieval-Augmented Generation) system
      that can answer questions using your own documentation.

WHAT YOU'LL LEARN:
- How to load documents into a RAG system
- How to split documents into chunks
- How to create embeddings (text → vectors)
- How to store vectors in ChromaDB
- How to query the RAG system

WHAT IS RAG?
-----------
RAG combines document retrieval with AI text generation:

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   User      │     │  Find       │     │   LLM       │
    │  Question   │ ──► │  Relevant   │ ──► │  Generates  │
    │             │     │  Documents  │     │  Answer     │
    └─────────────┘     └─────────────┘     └─────────────┘
                              │
                              ▼
                    Your Documentation
                    (Terraform, K8s, etc.)

Instead of relying on the LLM's training data, we give it
YOUR specific documentation to answer from!

REQUIREMENTS:
    pip install chromadb langchain langchain-community
    pip install sentence-transformers  # For free local embeddings
    pip install langchain-openai       # If using OpenAI
    pip install langchain-ollama       # If using Ollama

RUN THIS:
    python module-3-rag-fundamentals/exercises/ex1_basic_rag.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# LANGCHAIN IMPORTS - Each one has a specific job
# =============================================================================
# Document loading - reads files from disk
from langchain_community.document_loaders import TextLoader

# Text splitting - breaks documents into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings - converts text to vectors (numbers)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector storage - stores and searches vectors
from langchain_community.vectorstores import Chroma

# Chains - combines components into a workflow
from langchain.chains import RetrievalQA


def create_sample_documentation():
    """
    Create sample DevOps documentation for testing.

    In a real project, you would have actual documentation files.
    For this exercise, we create sample content.

    Returns:
        Path to the created documentation file
    """
    # Create a temporary directory for our docs
    docs_dir = Path(__file__).parent / "temp_docs"
    docs_dir.mkdir(exist_ok=True)

    # Sample Terraform documentation
    terraform_docs = """# Terraform Common Errors and Solutions

## Error: Resource Already Exists

### Description
This error occurs when Terraform tries to create a resource that already
exists in your cloud provider but is not tracked in Terraform state.

### Error Message
```
Error: A resource with the ID "/subscriptions/.../myresource" already exists
```

### Common Causes
1. **Manual Creation**: Resource was created manually via cloud console
2. **Lost State**: State file was deleted or corrupted
3. **Import Incomplete**: Previous import operation didn't complete
4. **Multiple Workspaces**: Another workspace created the resource

### Solutions

#### Solution 1: Import the Existing Resource
If you want Terraform to manage the existing resource:
```bash
# Find the resource ID in your cloud console
terraform import aws_instance.example i-1234567890abcdef0
```

#### Solution 2: Remove from State
If the resource in state is a duplicate:
```bash
terraform state rm aws_instance.example
```

#### Solution 3: Use Data Source
If you only need to read the resource (not manage it):
```hcl
data "aws_instance" "existing" {
  instance_id = "i-1234567890abcdef0"
}
```

---

## Error: State Lock

### Description
This error occurs when another Terraform process holds the state lock,
preventing concurrent modifications.

### Error Message
```
Error: Error locking state: Error acquiring the state lock
Lock Info:
  ID:        12345678-1234-1234-1234-123456789012
  Path:      terraform.tfstate
  Operation: OperationTypePlan
```

### Common Causes
1. **Interrupted Process**: Previous terraform command was interrupted (Ctrl+C)
2. **Concurrent Runs**: Another team member or CI/CD pipeline is running
3. **Stale Lock**: Lock wasn't released due to crash

### Solutions

#### Solution 1: Wait and Retry
If someone else is running Terraform, wait for them to finish.

#### Solution 2: Force Unlock (Use with Caution!)
Only use this if you're SURE no one else is running Terraform:
```bash
terraform force-unlock 12345678-1234-1234-1234-123456789012
```

#### Solution 3: Check CI/CD Pipelines
Verify no automated pipelines are currently running.

---

## Error: Provider Configuration Not Found

### Description
Terraform cannot find the required provider configuration.

### Error Message
```
Error: Provider configuration not found
To work with aws_instance.example, the provider must be configured.
```

### Solutions
Add provider configuration to your Terraform files:
```hcl
provider "aws" {
  region = "us-west-2"
}
```

Or set environment variables:
```bash
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```
"""

    # Write the documentation
    docs_path = docs_dir / "terraform_errors.md"
    docs_path.write_text(terraform_docs)

    print(f"✅ Created sample documentation at: {docs_path}")
    return str(docs_path)


class SimpleRAG:
    """
    A simple RAG (Retrieval-Augmented Generation) system.

    This class demonstrates the core components of RAG:
    1. Document loading and chunking
    2. Embedding creation (text → vectors)
    3. Vector storage (ChromaDB)
    4. Retrieval and generation

    Attributes:
        vectorstore: The ChromaDB vector store
        embeddings: The embedding model
        llm: The language model for generation

    Example:
        >>> rag = SimpleRAG("path/to/docs.md")
        >>> answer = rag.query("How do I fix state lock errors?")
        >>> print(answer)
    """

    def __init__(self, docs_path: str):
        """
        Initialize the RAG system.

        This method:
        1. Loads the document
        2. Splits it into chunks
        3. Creates embeddings
        4. Stores in vector database

        Args:
            docs_path: Path to the documentation file
        """
        print("\n" + "=" * 60)
        print("INITIALIZING RAG SYSTEM")
        print("=" * 60)

        # =====================================================================
        # STEP 1: LOAD DOCUMENTS
        # =====================================================================
        # TextLoader reads a text/markdown file into memory
        print("\n📄 Step 1: Loading documents...")
        loader = TextLoader(docs_path, encoding='utf-8')
        documents = loader.load()
        print(f"   Loaded {len(documents)} document(s)")
        print(f"   Total characters: {len(documents[0].page_content)}")

        # =====================================================================
        # STEP 2: SPLIT INTO CHUNKS
        # =====================================================================
        # Why chunk?
        # - LLMs have context limits
        # - We want to retrieve SPECIFIC relevant parts, not whole documents
        # - Smaller chunks = more precise retrieval
        print("\n✂️  Step 2: Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Target ~500 characters per chunk
            chunk_overlap=50,    # 50 char overlap to maintain context
            separators=["\n\n", "\n", ". ", " ", ""]  # Split at these first
        )
        chunks = splitter.split_documents(documents)
        print(f"   Created {len(chunks)} chunks")
        print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

        # =====================================================================
        # STEP 3: CREATE EMBEDDINGS
        # =====================================================================
        # Embeddings convert text to vectors (lists of numbers)
        # Similar text = similar vectors
        # We use HuggingFace embeddings (free, runs locally)
        print("\n🔢 Step 3: Creating embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions
        )
        print("   Using model: all-MiniLM-L6-v2")
        print("   Vector dimensions: 384")

        # =====================================================================
        # STEP 4: CREATE VECTOR STORE
        # =====================================================================
        # ChromaDB stores our vectors for fast similarity search
        print("\n💾 Step 4: Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            # persist_directory="./chroma_db"  # Uncomment to save to disk
        )
        print(f"   Stored {len(chunks)} vectors in ChromaDB")

        # =====================================================================
        # STEP 5: SET UP LLM
        # =====================================================================
        print("\n🤖 Step 5: Setting up LLM...")
        self._setup_llm()

        print("\n" + "=" * 60)
        print("✅ RAG SYSTEM READY!")
        print("=" * 60)

    def _setup_llm(self):
        """
        Set up the LLM based on environment configuration.

        Supports:
        - OpenAI (paid, cloud)
        - Ollama (free, local)
        """
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        if use_ollama:
            # Ollama - free, runs locally
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "llama2"),
                    temperature=0.2
                )
                print("   Using: Ollama (local)")
            except ImportError:
                print("   ⚠️  langchain-ollama not installed")
                print("   Run: pip install langchain-ollama")
                self.llm = None
        else:
            # OpenAI - paid, cloud
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=0.2
                )
                print("   Using: OpenAI (cloud)")
            except ImportError:
                print("   ⚠️  langchain-openai not installed")
                print("   Run: pip install langchain-openai")
                self.llm = None

    def query(self, question: str) -> dict:
        """
        Query the RAG system with a question.

        This method:
        1. Converts question to a vector
        2. Finds similar document chunks
        3. Sends chunks + question to LLM
        4. Returns the answer

        Args:
            question: The user's question

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        print(f"\n🔍 Querying: {question}")

        # =====================================================================
        # RETRIEVE RELEVANT DOCUMENTS
        # =====================================================================
        # similarity_search finds chunks most similar to our question
        print("\n   📚 Finding relevant documents...")
        relevant_docs = self.vectorstore.similarity_search(question, k=3)

        print(f"   Found {len(relevant_docs)} relevant chunks:")
        for i, doc in enumerate(relevant_docs, 1):
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"   {i}. {preview}...")

        # =====================================================================
        # BUILD CONTEXT FROM RETRIEVED DOCUMENTS
        # =====================================================================
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

        # =====================================================================
        # GENERATE ANSWER
        # =====================================================================
        if self.llm is None:
            return {
                "answer": "LLM not configured. Please set up OpenAI or Ollama.",
                "sources": relevant_docs,
                "context": context
            }

        # Build the prompt with retrieved context
        prompt = f"""You are a helpful DevOps assistant. Answer the question based ONLY on the
following documentation. If the documentation doesn't contain the answer, say
"I don't have information about that in my documentation."

Documentation:
{context}

Question: {question}

Answer:"""

        print("\n   🤖 Generating answer...")
        response = self.llm.invoke(prompt)

        # Extract answer text
        answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer,
            "sources": relevant_docs,
            "context": context
        }

    def interactive_chat(self):
        """
        Run an interactive chat session.

        Users can ask questions and get answers from the RAG system.
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE RAG CHAT")
        print("=" * 60)
        print("Ask questions about Terraform errors!")
        print("Type 'quit' to exit, 'sources' to see last sources.\n")

        last_sources = []

        while True:
            user_input = input("\n❓ Your question: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'sources':
                if last_sources:
                    print("\n📚 Sources from last query:")
                    for i, doc in enumerate(last_sources, 1):
                        print(f"\n--- Source {i} ---")
                        print(doc.page_content[:300])
                else:
                    print("No sources yet. Ask a question first!")
                continue
            elif not user_input:
                continue

            result = self.query(user_input)
            print(f"\n💡 Answer:\n{result['answer']}")
            last_sources = result['sources']


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE 1: Basic RAG Pipeline")
    print("=" * 60)

    # Create sample documentation
    docs_path = create_sample_documentation()

    # Initialize RAG system
    rag = SimpleRAG(docs_path)

    # Test with some questions
    print("\n" + "=" * 60)
    print("TESTING THE RAG SYSTEM")
    print("=" * 60)

    test_questions = [
        "How do I fix the 'resource already exists' error in Terraform?",
        "What causes state lock errors?",
        "How do I import an existing resource into Terraform state?",
    ]

    for question in test_questions:
        result = rag.query(question)
        print(f"\n{'─' * 60}")
        print(f"💡 Answer:\n{result['answer']}")

    # Interactive mode
    print("\n" + "=" * 60)
    choice = input("\nWould you like to try interactive mode? (y/n): ")
    if choice.lower() == 'y':
        rag.interactive_chat()

    print("""
=" * 60
WHAT YOU LEARNED:
=" * 60

1. DOCUMENT LOADING
   - TextLoader reads files into memory

2. CHUNKING
   - Split documents into smaller pieces for precise retrieval

3. EMBEDDINGS
   - Convert text to vectors using HuggingFace models (free!)

4. VECTOR STORAGE
   - ChromaDB stores vectors for fast similarity search

5. RAG QUERY FLOW
   Question → Find similar docs → Add to prompt → Generate answer

NEXT STEPS:
- Try ex2_document_loading.py to learn about different file formats
- Try ex3_chunking_strategies.py to experiment with chunk sizes
""")

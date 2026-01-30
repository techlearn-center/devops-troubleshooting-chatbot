#!/usr/bin/env python3
"""
SOLUTION: Exercise 1 - Basic RAG Pipeline
==========================================

This is the complete, production-ready RAG implementation.

FEATURES:
- Document loading and chunking
- Embedding creation with HuggingFace (free) or OpenAI
- ChromaDB vector storage with persistence
- Support for both OpenAI and Ollama LLMs
- Source tracking in responses
- Interactive chat mode

RUN THIS:
    python module-3-rag-fundamentals/solutions/ex1_basic_rag.py

    # With Ollama (free, local)
    USE_OLLAMA=true python module-3-rag-fundamentals/solutions/ex1_basic_rag.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate


class DevOpsRAG:
    """
    Production-ready RAG system for DevOps documentation.

    Features:
    - Configurable embedding models
    - Persistent vector storage
    - OpenAI or Ollama LLM support
    - Source tracking
    - Custom prompt templates
    """

    # System prompt for DevOps expertise
    SYSTEM_TEMPLATE = """You are an expert DevOps engineer helping troubleshoot issues.

Use ONLY the following documentation to answer the question. If the documentation
doesn't contain enough information to fully answer, say what you CAN answer based
on the docs and note what additional information would help.

Documentation:
{context}

Question: {question}

Instructions:
1. Answer based on the documentation provided
2. Include specific commands or code when relevant
3. Explain WHY the solution works
4. If multiple solutions exist, list them in order of preference

Answer:"""

    def __init__(
        self,
        docs_path: str = None,
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        """
        Initialize the RAG system.

        Args:
            docs_path: Path to documentation file or directory
            persist_directory: Where to save ChromaDB data
            embedding_model: HuggingFace embedding model name
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embeddings
        print("📦 Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Initialize or load vector store
        if docs_path:
            self._create_vectorstore(docs_path)
        else:
            self._load_existing_vectorstore()

        # Initialize LLM
        self._setup_llm()

        # Create prompt template
        self.prompt_template = PromptTemplate(
            template=self.SYSTEM_TEMPLATE,
            input_variables=["context", "question"]
        )

    def _create_vectorstore(self, docs_path: str):
        """Load documents and create vector store."""
        print(f"📄 Loading documents from: {docs_path}")

        # Determine if path is file or directory
        path = Path(docs_path)
        if path.is_file():
            loader = TextLoader(str(path), encoding='utf-8')
            documents = loader.load()
        elif path.is_dir():
            loader = DirectoryLoader(
                str(path),
                glob="**/*.md",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
        else:
            raise ValueError(f"Invalid path: {docs_path}")

        print(f"   Loaded {len(documents)} document(s)")

        # Split into chunks
        print("✂️  Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"   Created {len(chunks)} chunks")

        # Create vector store
        print("💾 Creating vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"   ✅ Vector store created and persisted")

    def _load_existing_vectorstore(self):
        """Load existing vector store from disk."""
        print(f"💾 Loading existing vector store from: {self.persist_directory}")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )

    def _setup_llm(self):
        """Set up the LLM based on configuration."""
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        if use_ollama:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "llama2"),
                    temperature=0.2
                )
                print("🤖 Using Ollama (local)")
            except ImportError:
                print("⚠️  langchain-ollama not installed")
                self.llm = None
        else:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=0.2
                )
                print("🤖 Using OpenAI")
            except ImportError:
                print("⚠️  langchain-openai not installed")
                self.llm = None

    def query(self, question: str, k: int = 4) -> dict:
        """
        Query the RAG system.

        Args:
            question: User's question
            k: Number of documents to retrieve

        Returns:
            Dictionary with answer, sources, and context
        """
        # Retrieve relevant documents
        docs = self.vectorstore.similarity_search(question, k=k)

        # Build context from documents
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])

        # Generate answer
        if self.llm is None:
            return {
                "answer": "LLM not configured. Retrieved documents shown below.",
                "sources": docs,
                "context": context
            }

        prompt = self.prompt_template.format(context=context, question=question)
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, 'content') else str(response)

        return {
            "answer": answer,
            "sources": docs,
            "context": context
        }

    def add_documents(self, docs_path: str):
        """Add more documents to the existing vector store."""
        path = Path(docs_path)
        if path.is_file():
            loader = TextLoader(str(path), encoding='utf-8')
        else:
            loader = DirectoryLoader(str(path), glob="**/*.md", loader_cls=TextLoader)

        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore.add_documents(chunks)
        print(f"✅ Added {len(chunks)} chunks to vector store")

    def interactive_chat(self):
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("RAG INTERACTIVE CHAT")
        print("=" * 60)
        print("Commands: 'quit', 'sources' (show last sources)")

        last_sources = []

        while True:
            question = input("\n❓ You: ").strip()

            if question.lower() == 'quit':
                print("Goodbye!")
                break
            elif question.lower() == 'sources':
                if last_sources:
                    print("\n📚 Sources from last query:")
                    for i, doc in enumerate(last_sources, 1):
                        source = doc.metadata.get('source', 'unknown')
                        print(f"\n[{i}] {source}")
                        print(doc.page_content[:200] + "...")
                continue
            elif not question:
                continue

            result = self.query(question)
            print(f"\n💡 Answer:\n{result['answer']}")
            last_sources = result['sources']

            # Show sources
            print("\n📚 Based on:")
            for doc in last_sources[:2]:
                source = doc.metadata.get('source', 'unknown')
                print(f"   - {source}")


def create_sample_docs():
    """Create sample documentation for testing."""
    docs_dir = Path(__file__).parent / "sample_docs"
    docs_dir.mkdir(exist_ok=True)

    # Terraform docs
    (docs_dir / "terraform.md").write_text("""# Terraform Troubleshooting

## State Lock Error
Error acquiring the state lock means another process is running.

### Solutions
1. Wait for other process
2. Force unlock: `terraform force-unlock <LOCK_ID>`

## Resource Already Exists
The resource exists but isn't in state.

### Solutions
1. Import: `terraform import <type>.<name> <id>`
2. Remove from state: `terraform state rm <resource>`
""")

    # Kubernetes docs
    (docs_dir / "kubernetes.md").write_text("""# Kubernetes Troubleshooting

## CrashLoopBackOff
Container keeps crashing and restarting.

### Debugging
```bash
kubectl logs <pod> --previous
kubectl describe pod <pod>
```

### Common Causes
- Missing environment variables
- Application error on startup
- OOMKilled (out of memory)

## ImagePullBackOff
Cannot pull container image.

### Causes
- Wrong image name/tag
- Private registry without auth
- Network issues
""")

    return str(docs_dir)


if __name__ == "__main__":
    # Create sample docs
    docs_path = create_sample_docs()

    # Initialize RAG
    rag = DevOpsRAG(docs_path=docs_path)

    # Test queries
    print("\n" + "=" * 60)
    print("TESTING RAG SYSTEM")
    print("=" * 60)

    questions = [
        "How do I fix state lock errors in Terraform?",
        "My pod shows CrashLoopBackOff, what should I check?",
        "How do I import existing resources into Terraform?",
    ]

    for q in questions:
        print(f"\n❓ {q}")
        result = rag.query(q)
        print(f"\n💡 {result['answer'][:300]}...")

    # Interactive mode
    choice = input("\n\nStart interactive chat? (y/n): ")
    if choice.lower() == 'y':
        rag.interactive_chat()

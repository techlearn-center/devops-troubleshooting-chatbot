#!/usr/bin/env python3
"""
SOLUTION: Exercise 4 - Knowledge Base Query
============================================

RAG query engine that retrieves relevant docs from the knowledge base
and generates answers using an LLM. Supports OpenAI and Ollama.

RUN THIS:
    # Build KB + run test queries (OpenAI)
    python module-4-knowledge-base/solutions/ex4_kb_query.py

    # With Ollama (free, local)
    USE_OLLAMA=true python module-4-knowledge-base/solutions/ex4_kb_query.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate


class KBQueryEngine:
    """
    RAG-powered query engine over the DevOps knowledge base.

    Pipeline:
        1. Build / load a ChromaDB vector store from knowledge-base/
        2. Retrieve relevant chunks for a user question
        3. Augment a prompt with retrieved context
        4. Generate an answer with an LLM (OpenAI or Ollama)
    """

    PROMPT_TEMPLATE = """You are an expert DevOps engineer.
Use ONLY the following documentation to answer the question.
If the docs are insufficient, say so and suggest what to check.

Documentation:
{context}

Question: {question}

Provide a concise, actionable answer with commands where relevant."""

    def __init__(
        self,
        kb_path: str = None,
        persist_directory: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        project_root = Path(__file__).parent.parent.parent

        self.kb_path = Path(kb_path) if kb_path else project_root / "knowledge-base"
        self.persist_directory = persist_directory or str(project_root / "data" / "chroma_query")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embeddings
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        # Vectorstore (populated by build_kb)
        self.vectorstore = None

        # LLM
        self.llm = None
        self._setup_llm()

        # Prompt
        self.prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )

    # ------------------------------------------------------------------
    # LLM setup
    # ------------------------------------------------------------------

    def _setup_llm(self):
        """Initialise OpenAI or Ollama based on USE_OLLAMA env var."""
        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        if use_ollama:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=os.getenv("OLLAMA_MODEL", "llama2"),
                    temperature=0.2,
                )
                print("Using Ollama (local)")
            except ImportError:
                print("langchain-ollama not installed -- LLM disabled")
        else:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    temperature=0.2,
                )
                print("Using OpenAI")
            except ImportError:
                print("langchain-openai not installed -- LLM disabled")

    # ------------------------------------------------------------------
    # Build the knowledge base
    # ------------------------------------------------------------------

    def build_kb(self) -> int:
        """Load, chunk, embed, and store the knowledge base. Returns chunk count."""
        documents = []
        for md_file in sorted(self.kb_path.rglob("*.md")):
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                docs = loader.load()
                category = md_file.parent.name
                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["topic"] = md_file.stem
                    doc.metadata["filename"] = md_file.name
                documents.extend(docs)
            except Exception as e:
                print(f"  Skipping {md_file}: {e}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"KB built: {len(chunks)} chunks from {len(documents)} documents")
        return len(chunks)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, k: int = 4) -> dict:
        """
        Retrieve relevant docs and generate an LLM answer.

        Returns:
            dict with keys: answer, sources, context
        """
        if self.vectorstore is None:
            return {"answer": "Knowledge base not built. Call build_kb() first.",
                    "sources": [], "context": ""}

        # Retrieve
        docs = self.vectorstore.similarity_search(question, k=k)

        # Build context string
        context = "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('category','?')}/{d.metadata.get('filename','?')}]\n{d.page_content}"
            for d in docs
        )

        # Generate answer
        if self.llm is None:
            answer = "(LLM not configured -- showing retrieved context only)\n\n" + context
        else:
            formatted = self.prompt.format(context=context, question=question)
            response = self.llm.invoke(formatted)
            answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer, "sources": docs, "context": context}

    # ------------------------------------------------------------------
    # Interactive chat
    # ------------------------------------------------------------------

    def interactive_chat(self):
        """REPL loop: ask questions, see answers and sources."""
        print("\n" + "=" * 60)
        print("INTERACTIVE KB CHAT")
        print("=" * 60)
        print("Type a question, or 'quit' to exit.\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            result = self.query(question)

            print(f"\nAnswer:\n{result['answer']}\n")

            # Show sources
            print("Sources:")
            for doc in result["sources"]:
                cat = doc.metadata.get("category", "?")
                fname = doc.metadata.get("filename", "?")
                print(f"  - {cat}/{fname}")
            print()


if __name__ == "__main__":
    engine = KBQueryEngine()
    engine.build_kb()

    # Test queries
    test_questions = [
        "How do I fix a CrashLoopBackOff in Kubernetes?",
        "Terraform state lock error - how to resolve?",
        "Docker build fails with COPY file not found",
        "GitHub Actions permission denied error",
    ]

    print("\n" + "=" * 60)
    print("TEST QUERIES")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQ: {q}")
        result = engine.query(q)
        # Show first 300 chars of the answer
        preview = result["answer"][:300]
        print(f"A: {preview}{'...' if len(result['answer']) > 300 else ''}")
        cats = {d.metadata.get("category") for d in result["sources"]}
        print(f"   (sources: {', '.join(cats)})")

    # Offer interactive mode
    choice = input("\n\nStart interactive chat? (y/n): ").strip().lower()
    if choice == "y":
        engine.interactive_chat()

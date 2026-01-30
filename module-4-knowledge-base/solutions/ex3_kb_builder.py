#!/usr/bin/env python3
"""
SOLUTION: Exercise 3 - Knowledge Base Builder
==============================================

Full pipeline that loads markdown docs from knowledge-base/,
chunks them with RecursiveCharacterTextSplitter, embeds with
HuggingFace, and persists in ChromaDB.

RUN THIS:
    python module-4-knowledge-base/solutions/ex3_kb_builder.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class KnowledgeBaseBuilder:
    """
    End-to-end pipeline: load -> chunk -> index -> save.
    Produces a persistent ChromaDB vector store.
    """

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
        self.persist_directory = persist_directory or str(project_root / "data" / "chroma_kb")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Raw and processed data
        self.raw_documents = []
        self.chunks = []
        self.vectorstore = None

        # Embeddings (loaded lazily on first index)
        self._embedding_model_name = embedding_model
        self._embeddings = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            print("Loading embedding model...")
            self._embeddings = HuggingFaceEmbeddings(model_name=self._embedding_model_name)
        return self._embeddings

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load(self) -> "KnowledgeBaseBuilder":
        """Step 1: Walk knowledge-base/ and load .md files with metadata."""
        self.raw_documents = []
        print(f"Loading from: {self.kb_path}")

        for md_file in sorted(self.kb_path.rglob("*.md")):
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                docs = loader.load()

                category = md_file.parent.name

                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["topic"] = md_file.stem
                    doc.metadata["filename"] = md_file.name

                self.raw_documents.extend(docs)
                print(f"  Loaded: {category}/{md_file.name}")
            except Exception as e:
                print(f"  Error loading {md_file}: {e}")

        print(f"Loaded {len(self.raw_documents)} raw document(s)")
        return self

    def chunk(self) -> "KnowledgeBaseBuilder":
        """Step 2: Split documents into chunks suitable for embedding."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self.chunks = splitter.split_documents(self.raw_documents)
        print(f"Created {len(self.chunks)} chunks from {len(self.raw_documents)} document(s)")
        return self

    def index(self) -> "KnowledgeBaseBuilder":
        """Step 3: Embed chunks and store in ChromaDB."""
        print("Indexing into ChromaDB (this may take a moment)...")
        self.vectorstore = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
        )
        print(f"Indexed {len(self.chunks)} chunks")
        return self

    def save(self) -> "KnowledgeBaseBuilder":
        """Step 4: Persist the vector store to disk."""
        if self.vectorstore is not None:
            # Chroma auto-persists when persist_directory is set,
            # but we call persist() explicitly for older versions.
            if hasattr(self.vectorstore, "persist"):
                self.vectorstore.persist()
            print(f"Saved to: {self.persist_directory}")
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def build(self) -> "KnowledgeBaseBuilder":
        """Run the full pipeline: load -> chunk -> index -> save."""
        return self.load().chunk().index().save()

    def build_stats(self) -> dict:
        """Return stats about the built knowledge base."""
        cat_chunks = defaultdict(int)
        for c in self.chunks:
            cat_chunks[c.metadata.get("category", "unknown")] += 1

        return {
            "raw_documents": len(self.raw_documents),
            "total_chunks": len(self.chunks),
            "chunks_by_category": dict(sorted(cat_chunks.items())),
            "persist_directory": self.persist_directory,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    builder.build()

    # Print build summary
    stats = builder.build_stats()

    print("\n" + "=" * 55)
    print("KNOWLEDGE BASE BUILD SUMMARY")
    print("=" * 55)
    print(f"  Raw documents:    {stats['raw_documents']}")
    print(f"  Total chunks:     {stats['total_chunks']}")
    print(f"  Chunk size:       {stats['chunk_size']}")
    print(f"  Chunk overlap:    {stats['chunk_overlap']}")
    print(f"  Persist dir:      {stats['persist_directory']}")
    print(f"\n  Chunks by category:")
    for cat, count in stats["chunks_by_category"].items():
        print(f"    {cat}: {count}")

    # Quick smoke test: search for something
    if builder.vectorstore:
        print("\n" + "-" * 55)
        print("QUICK SEARCH TEST")
        print("-" * 55)
        query = "pod keeps crashing"
        results = builder.vectorstore.similarity_search(query, k=2)
        print(f"  Query: '{query}'")
        for i, doc in enumerate(results, 1):
            cat = doc.metadata.get("category", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"  [{i}] ({cat}) {preview}...")

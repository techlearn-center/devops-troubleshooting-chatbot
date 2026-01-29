"""
RAG Engine for DevOps Troubleshooting Chatbot
=============================================
This module handles document loading, embedding, and retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional

# Vector store
import chromadb
from chromadb.config import Settings

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

# Embeddings
from sentence_transformers import SentenceTransformer


class RAGEngine:
    """RAG Engine for document retrieval and context augmentation."""

    def __init__(
        self,
        knowledge_base_path: str = "knowledge-base",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "devops_docs"
    ):
        """Initialize the RAG engine.

        Args:
            knowledge_base_path: Path to the knowledge base documents
            persist_directory: Path to store the vector database
            embedding_model: Name of the sentence-transformers model
            collection_name: Name of the ChromaDB collection
        """
        self.knowledge_base_path = Path(knowledge_base_path)
        self.persist_directory = Path(persist_directory)
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}...")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DevOps troubleshooting documentation"}
        )

        print(f"RAG Engine initialized. Collection has {self.collection.count()} documents.")

    def load_documents(self) -> List:
        """Load documents from the knowledge base.

        Returns:
            List of loaded documents
        """
        print(f"Loading documents from: {self.knowledge_base_path}")

        documents = []

        # Load markdown files
        for md_file in self.knowledge_base_path.rglob("*.md"):
            try:
                loader = UnstructuredMarkdownLoader(str(md_file))
                docs = loader.load()

                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = str(md_file)
                    doc.metadata["type"] = "markdown"

                    # Extract category from path
                    relative_path = md_file.relative_to(self.knowledge_base_path)
                    if len(relative_path.parts) > 1:
                        doc.metadata["category"] = relative_path.parts[0]

                documents.extend(docs)
                print(f"  Loaded: {md_file.name}")
            except Exception as e:
                print(f"  Error loading {md_file}: {e}")

        # Load text files
        for txt_file in self.knowledge_base_path.rglob("*.txt"):
            try:
                loader = TextLoader(str(txt_file))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = str(txt_file)
                    doc.metadata["type"] = "text"

                documents.extend(docs)
                print(f"  Loaded: {txt_file.name}")
            except Exception as e:
                print(f"  Error loading {txt_file}: {e}")

        print(f"Total documents loaded: {len(documents)}")
        return documents

    def chunk_documents(
        self,
        documents: List,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List:
        """Split documents into smaller chunks.

        Args:
            documents: List of documents to chunk
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List of document chunks
        """
        print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")

        return chunks

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()

    def index_documents(self, force_reindex: bool = False) -> int:
        """Index documents into the vector store.

        Args:
            force_reindex: If True, delete existing documents and reindex

        Returns:
            Number of documents indexed
        """
        # Check if already indexed
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reindex:
            print(f"Collection already has {existing_count} documents. Use force_reindex=True to reindex.")
            return existing_count

        # Clear existing documents if reindexing
        if force_reindex and existing_count > 0:
            print("Clearing existing documents...")
            # Delete collection and recreate
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DevOps troubleshooting documentation"}
            )

        # Load and chunk documents
        documents = self.load_documents()
        if not documents:
            print("No documents found to index!")
            return 0

        chunks = self.chunk_documents(documents)

        # Prepare for indexing
        ids = []
        texts = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            ids.append(f"doc_{i}")
            texts.append(chunk.page_content)
            metadatas.append(chunk.metadata)

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embed_texts(texts)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end]
            )
            print(f"  Indexed {end}/{len(ids)} documents...")

        print(f"Indexing complete! Total documents: {self.collection.count()}")
        return self.collection.count()

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filter_category: Optional[str] = None
    ) -> List[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query
            n_results: Number of results to return
            filter_category: Optional category filter (terraform, kubernetes, etc.)

        Returns:
            List of relevant documents with metadata and scores
        """
        # Generate query embedding
        query_embedding = self.embed_text(query)

        # Build where filter if category specified
        where_filter = None
        if filter_category:
            where_filter = {"category": filter_category}

        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })

        return formatted_results

    def get_context(
        self,
        query: str,
        n_results: int = 3,
        max_context_length: int = 4000
    ) -> str:
        """Get formatted context for a query.

        Args:
            query: User query
            n_results: Number of documents to retrieve
            max_context_length: Maximum length of context string

        Returns:
            Formatted context string
        """
        results = self.retrieve(query, n_results=n_results)

        if not results:
            return "No relevant documentation found."

        context_parts = []
        total_length = 0

        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result["metadata"].get("source", "Unknown")
            source_name = Path(source).name if source != "Unknown" else "Unknown"

            # Format this document
            doc_text = f"--- Source: {source_name} ---\n{content}"

            # Check if we can add this document
            if total_length + len(doc_text) > max_context_length:
                # Truncate if necessary
                remaining = max_context_length - total_length - 50
                if remaining > 100:
                    doc_text = doc_text[:remaining] + "\n[truncated...]"
                    context_parts.append(doc_text)
                break

            context_parts.append(doc_text)
            total_length += len(doc_text)

        return "\n\n".join(context_parts)


# Convenience function for quick setup
def create_rag_engine(
    knowledge_base_path: str = "knowledge-base",
    auto_index: bool = True
) -> RAGEngine:
    """Create and optionally initialize a RAG engine.

    Args:
        knowledge_base_path: Path to knowledge base
        auto_index: If True, automatically index documents if needed

    Returns:
        Initialized RAGEngine
    """
    engine = RAGEngine(knowledge_base_path=knowledge_base_path)

    if auto_index:
        engine.index_documents(force_reindex=False)

    return engine


if __name__ == "__main__":
    # Test the RAG engine
    print("Testing RAG Engine...")

    engine = RAGEngine()
    engine.index_documents(force_reindex=True)

    # Test retrieval
    test_queries = [
        "My pod is in CrashLoopBackOff",
        "Terraform resource already exists error",
        "Docker build COPY failed file not found",
        "GitHub Actions permission denied"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print("="*50)

        context = engine.get_context(query)
        print(context[:500] + "..." if len(context) > 500 else context)

#!/usr/bin/env python3
"""
SOLUTION: Exercise 2 - Document Loading
========================================

Complete document loading solution with multiple formats.

RUN THIS:
    python module-3-rag-fundamentals/solutions/ex2_document_loading.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document


class DocumentLoader:
    """
    Unified document loading with metadata enrichment.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.documents = []

    def load_file(self, file_path: str, metadata: dict = None) -> list:
        """Load a single file with optional metadata."""
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

        if metadata:
            for doc in docs:
                doc.metadata.update(metadata)

        self.documents.extend(docs)
        return docs

    def load_directory(self, pattern: str = "**/*.md", category: str = None) -> list:
        """Load all files matching pattern."""
        loader = DirectoryLoader(
            str(self.base_path),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()

        # Add category metadata if provided
        if category:
            for doc in docs:
                doc.metadata["category"] = category

        self.documents.extend(docs)
        return docs

    def load_with_structure(self) -> list:
        """
        Load documents preserving directory structure as categories.

        Structure:
            docs/
            ├── kubernetes/  → category: kubernetes
            ├── terraform/   → category: terraform
            └── docker/      → category: docker
        """
        for subdir in self.base_path.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                loader = DirectoryLoader(
                    str(subdir),
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"}
                )
                docs = loader.load()

                for doc in docs:
                    doc.metadata["category"] = subdir.name
                    # Extract topic from filename
                    source = Path(doc.metadata.get('source', ''))
                    doc.metadata["topic"] = source.stem

                self.documents.extend(docs)

        return self.documents

    def get_documents(self) -> list:
        """Return all loaded documents."""
        return self.documents

    def get_by_category(self, category: str) -> list:
        """Filter documents by category."""
        return [d for d in self.documents if d.metadata.get("category") == category]

    def summary(self):
        """Print summary of loaded documents."""
        print(f"\n📊 Document Summary:")
        print(f"   Total documents: {len(self.documents)}")
        print(f"   Total characters: {sum(len(d.page_content) for d in self.documents)}")

        # Group by category
        categories = {}
        for doc in self.documents:
            cat = doc.metadata.get("category", "uncategorized")
            categories[cat] = categories.get(cat, 0) + 1

        print(f"   Categories:")
        for cat, count in sorted(categories.items()):
            print(f"      - {cat}: {count} docs")


def create_sample_structure():
    """Create sample directory structure."""
    base = Path(__file__).parent / "sample_docs"

    # Create structure
    (base / "kubernetes").mkdir(parents=True, exist_ok=True)
    (base / "terraform").mkdir(parents=True, exist_ok=True)
    (base / "docker").mkdir(parents=True, exist_ok=True)

    # Add sample files
    (base / "kubernetes" / "pods.md").write_text("# Pod Troubleshooting\n\nGuide for pod issues...")
    (base / "kubernetes" / "services.md").write_text("# Services\n\nKubernetes services...")
    (base / "terraform" / "state.md").write_text("# State Management\n\nTerraform state...")
    (base / "terraform" / "modules.md").write_text("# Modules\n\nCreating modules...")
    (base / "docker" / "builds.md").write_text("# Docker Builds\n\nOptimizing builds...")

    return str(base)


if __name__ == "__main__":
    # Create sample structure
    base_path = create_sample_structure()

    # Load with structure preservation
    loader = DocumentLoader(base_path)
    docs = loader.load_with_structure()

    # Show summary
    loader.summary()

    # Show sample
    print("\n📄 Sample document:")
    if docs:
        doc = docs[0]
        print(f"   Source: {doc.metadata.get('source')}")
        print(f"   Category: {doc.metadata.get('category')}")
        print(f"   Topic: {doc.metadata.get('topic')}")

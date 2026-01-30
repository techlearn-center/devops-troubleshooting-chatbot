#!/usr/bin/env python3
"""
SOLUTION: Exercise 1 - Knowledge Base Loader
=============================================

Loads all markdown documents from the knowledge-base/ directory,
auto-detects categories from folder structure, and enriches each
document with metadata (category, topic, filename).

RUN THIS:
    python module-4-knowledge-base/solutions/ex1_kb_loader.py
"""

import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader


class DevOpsKnowledgeBase:
    """
    Loads and organizes DevOps documentation from a structured
    directory, auto-detecting categories from folder names.
    """

    def __init__(self, kb_path: str = None):
        """
        Args:
            kb_path: Path to knowledge-base/ directory.
                     Defaults to project-root/knowledge-base/
        """
        if kb_path is None:
            kb_path = str(Path(__file__).parent.parent.parent / "knowledge-base")
        self.kb_path = Path(kb_path)
        self.documents = []

    def load_documents(self) -> list:
        """
        Walk the knowledge-base/ directory and load every .md file.

        Category is derived from the parent folder name:
            knowledge-base/terraform/errors.md  -> category: terraform
            knowledge-base/kubernetes/pod-errors.md -> category: kubernetes

        Returns:
            List of LangChain Document objects with enriched metadata.
        """
        self.documents = []

        if not self.kb_path.exists():
            print(f"Knowledge base path not found: {self.kb_path}")
            return self.documents

        print(f"Loading documents from: {self.kb_path}")

        for md_file in sorted(self.kb_path.rglob("*.md")):
            try:
                loader = TextLoader(str(md_file), encoding="utf-8")
                docs = loader.load()

                # Auto-detect category from parent folder
                category = md_file.parent.name

                for doc in docs:
                    doc.metadata["category"] = category
                    doc.metadata["topic"] = md_file.stem       # e.g. "errors"
                    doc.metadata["filename"] = md_file.name    # e.g. "errors.md"

                self.documents.extend(docs)
                print(f"  Loaded: {category}/{md_file.name}")

            except Exception as e:
                print(f"  Error loading {md_file}: {e}")

        print(f"Total documents loaded: {len(self.documents)}")
        return self.documents

    def get_stats(self) -> dict:
        """
        Return document counts grouped by category.

        Returns:
            Dict mapping category name to document count.
        """
        counts = defaultdict(int)
        for doc in self.documents:
            counts[doc.metadata.get("category", "unknown")] += 1
        return dict(sorted(counts.items()))

    def get_by_category(self, category: str) -> list:
        """Filter loaded documents by category."""
        return [d for d in self.documents if d.metadata.get("category") == category]


if __name__ == "__main__":
    kb = DevOpsKnowledgeBase()
    docs = kb.load_documents()

    # Print stats
    print("\n" + "=" * 50)
    print("KNOWLEDGE BASE STATS")
    print("=" * 50)

    stats = kb.get_stats()
    for category, count in stats.items():
        print(f"  {category}: {count} document(s)")
    print(f"  TOTAL: {len(docs)} document(s)")

    # Show sample metadata
    if docs:
        print("\nSample document metadata:")
        d = docs[0]
        print(f"  source:   {d.metadata.get('source')}")
        print(f"  category: {d.metadata.get('category')}")
        print(f"  topic:    {d.metadata.get('topic')}")
        print(f"  filename: {d.metadata.get('filename')}")
        print(f"  chars:    {len(d.page_content)}")

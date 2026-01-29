# Module 7: Building the Complete DevOps Chatbot

**Time Required: 3 hours**

This is it! Time to combine everything into a production-ready DevOps troubleshooting chatbot.

---

## Learning Objectives

By the end of this module, you will have built:
- A complete RAG-powered chatbot
- Category-aware retrieval
- Rich CLI interface
- Error classification
- Conversation history with context management

---

## Architecture Overview

```
+------------------------------------------------------------------+
|                    COMPLETE DEVOPS CHATBOT                        |
+------------------------------------------------------------------+
|                                                                   |
|  User Input                                                       |
|      |                                                            |
|      v                                                            |
|  +------------------+                                             |
|  | Error Classifier |  Determines: Terraform? K8s? Docker? CI/CD? |
|  +------------------+                                             |
|      |                                                            |
|      v                                                            |
|  +------------------+     +------------------+                    |
|  | RAG Retriever    | --> | Vector Database  |                    |
|  | (Category-aware) |     | (Knowledge Base) |                    |
|  +------------------+     +------------------+                    |
|      |                                                            |
|      | Relevant Documents                                         |
|      v                                                            |
|  +------------------+                                             |
|  | Prompt Builder   |  Combines: Question + Context + History     |
|  +------------------+                                             |
|      |                                                            |
|      v                                                            |
|  +------------------+                                             |
|  | LLM (GPT/Ollama) |  Generates informed response                |
|  +------------------+                                             |
|      |                                                            |
|      v                                                            |
|  +------------------+                                             |
|  | Response Display |  Rich formatting, code blocks, steps        |
|  +------------------+                                             |
|                                                                   |
+------------------------------------------------------------------+
```

---

## The Complete Chatbot Code

Create `chatbot/main.py`:

```python
#!/usr/bin/env python3
"""
DevOps Troubleshooting Chatbot
==============================
A RAG-powered chatbot for DevOps troubleshooting.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.rag_engine import DevOpsRAGEngine
from chatbot.prompts import PromptBuilder

load_dotenv()


class DevOpsChatbot:
    """Complete DevOps troubleshooting chatbot."""

    def __init__(self, knowledge_base_path: str = "knowledge-base"):
        """Initialize the chatbot with RAG engine."""
        self.console = Console()
        self.conversation_history: List[Dict] = []
        self.max_history = 10

        # Initialize RAG engine
        self.console.print("[yellow]Initializing knowledge base...[/yellow]")
        self.rag = DevOpsRAGEngine(knowledge_base_path)
        self.prompt_builder = PromptBuilder()

        self.console.print("[green]Ready![/green]\n")

    def classify_query(self, query: str) -> str:
        """
        Classify the query into a category.

        Returns:
            Category: terraform, kubernetes, docker, cicd, or general
        """
        query_lower = query.lower()

        keywords = {
            "terraform": ["terraform", "tf", "hcl", "state", "plan", "apply", "provider"],
            "kubernetes": ["kubernetes", "k8s", "kubectl", "pod", "deployment", "service", "ingress"],
            "docker": ["docker", "container", "dockerfile", "compose", "image", "build"],
            "cicd": ["github actions", "jenkins", "gitlab", "pipeline", "workflow", "ci/cd", "ci cd"]
        }

        for category, words in keywords.items():
            if any(word in query_lower for word in words):
                return category

        return "general"

    def get_response(self, user_message: str) -> str:
        """
        Get a response using RAG.

        Args:
            user_message: User's question

        Returns:
            Generated response
        """
        # Classify the query
        category = self.classify_query(user_message)

        # Retrieve relevant documents
        context_docs = self.rag.retrieve(
            query=user_message,
            category=category if category != "general" else None,
            k=5
        )

        # Build the prompt
        prompt = self.prompt_builder.build_rag_prompt(
            question=user_message,
            context=context_docs,
            history=self.conversation_history[-4:]  # Last 2 exchanges
        )

        # Get LLM response
        response = self.rag.generate(prompt)

        # Update history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        return response

    def display_welcome(self):
        """Display welcome message."""
        welcome_text = """
[bold green]DevOps Troubleshooting Chatbot[/bold green]

I can help you troubleshoot issues with:
• [cyan]Terraform[/cyan] - State management, errors, best practices
• [cyan]Kubernetes[/cyan] - Pod issues, deployments, networking
• [cyan]Docker[/cyan] - Build errors, runtime issues, Compose
• [cyan]CI/CD[/cyan] - GitHub Actions, Jenkins, GitLab CI

[dim]Commands:[/dim]
  [bold]clear[/bold]  - Clear conversation history
  [bold]help[/bold]   - Show this message
  [bold]quit[/bold]   - Exit the chatbot
"""
        self.console.print(Panel(welcome_text, border_style="green"))

    def display_response(self, response: str, category: str):
        """Display the response with rich formatting."""
        # Add category badge
        category_colors = {
            "terraform": "purple",
            "kubernetes": "blue",
            "docker": "cyan",
            "cicd": "yellow",
            "general": "white"
        }

        color = category_colors.get(category, "white")
        title = f"[bold {color}]DevOps Bot[/bold {color}] [{category}]"

        self.console.print(Panel(
            Markdown(response),
            title=title,
            border_style=color
        ))

    def run(self):
        """Run the interactive chatbot."""
        self.display_welcome()

        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()

                # Handle commands
                if not user_input:
                    continue
                elif user_input.lower() == "quit":
                    self.console.print("[yellow]Goodbye![/yellow]")
                    break
                elif user_input.lower() == "clear":
                    self.conversation_history.clear()
                    self.console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                elif user_input.lower() == "help":
                    self.display_welcome()
                    continue

                # Classify and get response
                category = self.classify_query(user_input)

                with self.console.status(f"[green]Searching {category} knowledge..."):
                    response = self.get_response(user_input)

                self.display_response(response, category)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")


def main():
    """Entry point."""
    chatbot = DevOpsChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
```

---

## The RAG Engine

Create `chatbot/rag_engine.py`:

```python
"""
RAG Engine for DevOps Chatbot
=============================
Handles document retrieval and LLM generation.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


class DevOpsRAGEngine:
    """RAG engine for DevOps documentation."""

    def __init__(self, knowledge_base_path: str):
        """
        Initialize the RAG engine.

        Args:
            knowledge_base_path: Path to knowledge base directory
        """
        self.kb_path = Path(knowledge_base_path)
        self.persist_dir = "./chroma_db"

        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )

        # Initialize LLM
        if os.getenv("USE_OLLAMA", "false").lower() == "true":
            from langchain_community.llms import Ollama
            self.llm = Ollama(model=os.getenv("OLLAMA_MODEL", "llama2"))
        else:
            self.llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0.2
            )

        # Load or create vector store
        self._init_vectorstore()

    def _init_vectorstore(self):
        """Initialize or load the vector store."""
        if Path(self.persist_dir).exists():
            # Load existing
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        else:
            # Create new from documents
            documents = self._load_documents()
            chunks = self._chunk_documents(documents)
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )

    def _load_documents(self) -> list:
        """Load documents from knowledge base."""
        documents = []

        for category in ["terraform", "kubernetes", "docker", "cicd"]:
            category_path = self.kb_path / category
            if category_path.exists():
                loader = DirectoryLoader(
                    str(category_path),
                    glob="**/*.md",
                    loader_cls=UnstructuredMarkdownLoader
                )
                docs = loader.load()

                # Add category metadata
                for doc in docs:
                    doc.metadata["category"] = category

                documents.extend(docs)

        return documents

    def _chunk_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", " "]
        )
        return splitter.split_documents(documents)

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            category: Optional category filter
            k: Number of results

        Returns:
            List of relevant document contents
        """
        search_kwargs = {"k": k}

        if category:
            search_kwargs["filter"] = {"category": category}

        results = self.vectorstore.similarity_search(query, **search_kwargs)
        return [doc.page_content for doc in results]

    def generate(self, prompt: str) -> str:
        """
        Generate response using LLM.

        Args:
            prompt: Full prompt with context

        Returns:
            Generated response
        """
        if hasattr(self.llm, 'invoke'):
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        else:
            return self.llm(prompt)
```

---

## Prompt Templates

Create `chatbot/prompts.py`:

```python
"""
Prompt Templates for DevOps Chatbot
===================================
"""

from typing import List, Dict


class PromptBuilder:
    """Builds optimized prompts for DevOps troubleshooting."""

    SYSTEM_PROMPT = """You are an expert DevOps engineer specializing in troubleshooting.

Your expertise covers:
- Terraform and Infrastructure as Code
- Kubernetes and container orchestration
- Docker and containerization
- CI/CD pipelines (GitHub Actions, Jenkins, GitLab CI)

When helping users:
1. Analyze the error/issue carefully
2. Explain what's happening and why
3. Provide step-by-step solutions with commands
4. Include code examples when helpful
5. Suggest preventive measures
6. Be concise but thorough

Format your responses in Markdown for readability.
Use code blocks with language specification for commands."""

    RAG_TEMPLATE = """Based on the following documentation and conversation context, help the user with their DevOps issue.

## Relevant Documentation:
{context}

## Conversation History:
{history}

## Current Question:
{question}

Provide a helpful, accurate response based on the documentation. If the documentation doesn't contain relevant information, use your general knowledge but mention that.

## Response:"""

    def build_rag_prompt(
        self,
        question: str,
        context: List[str],
        history: List[Dict] = None
    ) -> str:
        """
        Build a RAG prompt with context and history.

        Args:
            question: User's question
            context: Retrieved document chunks
            history: Conversation history

        Returns:
            Formatted prompt
        """
        # Format context
        context_str = "\n\n---\n\n".join(context) if context else "No specific documentation found."

        # Format history
        history_str = ""
        if history:
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n\n"

        return self.RAG_TEMPLATE.format(
            context=context_str,
            history=history_str or "No previous conversation.",
            question=question
        )
```

---

## Testing Your Chatbot

Run the complete chatbot:

```bash
cd chatbot
python main.py
```

Test with these queries:
1. "My Terraform apply shows 'Error: resource already exists'"
2. "Kubernetes pod stuck in CrashLoopBackOff"
3. "Docker build fails with 'no space left on device'"
4. "GitHub Actions workflow failing with permission denied"

---

## Key Takeaways

Congratulations! You've built a complete DevOps troubleshooting chatbot that:

1. **Uses RAG** - Retrieves relevant documentation
2. **Classifies queries** - Routes to appropriate knowledge
3. **Maintains context** - Remembers conversation history
4. **Formats beautifully** - Rich CLI interface
5. **Handles errors** - Graceful error handling

---

## Next Steps

Ideas for extending your chatbot:
- Add a web interface (FastAPI + React)
- Integrate with Slack/Discord
- Add more documentation sources
- Implement feedback collection
- Fine-tune on your team's specific issues

**You did it!** You've built an AI-powered DevOps assistant from scratch!

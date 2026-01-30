# Module 7: Building the Complete DevOps Chatbot

**Time Required: 3 hours**

This is the capstone module! You'll combine everything from Modules 1-6 into a production-ready DevOps troubleshooting chatbot. This is where all the pieces click together.

---

## What You'll Build

By the end of this module, you will have built a complete chatbot that:
- Automatically classifies user questions (Terraform? Kubernetes? Docker? CI/CD?)
- Retrieves relevant documentation using RAG
- Constructs optimized prompts using advanced techniques
- Generates accurate, actionable responses
- Displays beautiful formatted output in the terminal
- Manages conversation history intelligently
- Handles errors gracefully

---

## How Everything Fits Together

Remember all those modules? Here's how they connect:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR COMPLETE CHATBOT                          │
│                                                                         │
│  Module 2: Rich CLI    ┌──────────────────────────────────────────┐    │
│  ┌──────────────┐      │         THE BRAIN (Backend)              │    │
│  │              │      │                                          │    │
│  │  User types  │──────│──→ Module 7: Error Classifier            │    │
│  │  a question  │      │       "Is this about Terraform or K8s?"  │    │
│  │              │      │           │                              │    │
│  │  ┌────────┐  │      │           ▼                              │    │
│  │  │ > help │  │      │    Module 3-5: RAG Pipeline              │    │
│  │  │  my    │  │      │       "Find relevant docs"               │    │
│  │  │  pod   │  │      │           │                              │    │
│  │  │  is... │  │      │           ▼                              │    │
│  │  └────────┘  │      │    Module 6: Prompt Engine               │    │
│  │              │      │       "Build the perfect prompt"         │    │
│  │              │      │           │                              │    │
│  │              │      │           ▼                              │    │
│  │              │      │    Module 1: LLM API Call                │    │
│  │  Beautiful   │◀─────│──     "Generate the answer"              │    │
│  │  formatted   │      │                                          │    │
│  │  response    │      └──────────────────────────────────────────┘    │
│  │  with code   │                                                      │
│  │  blocks!     │      Module 4: Knowledge Base                        │
│  └──────────────┘      ┌──────────────────────────────────────────┐    │
│                        │  📁 terraform/  📁 kubernetes/            │    │
│                        │  📁 docker/     📁 cicd/                  │    │
│                        │  (12 markdown files with solutions)       │    │
│                        └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Architecture (Simplified)

When a user asks a question, here's exactly what happens:

```
Step 1: USER INPUT
  "My Kubernetes pod is in CrashLoopBackOff"
         │
         ▼
Step 2: ERROR CLASSIFIER (Module 7)
  Scans for keywords: "kubernetes", "pod", "CrashLoopBackOff"
  Result: category = "kubernetes", confidence = 0.95
         │
         ▼
Step 3: RAG RETRIEVER (Modules 3-5)
  Searches vector database for "kubernetes CrashLoopBackOff"
  BUT only in the "kubernetes" category (thanks to classifier!)
  Returns: Top 5 relevant documentation chunks
         │
         ▼
Step 4: PROMPT BUILDER (Module 6)
  Combines:
  - System prompt: "You are a DevOps expert..."
  - Retrieved docs: "CrashLoopBackOff means the container is crashing..."
  - Conversation history: (previous Q&A for context)
  - User question: "My Kubernetes pod is in CrashLoopBackOff"
  Result: One big, optimized prompt
         │
         ▼
Step 5: LLM GENERATION (Module 1)
  Sends the prompt to GPT/Ollama
  Gets back: Detailed troubleshooting response
         │
         ▼
Step 6: DISPLAY (Module 2)
  Formats with Rich: headers, code blocks, colored panels
  Shows: Category badge, source documents, response
```

---

## Component 1: Error Classifier

The classifier is like a **mail sorter** - it reads the question and puts it in the right mailbox:

```python
class ErrorClassifier:
    """
    Sorts user questions into categories so we search
    the right part of our knowledge base.

    Without classification:
      "pod crash" → searches ALL documents → slower, less relevant

    With classification:
      "pod crash" → classified as "kubernetes" → searches ONLY
      kubernetes docs → faster, more relevant results!
    """

    CATEGORIES = {
        "terraform": ["terraform", "tf", "hcl", "state", "plan",
                       "apply", "provider", "module", "resource"],
        "kubernetes": ["kubernetes", "k8s", "kubectl", "pod",
                        "deployment", "service", "ingress", "helm"],
        "docker": ["docker", "container", "dockerfile", "compose",
                    "image", "build", "volume", "network"],
        "cicd": ["github actions", "jenkins", "gitlab", "pipeline",
                  "workflow", "ci/cd", "deploy", "build"]
    }

    def classify(self, query: str) -> tuple:
        """Returns (category, confidence)."""
        query_lower = query.lower()
        scores = {}

        for category, keywords in self.CATEGORIES.items():
            matches = sum(1 for kw in keywords if kw in query_lower)
            scores[category] = matches / len(keywords)

        if max(scores.values()) > 0:
            best = max(scores, key=scores.get)
            return best, scores[best]

        return "general", 0.0
```

---

## Component 2: RAG Pipeline

The RAG pipeline is the **search engine** of your chatbot:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DevOpsRAGEngine:
    """
    RAG = Retrieval-Augmented Generation

    Instead of the LLM guessing from its training data,
    we FIND the relevant documentation first, then give it
    to the LLM as context. Like an open-book exam!
    """

    def __init__(self, knowledge_base_path):
        # Free, local embeddings (no API key needed!)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Vector database (stores document embeddings)
        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def retrieve(self, query, category=None, k=5):
        """Find the most relevant documentation chunks."""
        search_kwargs = {"k": k}

        # If we know the category, only search those docs
        if category:
            search_kwargs["filter"] = {"category": category}

        results = self.vectorstore.similarity_search(
            query, **search_kwargs
        )
        return [doc.page_content for doc in results]
```

---

## Component 3: Prompt Builder

The prompt builder is like a **recipe assembler** - it combines all the ingredients into the perfect prompt:

```python
class PromptBuilder:
    """
    Assembles the final prompt from multiple pieces:

    ┌─────────────────────────────────────────────┐
    │  FINAL PROMPT =                              │
    │                                               │
    │  System Prompt (who the LLM should be)        │
    │  +                                            │
    │  Retrieved Context (relevant documentation)   │
    │  +                                            │
    │  Conversation History (previous Q&A)          │
    │  +                                            │
    │  User Question (what they're asking now)       │
    │                                               │
    │  = One big prompt sent to the LLM             │
    └─────────────────────────────────────────────┘
    """

    SYSTEM_PROMPT = """You are an expert DevOps engineer.
Your expertise covers Terraform, Kubernetes, Docker, and CI/CD.
When helping users:
1. Explain what's happening and why
2. Provide step-by-step solutions with commands
3. Include code examples
4. Suggest preventive measures
Format your responses in Markdown."""

    RAG_TEMPLATE = """Based on the following documentation, help the user.

## Relevant Documentation:
{context}

## Conversation History:
{history}

## Current Question:
{question}

## Response:"""

    def build(self, question, context_docs, history=None):
        """Assemble all the pieces into one prompt."""
        context_str = "\n\n---\n\n".join(context_docs)
        history_str = self._format_history(history)

        return self.RAG_TEMPLATE.format(
            context=context_str or "No docs found.",
            history=history_str or "No previous conversation.",
            question=question
        )
```

---

## Component 4: The Main Chatbot Loop

This ties everything together:

```python
class DevOpsChatbot:
    """
    The main chatbot that orchestrates all components.

    Think of it as the CONDUCTOR of an orchestra:
    - It doesn't play any instruments (components do the work)
    - It coordinates WHEN each component plays
    - It makes sure everything works together harmoniously
    """

    def __init__(self):
        self.classifier = ErrorClassifier()
        self.rag = DevOpsRAGEngine("knowledge-base")
        self.prompt_builder = PromptBuilder()
        self.history = []

    def handle_message(self, user_message):
        """Process one user message through the full pipeline."""

        # Step 1: Classify
        category, confidence = self.classifier.classify(user_message)

        # Step 2: Retrieve relevant docs
        context = self.rag.retrieve(
            query=user_message,
            category=category if category != "general" else None
        )

        # Step 3: Build prompt
        prompt = self.prompt_builder.build(
            question=user_message,
            context_docs=context,
            history=self.history[-4:]  # Last 2 exchanges
        )

        # Step 4: Generate response
        response = self._call_llm(prompt)

        # Step 5: Update history
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": response})

        return response, category
```

---

## Slash Commands

A good CLI chatbot supports special commands:

```
┌─────────────────────────────────────────────────────────────┐
│  AVAILABLE COMMANDS                                          │
│                                                              │
│  /help      Show available commands and topics               │
│  /clear     Clear conversation history and start fresh       │
│  /history   Show conversation history                        │
│  /stats     Show knowledge base statistics                   │
│  /category  Show current detected category                   │
│  /sources   Show source documents for last response          │
│  /export    Save conversation to a markdown file             │
│  /feedback  Rate the last response (1-5 stars)               │
│  quit       Exit the chatbot                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Testing Your Chatbot

Once built, test with these queries to verify all components work:

### Terraform Questions
1. "My Terraform apply shows 'Error: resource already exists'"
2. "How do I fix Terraform state drift?"
3. "Terraform plan shows changes I didn't make"

### Kubernetes Questions
4. "Pod stuck in CrashLoopBackOff"
5. "Service can't connect to pods"
6. "How do I debug a pod that won't start?"

### Docker Questions
7. "Docker build fails with 'no space left on device'"
8. "Container exits immediately after starting"
9. "Docker compose services can't communicate"

### CI/CD Questions
10. "GitHub Actions workflow failing with permission denied"
11. "Jenkins pipeline stuck in queue"
12. "GitLab CI runner not picking up jobs"

### Cross-Category (tests classifier)
13. "How do I deploy a Docker container to Kubernetes using Terraform?"
14. "What's the difference between a pod and a container?"

---

## Exercises

The `exercises/` folder contains 4 hands-on exercises that build the chatbot step by step:

| Exercise | Component | What You'll Build |
|----------|-----------|-------------------|
| `ex1_error_classifier.py` | Classifier | Keyword-based error classification with confidence scoring |
| `ex2_rag_pipeline.py` | RAG Pipeline | Complete load → chunk → embed → store → retrieve pipeline |
| `ex3_prompt_engine.py` | Prompt Engine | Prompt assembly with history management and templates |
| `ex4_complete_chatbot.py` | Full Chatbot | Everything combined into a working chatbot! |

Each exercise builds a standalone component. Exercise 4 combines them all into the final product.

Check `solutions/` for complete implementations.

---

## The Production Chatbot

After completing the exercises, check out the `chatbot/` directory in the project root for the production version:

```
chatbot/
├── __init__.py         # Package initialization
├── main.py             # Main chatbot class (DevOpsChatbot)
├── rag_engine.py       # RAG engine (DevOpsRAGEngine)
└── prompts.py          # Prompt templates (PromptBuilder)
```

Run it with:
```bash
cd ..
python -m chatbot.main
```

---

## Key Takeaways

Congratulations! You've built a complete AI-powered DevOps troubleshooting chatbot!

Here's what you learned across all 7 modules:

| Module | What You Learned | Key Skill |
|--------|-----------------|-----------|
| Module 0 | Environment setup | Python, APIs, tools |
| Module 1 | LLM API basics | API calls, parameters, tokens |
| Module 2 | CLI chatbot | Rich library, streaming, UX |
| Module 3 | RAG fundamentals | Vectors, embeddings, retrieval |
| Module 4 | Knowledge base | Document loading, chunking, storage |
| Module 5 | Vector search | Similarity metrics, ChromaDB, quality |
| Module 6 | Advanced prompting | Few-shot, CoT, structured output |
| **Module 7** | **Complete chatbot** | **Composition, architecture, production** |

---

## Next Steps

Ideas for extending your chatbot:

1. **Add a Web Interface** - Use FastAPI + React/HTML for a browser-based UI
2. **Integrate with Slack/Discord** - Deploy as a bot in your team's chat
3. **Add More Knowledge** - Expand the knowledge base with your team's runbooks
4. **Implement Feedback Loop** - Collect user ratings to improve responses
5. **Fine-Tune on Your Data** - Train a custom model on your organization's issues
6. **Add Multi-Modal Support** - Accept screenshots of error messages
7. **Deploy to Production** - Containerize with Docker, deploy to Kubernetes

**You did it!** You've gone from zero to building an AI-powered DevOps assistant. Every concept was explained, every piece was built by hand, and now you understand how it all works together.

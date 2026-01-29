#!/usr/bin/env python3
"""
DevOps Troubleshooting Chatbot - Runner Script
==============================================
This script runs the chatbot and validates module completions.

Usage:
    python run.py                    # Run the chatbot
    python run.py --verify           # Verify all modules
    python run.py --verify --module 1  # Verify specific module
    python run.py --reindex          # Reindex knowledge base
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(success: bool, message: str):
    """Print a check result."""
    icon = "✓" if success else "✗"
    color_code = "\033[92m" if success else "\033[91m"
    reset_code = "\033[0m"
    print(f"  {color_code}{icon}{reset_code} {message}")


def verify_module_0():
    """Verify Module 0: Setup."""
    print("\n[Module 0] Prerequisites & Setup")
    results = []

    # Check Python version
    import sys
    version = sys.version_info
    success = version.major >= 3 and version.minor >= 9
    print_result(success, f"Python {version.major}.{version.minor}")
    results.append(success)

    # Check required packages
    packages = [
        ("openai", "openai"),
        ("langchain", "langchain"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("dotenv", "python-dotenv"),
        ("rich", "rich"),
    ]

    for import_name, package_name in packages:
        try:
            __import__(import_name)
            print_result(True, f"{package_name} installed")
            results.append(True)
        except ImportError:
            print_result(False, f"{package_name} NOT installed")
            results.append(False)

    # Check .env file
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        print_result(True, ".env file exists")
        results.append(True)
    else:
        print_result(False, ".env file not found")
        results.append(False)

    return all(results)


def verify_module_1():
    """Verify Module 1: LLM Basics."""
    print("\n[Module 1] LLM Fundamentals")
    results = []

    # Check if exercises directory exists
    exercises_dir = PROJECT_ROOT / "module-1-llm-basics" / "exercises"
    solutions_dir = PROJECT_ROOT / "module-1-llm-basics" / "solutions"

    if exercises_dir.exists() or solutions_dir.exists():
        print_result(True, "Module 1 directory structure exists")
        results.append(True)
    else:
        print_result(False, "Module 1 directory not found")
        results.append(False)

    # Check if solution file exists
    solution_file = solutions_dir / "ex1_first_call.py"
    if solution_file.exists():
        print_result(True, "Exercise 1 solution exists")
        results.append(True)
    else:
        print_result(False, "Exercise 1 solution not found")
        results.append(False)

    return all(results)


def verify_module_2():
    """Verify Module 2: Simple Chatbot."""
    print("\n[Module 2] Simple Chatbot")
    results = []

    # Check if module directory exists
    module_dir = PROJECT_ROOT / "module-2-simple-chatbot"
    if module_dir.exists():
        print_result(True, "Module 2 directory exists")
        results.append(True)
    else:
        print_result(False, "Module 2 directory not found")
        results.append(False)

    return all(results)


def verify_module_3():
    """Verify Module 3: RAG Fundamentals."""
    print("\n[Module 3] RAG Fundamentals")
    results = []

    module_dir = PROJECT_ROOT / "module-3-rag-fundamentals"
    if module_dir.exists():
        print_result(True, "Module 3 directory exists")
        results.append(True)
    else:
        print_result(False, "Module 3 directory not found")
        results.append(False)

    return all(results)


def verify_module_4():
    """Verify Module 4: Knowledge Base."""
    print("\n[Module 4] DevOps Knowledge Base")
    results = []

    # Check knowledge base directory
    kb_dir = PROJECT_ROOT / "knowledge-base"
    if kb_dir.exists():
        print_result(True, "Knowledge base directory exists")
        results.append(True)

        # Check for documentation files
        categories = ["terraform", "kubernetes", "docker", "cicd"]
        for category in categories:
            cat_dir = kb_dir / category
            if cat_dir.exists() and any(cat_dir.iterdir()):
                print_result(True, f"  {category}/ has content")
                results.append(True)
            else:
                print_result(False, f"  {category}/ missing or empty")
                results.append(False)
    else:
        print_result(False, "Knowledge base directory not found")
        results.append(False)

    return all(results)


def verify_module_5():
    """Verify Module 5: Vector Search."""
    print("\n[Module 5] Vector Search")
    results = []

    module_dir = PROJECT_ROOT / "module-5-vector-search"
    if module_dir.exists():
        print_result(True, "Module 5 directory exists")
        results.append(True)
    else:
        print_result(False, "Module 5 directory not found")
        results.append(False)

    # Check if ChromaDB can be used
    try:
        import chromadb
        client = chromadb.Client()
        collection = client.create_collection("test_verify")
        collection.add(documents=["test"], ids=["1"])
        results_query = collection.query(query_texts=["test"], n_results=1)
        print_result(True, "ChromaDB working correctly")
        results.append(True)
    except Exception as e:
        print_result(False, f"ChromaDB error: {str(e)[:50]}")
        results.append(False)

    return all(results)


def verify_module_6():
    """Verify Module 6: Advanced Prompting."""
    print("\n[Module 6] Advanced Prompting")
    results = []

    module_dir = PROJECT_ROOT / "module-6-advanced-prompting"
    if module_dir.exists():
        print_result(True, "Module 6 directory exists")
        results.append(True)
    else:
        print_result(False, "Module 6 directory not found")
        results.append(False)

    # Check prompts.py
    prompts_file = PROJECT_ROOT / "chatbot" / "prompts.py"
    if prompts_file.exists():
        print_result(True, "prompts.py exists")
        results.append(True)

        # Check for required prompt templates
        content = prompts_file.read_text()
        templates = [
            "SYSTEM_PROMPT",
            "RAG_PROMPT_TEMPLATE",
            "FEW_SHOT_EXAMPLES",
        ]
        for template in templates:
            if template in content:
                print_result(True, f"  {template} defined")
                results.append(True)
            else:
                print_result(False, f"  {template} not found")
                results.append(False)
    else:
        print_result(False, "prompts.py not found")
        results.append(False)

    return all(results)


def verify_module_7():
    """Verify Module 7: Complete Chatbot."""
    print("\n[Module 7] Complete Chatbot")
    results = []

    # Check chatbot directory
    chatbot_dir = PROJECT_ROOT / "chatbot"
    if chatbot_dir.exists():
        print_result(True, "chatbot/ directory exists")
        results.append(True)

        # Check required files
        required_files = ["main.py", "rag_engine.py", "prompts.py", "__init__.py"]
        for filename in required_files:
            filepath = chatbot_dir / filename
            if filepath.exists():
                print_result(True, f"  {filename} exists")
                results.append(True)
            else:
                print_result(False, f"  {filename} not found")
                results.append(False)
    else:
        print_result(False, "chatbot/ directory not found")
        results.append(False)

    return all(results)


def verify_all():
    """Run all verification checks."""
    print_header("DevOps Troubleshooting Chatbot - Verification")

    modules = [
        (0, verify_module_0),
        (1, verify_module_1),
        (2, verify_module_2),
        (3, verify_module_3),
        (4, verify_module_4),
        (5, verify_module_5),
        (6, verify_module_6),
        (7, verify_module_7),
    ]

    results = {}
    for module_num, verify_func in modules:
        try:
            results[module_num] = verify_func()
        except Exception as e:
            print_result(False, f"Module {module_num} verification failed: {e}")
            results[module_num] = False

    # Summary
    print_header("Summary")
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for module_num, success in results.items():
        status = "PASS" if success else "FAIL"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"  Module {module_num}: {color}{status}{reset}")

    print(f"\n  Total: {passed}/{total} modules passed")

    if passed == total:
        print("\n  \033[92m✓ All modules verified successfully!\033[0m")
        return 0
    else:
        print(f"\n  \033[91m✗ {total - passed} module(s) need attention.\033[0m")
        return 1


def verify_module(module_num: int):
    """Verify a specific module."""
    verifiers = {
        0: verify_module_0,
        1: verify_module_1,
        2: verify_module_2,
        3: verify_module_3,
        4: verify_module_4,
        5: verify_module_5,
        6: verify_module_6,
        7: verify_module_7,
    }

    if module_num not in verifiers:
        print(f"Invalid module number: {module_num}")
        print(f"Valid modules: {list(verifiers.keys())}")
        return 1

    print_header(f"Verifying Module {module_num}")
    try:
        success = verifiers[module_num]()
        return 0 if success else 1
    except Exception as e:
        print_result(False, f"Verification failed: {e}")
        return 1


def run_chatbot():
    """Run the chatbot."""
    from chatbot.main import run_cli
    run_cli()


def reindex_knowledge_base():
    """Reindex the knowledge base."""
    from chatbot.rag_engine import create_rag_engine

    print_header("Reindexing Knowledge Base")

    engine = create_rag_engine(auto_index=False)
    count = engine.index_documents(force_reindex=True)

    print(f"\n  Indexed {count} documents.")
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DevOps Troubleshooting Chatbot Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                     Run the chatbot
  python run.py --verify            Verify all modules
  python run.py --verify --module 1 Verify specific module
  python run.py --reindex           Reindex knowledge base
        """
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification checks"
    )
    parser.add_argument(
        "--module",
        type=int,
        help="Specific module to verify (0-7)"
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reindex the knowledge base"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama instead of OpenAI"
    )

    args = parser.parse_args()

    # Set Ollama environment variable if specified
    if args.ollama:
        os.environ["USE_OLLAMA"] = "true"

    # Handle commands
    if args.verify:
        if args.module is not None:
            return verify_module(args.module)
        else:
            return verify_all()

    if args.reindex:
        return reindex_knowledge_base()

    # Default: run the chatbot
    try:
        run_chatbot()
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry running: python run.py --verify")
        return 1


if __name__ == "__main__":
    sys.exit(main())

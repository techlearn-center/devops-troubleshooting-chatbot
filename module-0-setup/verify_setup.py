#!/usr/bin/env python3
"""
DevOps Chatbot - Setup Verification Script
==========================================
Verifies that your environment is correctly configured.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header():
    """Print the verification header."""
    print("\n" + "=" * 50)
    print("  DevOps Chatbot - Setup Verification")
    print("=" * 50 + "\n")


def print_result(success: bool, message: str):
    """Print a check result."""
    icon = "✓" if success else "✗"
    color_code = "\033[92m" if success else "\033[91m"
    reset_code = "\033[0m"
    print(f"  {color_code}{icon}{reset_code} {message}")


def check_python_version():
    """Check Python version is 3.9+."""
    print("[1/6] Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print_result(True, f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_result(False, f"Python {version.major}.{version.minor} (need 3.9+)")
        return False


def check_packages():
    """Check required packages are installed."""
    print("\n[2/6] Checking required packages...")

    required_packages = [
        ("openai", "openai"),
        ("langchain", "langchain"),
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("dotenv", "python-dotenv"),
        ("rich", "rich"),
    ]

    all_installed = True
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print_result(True, f"{package_name} installed")
        except ImportError:
            print_result(False, f"{package_name} NOT installed")
            all_installed = False

    return all_installed


def check_env_file():
    """Check .env file exists and has required variables."""
    print("\n[3/6] Checking environment variables...")

    # Check .env file exists
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print_result(False, ".env file not found (copy from .env.example)")
        return False

    print_result(True, ".env file found")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(env_path)

    # Check for API key or Ollama
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        print_result(True, "Using Ollama (local LLM)")
        return True
    else:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and api_key != "sk-your-api-key-here":
            print_result(True, "OPENAI_API_KEY configured")
            return True
        else:
            print_result(False, "OPENAI_API_KEY not set (edit .env file)")
            return False


def check_openai_connection():
    """Test OpenAI API connection."""
    print("\n[4/6] Testing LLM connection...")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # Test Ollama connection
        try:
            import requests
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print_result(True, "Successfully connected to Ollama")
                return True
            else:
                print_result(False, "Ollama connection failed")
                return False
        except Exception as e:
            print_result(False, f"Ollama error: {str(e)[:50]}")
            return False
    else:
        # Test OpenAI connection
        try:
            from openai import OpenAI
            client = OpenAI()

            # Make a minimal API call
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=5
            )
            print_result(True, "Successfully connected to OpenAI")
            return True
        except Exception as e:
            error_msg = str(e)[:50]
            print_result(False, f"OpenAI error: {error_msg}")
            return False


def check_embeddings():
    """Test embedding model."""
    print("\n[5/6] Testing embeddings...")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")

    try:
        if embedding_provider == "local":
            from sentence_transformers import SentenceTransformer
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
            embedding = model.encode("test")
            print_result(True, f"Local embedding model loaded ({model_name})")
        else:
            from openai import OpenAI
            client = OpenAI()
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input="test"
            )
            print_result(True, "OpenAI embeddings working")
        return True
    except Exception as e:
        print_result(False, f"Embedding error: {str(e)[:50]}")
        return False


def check_chromadb():
    """Test ChromaDB."""
    print("\n[6/6] Testing ChromaDB...")

    try:
        import chromadb

        # Create a temporary in-memory client
        client = chromadb.Client()
        collection = client.create_collection("test_collection")

        # Test adding and querying
        collection.add(
            documents=["test document"],
            ids=["test_id"]
        )
        results = collection.query(query_texts=["test"], n_results=1)

        print_result(True, "ChromaDB working")
        return True
    except Exception as e:
        print_result(False, f"ChromaDB error: {str(e)[:50]}")
        return False


def main():
    """Run all verification checks."""
    print_header()

    checks = [
        check_python_version,
        check_packages,
        check_env_file,
        check_openai_connection,
        check_embeddings,
        check_chromadb,
    ]

    results = []
    for check in checks:
        try:
            results.append(check())
        except Exception as e:
            print_result(False, f"Check failed with error: {e}")
            results.append(False)

    # Print summary
    print("\n" + "=" * 50)
    if all(results):
        print("  \033[92m✓ All checks passed! You're ready to go!\033[0m")
    else:
        failed = len([r for r in results if not r])
        print(f"  \033[91m✗ {failed} check(s) failed. Please fix and retry.\033[0m")
    print("=" * 50 + "\n")

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
DevOps Chatbot - Setup Verification Script
==========================================

This script verifies that your development environment is correctly set up
to run the DevOps Troubleshooting Chatbot.

WHAT THIS SCRIPT DOES:
---------------------
1. Checks Python version (need 3.9+)
2. Verifies required packages are installed
3. Checks .env file exists with API keys
4. Tests connection to LLM (OpenAI or Ollama)
5. Tests embedding model for RAG
6. Tests vector database (ChromaDB)

HOW TO RUN:
----------
    python module-0-setup/verify_setup.py

    Or from the run.py script:
    python run.py --verify --module 0

UNDERSTANDING THE OUTPUT:
------------------------
    ✓ = Check passed (green)
    ✗ = Check failed (red) - read the message for what to fix

TROUBLESHOOTING:
---------------
- If packages are missing: pip install -r requirements.txt
- If .env not found: cp .env.example .env
- If OpenAI fails: Check your API key in .env
- If Ollama fails: Make sure Ollama is running (ollama serve)
"""

import sys
import os
from pathlib import Path

# =============================================================================
# PATH SETUP
# =============================================================================
# We add the parent directory to Python's path so we can import our modules.
# This is needed because this script is in a subdirectory (module-0-setup/)
# but needs to access files in the parent directory.
#
# sys.path is a list of directories where Python looks for modules to import.
# By adding the parent directory, Python can find our project modules.
# =============================================================================
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_header():
    """
    Print a nice header for the verification output.

    This uses simple string multiplication to create divider lines.
    "=" * 50 creates a string of 50 equal signs.
    """
    print("\n" + "=" * 50)
    print("  DevOps Chatbot - Setup Verification")
    print("=" * 50 + "\n")


def print_result(success: bool, message: str):
    """
    Print a check result with color-coded icon.

    Args:
        success: True if the check passed, False if it failed
        message: Description of what was checked

    How the colors work:
    - \033[92m = ANSI escape code for green color
    - \033[91m = ANSI escape code for red color
    - \033[0m  = ANSI escape code to reset color back to default

    ANSI escape codes are special character sequences that terminals
    interpret as formatting commands rather than text to display.
    """
    # Choose icon based on success/failure
    icon = "✓" if success else "✗"

    # Choose color code based on success/failure
    # Green (92) for success, Red (91) for failure
    color_code = "\033[92m" if success else "\033[91m"

    # Reset code returns terminal to default color
    reset_code = "\033[0m"

    # Print the formatted result
    # f-strings allow us to embed variables directly in the string
    print(f"  {color_code}{icon}{reset_code} {message}")


def check_python_version():
    """
    Check that Python version is 3.9 or higher.

    WHY 3.9+?
    - We use features like type hints that require Python 3.9+
    - Many AI/ML libraries require Python 3.9+
    - It's the minimum version supported by most modern packages

    HOW IT WORKS:
    - sys.version_info returns a tuple: (major, minor, micro, releaselevel, serial)
    - For Python 3.11.5, this would be (3, 11, 5, 'final', 0)
    - We check if major is 3 and minor is 9 or higher
    """
    print("[1/6] Checking Python version...")

    # Get version information from sys module
    # sys is a built-in module that provides access to Python interpreter info
    version = sys.version_info

    # Check if version meets our requirements
    # We need Python 3.9 or higher
    if version.major >= 3 and version.minor >= 9:
        # Version is good - print success message
        print_result(True, f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        # Version is too old - print failure message
        print_result(False, f"Python {version.major}.{version.minor} (need 3.9+)")
        return False


def check_packages():
    """
    Check that all required Python packages are installed.

    WHY THESE PACKAGES?
    - openai: Client library to call OpenAI's API
    - langchain: Framework for building LLM applications
    - chromadb: Vector database for storing embeddings (used in RAG)
    - sentence_transformers: For creating text embeddings locally
    - python-dotenv: For loading .env files with API keys
    - rich: For pretty terminal output in the chatbot

    HOW IT WORKS:
    - __import__(name) dynamically imports a module by name
    - If the import succeeds, the package is installed
    - If it raises ImportError, the package is missing

    NOTE: The import name and package name are sometimes different:
    - python-dotenv is installed as 'python-dotenv' but imported as 'dotenv'
    - sentence-transformers is imported as 'sentence_transformers' (underscore)
    """
    print("\n[2/6] Checking required packages...")

    # List of (import_name, package_name) tuples
    # import_name: how you import it in Python code
    # package_name: how you install it with pip
    required_packages = [
        ("openai", "openai"),                           # OpenAI API client
        ("langchain", "langchain"),                     # LLM application framework
        ("chromadb", "chromadb"),                       # Vector database
        ("sentence_transformers", "sentence-transformers"),  # Embedding models
        ("dotenv", "python-dotenv"),                    # Environment variable loading
        ("rich", "rich"),                               # Pretty terminal output
    ]

    all_installed = True  # Track if all packages are found

    for import_name, package_name in required_packages:
        try:
            # Try to import the package
            # __import__ is Python's built-in function for dynamic imports
            # It's equivalent to writing "import openai" but with a variable name
            __import__(import_name)

            # Import succeeded - package is installed
            print_result(True, f"{package_name} installed")

        except ImportError:
            # Import failed - package is not installed
            print_result(False, f"{package_name} NOT installed")
            all_installed = False

    return all_installed


def check_env_file():
    """
    Check that .env file exists and contains required variables.

    WHAT IS A .ENV FILE?
    - A text file that stores configuration and secrets
    - Format: KEY=value, one per line
    - Keeps sensitive data (API keys) out of your code
    - Should NEVER be committed to Git (add to .gitignore)

    WHY USE .ENV FILES?
    - Security: API keys aren't in your code/repository
    - Flexibility: Easy to change settings without editing code
    - Environment-specific: Different settings for dev/prod

    HOW PYTHON-DOTENV WORKS:
    1. load_dotenv() reads the .env file
    2. It sets the values as environment variables
    3. os.getenv("KEY") retrieves the values
    """
    print("\n[3/6] Checking environment variables...")

    # Build the path to .env file
    # Path(__file__) = this script's location
    # .parent = module-0-setup/ directory
    # .parent = project root directory
    # / ".env" = append the filename
    env_path = Path(__file__).parent.parent / ".env"

    # Check if .env file exists
    if not env_path.exists():
        print_result(False, ".env file not found (copy from .env.example)")
        return False

    print_result(True, ".env file found")

    # Load environment variables from .env file
    # This makes them available via os.getenv()
    from dotenv import load_dotenv
    load_dotenv(env_path)

    # Check which LLM backend to use: Ollama (local) or OpenAI (cloud)
    # os.getenv() returns the value of an environment variable
    # The second argument is the default if the variable isn't set
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # User wants to use Ollama (free, runs locally)
        print_result(True, "Using Ollama (local LLM)")
        return True
    else:
        # User wants to use OpenAI (paid, runs in cloud)
        api_key = os.getenv("OPENAI_API_KEY", "")

        # Check if key is set and not the placeholder value
        if api_key and api_key != "sk-your-api-key-here":
            print_result(True, "OPENAI_API_KEY configured")
            return True
        else:
            print_result(False, "OPENAI_API_KEY not set (edit .env file)")
            return False


def check_openai_connection():
    """
    Test actual connection to the LLM (OpenAI or Ollama).

    This makes a real API call to verify:
    - Network connectivity
    - API key validity
    - Service availability

    OPENAI VS OLLAMA:

    OpenAI:
    - Cloud-based service
    - Requires API key and payment
    - More powerful models (GPT-4, GPT-4o)
    - No local resources needed
    - Requires internet connection

    Ollama:
    - Runs locally on your computer
    - Free to use
    - Models: Llama2, Mistral, etc.
    - Requires 8GB+ RAM
    - Works offline

    HOW THE TEST WORKS:
    - OpenAI: Makes a minimal chat completion request
    - Ollama: Calls the /api/tags endpoint to list models
    """
    print("\n[4/6] Testing LLM connection...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Check which backend to use
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        # =====================================================================
        # OLLAMA CONNECTION TEST
        # =====================================================================
        # Ollama runs a local HTTP server (default: localhost:11434)
        # We call the /api/tags endpoint which lists available models
        # If this succeeds, Ollama is running and accessible
        # =====================================================================
        try:
            import requests  # HTTP library for making web requests

            # Get Ollama URL from env, default to localhost
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

            # Make a GET request to the tags endpoint
            # timeout=5 means give up if no response in 5 seconds
            response = requests.get(f"{base_url}/api/tags", timeout=5)

            # Check if request was successful (status code 200)
            if response.status_code == 200:
                print_result(True, "Successfully connected to Ollama")
                return True
            else:
                print_result(False, "Ollama connection failed")
                return False

        except Exception as e:
            # Something went wrong - maybe Ollama isn't running
            # str(e)[:50] truncates the error message to 50 characters
            print_result(False, f"Ollama error: {str(e)[:50]}")
            return False
    else:
        # =====================================================================
        # OPENAI CONNECTION TEST
        # =====================================================================
        # We make a minimal API call to test:
        # 1. API key is valid
        # 2. We can reach OpenAI's servers
        # 3. We have quota/credits available
        #
        # We ask for a very short response ("Say 'OK'") to minimize cost
        # max_tokens=5 limits the response length
        # =====================================================================
        try:
            from openai import OpenAI

            # Create OpenAI client
            # It automatically reads OPENAI_API_KEY from environment
            client = OpenAI()

            # Make a minimal API call
            # This tests that everything works without using much quota
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=5  # Limit response to save money
            )

            print_result(True, "Successfully connected to OpenAI")
            return True

        except Exception as e:
            # API call failed - could be invalid key, no quota, etc.
            error_msg = str(e)[:50]
            print_result(False, f"OpenAI error: {error_msg}")
            return False


def check_embeddings():
    """
    Test that the embedding model works.

    WHAT ARE EMBEDDINGS?
    - Numbers that represent the meaning of text
    - Similar texts have similar numbers
    - Used to find relevant documents in RAG

    Example:
        "cat" → [0.2, 0.8, 0.1, ...]
        "kitten" → [0.3, 0.7, 0.2, ...]  (similar to cat!)
        "car" → [0.9, 0.1, 0.5, ...]  (very different)

    WHY LOCAL EMBEDDINGS?
    - Faster: No network latency
    - Free: No API costs
    - Private: Text never leaves your computer
    - Offline: Works without internet

    THE MODEL WE USE:
    - all-MiniLM-L6-v2 from sentence-transformers
    - Small and fast (22M parameters)
    - Good quality for most use cases
    - Creates 384-dimensional vectors
    """
    print("\n[5/6] Testing embeddings...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Check which embedding provider to use
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local")

    try:
        if embedding_provider == "local":
            # =================================================================
            # LOCAL EMBEDDINGS (using sentence-transformers)
            # =================================================================
            # This runs entirely on your computer
            # First time: Downloads the model (~90MB)
            # After that: Loads from disk
            # =================================================================
            from sentence_transformers import SentenceTransformer

            # Get model name from env or use default
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

            # Load the model (downloads if first time)
            model = SentenceTransformer(model_name)

            # Test encoding some text
            # This converts "test" into a vector of numbers
            embedding = model.encode("test")

            print_result(True, f"Local embedding model loaded ({model_name})")

        else:
            # =================================================================
            # OPENAI EMBEDDINGS (cloud-based)
            # =================================================================
            # Uses OpenAI's embedding API
            # Costs money but very high quality
            # =================================================================
            from openai import OpenAI
            client = OpenAI()

            # Call the embeddings API
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
    """
    Test that ChromaDB (vector database) works.

    WHAT IS CHROMADB?
    - A database optimized for storing and searching vectors
    - Used in RAG to find relevant documents
    - Stores embeddings and lets you search by similarity

    WHY DO WE NEED IT?
    - Normal databases search by exact match
    - Vector databases search by meaning/similarity
    - Essential for the "Retrieval" part of RAG

    HOW IT WORKS:
    1. Store documents with their embeddings
    2. When user asks a question, embed the question
    3. Find documents with similar embeddings
    4. Return those documents as context for the LLM

    Example:
        Stored: "Kubernetes pods can crash due to OOM" → [0.3, 0.7, ...]
        Query:  "Why is my pod crashing?"             → [0.4, 0.6, ...]
        Result: High similarity! Return this document.
    """
    print("\n[6/6] Testing ChromaDB...")

    try:
        import chromadb

        # Create a temporary in-memory client
        # In production, we use PersistentClient to save to disk
        # For testing, in-memory is faster and cleaner
        client = chromadb.Client()

        # Create a test collection
        # Collections are like tables in regular databases
        # Each collection stores related documents
        collection = client.create_collection("test_collection")

        # Test adding a document
        # - documents: The actual text we're storing
        # - ids: Unique identifier for each document
        # Normally we'd also pass embeddings, but ChromaDB can generate them
        collection.add(
            documents=["test document"],
            ids=["test_id"]
        )

        # Test querying
        # This searches for documents similar to "test"
        # n_results=1 means return only the top 1 result
        results = collection.query(query_texts=["test"], n_results=1)

        print_result(True, "ChromaDB working")
        return True

    except Exception as e:
        print_result(False, f"ChromaDB error: {str(e)[:50]}")
        return False


def main():
    """
    Run all verification checks.

    This function:
    1. Prints a header
    2. Runs each check function
    3. Collects results
    4. Prints a summary
    5. Returns exit code (0 = success, 1 = failure)

    EXIT CODES:
    - 0: All checks passed
    - 1: One or more checks failed

    Exit codes are important for automation:
    - Scripts can check if setup succeeded
    - CI/CD pipelines can fail on non-zero exit
    """
    print_header()

    # List of all check functions to run
    checks = [
        check_python_version,    # Check 1/6
        check_packages,          # Check 2/6
        check_env_file,          # Check 3/6
        check_openai_connection, # Check 4/6
        check_embeddings,        # Check 5/6
        check_chromadb,          # Check 6/6
    ]

    # Run each check and collect results
    results = []
    for check in checks:
        try:
            # Call the check function and store result
            results.append(check())
        except Exception as e:
            # If a check crashes, mark it as failed
            print_result(False, f"Check failed with error: {e}")
            results.append(False)

    # Print summary
    print("\n" + "=" * 50)

    if all(results):
        # All checks passed! (all() returns True if every item is True)
        print("  \033[92m✓ All checks passed! You're ready to go!\033[0m")
    else:
        # Some checks failed
        # Count how many failed using list comprehension
        failed = len([r for r in results if not r])
        print(f"  \033[91m✗ {failed} check(s) failed. Please fix and retry.\033[0m")

    print("=" * 50 + "\n")

    # Return exit code
    # 0 = success (all True), 1 = failure (at least one False)
    return 0 if all(results) else 1


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================
# This block only runs when the script is executed directly:
#   python verify_setup.py    → runs main()
#   import verify_setup       → does NOT run main()
#
# This is a Python best practice that allows the file to be:
# 1. Run as a script (executes main)
# 2. Imported as a module (just loads the functions)
# =============================================================================
if __name__ == "__main__":
    sys.exit(main())

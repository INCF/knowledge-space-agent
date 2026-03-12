#!/usr/bin/env python3
"""
KnowledgeSpace Agent - Basic Demo
=================================

This script demonstrates how to use the KnowledgeSpace AI Agent programmatically.
It shows the core workflow: initializing the agent, sending a query, and 
inspecting the response.

Requirements:
    - Python 3.11+
    - Dependencies installed (uv sync or pip install)
    - GOOGLE_API_KEY environment variable set

Usage:
    cd examples
    python basic_demo.py

For more details, see examples/README.md
"""

import os
import sys
import asyncio

# Add the backend directory to Python path so we can import the agent
# This allows running the script from the examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv is optional


def print_separator(title: str = "") -> None:
    """Print a visual separator for better output readability."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def check_environment() -> bool:
    """
    Verify that required environment variables are set.
    Returns True if environment is properly configured.
    """
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    use_vertex = os.getenv("GEMINI_USE_VERTEX", "false").lower() in ("true", "1", "yes")
    
    if use_vertex:
        # Vertex AI mode requires GCP_PROJECT_ID
        project_id = os.getenv("GCP_PROJECT_ID")
        if not project_id:
            print("❌ Error: GEMINI_USE_VERTEX is enabled but GCP_PROJECT_ID is not set.")
            print("   Set GCP_PROJECT_ID or disable Vertex mode with GEMINI_USE_VERTEX=false")
            return False
        print(f"✓ Using Vertex AI mode (project: {project_id})")
    else:
        # Standard API key mode
        if not api_key:
            print("❌ Error: GOOGLE_API_KEY environment variable is not set.")
            print("")
            print("To fix this:")
            print("  1. Get a free API key from: https://aistudio.google.com/apikey")
            print("  2. Set the environment variable:")
            print("")
            print("     Windows PowerShell:")
            print('       $env:GOOGLE_API_KEY = "your_key_here"')
            print("")
            print("     Windows CMD:")
            print("       set GOOGLE_API_KEY=your_key_here")
            print("")
            print("     Linux/macOS:")
            print("       export GOOGLE_API_KEY=your_key_here")
            print("")
            print("  Or add it to a .env file in the project root.")
            return False
        print("✓ Using Google API Key mode")
    
    return True


async def run_demo() -> None:
    """
    Main demo function that shows how to use the KnowledgeSpace Agent.
    
    This demonstrates:
    1. Initializing the NeuroscienceAssistant
    2. Sending a sample neuroscience query
    3. Displaying the response
    """
    
    print_separator("KnowledgeSpace Agent - Basic Demo")
    
    # Step 1: Check environment
    print("\nChecking environment configuration...")
    if not check_environment():
        return
    
    # Step 2: Import and initialize the agent
    # Note: We import here (after path setup) to avoid import errors
    print("\nInitializing the NeuroscienceAssistant...")
    
    try:
        from agents import NeuroscienceAssistant
        assistant = NeuroscienceAssistant()
        print("✓ Agent initialized successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've installed dependencies: uv sync")
        return
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    # Step 3: Define a sample query
    # You can modify this to test different queries
    sample_queries = [
        "Find datasets about hippocampus neurons in mice",
        # Alternative queries you can try:
        # "Show me human EEG datasets with BIDS format",
        # "What fMRI datasets are available with CC0 license?",
        # "Find electrophysiology recordings from rat prefrontal cortex",
    ]
    
    query = sample_queries[0]
    
    print(f'\nSending query: "{query}"')
    print("\nProcessing... (this may take a few seconds)")
    
    # Step 4: Send the query to the agent
    # The handle_chat method is async, so we await it
    try:
        response = await assistant.handle_chat(
            session_id="demo_session",  # Unique session ID for conversation history
            query=query,
            reset=True,  # Start fresh (clear any previous conversation)
        )
    except Exception as e:
        print(f"\n❌ Error during query processing: {e}")
        print("   This might be due to API rate limits or network issues.")
        print("   Wait a moment and try again.")
        return
    
    # Step 5: Display the response
    print_separator("Agent Response")
    print(response)
    
    # Step 6: (Optional) Inspect internal session state for learning/debugging
    #
    # NOTE:
    # The following section accesses internal session memory for educational purposes.
    # This is NOT part of the public API and may change in future versions.
    # New users can safely ignore this section.
    print_separator("Session Details (for debugging)")
    
    session_memory = {}

    if hasattr(assistant, "session_memory"):
        session_memory = assistant.session_memory.get("demo_session", {})
    
    if session_memory:
        print(f"\n📌 Effective Query: {session_memory.get('effective_query', 'N/A')}")
        # Best-effort fields: availability may vary depending on agent configuration
        print(f"📌 Detected Intents: {session_memory.get('intents', [])}")
        print(f"📌 Extracted Keywords: {session_memory.get('keywords', [])}")
        print(f"📌 Total Results Found: {len(session_memory.get('all_results', []))}")
    else:
        print("(No session memory available)")
    
    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("  - Modify the 'query' variable above to try different searches")
    print("  - Explore backend/agents.py to understand the full implementation")
    print("  - Visit https://chat.knowledge-space.org/ for the full web interface")


def main():
    """Entry point for the demo script."""
    try:
        # Run the async demo function
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

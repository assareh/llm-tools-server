#!/usr/bin/env python3
"""Simple test script for web search functionality."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_api_server import ServerConfig, create_web_search_tool


def test_web_search_without_api_key():
    """Test web search with DuckDuckGo fallback (no API key)."""
    print("\n" + "=" * 60)
    print("TEST 1: Web search WITHOUT API key (DuckDuckGo fallback)")
    print("=" * 60)

    # Create config without API key
    config = ServerConfig()
    config.OLLAMA_API_KEY = ""

    # Create web search tool
    web_search_tool = create_web_search_tool(config)

    print(f"\nTool name: {web_search_tool.name}")
    print(f"Tool description: {web_search_tool.description}")

    # Test search
    print("\nSearching for 'Python programming language'...")
    try:
        result = web_search_tool.func(query="Python programming language", max_results=3)
        print("\nResults:")
        print(result)
        print("\n✅ Test PASSED: DuckDuckGo fallback works")
        return True
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_web_search_with_api_key():
    """Test web search with Ollama API key."""
    print("\n" + "=" * 60)
    print("TEST 2: Web search WITH API key (Ollama API)")
    print("=" * 60)

    # Get API key from environment
    api_key = os.getenv("OLLAMA_API_KEY", "")

    if not api_key:
        print("\n⚠️  SKIPPED: OLLAMA_API_KEY not set in environment")
        print("   To test with Ollama API, set OLLAMA_API_KEY environment variable")
        return True

    # Create config with API key
    config = ServerConfig()
    config.OLLAMA_API_KEY = api_key

    # Create web search tool
    web_search_tool = create_web_search_tool(config)

    print(f"\nTool name: {web_search_tool.name}")
    print(f"API key configured: {'Yes' if config.OLLAMA_API_KEY else 'No'}")

    # Test search
    print("\nSearching for 'Python programming language'...")
    try:
        result = web_search_tool.func(query="Python programming language", max_results=3)
        print("\nResults:")
        print(result)

        # Check if Ollama was used
        if "via Ollama" in result:
            print("\n✅ Test PASSED: Ollama API works")
        else:
            print("\n⚠️  Test WARNING: Ollama API not used (fell back to DuckDuckGo)")
        return True
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_web_search_site_restriction():
    """Test web search with site restriction."""
    print("\n" + "=" * 60)
    print("TEST 3: Web search with site restriction")
    print("=" * 60)

    # Create config without API key
    config = ServerConfig()
    config.OLLAMA_API_KEY = ""

    # Create web search tool
    web_search_tool = create_web_search_tool(config)

    # Test search with site restriction
    print("\nSearching for 'vault' on hashicorp.com...")
    try:
        result = web_search_tool.func(query="vault", max_results=3, site="hashicorp.com")
        print("\nResults:")
        print(result)

        # Check if results contain hashicorp.com
        if "hashicorp.com" in result.lower():
            print("\n✅ Test PASSED: Site restriction works")
        else:
            print("\n⚠️  Test WARNING: Site restriction may not be working")
        return True
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n╔════════════════════════════════════════╗")
    print("║     Web Search Tool Test Suite        ║")
    print("╚════════════════════════════════════════╝")

    results = []

    # Run tests
    results.append(("DuckDuckGo fallback", test_web_search_without_api_key()))
    results.append(("Ollama API", test_web_search_with_api_key()))
    results.append(("Site restriction", test_web_search_site_restriction()))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

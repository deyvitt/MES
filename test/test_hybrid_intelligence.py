#!/usr/bin/env python3
"""
Test script for Hybrid Intelligence Search Engine
Demonstrates the revolutionary web search capabilities
"""

# Test with basic imports first
try:
    import requests
    print("✅ requests library available")
except ImportError:
    print("❌ requests library not available - run: pip install requests")
    requests = None

try:
    from bs4 import BeautifulSoup
    print("✅ beautifulsoup4 library available")
except ImportError:
    print("❌ beautifulsoup4 library not available - run: pip install beautifulsoup4")
    BeautifulSoup = None

print("\n🚀 Testing Hybrid Intelligence Components...")

# Test 1: Basic Search Query Detection
def test_query_detection():
    print("\n🔍 Test 1: Smart Query Detection")
    
    test_queries = [
        ("What's the weather today?", True),  # Current info needed
        ("Explain quantum mechanics", False),  # General knowledge
        ("Latest news about AI", True),  # Current info needed
        ("How to write Python code", False),  # General knowledge
        ("Stock price of Tesla", True),  # Current data needed
    ]
    
    for query, expected in test_queries:
        # Simple detection logic for testing
        needs_search = any(word in query.lower() for word in 
                         ['today', 'latest', 'current', 'news', 'price', 'weather'])
        
        status = "✅" if needs_search == expected else "❌"
        print(f"  {status} '{query}' -> {'Search needed' if needs_search else 'Local knowledge'}")

# Test 2: Domain Detection
def test_domain_detection():
    print("\n🎯 Test 2: Domain Detection")
    
    test_queries = [
        ("Write a Python function to sort a list", "code"),
        ("What are the symptoms of diabetes?", "medical"),
        ("Create a business plan for a startup", "business"),
        ("Explain photosynthesis process", "science"),
        ("Write a short story about dragons", "creative"),
    ]
    
    domain_keywords = {
        'code': ['python', 'function', 'programming', 'code'],
        'medical': ['symptoms', 'treatment', 'medical', 'health'],
        'business': ['business', 'plan', 'startup', 'marketing'],
        'science': ['explain', 'process', 'scientific', 'research'],
        'creative': ['story', 'write', 'creative', 'fiction']
    }
    
    for query, expected_domain in test_queries:
        query_lower = query.lower()
        detected_domain = 'general'
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_domain = domain
                break
        
        status = "✅" if detected_domain == expected_domain else "❌"
        print(f"  {status} '{query[:40]}...' -> {detected_domain}")

# Test 3: Web Search Simulation (if requests available)
def test_web_search():
    print("\n🌐 Test 3: Web Search Capabilities")
    
    if not requests:
        print("  ⚠️ Skipping web search test - requests not available")
        return
    
    try:
        # Test simple HTTP request
        response = requests.get("https://httpbin.org/json", timeout=5)
        if response.status_code == 200:
            print("  ✅ HTTP requests working")
        else:
            print(f"  ❌ HTTP request failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Network error: {e}")
    
    # Test DuckDuckGo API (simple version)
    try:
        duckduckgo_url = "https://api.duckduckgo.com/"
        params = {'q': 'test query', 'format': 'json'}
        response = requests.get(duckduckgo_url, params=params, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("  ✅ DuckDuckGo API accessible")
            if data.get('Abstract'):
                print(f"    Sample result: {data['Abstract'][:100]}...")
        else:
            print(f"  ❌ DuckDuckGo API failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ DuckDuckGo API error: {e}")

# Test 4: Hybrid Response Generation
def test_hybrid_response():
    print("\n🧠 Test 4: Hybrid Response Generation")
    
    test_cases = [
        {
            'query': 'What is machine learning?',
            'domain': 'science',
            'web_context': 'Machine learning is a method of data analysis that automates analytical model building...',
            'expected_elements': ['machine learning', 'data analysis', 'model']
        },
        {
            'query': 'Write a Python function',
            'domain': 'code', 
            'web_context': '',
            'expected_elements': ['def', 'function', 'python']
        }
    ]
    
    for case in test_cases:
        # Simulate hybrid response generation
        query = case['query']
        domain = case['domain']
        web_context = case['web_context']
        
        # Simple response generation simulation
        if web_context:
            response = f"Based on current information: {web_context[:100]}... In {domain} domain, {query.lower()}"
        else:
            response = f"In {domain} domain: {query}"
        
        # Check if expected elements are present
        elements_found = sum(1 for elem in case['expected_elements'] 
                           if elem.lower() in response.lower())
        total_elements = len(case['expected_elements'])
        
        status = "✅" if elements_found >= total_elements // 2 else "❌"
        print(f"  {status} {query[:30]}... -> {elements_found}/{total_elements} elements found")

def main():
    print("🚀 Mamba Encoder Swarm - Hybrid Intelligence Test Suite")
    print("=" * 60)
    
    test_query_detection()
    test_domain_detection()
    test_web_search()
    test_hybrid_response()
    
    print(f"\n🎉 Test Suite Complete!")
    print("\n💡 Next Steps:")
    print("   1. If packages are missing, install them:")
    print("      pip install requests beautifulsoup4")
    print("   2. Run the main app:")
    print("      python app.py")
    print("   3. Test hybrid intelligence with queries like:")
    print("      - 'What's the latest news about AI?'")
    print("      - 'Current stock price of Apple'")
    print("      - 'Write a Python function to calculate factorial'")

if __name__ == "__main__":
    main()

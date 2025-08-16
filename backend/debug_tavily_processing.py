#!/usr/bin/env python3
"""
Debug the Tavily result processing pipeline.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

# Load environment variables
load_dotenv()

def debug_tavily_processing():
    """Debug the complete Tavily processing pipeline."""
    
    print("🔍 Debugging Tavily Processing Pipeline")
    print("=" * 60)
    
    try:
        # Import our modules
        from langchain_tavily import TavilySearch
        from agent.tavily_processor import (
            process_search_results_for_ai,
            validate_and_clean_content,
            extract_tavily_results
        )
        
        # Test query
        test_query = "人工智能 2024"
        print(f"🔍 Testing query: '{test_query}'")
        
        # Step 1: Raw Tavily search
        print(f"\n1️⃣ Raw Tavily Search...")
        tavily_search = TavilySearch(max_results=5)
        raw_response = tavily_search.invoke(test_query)
        
        print(f"✅ Raw response type: {type(raw_response)}")
        if isinstance(raw_response, dict):
            results = raw_response.get('results', [])
            print(f"📊 Raw results count: {len(results)}")
            
            if results:
                print(f"📄 First raw result:")
                first_result = results[0]
                print(f"   Title: {first_result.get('title', 'No title')}")
                print(f"   URL: {first_result.get('url', 'No URL')}")
                print(f"   Content length: {len(first_result.get('content', ''))}")
                print(f"   Content preview: {first_result.get('content', '')[:200]}...")
        
        # Step 2: Extract results
        print(f"\n2️⃣ Extract Results...")
        extracted_results = extract_tavily_results(raw_response)
        print(f"📊 Extracted results count: {len(extracted_results)}")
        
        if extracted_results:
            first_extracted = extracted_results[0]
            print(f"📄 First extracted result:")
            print(f"   Title: {first_extracted.get('title', 'No title')}")
            print(f"   URL: {first_extracted.get('url', 'No URL')}")
            print(f"   Content length: {len(first_extracted.get('content', ''))}")
        
        # Step 3: Validate and clean content
        print(f"\n3️⃣ Validate and Clean Content...")
        if extracted_results:
            raw_content = extracted_results[0].get('content', '')
            cleaned_content = validate_and_clean_content(raw_content)
            
            print(f"📊 Raw content length: {len(raw_content)}")
            print(f"📊 Cleaned content length: {len(cleaned_content)}")
            print(f"📄 Raw content preview: {raw_content[:200]}...")
            print(f"📄 Cleaned content preview: {cleaned_content[:200]}...")
            
            # Check if cleaning is too aggressive
            if len(cleaned_content) < 20:
                print(f"⚠️ ISSUE FOUND: Cleaned content too short ({len(cleaned_content)} chars)")
                print(f"🔧 This would be filtered out by min_content_length=20")
        
        # Step 4: Full processing pipeline
        print(f"\n4️⃣ Full Processing Pipeline...")
        formatted_sources, sources_metadata = process_search_results_for_ai(
            raw_response,
            max_results=5,
            max_content_length=1500,
            min_content_length=20
        )
        
        print(f"📊 Final formatted sources: {len(formatted_sources)}")
        print(f"📊 Final sources metadata: {len(sources_metadata)}")
        
        if not formatted_sources:
            print(f"❌ PROBLEM: No sources after processing!")
            print(f"🔧 Let's test with lower min_content_length...")
            
            # Try with very low minimum
            formatted_sources_low, sources_metadata_low = process_search_results_for_ai(
                raw_response,
                max_results=5,
                max_content_length=1500,
                min_content_length=1  # Very low threshold
            )
            
            print(f"📊 With min_length=1: {len(formatted_sources_low)} sources")
            
            if formatted_sources_low:
                print(f"✅ Issue is min_content_length threshold!")
                print(f"📄 Sample processed source:")
                print(formatted_sources_low[0][:300] + "...")
            else:
                print(f"❌ Still no sources - deeper issue in processing")
        else:
            print(f"✅ Processing successful!")
            print(f"📄 Sample processed source:")
            print(formatted_sources[0][:300] + "...")
        
        return len(formatted_sources) > 0
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_tavily_processing()
    if success:
        print(f"\n🏆 SUCCESS: Tavily processing is working!")
    else:
        print(f"\n❌ ISSUE: Found problem in Tavily processing pipeline")
        print(f"💡 Check the debug output above for specific issues")
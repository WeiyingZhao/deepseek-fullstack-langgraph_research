"""Tavily Search Results Processing Utilities

This module provides utilities for processing and formatting Tavily search results
to ensure optimal consumption by AI models, particularly DeepSeek.
"""

from typing import List, Dict, Any, Optional
import json
import re


def validate_and_clean_content(content: str) -> str:
    """Clean and validate content from search results.
    
    Args:
        content: Raw content from search result
        
    Returns:
        Cleaned and formatted content string
    """
    if not content or not isinstance(content, str):
        return ""
    
    # Remove excessive whitespace and normalize
    content = re.sub(r'\s+', ' ', content.strip())
    
    # Remove common noise patterns
    noise_patterns = [
        r'\b(Cookie|Privacy Policy|Terms of Service)\b.*?(?=\.|$)',
        r'\b(Advertisement|Ad)\b.*?(?=\.|$)',
        r'JavaScript.*?(?=\.|$)',
        r'Enable JavaScript.*?(?=\.|$)',
        r'This website uses cookies.*?(?=\.|$)',
    ]
    
    for pattern in noise_patterns:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE)
    
    return content.strip()


def extract_tavily_results(search_response: Any) -> List[Dict[str, Any]]:
    """Extract results from various Tavily response formats.
    
    Args:
        search_response: Response from Tavily search API
        
    Returns:
        List of normalized search result dictionaries
    """
    results_to_process = []
    
    # Handle different response formats
    if isinstance(search_response, dict):
        if 'results' in search_response:
            results_to_process = search_response['results']
        elif 'answer' in search_response:
            # Handle direct answer format
            results_to_process = [{
                'title': 'Direct Answer',
                'url': '',
                'content': search_response['answer']
            }]
    elif isinstance(search_response, list):
        results_to_process = search_response
    elif isinstance(search_response, str):
        # Try to parse JSON string
        try:
            parsed = json.loads(search_response)
            if isinstance(parsed, dict) and 'results' in parsed:
                results_to_process = parsed['results']
            else:
                results_to_process = [{
                    'title': 'Search Result',
                    'url': '',
                    'content': search_response
                }]
        except json.JSONDecodeError:
            results_to_process = [{
                'title': 'Search Result',
                'url': '',
                'content': search_response
            }]
    
    return results_to_process


def process_search_results_for_ai(
    search_response: Any,
    max_results: int = 5,
    max_content_length: int = 1500,
    min_content_length: int = 20
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Process Tavily search results for optimal AI model consumption.
    
    Args:
        search_response: Raw response from Tavily API
        max_results: Maximum number of results to process
        max_content_length: Maximum length for individual content pieces
        min_content_length: Minimum length for content to be included
        
    Returns:
        Tuple of (formatted_sources, sources_metadata)
    """
    raw_results = extract_tavily_results(search_response)
    
    formatted_sources = []
    sources_gathered = []
    
    valid_result_count = 0
    
    for i, result in enumerate(raw_results):
        if valid_result_count >= max_results:
            break
            
        if not isinstance(result, dict):
            continue
            
        # Extract and validate basic fields
        title = result.get('title', f'Search Result {valid_result_count + 1}').strip()
        url = result.get('url', '').strip()
        raw_content = result.get('content', '').strip()
        
        # Clean and validate content
        clean_content = validate_and_clean_content(raw_content)
        
        # Skip if content is too short or empty
        if len(clean_content) < min_content_length:
            continue
            
        # Truncate if too long
        if len(clean_content) > max_content_length:
            clean_content = clean_content[:max_content_length] + "..."
        
        valid_result_count += 1
        source_id = f"[{valid_result_count}]"
        
        # Format source for AI model consumption with clear structure
        formatted_source = f"""=== Source {valid_result_count} ===
Title: {title}
URL: {url}
Content: {clean_content}
=================="""
        
        formatted_sources.append(formatted_source)
        
        # Store metadata for citation processing
        sources_gathered.append({
            "title": title,
            "url": url,
            "content": clean_content,
            "short_url": source_id,
            "value": url,
            "label": title
        })
    
    return formatted_sources, sources_gathered


def create_structured_search_prompt(
    search_query: str,
    formatted_sources: List[str],
    current_date: str,
    instruction_type: str = "analysis"
) -> str:
    """Create a structured prompt for AI model to process search results.
    
    Args:
        search_query: The original search query
        formatted_sources: List of formatted source content
        current_date: Current date string
        instruction_type: Type of instruction ("analysis", "answer", "summary")
        
    Returns:
        Formatted prompt string optimized for DeepSeek processing
    """
    
    if instruction_type == "answer":
        prompt_template = """You are a professional research analyst. Please carefully analyze the following search results and answer the user's question.

User Question: {search_query}
Current Date: {current_date}

Please process the search results according to the following requirements:
1. Carefully read the content of each source
2. Extract key information and data
3. Synthesize and analyze information from multiple sources
4. Write detailed and accurate answers in English
5. Use [1], [2] etc. markers to cite sources when referencing information

Search Results:
{sources}

Please provide a detailed answer:"""
    
    elif instruction_type == "summary":
        prompt_template = """You are a professional information summary expert. Please carefully analyze the following search results and provide a comprehensive summary about "{search_query}".

Current Date: {current_date}
Research Topic: {search_query}

Please process the search results according to the following requirements:
1. Carefully read the content of each source
2. Identify key themes and important information
3. Synthesize viewpoints from multiple sources
4. Write structured summary reports in English
5. Use [1], [2] etc. markers to cite sources when referencing information

Search Results:
{sources}

Please provide a comprehensive summary:"""
    
    else:  # analysis
        prompt_template = """You are a professional research analyst. Please carefully analyze the following search results and provide a research summary about "{search_query}".

Important Note: This is one step in a multi-step research process. You only need to provide a key information summary for this specific query, not a complete standalone report. The final report will be integrated by subsequent steps that combine all research results.

Current Date: {current_date}
Research Query: {search_query}

Please process the search results according to the following requirements:
1. Carefully read the content of each source
2. Extract key information and data relevant to the query
3. Organize into a concise information summary (300-500 words)
4. Focus on facts, data, and specific cases
5. Use [1], [2] etc. markers to cite sources when referencing information

Search Results:
{sources}

Please provide a concise research summary (do not write executive summary, titles, or complete report format):"""
    
    return prompt_template.format(
        search_query=search_query,
        current_date=current_date,
        sources="\n".join(formatted_sources)
    )


def validate_ai_response(response_content: str, min_length: int = 100) -> bool:
    """Validate the quality of AI model response.
    
    Args:
        response_content: Response content from AI model
        min_length: Minimum acceptable response length
        
    Returns:
        True if response is valid, False otherwise
    """
    if not response_content or not isinstance(response_content, str):
        return False
        
    clean_content = response_content.strip()
    
    # Check minimum length
    if len(clean_content) < min_length:
        return False
        
    # Check for common error patterns
    error_patterns = [
        r"^(Error|Sorry|Cannot|Unable)",
        r"API.*?(error|failed)",
        r"^(Search.*?failed)",
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, clean_content, re.IGNORECASE):
            return False
    
    return True


def process_citations_in_response(
    response_content: str, 
    sources_metadata: List[Dict[str, Any]]
) -> str:
    """Process and normalize citations in AI response.
    
    Args:
        response_content: Raw response from AI model
        sources_metadata: List of source metadata dictionaries
        
    Returns:
        Response with properly formatted citations
    """
    processed_content = response_content
    
    # Replace any direct URL references with citation markers
    for i, source in enumerate(sources_metadata):
        url = source.get('url', '')
        short_url = source.get('short_url', f'[{i+1}]')
        
        if url and short_url:
            # Replace full URL with citation marker
            processed_content = processed_content.replace(url, short_url)
            
            # Also try to replace domain references
            try:
                domain = url.split('/')[2]
                if domain in processed_content and len(domain) > 10:  # Only replace meaningful domains
                    processed_content = processed_content.replace(domain, short_url)
                    
                # Replace common URL patterns
                url_patterns = [
                    f"https://{domain}",
                    f"http://{domain}",
                    domain
                ]
                for pattern in url_patterns:
                    if pattern in processed_content:
                        processed_content = processed_content.replace(pattern, short_url)
                        
            except (IndexError, AttributeError):
                pass
                
        # Also look for title references that could be cited
        title = source.get('title', '')
        if title and len(title) > 20:  # Only meaningful titles
            # If title appears in text but no citation nearby, add citation
            title_lower = title.lower()
            content_lower = processed_content.lower()
            if title_lower in content_lower:
                # Find the position and add citation if not already present
                title_pos = content_lower.find(title_lower)
                if title_pos != -1:
                    # Check if citation is already nearby (within 50 characters)
                    nearby_text = processed_content[max(0, title_pos-25):title_pos+len(title)+25]
                    if short_url not in nearby_text:
                        # Insert citation after the title
                        title_end = title_pos + len(title)
                        processed_content = (
                            processed_content[:title_end] + 
                            short_url + 
                            processed_content[title_end:]
                        )
    
    return processed_content
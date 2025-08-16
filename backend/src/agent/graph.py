import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_openai import ChatOpenAI
from agent.utils import (
    get_research_topic,
)
from agent.tavily_processor import (
    process_search_results_for_ai,
    create_structured_search_prompt,
    validate_ai_response,
    process_citations_in_response,
)

load_dotenv(override=True)

if os.getenv("DEEPSEEK_API_KEY") is None:
    raise ValueError("DEEPSEEK_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

# Initialize Tavily Search
tavily_search = TavilySearch(max_results=5)


# Nodes
async def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses DeepSeek to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init DeepSeek via OpenAI API
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    # Format the prompt
    current_date = get_current_date()
    research_topic = get_research_topic(state["messages"])
    print(f"DEBUG: Original user input: {research_topic}")
    
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=research_topic,
        number_queries=state["initial_search_query_count"],
    )
    
    # Try structured output first, with DeepSeek compatibility fallback
    try:
        structured_llm = llm.with_structured_output(SearchQueryList)
        result = await structured_llm.ainvoke(formatted_prompt)
        search_queries = result.query
    except Exception as e:
        print(f"INFO: Structured output not supported, using text parsing fallback: {str(e)}")
        # Fallback: Use regular LLM and parse manually
        simple_prompt = f"""Based on the following research topic, generate {state["initial_search_query_count"]} search queries:

Research Topic: {get_research_topic(state["messages"])}
Current Date: {current_date}

Please directly list {state["initial_search_query_count"]} search queries, one per line:"""
        
        response = await llm.ainvoke(simple_prompt)
        try:
            # Parse search queries from response
            lines = response.content.strip().split('\n')
            search_queries = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Search query'):
                    # Remove numbering and clean up
                    clean_line = line.split('.', 1)[-1].strip() if '.' in line else line
                    # Remove quotes that DeepSeek sometimes adds incorrectly
                    clean_line = clean_line.strip('"').strip("'").strip()
                    # Remove common unwanted suffixes DeepSeek adds
                    unwanted_suffixes = [' 2025', ' latest', ' report']
                    # Only remove 2024 if it wasn't in the original topic
                    if clean_line.endswith(' 2024') and '2024' not in research_topic:
                        clean_line = clean_line[:-5].strip()
                    
                    for suffix in unwanted_suffixes:
                        if clean_line.endswith(suffix):
                            clean_line = clean_line[:-len(suffix)].strip()
                    if clean_line:
                        search_queries.append(clean_line)
            
            # Ensure we have at least one query
            if not search_queries:
                search_queries = [get_research_topic(state["messages"])]
            elif len(search_queries) > state["initial_search_query_count"]:
                search_queries = search_queries[:state["initial_search_query_count"]]
                
        except Exception:
            search_queries = [get_research_topic(state["messages"])]
    
    print(f"DEBUG: Generated search queries: {search_queries}")
    return {"search_query": search_queries}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


async def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Tavily Search API.

    Executes a web search using Tavily Search API and then uses DeepSeek to analyze and summarize the results.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    
    search_query = state["search_query"]
    
    try:
        # Perform search using Tavily
        print(f"DEBUG: Starting search for query: '{search_query}'")
        print(f"DEBUG: Tavily config - max_results: {tavily_search.max_results}")
        print(f"DEBUG: Search query type: {type(search_query)}, length: {len(search_query)}")
        print(f"DEBUG: Query repr: {repr(search_query)}")
        
        # Use asyncio.to_thread to prevent blocking I/O in async environment
        import asyncio
        search_results = await asyncio.to_thread(tavily_search.invoke, search_query)
        print(f"DEBUG: Raw Tavily response type: {type(search_results)}")
        print(f"DEBUG: Raw Tavily response keys: {list(search_results.keys()) if isinstance(search_results, dict) else 'N/A'}")
        
        if isinstance(search_results, dict):
            raw_results = search_results.get('results', [])
            print(f"DEBUG: Raw Tavily found {len(raw_results)} results")
            if len(raw_results) == 0 and search_results:
                print(f"DEBUG: Full response for debugging: {search_results}")
        else:
            print(f"DEBUG: Unexpected Tavily response format: {search_results}")
            search_results = {"results": []}
        
        # Process search results using the new utility
        formatted_sources, sources_gathered = process_search_results_for_ai(
            search_results,
            max_results=5,
            max_content_length=1500,
            min_content_length=20
        )
        
        print(f"DEBUG: After processing: {len(formatted_sources)} formatted sources, {len(sources_gathered)} metadata")
        
        if not formatted_sources:
            print(f"WARNING: No valid search results found for query: {search_query}")
            return {
                "sources_gathered": [],
                "search_query": [search_query],
                "web_research_result": [f"Could not find relevant information about '{search_query}'. Please try using different keywords."],
            }
        
        # Create structured prompt for DeepSeek
        analysis_prompt = create_structured_search_prompt(
            search_query=search_query,
            formatted_sources=formatted_sources,
            current_date=get_current_date(),
            instruction_type="analysis"
        )
    
        # Use DeepSeek to analyze and summarize the search results
        llm = ChatOpenAI(
            model=configurable.query_generator_model,
            temperature=0.1,  # Slightly higher for better analysis
            max_retries=3,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        
        response = await llm.ainvoke(analysis_prompt)
        
        # Validate response quality using utility function
        if not validate_ai_response(response.content, min_length=100):
            return {
                "sources_gathered": sources_gathered,
                "search_query": [search_query],
                "web_research_result": [f"Analysis results incomplete. Search for '{search_query}' found {len(sources_gathered)} sources, but analysis encountered issues."],
            }
        
        # Process citations in the response
        analysis_result = process_citations_in_response(response.content, sources_gathered)
        
        return {
            "sources_gathered": sources_gathered,
            "search_query": [search_query],
            "web_research_result": [analysis_result],
        }
        
    except Exception as e:
        print(f"ERROR in web_research: {str(e)}")
        return {
            "sources_gathered": [],
            "search_query": [search_query],
            "web_research_result": [f"Error occurred during search: {str(e)}. Please try again later."],
        }


async def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    # Try structured output first, with DeepSeek compatibility fallback
    try:
        result = await llm.with_structured_output(Reflection).ainvoke(formatted_prompt)
        return {
            "is_sufficient": result.is_sufficient,
            "knowledge_gap": result.knowledge_gap,
            "follow_up_queries": result.follow_up_queries,
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }
    except Exception as e:
        print(f"INFO: Structured reflection not supported, using simplified analysis: {str(e)}")
        # Simplified reflection without structured output
        simple_prompt = f"""Analyze the completeness of the following research summaries:

Research Topic: {get_research_topic(state["messages"])}
Summary Content:
{chr(10).join(state["web_research_result"][:3])}

Please answer:
1. Is this information sufficient to answer the user's question? (Yes/No)
2. If not, what additional information is needed?"""
        
        response = await llm.ainvoke(simple_prompt)
        
        # Simple parsing based on keywords
        content = response.content.lower()
        is_sufficient = any(word in content for word in ['sufficient', 'enough', 'complete', 'yes', 'adequate'])
        
        # For demo purposes, limit research loops to avoid infinite loops
        max_loops = state.get("max_research_loops", 1)
        if state["research_loop_count"] >= max_loops:
            is_sufficient = True
        
        return {
            "is_sufficient": is_sufficient,
            "knowledge_gap": "Need more detailed information" if not is_sufficient else "",
            "follow_up_queries": [f"{get_research_topic(state['messages'])} detailed information"] if not is_sufficient else [],
            "research_loop_count": state["research_loop_count"],
            "number_of_ran_queries": len(state["search_query"]),
        }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


async def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to DeepSeek
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    result = await llm.ainvoke(formatted_prompt)

    # Process citations and create proper hyperlinks with references section
    unique_sources = []
    citation_map = {}  # Map citation numbers to sources
    
    for i, source in enumerate(state["sources_gathered"]):
        if source["short_url"] in result.content:
            # Create numbered citation reference
            citation_num = len(unique_sources) + 1
            citation_ref = f"[{citation_num}]"
            
            # Replace short URL with citation reference
            result.content = result.content.replace(
                source["short_url"], citation_ref
            )
            
            unique_sources.append(source)
            citation_map[citation_num] = source

    # Add references section at the end if we have sources
    if unique_sources:
        references_section = "\n\n## References\n\n"
        for citation_num, source in citation_map.items():
            title = source.get("title", "Untitled")
            url = source.get("url", source.get("value", ""))
            
            # Create markdown hyperlink format
            references_section += f"{citation_num}. [{title}]({url})\n"
        
        result.content += references_section

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")

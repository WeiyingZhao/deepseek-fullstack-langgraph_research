import os
from typing import TypedDict, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    answer_instructions,
)
from agent.utils import get_research_topic
from agent.tavily_processor import (
    process_search_results_for_ai,
    create_structured_search_prompt,
    validate_ai_response,
    process_citations_in_response,
)

load_dotenv(override=True)

# Check API keys
if os.getenv("DEEPSEEK_API_KEY") is None:
    raise ValueError("DEEPSEEK_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

# Initialize tools
tavily_search = TavilySearch(max_results=5)


async def research_agent(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    """Simplified research agent that combines query generation, search, and answer generation."""
    
    configurable = Configuration.from_runnable_config(config)
    
    # Get research topic from messages
    research_topic = get_research_topic(state["messages"])
    
    # Initialize DeepSeek LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.1,
        max_retries=2,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    
    try:
        # Perform search using Tavily
        search_response = tavily_search.invoke(research_topic)
        
        # Process search results using the new utility
        formatted_sources, sources_gathered = process_search_results_for_ai(
            search_response,
            max_results=5,
            max_content_length=1500,
            min_content_length=20
        )
        
        if not formatted_sources:
            return {
                "messages": [AIMessage(content=f"Could not find relevant information about '{research_topic}'. Please try using different keywords.")],
                "sources_gathered": [],
            }
        
        # Create structured prompt using utility function
        formatted_prompt = create_structured_search_prompt(
            search_query=research_topic,
            formatted_sources=formatted_sources,
            current_date=get_current_date(),
            instruction_type="answer"
        )
        
        response = await llm.ainvoke(formatted_prompt)
        
        # Validate response quality using utility function
        if not validate_ai_response(response.content, min_length=50):
            return {
                "messages": [AIMessage(content=f"Analysis results incomplete. Search for '{research_topic}' found {len(sources_gathered)} sources, but analysis encountered issues.")],
                "sources_gathered": sources_gathered,
            }
        
        # Process citations in the response
        analysis_result = process_citations_in_response(response.content, sources_gathered)
        
        return {
            "messages": [AIMessage(content=analysis_result)],
            "sources_gathered": sources_gathered,
        }
        
    except Exception as e:
        error_message = f"Error occurred during search: {str(e)}"
        return {
            "messages": [AIMessage(content=error_message)],
            "sources_gathered": [],
        }


# Create simplified graph
builder = StateGraph(OverallState)
builder.add_node("research_agent", research_agent)
builder.set_entry_point("research_agent")
builder.add_edge("research_agent", END)

graph = builder.compile()
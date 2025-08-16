# readme.md

This file provides guidance when working with code in this repository.

## Project Overview

This is a LangGraph-based research agent backend that uses DeepSeek LLM and Tavily Search API to perform intelligent web research. The system operates as a multi-node graph workflow that generates search queries, performs web research in parallel, reflects on results, and produces comprehensive research summaries.

## Architecture

The system follows a LangGraph state machine architecture with these key components:

- **Graph Workflow** (`src/agent/graph.py`): Main orchestration logic with nodes for query generation, web research, reflection, and answer finalization
- **State Management** (`src/agent/state.py`): TypedDict-based state containers for different workflow stages
- **FastAPI Application** (`src/agent/app.py`): Web server that serves both the LangGraph API and optional frontend static files
- **Configuration** (`src/agent/configuration.py`): Runtime configuration for models and parameters
- **Tools & Schemas** (`src/agent/tools_and_schemas.py`): Structured output schemas for LLM interactions
- **Prompts** (`src/agent/prompts.py`): Prompt templates for different workflow nodes

### Workflow Nodes

1. **generate_query**: Creates optimized search queries from user questions
2. **web_research**: Performs parallel web searches using Tavily API and summarizes results with DeepSeek
3. **reflection**: Analyzes research completeness and generates follow-up queries
4. **finalize_answer**: Produces final research report with citations

## Development Commands

### Environment Setup
```bash
# Note: There are dependency version conflicts in the current setup
# The original dependencies specified require incompatible versions

# Manual dependency installation (required due to conflicts):
pip install langchain-community langchain-openai tavily-python fastapi langgraph python-dotenv

# Or use uv for development (recommended):
uv sync

# Set required environment variables in .env file
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Development
```bash
# Run the LangGraph server locally
langgraph dev

# Run tests
make test
uv run --with-editable . pytest tests/unit_tests/

# Run specific test file
make test TEST_FILE=tests/unit_tests/test_specific.py

# Watch mode testing
make test_watch
```

### Code Quality
```bash
# Format code
make format
uv run ruff format .

# Lint code
make lint
uv run ruff check .

# Type checking
uv run mypy --strict src/

# Run all linting (ruff + mypy)
make lint
```

### LangGraph Configuration

The agent is configured via `langgraph.json`:
- Main graph entry point: `src/agent/graph.py:graph`
- FastAPI app: `src/agent/app.py:app`  
- Environment variables loaded from `.env`

### Key Dependencies

- **langgraph**: State graph orchestration framework  
- **langchain-community**: Community integrations including LLM providers
- **langchain-openai**: OpenAI and compatible API integrations (used for DeepSeek)
- **tavily-python**: Tavily search API client
- **fastapi**: Web application framework
- **python-dotenv**: Environment variable management

### Dependency Issues

The original `pyproject.toml` contains dependency conflicts:
- `langchain-deepseek` and `langchain-tavily` packages don't exist as separate packages
- Version conflicts between `langgraph` and `langchain-core` 
- Use `langchain-community` and `tavily-python` instead
- DeepSeek integration is available through the community package

### State Management

The system uses typed state containers:
- `OverallState`: Main workflow state with messages, search queries, results, and sources
- `ReflectionState`: Reflection analysis results
- `QueryGenerationState`: Generated search queries
- `WebSearchState`: Individual search operation state

### Frontend Integration

The FastAPI app can serve a React frontend from `/app` route, with build directory expected at `../frontend/dist`. If frontend build is not found, serves a 503 error message.

### Testing Strategy

Tests should be placed in `tests/unit_tests/` directory. Use pytest for test execution with the project's test configuration.
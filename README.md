# full-stack LangGraph deep research 

This file provides guidance when working with code in this repository.

## Project Overview

This is a full-stack LangGraph research agent application consisting of a React frontend and Python LangGraph backend. The system provides an intelligent web research assistant that uses DeepSeek LLM and Tavily Search API to perform comprehensive research and generate detailed reports.

## Architecture

### Full-Stack Structure
- **Frontend**: React 19 + TypeScript + Vite with LangGraph SDK for real-time streaming
- **Backend**: LangGraph-based agent with DeepSeek LLM and Tavily search integration
- **Deployment**: Docker Compose with Redis and PostgreSQL for production
- **Development**: Local development servers with hot reload

### Key System Flow
1. User submits research query via React frontend
2. LangGraph workflow generates optimized search queries
3. Parallel web research using Tavily API
4. Results reflection and quality assessment
5. Final comprehensive research report with citations
6. Real-time progress updates streamed to frontend

## Development Commands

### Initial Setup
```bash
# Frontend setup (creates node_modules/)
cd frontend && npm install

# Backend setup (creates .venv/)
cd backend && uv sync

# Environment variables (create .env in backend/)
DEEPSEEK_API_KEY=your_deepseek_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### Environment Recreation
These commands recreate the development environment after deleting large folders to reduce project size:

```bash
# Recreate backend virtual environment (if .venv deleted)
cd backend && uv sync

# Alternative backend setup with traditional venv
cd backend && python -m venv venv && source venv/bin/activate && pip install -e .

# Recreate frontend dependencies (if node_modules deleted)
cd frontend && npm install

# Quick setup for both (recommended)
cd backend && uv sync && cd ../frontend && npm install
```

### Size Optimization Commands
```bash
# Remove large folders to reduce project size
rm -rf backend/.venv
rm -rf frontend/node_modules

# Recreate when needed
cd backend && uv sync && cd ../frontend && npm install
```

### Development Servers
```bash
# Start both frontend and backend (recommended)
make dev

# Or individually:
make dev-frontend    # Frontend only (localhost:5173)
make dev-backend     # Backend only (localhost:2024)
```

### Backend Development
```bash
# From backend/ directory
langgraph dev        # Start LangGraph server
make test           # Run unit tests
make test_watch     # Run tests in watch mode
make lint           # Run ruff + mypy linting
make format         # Format code with ruff
```

### Frontend Development
```bash
# From frontend/ directory
npm run dev         # Development server
npm run build       # Production build
npm run lint        # ESLint
npm run preview     # Preview production build
```

### Production Deployment
```bash
# Docker deployment with Redis/PostgreSQL
docker-compose up

# Access at http://localhost:8123
```

## Code Architecture

### Backend Structure (`backend/src/agent/`)
- **`graph.py`**: Main LangGraph workflow with nodes for query generation, web research, reflection, and finalization
- **`state.py`**: TypedDict state management for workflow stages
- **`app.py`**: FastAPI server serving both LangGraph API and frontend static files
- **`configuration.py`**: Runtime configuration for models and search parameters
- **`tools_and_schemas.py`**: Structured output schemas for LLM interactions
- **`prompts.py`**: Prompt templates for different workflow nodes

### Frontend Structure (`frontend/src/`)
- **`App.tsx`**: Main application with LangGraph stream integration
- **`components/`**: UI components including WelcomeScreen, ChatMessagesView, InputForm, ActivityTimeline
- **`lib/utils.ts`**: Utility functions and configurations

### Key Integration Points
- **LangGraph SDK**: Real-time streaming between React frontend and LangGraph backend
- **State Synchronization**: Activity timeline shows live workflow progress
- **API Proxy**: Vite proxies `/api` requests to backend during development

## Testing Strategy

### Backend Testing
```bash
# Run specific test file
make test TEST_FILE=tests/unit_tests/test_specific.py

# Profile tests
make test_profile

# Extended tests
make extended_tests
```

### Frontend Testing
The frontend uses ESLint for code quality. Tests should be added for components and core functionality.

## Configuration Notes

### LangGraph Configuration (`backend/langgraph.json`)
- Graph entry: `src/agent/graph.py:graph`
- FastAPI app: `src/agent/app.py:app`
- Environment from `.env` file

### Docker Configuration
- **Production Image**: `gemini-fullstack-langgraph` (Note: Name should be updated to reflect DeepSeek)
- **Ports**: Frontend/Backend on 8123, PostgreSQL on 5433
- **Environment**: Requires GEMINI_API_KEY and LANGSMITH_API_KEY (Note: Should use DEEPSEEK_API_KEY)

### Frontend Proxy Configuration
- Development: Connects to `localhost:2024` (LangGraph dev server)
- Production: Backend serves frontend from `/app` route
- API requests proxied to backend during development

## Development Workflow

### Adding New Research Features
1. Update state definitions in `backend/src/agent/state.py`
2. Modify workflow graph in `backend/src/agent/graph.py`
3. Add corresponding UI updates in frontend components
4. Test with both development servers running

### Frontend-Backend Integration
- Use `useStream` hook from `@langchain/langgraph-sdk/react`
- Handle real-time events for activity timeline updates
- Maintain message history and research session state

## Dependencies & Known Issues (RESOLVED)

### Tavily + DeepSeek Integration Fix (CRITICAL)
**Issue**: DeepSeek AI model was unable to read/process search results from Tavily API

**Root Cause**: 
- Poor content formatting and structure in search result processing
- Aggressive content truncation losing critical information
- Suboptimal prompt structure for AI model consumption
- Inconsistent data validation and error handling

**Solution Implemented**:
- Created `tavily_processor.py` utility module for consistent result processing
- Improved content cleaning and validation (up to 1500 chars vs 500)
- Structured prompt templates optimized for DeepSeek consumption
- Enhanced citation processing and response validation
- Better error handling and fallback mechanisms

**Files Updated**:
- `src/agent/graph.py`: Main LangGraph workflow
- `src/agent/simple_graph.py`: Simplified research agent
- `src/agent/tavily_processor.py`: New utility module (ADDED)
- `test_tavily_fix.py`: Comprehensive test suite (ADDED)

**Testing**:
```bash
# Test core integration (simple graph)
cd backend && python test_simple_fix.py

# Test full workflow (complete graph)
cd backend && python test_full_workflow.py

# Comprehensive test suite
cd backend && python test_tavily_fix.py
```

**Test Results**:
- ✅ **Simple Graph**: DeepSeek successfully processes Tavily results with proper citations
- ✅ **Full Workflow**: Complete research pipeline working with structured output fallbacks
- ✅ **Core Issue Resolved**: AI model can now read and analyze search results effectively

### Backend Dependencies
- ✅ **FIXED**: `langchain_tavily` package is properly installed and working
- DeepSeek integration via `langchain_openai` with custom base URL
- Tavily integration via `langchain_tavily` package (confirmed working)

### Environment Variables
Backend requires:
- `DEEPSEEK_API_KEY`: For DeepSeek LLM integration
- `TAVILY_API_KEY`: For web search functionality
- Optional: `LANGSMITH_API_KEY` for tracing

### Docker Configuration Mismatch
- Docker compose references `GEMINI_API_KEY` but should use `DEEPSEEK_API_KEY`
- Image name `gemini-fullstack-langgraph` should be updated to reflect DeepSeek usage
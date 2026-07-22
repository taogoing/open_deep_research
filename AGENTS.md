# Open Deep Research Repository Overview

## Project Description
Open Deep Research is a configurable, fully open-source deep research agent that works across multiple model providers, search tools, and MCP (Model Context Protocol) servers. It enables automated research with parallel processing and comprehensive report generation.

## Repository Structure

### Root Directory
- `README.md` - Comprehensive project documentation with quickstart guide
- `pyproject.toml` - Python project configuration and dependencies
- `langgraph.json` - LangGraph configuration defining the main graph entry point
- `uv.lock` - UV package manager lock file
- `LICENSE` - MIT license
- `.env.example` - Environment variables template (not tracked)

### Core Implementation (`src/open_deep_research/`)
- `deep_researcher.py` - Main LangGraph implementation (entry point: `deep_researcher`)
- `configuration.py` - Configuration management and settings
- `state.py` - Graph state definitions and data structures  
- `prompts.py` - System prompts and prompt templates
- `utils.py` - Utility functions and helpers

### Security (`src/security/`)
- `auth.py` - Authentication handler for LangGraph deployment

### Testing (`tests/`)
- `run_evaluate.py` - Main evaluation script configured to run on deep research bench
- `evaluators.py` - Specialized evaluation functions  
- `prompts.py` - Evaluation prompts and criteria
- `pairwise_evaluation.py` - Comparative evaluation tools
- `supervisor_parallel_evaluation.py` - Multi-threaded evaluation

### Examples (`examples/`)
- `arxiv.md` - ArXiv research example
- `pubmed.md` - PubMed research example
- `inference-market.md` - Inference market analysis examples

## Architecture

The agent uses a **three-level hierarchical subgraph** architecture built on LangGraph 1.x:

1. **Main Graph** - User clarification → Research brief → Supervisor subgraph → Final report
2. **Supervisor Subgraph** - Plans research strategy, delegates to researcher subgraphs in parallel
3. **Researcher Subgraph** - Conducts focused research with search tools, compresses findings

Key LangGraph 1.x patterns used:
- `Command` objects for conditional routing (replaces `add_conditional_edges`)
- Nested subgraphs with `input`/`output` state separation
- Custom `override_reducer` for controlled state updates
- `asyncio.gather` for parallel researcher execution
- `config_schema` for runtime configuration via LangGraph Studio

## Key Technologies
- **LangGraph 1.x** - Workflow orchestration and graph execution
- **LangChain** - LLM integration and tool calling
- **Multiple LLM Providers** - OpenAI, Anthropic, Google, Groq, DeepSeek support
- **Search APIs** - Tavily, OpenAI/Anthropic native search
- **MCP Servers** - Model Context Protocol for extended capabilities

## Development Commands
- `uvx langgraph dev` - Start development server with LangGraph Studio
- `python tests/run_evaluate.py` - Run comprehensive evaluations
- `ruff check` - Code linting
- `mypy` - Type checking

## Configuration
All settings configurable via:
- Environment variables (`.env` file)
- Web UI in LangGraph Studio
- Direct configuration modification

Key settings include model selection, search API choice, concurrency limits, and MCP server configurations.

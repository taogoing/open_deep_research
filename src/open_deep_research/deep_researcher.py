"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_plan_prompt,
    research_system_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearchPlan,
    ResearcherOutputState,
    ResearcherState,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "base_url"),
)


def _model_runtime_config(
    model: str,
    max_tokens: int,
    configurable: Configuration,
    config: RunnableConfig,
) -> dict:
    """Build provider-aware runtime settings for a configurable chat model."""
    settings = {
        "model": model,
        "max_tokens": max_tokens,
        "api_key": get_api_key_for_model(model, config),
        "tags": ["langsmith:nostream"],
    }
    if model.lower().startswith("ollama:"):
        settings["base_url"] = configurable.ollama_base_url
    return settings


async def clarify_with_user(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["request_clarification", "write_research_plan"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_plan")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = _model_runtime_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        configurable,
        config,
    )
    
    # Configure model with structured output and retry logic
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        return Command(
            goto="request_clarification",
            update={"clarification": response},
        )
    return Command(
        goto="write_research_plan",
        update={"messages": [AIMessage(content=response.verification)]},
    )


async def request_clarification(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_plan"]]:
    """Pause for optional scope feedback before creating the plan."""
    clarification = state.get("clarification")
    payload = (
        clarification.model_dump()
        if hasattr(clarification, "model_dump")
        else dict(clarification or {})
    )
    decision = interrupt({"type": "clarification", **payload})
    action = decision.get("action", "skip") if isinstance(decision, dict) else "skip"
    if action == "supplement":
        selected = decision.get("selected", [])
        feedback = str(decision.get("feedback", "")).strip()
        details = "；".join([*selected, feedback] if feedback else selected)
        if details:
            return Command(
                goto="write_research_plan",
                update={"messages": [HumanMessage(content=f"补充研究要求：{details}")]},
            )
    return Command(goto="write_research_plan")


async def write_research_plan(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["review_research_plan"]]:
    """Generate a structured, user-visible research plan."""
    configurable = Configuration.from_runnable_config(config)
    research_model_config = _model_runtime_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        configurable,
        config,
    )
    research_model = (
        configurable_model
        .with_structured_output(ResearchPlan)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    prompt_content = research_plan_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
        max_search_calls=configurable.max_search_calls,
    )
    plan = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    return Command(
        goto="review_research_plan",
        update={"research_plan": plan},
    )


async def review_research_plan(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["write_research_plan", "research_supervisor"]]:
    """Pause for plan approval or revision, then initialize the supervisor."""
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("research_plan")
    if plan is None:
        return Command(goto="write_research_plan")
    if isinstance(plan, dict):
        plan = ResearchPlan.model_validate(plan)
    decision = interrupt({
        "type": "research_plan",
        "plan": plan.model_dump(),
        "search_budget": configurable.max_search_calls,
        "search_sources": [source.value for source in configurable.search_apis],
    })
    action = decision.get("action", "approve") if isinstance(decision, dict) else "approve"
    if action == "revise":
        feedback = str(decision.get("feedback", "")).strip()
        return Command(
            goto="write_research_plan",
            update={
                "messages": [HumanMessage(content=f"请按以下意见修改研究方案：{feedback}")],
                "plan_revision_count": state.get("plan_revision_count", 0) + 1,
            },
        )

    research_brief = f"{plan.title}\n\n{plan.objective}\n\n" + "\n".join(
        f"{index + 1}. {section.title}: {section.description}"
        for index, section in enumerate(plan.sections)
    )
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=research_brief),
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = _model_runtime_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        configurable,
        config,
    )
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            async def run_research_unit(tool_call):
                topic = tool_call["args"]["research_topic"]
                unit_id = tool_call["id"]
                await adispatch_custom_event(
                    "research_activity",
                {
                    "id": f"unit:{unit_id}",
                    "type": "research.unit.started",
                        "status": "running",
                        "unit_id": unit_id,
                        "title": topic[:120],
                        "detail": "子研究任务已开始",
                    },
                    config=config,
                )
                result = await researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=topic)
                    ],
                    "research_topic": topic,
                }, config)
                await adispatch_custom_event(
                    "research_activity",
                {
                    "id": f"unit:{unit_id}",
                    "type": "research.unit.completed",
                        "status": "completed",
                        "unit_id": unit_id,
                        "title": topic[:120],
                        "detail": "已完成资料整理与摘要",
                    },
                    config=config,
                )
                return result

            research_tasks = [
                run_research_unit(tool_call)
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # A failed research phase must not be presented as a successful empty report.
            logging.exception("Research execution failed")
            raise RuntimeError(f"Research execution failed: {e}") from e
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = _model_runtime_config(
        configurable.research_model,
        configurable.research_model_max_tokens,
        configurable,
        config,
    )
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    search_call_count = state.get("search_call_count", 0)
    search_calls_this_step = 0

    async def execute_tool_call(tool_call):
        nonlocal search_calls_this_step
        selected_tool = tools_by_name.get(tool_call["name"])
        if selected_tool is None:
            return f"Error executing tool: unknown tool '{tool_call['name']}'"
        metadata = getattr(selected_tool, "metadata", {}) or {}
        is_search = metadata.get("type") == "search" or tool_call["name"] in {
            "tavily_search", "duckduckgo_search", "web_search"
        }
        if is_search:
            if search_call_count + search_calls_this_step >= configurable.max_search_calls:
                return "Search budget reached. Complete this unit with the collected sources."
            search_calls_this_step += 1
            search_terms = tool_call["args"].get("query") or tool_call["args"].get("queries")
            if isinstance(search_terms, list):
                search_terms = " / ".join(str(term) for term in search_terms)
            await adispatch_custom_event(
                "research_activity",
                {
                    "id": f"search:{tool_call['id']}",
                    "type": "search.started",
                    "status": "running",
                    "title": "正在搜索资料",
                    "detail": str(search_terms or tool_call["args"]),
                    "source": metadata.get("source", tool_call["name"]),
                    "search_index": search_call_count + search_calls_this_step,
                    "search_budget": configurable.max_search_calls,
                },
                config=config,
            )
        result = await execute_tool_safely(selected_tool, tool_call["args"], config)
        if is_search:
            await adispatch_custom_event(
                "research_activity",
                {
                    "id": f"search:{tool_call['id']}",
                    "type": "search.completed",
                    "status": "completed",
                    "title": "搜索完成",
                    "detail": "已获取并整理候选来源",
                    "source": metadata.get("source", tool_call["name"]),
                    "search_index": search_call_count + search_calls_this_step,
                    "search_budget": configurable.max_search_calls,
                },
                config=config,
            )
        return result

    tool_execution_tasks = [execute_tool_call(call) for call in tool_calls]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={
                "researcher_messages": tool_outputs,
                "search_call_count": search_call_count + search_calls_this_step,
            }
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={
            "researcher_messages": tool_outputs,
            "search_call_count": search_call_count + search_calls_this_step,
        }
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config(_model_runtime_config(
        configurable.compression_model,
        configurable.compression_model_max_tokens,
        configurable,
        config,
    ))
    
    # Step 2: Prepare messages for compression
    # Do not mutate the message list owned by the LangGraph state/checkpoint.
    researcher_messages = list(state.get("researcher_messages", []))
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
    context_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = _model_runtime_config(
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        configurable,
        config,
    )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await configurable_model.with_config(writer_model_config).ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState,
    input_schema=AgentInputState,
    context_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("request_clarification", request_clarification)
deep_researcher_builder.add_node("write_research_plan", write_research_plan)
deep_researcher_builder.add_node("review_research_plan", review_research_plan)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile with checkpointing so clarification and plan approval can pause/resume.
deep_researcher = deep_researcher_builder.compile(checkpointer=InMemorySaver())

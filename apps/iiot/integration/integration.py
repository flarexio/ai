from pydantic import Field
from typing import Literal, Optional, TypedDict

from langchain.chat_models import BaseChatModel
from langchain.messages import ToolMessage, SystemMessage, trim_messages
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from protocol import ChatContext
from .mapping import create_mapping_agent
from .connectivity import create_connectivity_agent
from ..model import IIoTRepositoryProtocol, IIoTState


class RouteIntent(TypedDict):
    """Define which agent to activate next."""
    route: Literal[
        "mapping", 
        "connectivity", 
        # "deployment",
    ]
    supervisor_message: str = Field(description="The message from the supervisor to the agent.")


INTEGRATION_SUPERVISOR_PROMPT = """
You are the **Integration Supervisor** for an IIoT system.

Transform a surveyed factory (`SurveyFactory`) into a deployable IIoT factory (`Factory`) through:

1. **Factory Mapping** - Structure factory objects (machines, zones, controllers, points)
2. **Connectivity Testing** - Verify controller connections and test data points  
3. **Deployment** - Generate final deployment configuration

Current Factory State:
<factory>
{existing_factory}
</factory>

**Rules**:
- **Always use RouteIntent tool** to delegate work - never describe plans
- **Only use data provided by user** - ask for missing information instead of assuming
- **Wait for user confirmation** before proceeding to next phase

**CRITICAL - Valid Routes ONLY**:
- **"mapping"** - For factory structure, driver configuration, point updates, or any factory model changes
- **"connectivity"** - For connection testing and validation only

**Route Decision Logic**:
- Need to create/modify factory structure? → Use "mapping"
- Need to configure drivers or points? → Use "mapping" 
- Need to test connections? → Use "connectivity"
- Need to update any factory data? → Use "mapping"

**Examples**:
- "Configure temperature points" → "mapping"
- "Set controller addresses" → "mapping"
- "Test PLC connection" → "connectivity"
- "Update point configuration" → "mapping"

**Important**: After mapping is complete, present results to user for confirmation before connectivity testing.
"""


def create_integration_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "integration_agent",
) -> CompiledStateGraph:

    # Create the agents
    mapping_agent = create_mapping_agent(model, tools, repo)
    connectivity_agent = create_connectivity_agent(model, tools, repo)
    # deployment_agent = create_deployment_agent(model, repo)

    def call_supervisor(state: IIoTState, runtime: Runtime[ChatContext]) -> IIoTState:
        ctx = runtime.context

        factories = repo.list_factories(ctx.customer_id)

        # TODO: only one factory for now
        existing_factory = None
        if len(factories) > 0:
            existing_factory = factories[0].model_dump()
        
        system_prompt = INTEGRATION_SUPERVISOR_PROMPT.format(
            existing_factory=existing_factory,
        )

        if state["supervisor_message"] is not None:
            system_prompt += f"\n\nPlease review the following message from the supervisor agent: {state['supervisor_message']}"

        model_with_tools = model.bind_tools([RouteIntent], parallel_tool_calls=False)

        # add short-term memory to the model
        recent_messages = trim_messages(
            state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
        )

        # Add the system prompt and recent messages to the model
        messages = [SystemMessage(content=system_prompt)] + recent_messages
        response = model_with_tools.invoke(messages)

        return {"messages": [ response ]}


    def should_continue(state: IIoTState) -> Literal[
        "__end__", 
        "mapping_agent", 
        "connectivity_agent",
        # "deployment_agent",
        "handle_error",
    ]:
        message = state["messages"][-1]
        if len(message.tool_calls) == 0:
            return "__end__"
        else:
            tool_call = message.tool_calls[0]
            route = tool_call["args"]["route"]

            if route == "mapping":
                return "mapping_agent"
            elif route == "connectivity":
                return "connectivity_agent"
            # elif route == "deployment":
            #     return "deployment_agent"
            else:
                return "handle_error"

    def handle_error(state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content=f"error: unknown route {tool_call['args'].get('route', 'unknown')}",
            tool_call_id=tool_call["id"],
        )
        return {"messages": [tool_msg]}


    async def call_mapping_agent(state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content="initializing mapping agent",
            tool_call_id=tool_call["id"],
        )

        try:
            response = await mapping_agent.ainvoke({
                "messages": state["messages"][:-1],
                "supervisor_message": tool_call["args"]["supervisor_message"],
            })
            tool_msg.content = response["messages"][-1].content
        
        except Exception as e:
            print(f"Error in mapping agent: {e}")
            tool_msg.content = f"error: {e}"

        return {"messages": [tool_msg]}


    async def call_connectivity_agent(state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content="initializing connectivity agent",
            tool_call_id=tool_call["id"],
        )

        try:
            response = await connectivity_agent.ainvoke({
                "messages": state["messages"][:-1],
                "supervisor_message": tool_call["args"]["supervisor_message"],
            })
            tool_msg.content = response["messages"][-1].content

        except Exception as e:
            print(f"Error in connectivity agent: {e}")
            tool_msg.content = f"error: {e}"

        return {"messages": [tool_msg]}


    # Create the workflow
    workflow = StateGraph(IIoTState, ChatContext)
    
    # Add nodes
    workflow.add_node("supervisor", call_supervisor)
    workflow.add_node("mapping_agent", call_mapping_agent)
    workflow.add_node("connectivity_agent", call_connectivity_agent)
    # workflow.add_node("deployment_agent", call_deployment_agent)
    workflow.add_node("handle_error", handle_error)

    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", should_continue)
    workflow.add_edge("mapping_agent", "supervisor")
    workflow.add_edge("connectivity_agent", "supervisor")
    # workflow.add_edge("deployment_agent", "supervisor")
    workflow.add_edge("handle_error", "supervisor")

    return workflow.compile(
        checkpointer,
        store=store,
        name=name,
    )

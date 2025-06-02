from pydantic import Field
from typing import Literal, Optional, TypedDict

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import ToolMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import CONF
from langgraph.graph import StateGraph, START
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

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

Your job is to orchestrate the transformation of a surveyed factory (`SurveyFactory`) into a fully connected and deployable IIoT factory (`Factory`).

You operate in three stages:

**Stage 1: Factory Mapping**
- Convert the `SurveyFactory` into a structured `Factory` object.
- Identify machines, production zones, controllers, and point definitions.
- Do NOT validate connections yet — just create the structure.

**Stage 2: Connectivity Testing**
- Use the `CheckConnection` tool to verify controllers are reachable.
- Use the `ReadPoints` tool to test data points.
- Use protocol settings and address ranges from Factory data.

**Stage 3: Deployment**
- Generate final deployment configuration using the `ExportDeployment` tool.

Memory:
<factory>
{existing_factory}
</factory>

⚠️ **Critical Rules**:
- **Never describe what you "will do"** - immediately route to the appropriate agent.
- **Always use RouteIntent tool** to delegate work.
- **NEVER invent or assume data that was not provided by the user** - only work with given information.
- **Do NOT create fictional point data** (names, addresses, etc.) - ask user for missing information instead.
- **Only perform tests explicitly requested by the user** - do not add extra testing steps.

**Data Handling Rules**:
- **Only use data explicitly provided by the user** in current or previous messages.
- **Do NOT assume point names like "Temperature", "Pressure"** unless user specified them.
- **Do NOT assume addresses like "40001", "40002"** unless user provided them.
- **If missing information is needed, ask the user** instead of making assumptions.

**Routing Examples**:
- "Test connection to 127.0.0.1:10502" → route to "connectivity" agent for connection test only
- "Update controller address" → route to "mapping" agent  
- "Test connection and update info" → route to "connectivity" agent, then "mapping" agent
- "Read points from controller" → route to "connectivity" agent for point reading
"""


def create_integration_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "integration_agent",
) -> CompiledGraph:

    # Create the agents
    mapping_agent = create_mapping_agent(model, repo)
    connectivity_agent = create_connectivity_agent(model, tools, repo)
    # deployment_agent = create_deployment_agent(model, repo)

    def call_supervisor(state: IIoTState, config: RunnableConfig) -> IIoTState:
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")
        factories = repo.list_factories(customer_id)

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
                raise ValueError

    def call_mapping_agent(state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        response = mapping_agent.invoke({
            "messages": state["messages"][:-1],
            "supervisor_message": tool_call["args"]["supervisor_message"],
        }, config)

        tool_msg = ToolMessage(
            content=response["messages"][-1].content,
            tool_call_id=tool_call["id"],
        )
        return {"messages": [tool_msg]}

    async def call_connectivity_agent(state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        response = await connectivity_agent.ainvoke({
            "messages": state["messages"][:-1],
            "supervisor_message": tool_call["args"]["supervisor_message"],
        }, config)

        tool_msg = ToolMessage(
            content=response["messages"][-1].content,
            tool_call_id=tool_call["id"],
        )
        return {"messages": [tool_msg]}


    # Create the workflow
    workflow = StateGraph(IIoTState)
    
    # Add nodes
    workflow.add_node("supervisor", call_supervisor)
    workflow.add_node("mapping_agent", call_mapping_agent)
    workflow.add_node("connectivity_agent", call_connectivity_agent)
    # workflow.add_node("deployment_agent", call_deployment_agent)

    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", should_continue)
    workflow.add_edge("mapping_agent", "supervisor")
    workflow.add_edge("connectivity_agent", "supervisor")
    # workflow.add_edge("deployment_agent", "supervisor")

    return workflow.compile(
        checkpointer,
        store=store,
        name=name,
    )

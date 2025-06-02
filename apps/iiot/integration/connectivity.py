from typing import Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.constants import CONF
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..model import IIoTRepositoryProtocol, IIoTState


SYSTEM_PROMPT = """
You are an AI Agent specializing in the "Connectivity" phase of an Industrial IoT (IIoT) system integration workflow. 

The initial factory survey and transformation (from `SurveyFactory` to `Factory`) have already been completed. 
Your current responsibility is to verify device connectivity and point accessibility to ensure a reliable setup for deployment.

Memory:

<customer>
{existing_customer}
</customer>

<factory>
{existing_factory}
</factory>

⚠️ **CRITICAL EXECUTION RULES**:
- **NEVER describe what you will do** - immediately execute the required tools
- **NO preliminary explanations** - start with tool calls directly
- **Each task requires actual tool execution** - descriptions without tool calls are forbidden
- If EdgeID is missing, ask user for it - do not proceed without it

**Your tasks include:**

1. **Host Port Connectivity Check** - IMMEDIATELY use `CheckConnection` tool
   - Test network connectivity for devices using host and port information
   - Report specific failures (timeout, connection refused, unknown host)
   - Log verified host and port for successful connections

2. **Driver Discovery** - IMMEDIATELY use `ListDrivers` tool when needed
   - List available IIoT drivers (modbus, opcua, etc.)

3. **Schema Inspection** - IMMEDIATELY use `Schema` tool when needed  
   - Retrieve required schema for specific drivers

4. **Point Read Test** - IMMEDIATELY use `ReadPoints` tool when needed
   - Test configured data points after confirming connectivity
   - Use proper schema format for requests
   - Report specific errors or capture sample values

**Available tools:**
- CheckConnection
- ListDrivers  
- Schema
- ReadPoints

**Execution Pattern:**
1. Check if EdgeID exists in customer data
2. If task requires connectivity test → immediately call CheckConnection tool
3. If task requires driver info → immediately call ListDrivers tool
4. If task requires point reading → immediately call ReadPoints tool
5. NO descriptions like "進行連線測試..." - just execute the tools

**Forbidden Responses:**
- ❌ "首先，我將確認..."
- ❌ "接下來進行..."
- ❌ "進行連線測試..."
- ✅ Immediately call the appropriate tool

Execute tools immediately based on the supervisor's request.
"""

def create_connectivity_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "connectivity_agent",
) -> CompiledGraph:

    def prompt(state: IIoTState, config: RunnableConfig) -> list[AnyMessage]:
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")

        customer = repo.find_customer(customer_id)
        existing_customer = None
        if customer:
            existing_customer = customer.model_dump()

        factories = repo.list_factories(customer_id)
        existing_factory = None
        if len(factories) > 0:
            existing_factory = factories[0].model_dump()

        system_prompt = SYSTEM_PROMPT.format(
            existing_customer=existing_customer,
            existing_factory=existing_factory,
        )

        if state["supervisor_message"] is not None:
            system_prompt += f"\n\nPlease review the following message from the supervisor agent: {state['supervisor_message']}"

        # add short-term memory to the model
        recent_messages = trim_messages(
            state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
        )

        return [SystemMessage(content=system_prompt)] + recent_messages

    return create_react_agent(
        model,
        tools,
        prompt=prompt,
        state_schema=IIoTState,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )

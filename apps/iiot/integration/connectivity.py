from typing import Any, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import before_model
from langchain.chat_models import BaseChatModel
from langchain.messages import trim_messages, RemoveMessage, SystemMessage
from langchain.tools import BaseTool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from protocol import ChatContext
from ..model import IIoTRepositoryProtocol, IIoTState


SYSTEM_PROMPT = """
You are the **Connectivity Agent** for IIoT controller connection testing.

**Tasks**:
1. Listen to user instructions about which controllers to test
2. Test ONLY the controllers specified by the user
3. Follow proper tool usage sequence
4. Report results and STOP

**Current State**:
<customer>{existing_customer}</customer>
<factory>{existing_factory}</factory>

**Tool Usage Sequence** (for each specified controller):
1. Get Controller and Points Options to understand configuration
2. Check Schema for the specific driver to understand request format
3. Generate request JSON based on Schema requirements
4. Use ReadPoints with the generated request JSON
5. Test ONCE only - no retries

**Connection Testing Process**:
1. Ask user which controllers need testing (if not specified)
2. For each user-specified controller:
   - Follow the Tool Usage Sequence above
   - Use controller.options for connection parameters
   - Use point.options for data reading parameters
3. **MANDATORY**: When ALL specified controllers tested, respond with EXACTLY: "Connectivity testing completed" and STOP

**CRITICAL Rules**:
- Only test controllers explicitly requested by user
- ALWAYS check Schema before generating request JSON
- Generate proper request JSON based on Schema format
- One test attempt per controller only
- **STOP immediately after saying "Connectivity testing completed"**
- Skip controllers not mentioned by user
"""

def create_connectivity_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "connectivity_agent",
) -> CompiledStateGraph:

    @before_model
    def prompt(state: IIoTState, runtime: Runtime[ChatContext]) -> dict[str | Any] | None:
        ctx = runtime.context

        customer = repo.find_customer(ctx.customer_id)
        existing_customer = None
        if customer:
            existing_customer = customer.model_dump()

        factories = repo.list_factories(ctx.customer_id)
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

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                SystemMessage(content=system_prompt),
                *recent_messages,
            ]
        }

    return create_agent(
        model,
        tools,
        middleware=[prompt],
        state_schema=IIoTState,
        context_schema=ChatContext,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )

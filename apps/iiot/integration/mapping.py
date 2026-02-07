from typing import Any, Optional

from langchain_core.messages import merge_message_runs
from langchain.agents import create_agent
from langchain.agents.middleware import before_model 
from langchain.chat_models import BaseChatModel
from langchain.messages import trim_messages, RemoveMessage, SystemMessage
from langchain.tools import tool, BaseTool, ToolRuntime
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from trustcall import create_extractor

from protocol import ChatContext
from ..model import Factory, IIoTRepositoryProtocol, IIoTState


SYSTEM_PROMPT = """
You are the **Mapping Agent** - transform Survey data into structured Factory model.

**Process** (step-by-step):
1. Production lines and areas
2. Machines (mapped to lines)  
3. Controllers (PLCs, CNCs) with driver options
4. Points (data signals) with driver options

**Driver Configuration**:
1. Use `ListDrivers()` to see available drivers
2. **REQUIRED**: Get BOTH `Schema(driver)` AND `Instruction(driver)` for each driver
3. Configure Controller.options and Point.options using both Schema and Instruction
4. **For Points**: Must reference BOTH Schema AND Instruction to get correct configuration format

**Rules**:
- Analyze gaps between Survey and Factory first
- Complete each level before moving to next
- **CRITICAL**: Always call both Schema() and Instruction() for driver configuration
- Call `update_factory` for each component immediately
- STOP when all Survey components are mapped
- Don't retry successful operations

**Current State**:
<customer>{existing_customer}</customer>
<survey>{existing_survey}</survey>
<factory>{existing_factory}</factory>

**Completion**: When all mapped, respond "Mapping completed successfully" and STOP.
"""


def create_mapping_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "mapping_agent",
) -> CompiledStateGraph:

    # Create the extractor
    extractor = create_extractor(model, tools=[Factory], tool_choice="Factory")

    @before_model
    def prompt(state: IIoTState, runtime: Runtime[ChatContext]) -> dict[str | Any] | None:
        ctx = runtime.context

        customer = repo.find_customer(ctx.customer_id)
        existing_customer = None
        if customer:
            existing_customer = customer.model_dump()

        surveys = repo.list_surveys(ctx.customer_id)
        existing_survey = None
        if len(surveys) > 0:
            existing_survey = surveys[0].model_dump()

        factories = repo.list_factories(ctx.customer_id)
        existing_factory = None
        if len(factories) > 0:
            existing_factory = factories[0].model_dump()

        system_prompt = SYSTEM_PROMPT.format(
            existing_customer=existing_customer,
            existing_survey=existing_survey,
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


    @tool("update_factory", description="Update the factory model based on the Agent's instructions and driver configuration.")
    def update_factory(runtime: ToolRuntime[ChatContext]) -> str:
        ctx = runtime.context

        surveys = repo.list_surveys(ctx.customer_id)
        existing_survey = None
        if len(surveys) > 0:
            existing_survey = surveys[0].model_dump()

        factories = repo.list_factories(ctx.customer_id)
        factory = factories[0] if len(factories) > 0 else None
        existing = {
            "Factory": factory.model_dump() if factory else None,
        }

        # 從對話中獲取最新的指令和配置資訊
        recent_messages = trim_messages(
            runtime.state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
            end_on=["human", "ai"],
        )

        prompt = f"""
        You are assisting in constructing a structured `Factory` model from survey data and agent instructions.

        Survey Data:
        <survey>
        {existing_survey}
        </survey>

        **Instructions**:
        - Follow the agent's instructions from the conversation history
        - The agent has already prepared the driver configuration information
        - Update the factory model according to the agent's specifications
        - Focus on the specific component/level mentioned by the agent
        - **IMPORTANT**: All data written to memory must be in English only

        **Tool Use**:
        - Always include `"json_doc_id": "Factory"` in the tool call
        - Update the factory with the configuration provided by the agent

        Work based on the agent's prepared configuration and instructions.
        """

        updated_messages = list(merge_message_runs(messages=[SystemMessage(content=prompt)] + recent_messages[:-1]))

        result = extractor.invoke({
            "messages": updated_messages,
            "existing": existing,
        })

        # Store the factory in the repository
        for resp in result["responses"]:
            factory = Factory.model_validate(resp)
            repo.store_factory(factory)

        return "Factory updated successfully."


    return create_agent(
        model,
        [update_factory] + tools,
        middleware=[prompt],
        state_schema=IIoTState,
        context_schema=ChatContext,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )

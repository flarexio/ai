from typing import Annotated, Dict, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, merge_message_runs, trim_messages
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langgraph.constants import CONF
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from trustcall import create_extractor

from ..model import Factory, IIoTRepositoryProtocol, IIoTState


SYSTEM_PROMPT = """
You are the **Mapping Agent** responsible for transforming a `SurveyFactory` into a clean, structured `Factory` model for IIoT integration.

Your responsibilities:
1. Parse the `SurveyFactory`, which contains semi-structured field survey data.
2. **Compare Survey vs Factory** to identify what's missing or incomplete.
3. Identify and extract IIoT components step-by-step:
   - Production lines and areas.
   - Machines and their mappings to lines.
   - Controllers (e.g., PLCs, CNCs) and their configuration.
   - Points (data signals), including address, type, role, and direction.

⚠️ **Critical Working Rules**:
- **ALWAYS analyze gaps** between Survey and Factory data before concluding.
- Work in layers: begin with lines and areas, then machines, then controllers, then points.
- For each component that is confident and well-understood, call `update_factory` immediately.
- **Continue calling `update_factory`** until all Survey components are properly mapped to Factory.
- **Do NOT claim completion** just because one update_factory call succeeded.

💡 **Gap Analysis Process**:
1. Compare Survey areas with Factory production_lines
2. Compare Survey machines with Factory machines (accounting for quantity)
3. Compare Survey controllers with Factory controllers
4. Compare Survey points with Factory points
5. If gaps exist, continue with the next appropriate update_factory call

Memory:
<survey>
{existing_survey}
</survey>

<factory>
{existing_factory}
</factory>

**Before responding to user**: Check if there are Survey components not yet reflected in the Factory. If yes, continue with update_factory calls. Only stop when Survey→Factory mapping is complete.
"""


def create_mapping_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "mapping_agent",
) -> CompiledGraph:

    # Create the extractor
    extractor = create_extractor(model, tools=[Factory], tool_choice="Factory")

    def prompt(state: IIoTState, config: RunnableConfig) -> list[AnyMessage]:
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")

        surveys = repo.list_surveys(customer_id)
        existing_survey = None
        if len(surveys) > 0:
            existing_survey = surveys[0].model_dump()

        factories = repo.list_factories(customer_id)
        existing_factory = None
        if len(factories) > 0:
            existing_factory = factories[0].model_dump()

        system_prompt = SYSTEM_PROMPT.format(
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

        return [SystemMessage(content=system_prompt)] + recent_messages


    def update_factory(
        *, 
        state: Annotated[IIoTState, InjectedState], 
        config: RunnableConfig,
    ) -> str:
        """Update the factory model based on the survey data and existing factory information."""

        conf = config.get(CONF)
        customer_id = conf.get("customer_id")

        surveys = repo.list_surveys(customer_id)
        existing_survey = None
        if len(surveys) > 0:
            existing_survey = surveys[0].model_dump()

        factories = repo.list_factories(customer_id)
        factory = factories[0] if len(factories) > 0 else None
        existing = {
            "Factory": factory.model_dump() if factory else None,
        }

        prompt = f"""
        You are assisting in incrementally constructing a structured `Factory` model from survey data.

        Memory:
        <survey>
        {existing_survey}
        </survey>

        Instructions:
        - Your task is to **update one level of the Factory at a time**.
        - You may only focus on one of the following levels in a single update:
            - Production lines (with only line-level info, no machines yet)
            - Machines (must be attached to an existing line)
            - Controllers (must be attached to an existing machine)
            - Points (must be attached to a specific controller)
        - **Do NOT attempt to create multiple layers in one turn**.
        - **Do NOT guess or invent missing context from deeper levels** — skip them if not confirmed.

        Tool Use:
        - Always include `"json_doc_id": "Factory"` in the tool call.
        - Each update can add, overwrite, or extend the corresponding part.

        You are not allowed to build the entire Factory structure at once.
        Work in controlled, confident steps only.
        """

        # add short-term memory to the model
        recent_messages = trim_messages(
            state["messages"], 
            max_tokens=10,
            token_counter=len,
            start_on="human",
            end_on=["human", "ai"],
        )
        updated_messages = list(merge_message_runs(messages=[SystemMessage(content=prompt)] + recent_messages[:-1]))

        result = extractor.invoke({
            "messages": updated_messages,
            "existing": existing,
        })

        # Store the survey in the memory
        for resp in result["responses"]:
            factory = Factory.model_validate(resp)
            repo.store_factory(factory)

        return "Factory updated successfully."


    return create_react_agent(
        model,
        [update_factory] + tools,
        prompt=prompt,
        state_schema=IIoTState,
        checkpointer=checkpointer,
        store=store,
        name=name,
    )

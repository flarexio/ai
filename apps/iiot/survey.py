from typing import Literal, Optional, TypedDict

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import ToolMessage, SystemMessage, trim_messages, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.constants import CONF
from langgraph.graph.graph import CompiledGraph
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from trustcall import create_extractor

from .model import IIoTRepositoryProtocol, IIoTState, SurveyFactory


class UpdateMemory(TypedDict):
    """ Decision on what memory type to update """
    update_type: Literal["survey"]


SYSTEM_PROMPT = """
You are an IIoT Survey Assistant helping engineers conduct field surveys at client factories. 

Your role is to collect and organize information about a factory's IIoT-connectable assets, including production areas, equipment, and control systems.

Instructions:
- Review the existing factory survey data provided in the memory section.
- When new information is available, or corrections are needed, use the UpdateMemory tool to propose updates to the survey data.
- Always set `update_type` to `"survey"` — this agent is only responsible for factory-related technical details.
- Do not attempt to update or reference customer-level information such as company name, contact, or profile.
- Only use one update_type per tool call.

Memory:
<survey>
{existing_survey}
</survey>

Be concise, structured, and accurate in your updates. If no update is required, do not invoke the tool.
"""

def create_survey_agent(
    model: BaseChatModel,
    repo: IIoTRepositoryProtocol,
    *,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    name: str = "survey_agent",
) -> CompiledGraph:

    # Create the extractor
    extractor = create_extractor(model, tools=[SurveyFactory], tool_choice="SurveyFactory")

    def call_model(state: IIoTState, config: RunnableConfig) -> IIoTState:
        # add long-term memory to the model
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")
        surveys = repo.list_surveys(customer_id)

        # TODO: only one survey for now
        existing_survey = None
        if len(surveys) > 0:
            existing_survey = surveys[0].model_dump()

        system_prompt = SYSTEM_PROMPT.format(
            existing_survey=existing_survey,
        )
        if state["supervisor_message"] is not None:
            system_prompt += f"\n\nPlease review the following message from the supervisor agent: {state['supervisor_message']}"

        model_with_tools = model.bind_tools([UpdateMemory], parallel_tool_calls=False)

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
        "update_survey",
        "handle_error",
    ]:
        message = state["messages"][-1]
        if len(message.tool_calls) == 0:
            return "__end__"
        else:
            tool_call = message.tool_calls[0]
            if tool_call["args"]["update_type"] == "survey":
                return "update_survey"
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


    def update_survey(state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        # add long-term memory to the model
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")
        surveys = repo.list_surveys(customer_id)
        
        # TODO: only one survey for now
        survey = surveys[0] if len(surveys) > 0 else None

        existing = {
            "SurveyFactory": survey.model_dump() if survey else {},
        }

        # Prepare the prompt for LLM to generate structured survey - 簡化提示
        prompt = """
        Update survey factory information based on the conversation.
        
        IMPORTANT:
        1. Always set "json_doc_id": "SurveyFactory"
        2. If the user input does not specify an area or production line, assign the machines to a common area.
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

        tool_msg = ToolMessage(
            content="no updates",
            tool_call_id=tool_call["id"],
        )

        if len(result["responses"]) == 0:
            return {"messages": [tool_msg]}

        # Store the survey in the memory
        for resp in result["responses"]:
            survey = SurveyFactory.model_validate(resp)
            repo.store_survey(survey)

        tool_msg.content = "updated survey factory"
        return {"messages": [tool_msg]}


    # Create the workflow
    workflow = StateGraph(IIoTState)
    
    # Add nodes
    workflow.add_node("model", call_model)
    workflow.add_node("update_survey", update_survey)
    workflow.add_node("handle_error", handle_error)

    # Add edges
    workflow.add_edge(START, "model")
    workflow.add_conditional_edges("model", should_continue)
    workflow.add_edge("update_survey", "model")
    workflow.add_edge("handle_error", "model")

    return workflow.compile(
        checkpointer,
        store=store,
        name=name,
    )

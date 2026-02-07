from pydantic import Field
from typing import Literal, TypedDict

from langchain_core.messages import merge_message_runs
from langchain.chat_models import init_chat_model
from langchain.messages import trim_messages, SystemMessage, ToolMessage
from langchain.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START
from langgraph.runtime import Runtime
from trustcall import create_extractor

from protocol import ChatContext
from ..base import BaseAIApp
from .model import Customer, IIoTRepositoryProtocol, IIoTState
from .survey import create_survey_agent
from .integration import create_integration_agent


class RouteIntent(TypedDict):
    """Define which agent to activate next."""
    route: Literal[
        "update_customer",
        "survey", 
        # "quotation", 
        "integration",
    ]
    supervisor_message: str = Field(description="The message from the supervisor to the agent.")


SUPERVISOR_INSTRUCTION = """
You are the **Supervisor** in an AI-powered IIoT project system.

**Routing Rules:**
1. **Check `customer.status` first** - determines workflow stage
2. **Route by data type and stage:**
   - `update_customer`: Customer business info (name, industry, contact) + status transitions
   - `survey`: Survey data (equipment, areas, controllers) when status starts with `survey_`
   - `integration`: Factory configuration when status starts with `integration_`

3. **Stage transition control:**
   - Always confirm before changing customer status
   - Example: "You're in 'survey_completed'. Proceed to 'integration_in_progress'?"

4. **Key examples:**
   - "Update survey date" → survey agent (if survey stage)
   - "Add machine to area A" → survey agent (if survey stage)  
   - "Change company name" → update_customer
   - "Change status to integration" → update_customer (with confirmation)

**Memory:**
<customer>
{existing_customer}
</customer>

**Available agents:**
- `update_customer`: Customer data & status transitions ✅
- `survey`: Survey & factory data collection ✅  
- `quotation`: Cost estimation ❌ Unavailable
- `integration`: Device configuration & deployment ✅

Call agents sequentially as needed, then respond naturally to user.
"""

class IIoTAIApp(BaseAIApp):
    def __init__(self, memory: BaseCheckpointSaver, repo: IIoTRepositoryProtocol, toolkit: dict[str, list[BaseTool]]):
        # Setup the LLM
        model = init_chat_model("openai:gpt-5-mini")

        tools = toolkit["iiot"]

        # Create the agents
        customer_extractor = create_extractor(model, tools=[Customer], tool_choice="Customer")
        survey_agent = create_survey_agent(model, repo)
        # quotation_agent = None # Not implemented yet
        integration_agent = create_integration_agent(model, tools, repo)

        # Create the workflow
        workflow = StateGraph(IIoTState, ChatContext)
        
        # Add nodes
        workflow.add_node("supervisor", self.call_supervisor)
        workflow.add_node("update_customer", self.update_customer)
        workflow.add_node("survey_agent", self.call_survey_agent)
        # workflow.add_node("quotation_agent", self.call_quotation_agent)
        workflow.add_node("integration_agent", self.call_integration_agent)
        workflow.add_node("handle_error", self.handle_error)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges("supervisor", self.should_continue)
        workflow.add_edge("update_customer", "supervisor")
        workflow.add_edge("survey_agent", "supervisor")
        # workflow.add_edge("quotation_agent", "supervisor")
        workflow.add_edge("integration_agent", "supervisor")
        workflow.add_edge("handle_error", "supervisor")

        # Compile the workflow
        app = workflow.compile(checkpointer=memory, store=repo)

        self.app = app
        self.customer_extractor = customer_extractor
        self.survey_agent = survey_agent
        # self.quotation_agent = quotation_agent
        self.integration_agent = integration_agent
        self.model = model
        self.repo = repo

    def call_supervisor(self, state: IIoTState, runtime: Runtime[ChatContext]) -> IIoTState:
        ctx = runtime.context

        customer = self.repo.find_customer(ctx.customer_id)

        existing_customer = None
        if customer:
            existing_customer = customer.model_dump()

        system_prompt = SUPERVISOR_INSTRUCTION.format(
            existing_customer=existing_customer,
        )

        model_with_tools = self.model.bind_tools([RouteIntent], parallel_tool_calls=False)

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


    def should_continue(self, state: IIoTState) -> Literal[
        "__end__", 
        "update_customer", 
        "survey_agent",
        # "quotation_agent",
        "integration_agent",
        "handle_error",
    ]:
        message = state["messages"][-1]
        if len(message.tool_calls) == 0:
            return "__end__"
        else:
            tool_call = message.tool_calls[0]
            route = tool_call["args"]["route"]

            if route == "update_customer":
                return "update_customer"
            elif route == "survey":
                return "survey_agent"
            # elif route == "quotation":
            #     return "quotation_agent"
            elif route == "integration":
                return "integration_agent"
            else:
                return "handle_error"


    def handle_error(self, state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content=f"error: unknown route {tool_call['args'].get('route', 'unknown')}",
            tool_call_id=tool_call["id"],
        )
        return {"messages": [tool_msg]}


    def update_customer(self, state: IIoTState, runtime: Runtime[ChatContext]) -> IIoTState:
        ctx = runtime.context

        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        # add long-term memory to the model
        customer = self.repo.find_customer(ctx.customer_id)

        existing = {
            "Customer": customer.model_dump() if customer else {},
        }

        # Prepare the prompt for LLM to generate structured survey
        prompt = f"""
        Update customer information based on the conversation.
        
        IMPORTANT: Always set "json_doc_id": "Customer"
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

        tool_msg = ToolMessage(
            content="no updates",
            tool_call_id=tool_call["id"],
        )

        try: 
            result = self.customer_extractor.invoke({
                "messages": updated_messages,
                "existing": existing,
            })

            if len(result["responses"]) == 0:
                return {"messages": [tool_msg]}

            # Store the customer in the memory
            updated_customer = Customer.model_validate(result["responses"][0])
            self.repo.store_customer(updated_customer)

            tool_msg.content = "updated customer"

        except Exception as e:
            print(f"Error in update_customer: {e}")
            tool_msg.content = f"error: {e}"

        return {"messages": [tool_msg]}


    def call_survey_agent(self, state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content="initializing survey agent",
            tool_call_id=tool_call["id"],
        )

        try:
            response = self.survey_agent.invoke({
                "messages": state["messages"][:-1],
                "supervisor_message": tool_call["args"]["supervisor_message"],
            })
            tool_msg.content = response["messages"][-1].content

        except Exception as e:
            print(f"Error in call_survey_agent: {e}")
            tool_msg.content = f"error: {e}"

        return {"messages": [tool_msg]}


    async def call_integration_agent(self, state: IIoTState) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]
        tool_msg = ToolMessage(
            content="initializing integration agent",
            tool_call_id=tool_call["id"],
        )

        try:
            response = await self.integration_agent.ainvoke({
                "messages": state["messages"][:-1],
                "supervisor_message": tool_call["args"]["supervisor_message"],
            })
            tool_msg.content = response["messages"][-1].content

        except Exception as e:
            print(f"Error in call_integration_agent: {e}")
            tool_msg.content = f"error: {e}"

        return {"messages": [tool_msg]}

    def id(self) -> str:
        return "iiot"

    def name(self) -> str:
        return "IIoT"

    def description(self) -> str:
        return "An AI app for managing IIoT projects, including customer updates, surveys, and integration tasks."
    
    def version(self) -> str:
        return "1.0.0"

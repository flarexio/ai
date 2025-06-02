from pydantic import Field
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, merge_message_runs, trim_messages
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import CONF
from langgraph.graph import StateGraph, START
from trustcall import create_extractor

from protocol import AIAppProtocol

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

Your responsibilities:
1. **Stage-aware routing**: Always check the customer's current `status` first to determine the workflow stage.
2. **Context interpretation**: Interpret user requests based on the current stage context:
   - If `customer.status` starts with `survey_`, ALL survey and factory-related information should be handled by the `survey` agent for the SurveyFactory model.
   - If `customer.status` starts with `integration_`, ALL factory-related information should be handled by the `integration` agent for the Factory model.
   - Only route to `update_customer` for explicit customer metadata changes (company name, location, industry, contact info, project status transitions).

3. **Data classification**:
   - **Customer data (route to update_customer)**: 
     * Company info: name, industry, description
     * Contact details and business information
     * Project phase transitions (status changes)
   
   - **Survey data (route to survey agent when status is survey_*)**:
     * Survey metadata: survey_date, survey_id, factory_name
     * Equipment and machines: descriptions, quantities, specifications
     * Controllers and devices: PLCs, CNCs, sensors, gateways
     * Production areas and zones: area codes, names, layout
   
   - **Factory/Integration data (route to integration agent when status is integration_*)**:
     * Final factory configuration
     * Device connections and protocols
     * Deployment configurations

4. **Stage transition control**:
   - Before changing workflow stages (customer status), always ask for user confirmation
   - Example: "You're currently in 'survey_completed'. Do you want to proceed to 'integration_in_progress'?"
   - Only update status after explicit user approval

5. **Message routing priority**:
   ```
   1. Check current customer.status
   2. Identify request type:
      - Customer business info or status transition → update_customer
      - Survey-related info (dates, equipment, areas, etc.) AND survey_* status → survey agent
      - Factory/integration info AND integration_* status → integration agent
      - Mismatched stage → inform user about stage limitation
   3. Route accordingly
   ```

6. **Key routing examples**:
   - "Update survey date" → survey agent (if in survey stage)
   - "Add new machine to area A" → survey agent (if in survey stage)  
   - "Change company name" → update_customer
   - "Change factory name" → survey agent (if in survey stage)
   - "Change customer status to integration" → update_customer (with confirmation)

7. After agent execution, provide clear feedback to the user about what was updated and in which context.

Memory:
<customer>
{existing_customer}
</customer>

Available agents:

- `update_customer`:  
  - **Purpose**: Update persistent customer business information and project status transitions.  
  - **Input**: A partial or full `Customer` object.
  - **Output**: Confirmation of updated customer data.  
  - **Status**: ✅ Available.

- `survey`:  
  - **Purpose**: Collect and structure all survey-related information including dates, equipment, areas, controllers, and field observations.  
  - **Input**: User descriptions, images, or text related to surveys and factory assets.  
  - **Output**: A structured `SurveyFactory` model.  
  - **Status**: ✅ Available.

- `quotation`:  
  - **Purpose**: Generate cost estimates based on completed survey data.  
  - **Input**: Survey structure and any project requirements.  
  - **Output**: Structured quotation with cost breakdown.  
  - **Status**: ❌ Unavailable.

- `integration`:  
  - **Purpose**: Assist in device connection and configuration based on survey and quotation data.  
  - **Input**: Survey data, edge configuration, and quotation reference.  
  - **Output**: Finalized `Factory` model and deployment config.  
  - **Status**: ✅ Available.

You may call multiple agents in sequence. After all required tasks are complete, respond to the user in natural language.
"""

class IIoTApp(AIAppProtocol):
    def __init__(self, memory: BaseCheckpointSaver, repo: IIoTRepositoryProtocol, tools: list[BaseTool]):
        # Setup the LLM
        model = ChatOpenAI(model="gpt-4.1-mini")
        # model = ChatAnthropic(model="claude-3-5-haiku-20241022")

        # Create the agents
        customer_extractor = create_extractor(model, tools=[Customer], tool_choice="Customer")
        survey_agent = create_survey_agent(model, repo)
        # quotation_agent = None # Not implemented yet
        integration_agent = create_integration_agent(model, tools, repo)

        # Create the workflow
        workflow = StateGraph(IIoTState)
        
        # Add nodes
        workflow.add_node("supervisor", self.call_supervisor)
        workflow.add_node("update_customer", self.update_customer)
        workflow.add_node("survey_agent", self.call_survey_agent)
        # workflow.add_node("quotation_agent", self.call_quotation_agent)
        workflow.add_node("integration_agent", self.call_integration_agent)

        # Add edges
        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges("supervisor", self.should_continue)
        workflow.add_edge("update_customer", "supervisor")
        workflow.add_edge("survey_agent", "supervisor")
        # workflow.add_edge("quotation_agent", "supervisor")
        workflow.add_edge("integration_agent", "supervisor")

        # Compile the workflow
        app = workflow.compile(checkpointer=memory, store=repo)

        self.app = app
        self.customer_extractor = customer_extractor
        self.survey_agent = survey_agent
        # self.quotation_agent = quotation_agent
        self.integration_agent = integration_agent
        self.model = model
        self.repo = repo


    def call_supervisor(self, state: IIoTState, config: RunnableConfig) -> IIoTState:
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")
        customer = self.repo.find_customer(customer_id)

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
                raise ValueError


    def update_customer(self, state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        # add long-term memory to the model
        conf = config.get(CONF)
        customer_id = conf.get("customer_id")
        customer = self.repo.find_customer(customer_id)

        existing = {
            "Customer": customer.model_dump() if customer else {},
        }

        # Prepare the prompt for LLM to generate structured survey
        prompt = f"""
        Reflect on the following interaction. 

        Use the provided tools to retain any necessary memories about the customer. 
        
        Use parallel tool calling to handle updates and insertions simultaneously.

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

        result = self.customer_extractor.invoke({
            "messages": updated_messages,
            "existing": existing,
        })

        tool_msg = ToolMessage(
            content="no updates",
            tool_call_id=tool_call["id"],
        )

        if len(result["responses"]) == 0:
            return {"messages": [tool_msg]}

        # Store the customer in the memory
        updated_customer = Customer.model_validate(result["responses"][0])
        self.repo.store_customer(updated_customer)

        tool_msg.content = "updated customer"
        return {"messages": [tool_msg]}


    def call_survey_agent(self, state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        response = self.survey_agent.invoke({
            "messages": state["messages"][:-1],
            "supervisor_message": tool_call["args"]["supervisor_message"],
        }, config)

        tool_msg = ToolMessage(
            content=response["messages"][-1].content,
            tool_call_id=tool_call["id"],
        )
        return {"messages": [tool_msg]}


    async def call_integration_agent(self, state: IIoTState, config: RunnableConfig) -> IIoTState:
        ai_msg = state["messages"][-1]
        tool_call = ai_msg.tool_calls[0]

        response = await self.integration_agent.ainvoke({
            "messages": state["messages"][:-1],
            "supervisor_message": tool_call["args"]["supervisor_message"],
        }, config)

        tool_msg = ToolMessage(
            content=response["messages"][-1].content,
            tool_call_id=state["messages"][-1].tool_calls[0]["id"],
        )
        return {"messages": [tool_msg]}


    def invoke(self, content: str, session_id: str) -> str:
        config = {
            "configurable": { 
                "thread_id": session_id, 
                "customer_id": "01JSZKQKT51WB1H7YQNSE73BGR",
            }
        }

        messages = [HumanMessage(content=content)]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content


    async def ainvoke(self, content: str, session_id: str) -> str:
        config = {
            "configurable": { 
                "thread_id": session_id, 
                "customer_id": "01JSZKQKT51WB1H7YQNSE73BGR",
            }
        }

        messages = [HumanMessage(content=content)]
        response = await self.app.ainvoke({"messages": messages}, config)
        return response["messages"][-1].content

import json
import httpx
import os
from typing import Annotated, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from protocol import AIAppProtocol


driver_url = os.environ.get("IIOT_DRIVER_URL")

@tool
def read_points(
    driver: Annotated[
        Literal["modbus", "opcua"], 
        "The driver to read points from. Available drivers: modbus, opcua"
    ],
    req: Annotated[str, "The request to read points from."],
) -> Annotated[list[any], "The points read from the request."]:
    """Read the points from the request."""

    with httpx.Client() as client:
        response = client.post(
            f"{driver_url}/drivers/{driver}/read",
            json=json.loads(req),
        )
        return response.json()

tools = [read_points]

class IIoTApp(AIAppProtocol):
    def __init__(self, memory: BaseCheckpointSaver):
        # Setup the LLM
        llm = ChatOpenAI(model="gpt-4o-mini")

        # Load the Modbus schema
        with open("schema/modbus.schema.json", "r") as f:
            self.modbus_schema = f.read()

        # Define system prompt
        self.system_prompt = f"""
        You are a helpful assistant that can help with Modbus requests.

        The following is the schema for the Modbus requests:
        {self.modbus_schema}

        Please follow these guidelines:
        1. Use Modbus TCP by default
        2. For Holding Register or Input Register:
           - Quantity and Unit must match:
             * When quantity = 1, use unit = "word" (2 bytes):
             * When quantity = 2, use unit = "dword" (4 bytes):
             * When quantity = 4, use unit = "qword" (8 bytes):
           - First read with default byte/word orders
           - If value seems incorrect, try alternative byte/word orders
           - Choose the most reasonable value based on the context and expected data type
        3. Follow the schema when generating the Modbus requests
        4. Use default values when available and generate the request directly. Only ask for more information if essential details are missing and no defaults are available
        5. Respond with ONLY valid JSON without any additional explanations or comments
        6. If clarification is needed, respond with a plain text question, not JSON
        7. After generating the Modbus request JSON, use the read_points tool to read the actual values
        """

        # Define the workflow
        workflow = StateGraph(MessagesState)

        # Add the nodes
        workflow.add_node("chatbot", self.chatbot)
        workflow.add_node("tools", ToolNode(tools))

        # Add the edges
        workflow.add_edge(START, "chatbot")
        workflow.add_edge("tools", "chatbot")
        workflow.add_edge("chatbot", END)

        # Add the conditional edge
        workflow.add_conditional_edges(
            "chatbot", tools_condition,
        )

        # Compile the workflow
        app = workflow.compile(checkpointer=memory)

        self.app = app
        self.llm = llm

    def chatbot(self, state: MessagesState) -> MessagesState:
        response = self.llm.bind_tools(tools).invoke(state["messages"])
        return {"messages": [response]}

    def init_session(self, session_id: str):
        config = { "configurable": { "thread_id": session_id } }
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="This is an initial message, no need to respond"),
        ]
        self.app.invoke({"messages": messages}, config)

    def invoke(self, content: str, session_id: str) -> str:
        config = { "configurable": { "thread_id": session_id } }
        messages = [
            HumanMessage(content=content),
        ]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content

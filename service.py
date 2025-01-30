from typing import Protocol
from ulid import ULID

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

class ChatServiceProtocol(Protocol):
    """A protocol for a chat service."""

    def create_session(self) -> str:
        """ Create a new chat session."""
        ...

    def send_message(self, session_id: str, content: str) -> str:
        """Send a message to the chat session."""
        ...


llm = ChatOpenAI(model="gpt-4o-mini")

def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


class ChatService(ChatServiceProtocol):
    __slots__ = ["app"]

    def __init__(self):
        # Define the workflow
        workflow = StateGraph(MessagesState)

        # Add the nodes and edges
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        workflow.add_edge("model", END)

        # Add memory
        memory = MemorySaver()
        self.app = workflow.compile(checkpointer=memory)

    def create_session(self) -> str:
        ulid = ULID()
        return str(ulid)

    def send_message(self, session_id: str, content: str) -> str:
        config = { "configurable": { "thread_id": session_id } }

        messages = [HumanMessage(content=content)]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content

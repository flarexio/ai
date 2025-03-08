from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from protocol import AIAppProtocol


class BasicApp(AIAppProtocol):
    def __init__(self, memory: BaseCheckpointSaver):
        # Setup the LLM
        llm = ChatOpenAI(model="gpt-4o-mini")

        # Define the workflow
        workflow = StateGraph(MessagesState)

        # Add the nodes and edges
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        workflow.add_edge("model", END)

        # Compile the workflow
        app = workflow.compile(checkpointer=memory)

        self.app = app
        self.llm = llm
        self.memory = memory

    def _call_model(self, state: MessagesState) -> MessagesState:
        response = self.llm.invoke(state["messages"])
        return {"messages": [response]}

    def init_session(self, session_id: str):
        pass

    def invoke(self, content: str, session_id: str) -> str:
        config = { "configurable": { "thread_id": session_id } }
        messages = [HumanMessage(content=content)]
        response = self.app.invoke({"messages": messages}, config)
        return response["messages"][-1].content

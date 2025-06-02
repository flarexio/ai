from pydantic import Field

from langgraph.prebuilt.chat_agent_executor import AgentState


class IIoTState(AgentState):
    supervisor_message: str = Field(description="The message from the supervisor agent to the agent.")

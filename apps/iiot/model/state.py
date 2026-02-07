from pydantic import Field

from langchain.agents import AgentState


class IIoTState(AgentState):
    supervisor_message: str = Field(description="The message from the supervisor agent to the agent.")

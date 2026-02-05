from enum import Enum
from typing import AsyncIterator, Optional, Protocol, Union
from pydantic import BaseModel, Field


class Role(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"

    def __str__(self) -> str:
        match self:
            case Role.HUMAN:
                return "Human"
            case Role.AI:
                return "AI"
            case Role.SYSTEM:
                return "System"
            case Role.TOOL:
                return "Tool"
            case _:
                raise ValueError(f"Unknown role: {self.value}")

class ToolCall(BaseModel):
    """A tool call in a chat session."""
    id: Optional[str] = None
    name: Optional[str] = None
    args: Optional[str] = None

class Message(BaseModel):
    """A message in a chat session."""
    role: Role
    content: str

class HumanMessage(Message):
    """A message from the human user."""
    role: Role = Field(Role.HUMAN, frozen=True)

class AIMessage(Message):
    """A message from the AI."""
    role: Role = Field(Role.AI, frozen=True)
    tool_calls: Optional[list[ToolCall]] = None

class SystemMessage(Message):
    """A system message."""
    role: Role = Field(Role.SYSTEM, frozen=True)

class ToolMessage(Message):
    """A message that represents a tool call."""
    role: Role = Field(Role.TOOL, frozen=True)
    tool_call_id: str

AnyMessage = Union[HumanMessage, AIMessage, SystemMessage, ToolMessage]

class MessageChunk(BaseModel):
    """A chunk of a streaming message."""
    nodes: list[str] = []
    role: Role
    content: str
    tool_calls: list[ToolCall] = []
    tool_call_id: Optional[str] = None
    is_new: bool = False

    def to_message(self) -> AnyMessage:
        """Convert this chunk to a full message."""
        if self.role == Role.HUMAN:
            return HumanMessage(role=self.role, content=self.content)
        elif self.role == Role.AI:
            return AIMessage(role=self.role, content=self.content, tool_calls=self.tool_calls)
        elif self.role == Role.SYSTEM:
            return SystemMessage(role=self.role, content=self.content)
        elif self.role == Role.TOOL:
            return ToolMessage(role=self.role, content=self.content, tool_call_id=self.tool_call_id)
        else:
            raise ValueError(f"Unknown role: {self.role}")

class ChatContext(BaseModel):
    """Context for a chat session."""
    session_id: Optional[str] = Field(None, 
        description="The session ID for the chat context",
    )
    user_id: Optional[str] = Field(None,
        description="The user ID for the chat context",
    )
    customer_id: Optional[str] = Field(None,
        description="The customer ID for the chat context",
    )
    workspace_id: Optional[str] = Field(None, 
        description="The workspace ID for the chat context",
    )

class AppInfo(BaseModel):
    """Information about an AI app."""
    id: str = Field(description="The unique identifier for the app")
    name: str = Field(description="The name of the app")
    description: str = Field(description="A brief description of the app")
    version: str = Field(default="1.0.0", description="The version of the app")

class AIAppProtocol(Protocol):
    def id(self) -> str:
        """Return the unique identifier for the app."""
        ...

    """A protocol for an AI app."""
    def name(self) -> str:
        """Return the name of the app."""
        ...

    def description(self) -> str:
        """Return the description of the app."""
        ...
        
    def version(self) -> str:
        """Return the version of the app."""
        ...
         
    def info(self) -> AppInfo:
        """Return information about the app."""
        return AppInfo(
            id=self.id(),
            name=self.name(),
            description=self.description(),
            version=self.version()
        )

    async def ainvoke(self, ctx: ChatContext, content: str) -> str:
        """Asynchronously invoke the app."""
        config = {
            "configurable": { 
                "thread_id": ctx.session_id, 
                "user_id": ctx.user_id,
                "customer_id": ctx.customer_id,
                "workspace_id": ctx.workspace_id,
            }
        }
        messages = [HumanMessage(content=content)]
        try:
            response = await self.app.ainvoke({"messages": messages}, config)
            return response["messages"][-1].content
        except Exception as e:
            print(f"Error in ainvoke: {e}")
            return f"error: {e}"

    async def astream(self, ctx: ChatContext, content: str) -> AsyncIterator[MessageChunk]:
        """Asynchronously stream the app's response."""
        yield ...


class Session(BaseModel):
    id: str
    app_name: str


class ChatServiceProtocol(Protocol):
    """A protocol for a chat service."""

    def add_app(self, name: str, app: AIAppProtocol):
        """Add an app to the chat service."""
        ...

    async def list_apps(self) -> list[AIAppProtocol]:
        """List all apps in the chat service."""
        ...

    async def find_app(self, name: str) -> AIAppProtocol:
        """Find an app by name."""
        ... 

    async def create_session(self, app_name: str) -> str:
        """ Create a new chat session."""
        ...

    async def list_sessions(self) -> list[Session]:
        """List all chat sessions."""
        ...

    async def send_message(self, ctx: ChatContext, content: str) -> str:
        """Send a message to the chat session."""
        ...

    async def stream_message(self, ctx: ChatContext, content: str) -> AsyncIterator[MessageChunk]:
        """Stream messages from the chat session."""
        ...

    async def list_messages(self, session_id: str) -> list[Message]:
        """List all messages from the chat session."""
        ...


class ChatRepositoryProtocol(Protocol):
    """A repository for storing and retrieving sessions and messages."""

    async def store_session(self, session: Session):
        """Store a session."""
        ...

    async def list_sessions(self) -> list[Session]:
        """List all sessions."""
        ...

    async def find_session(self, session_id: str) -> Session:
        """Find a session by id."""
        ...
    
    async def list_messages(self, session_id: str) -> list[Message]:
        """List all messages from a session."""
        ...

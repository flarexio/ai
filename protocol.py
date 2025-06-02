from enum import Enum
from typing import Protocol
from pydantic import BaseModel


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
                raise self.value


class Message(BaseModel):
    """A message in a chat session."""
    role: Role
    content: str


class AIAppProtocol(Protocol):
    """A protocol for an AI app."""

    def invoke(self, content: str, session_id: str) -> str:
        """Invoke the app."""
        ...

    async def ainvoke(self, content: str, session_id: str) -> str:
        """Asynchronously invoke the app."""
        ...


class ChatContext(BaseModel):
    session_id: str


class Session(BaseModel):
    id: str
    app_name: str


class ChatServiceProtocol(Protocol):
    """A protocol for a chat service."""

    def add_app(self, name: str, app: AIAppProtocol):
        """Add an app to the chat service."""
        ...

    def find_app(self, name: str) -> AIAppProtocol:
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

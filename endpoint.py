from pydantic import BaseModel, Field

from kit import Endpoint
from protocol import ChatContext, ChatServiceProtocol, Message


class CreateSessionRequest(BaseModel):
    app_name: str = Field(
        description="The name of the app to create a session for",
    )


class CreateSessionResponse(BaseModel):
    session_id: str


class CreateSessionEndpoint(Endpoint[ChatServiceProtocol, CreateSessionRequest, CreateSessionResponse]):
    async def handle(self, request: CreateSessionRequest) -> CreateSessionResponse:
        session_id = await self.service.create_session(request.app_name)
        return CreateSessionResponse(session_id=session_id)


class ListSessionsResponse(BaseModel):
    sessions: list[str]


class ListSessionsEndpoint(Endpoint[ChatServiceProtocol, None, ListSessionsResponse]):
    async def handle(self) -> ListSessionsResponse:
        sessions = await self.service.list_sessions()
        return ListSessionsResponse(sessions=sessions)


class SendMessageRequest(BaseModel):
    session_id: str | None = Field(
        description="The session ID to send the message to",
        default=None,
    )
    content: str = Field(
        description="The message to send to the session",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "Hi, my name is Mirror.",
                },
            ]
        }
    }


class SendMessageResponse(BaseModel):
    content: str


class SendMessageEndpoint(Endpoint[ChatServiceProtocol, SendMessageRequest, SendMessageResponse]):
    async def handle(self, request: SendMessageRequest) -> SendMessageResponse:
        ctx = ChatContext(session_id=request.session_id)
        resp = await self.service.send_message(ctx, request.content)
        return SendMessageResponse(content=resp)


class ListMessagesRequest(BaseModel):
    session_id: str = Field(
        description="The session ID to list the messages from",
    )


class ListMessagesResponse(BaseModel):
    messages: list[Message]


class ListMessagesEndpoint(Endpoint[ChatServiceProtocol, ListMessagesRequest, ListMessagesResponse]):
    async def handle(self, request: ListMessagesRequest) -> ListMessagesResponse:
        messages = await self.service.list_messages(request.session_id)
        return ListMessagesResponse(messages=messages)

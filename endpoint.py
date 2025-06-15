from pydantic import BaseModel, Field
from typing import AsyncIterator

from kit import Endpoint
from protocol import AppInfo, ChatContext, ChatServiceProtocol, Message, MessageChunk, Session


class ListAppsResponse(BaseModel):
    apps: list[AppInfo] = []

class ListAppsEndpoint(Endpoint[ChatServiceProtocol, None, ListAppsResponse]):
    async def handle(self) -> ListAppsResponse:
        apps = await self.service.list_apps()

        infos: list[AppInfo] = []
        for app in apps:
            info = app.info()
            infos.append(info)

        return ListAppsResponse(apps=infos)

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
    sessions: list[Session]


class ListSessionsEndpoint(Endpoint[ChatServiceProtocol, None, ListSessionsResponse]):
    async def handle(self) -> ListSessionsResponse:
        sessions = await self.service.list_sessions()
        return ListSessionsResponse(sessions=sessions)


class SendMessageRequest(BaseModel):
    ctx: ChatContext = Field(ChatContext, description="The chat context, if available")
    content: str = Field("", description="The message to send to the session")

class SendMessageResponse(BaseModel):
    content: str

class SendMessageEndpoint(Endpoint[ChatServiceProtocol, SendMessageRequest, SendMessageResponse]):
    async def handle(self, request: SendMessageRequest) -> SendMessageResponse:
        resp = await self.service.send_message(request.ctx, request.content)
        return SendMessageResponse(content=resp)


class StreamMessageRequest(BaseModel):
    ctx: ChatContext = Field(ChatContext, description="The chat context, if available")
    content: str = Field(description="The message to stream")

class StreamMessageEndpoint(Endpoint[ChatServiceProtocol, StreamMessageRequest, AsyncIterator[MessageChunk]]):
    async def handle(self, request: StreamMessageRequest) -> AsyncIterator[MessageChunk]:
        stream: AsyncIterator[MessageChunk] = self.service.stream_message(request.ctx, request.content)
        async for chunk in stream:
            yield chunk

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

from nats.aio.msg import Msg
from typing import AsyncIterator
import nats

from endpoint import (
    CreateSessionRequest,
    CreateSessionResponse,
    ListAppsResponse,
    ListMessagesRequest, 
    ListMessagesResponse,
    ListSessionsResponse,
    SendMessageRequest,
    StreamMessageRequest,
)
from kit import EndpointProtocol
from protocol import MessageChunk


class NATSTransport:
    __slots__ = ["endpoints", "nc", "url", "creds"]

    def __init__(self, url: str, creds: str):
        self.endpoints: dict[str, EndpointProtocol] = {}
        self.url = url
        self.creds = creds

    def set_endpoints(self, endpoints: dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def list_apps_handler(self, msg: Msg):
        endpoint = self.endpoints["list_apps"]
        resp: ListAppsResponse = await endpoint.handle()
        await msg.respond(resp.model_dump_json().encode("utf-8"))

    async def create_session_handler(self, msg: Msg):
        endpoint = self.endpoints["create_session"]
        req = CreateSessionRequest.model_validate_json(msg.data)
        resp: CreateSessionResponse = await endpoint.handle(req)
        await msg.respond(resp.session_id.encode("utf-8"))

    async def list_sessions_handler(self, msg: Msg):
        endpoint = self.endpoints["list_sessions"]
        resp: ListSessionsResponse = await endpoint.handle()
        await msg.respond(resp.model_dump_json().encode("utf-8"))

    async def send_message_handler(self, msg: Msg):
        endpoint = self.endpoints["send_message"]
        req = SendMessageRequest.model_validate_json(msg.data)
        req.ctx.session_id = msg.subject.split(".")[2]
        resp = await endpoint.handle(req)
        await msg.respond(resp.content.encode("utf-8"))

    async def stream_message_handler(self, msg: Msg):
        endpoint = self.endpoints["stream_message"]
        req = StreamMessageRequest.model_validate_json(msg.data)
        req.ctx.session_id = msg.subject.split(".")[2]

        stream: AsyncIterator[MessageChunk] = endpoint.handle(req)
        async for chunk in stream:
            await self.nc.publish(msg.reply, chunk.model_dump_json().encode("utf-8"))

        await msg.respond(b"[DONE]")

    async def list_messages_handler(self, msg: Msg):
        endpoint = self.endpoints["list_messages"]
        req = ListMessagesRequest(
            session_id=msg.subject.split(".")[2],
        )
        resp: ListMessagesResponse = await endpoint.handle(req)
        await msg.respond(resp.model_dump_json().encode("utf-8"))

    async def serve(self):
        nc = await nats.connect(self.url, user_credentials=self.creds)
        await nc.subscribe("ai.apps.list", cb=self.list_apps_handler)
        await nc.subscribe("ai.sessions.create", cb=self.create_session_handler)
        await nc.subscribe("ai.sessions.list", cb=self.list_sessions_handler)
        await nc.subscribe("ai.sessions.*.messages.send", cb=self.send_message_handler)
        await nc.subscribe("ai.sessions.*.messages.stream", cb=self.stream_message_handler)
        await nc.subscribe("ai.sessions.*.messages.list", cb=self.list_messages_handler)
        self.nc = nc

    async def shutdown(self):
        await self.nc.drain()

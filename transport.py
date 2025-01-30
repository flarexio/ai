from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from endpoint import SendMessageRequest
from kit import EndpointProtocol

class TransportSendMessageRequest(BaseModel):
    content: str

class HTTPTransport:
    __slots__ = ["app", "endpoints"]

    def __init__(self):
        self.app = FastAPI()
        self.endpoints: Dict[str, EndpointProtocol] = {}
        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/sessions")
        async def create_session() -> str:
            return await self._create_session_handler()

        @self.app.put("/sessions/{session_id}/messages")
        async def send_message(session_id: str, request: TransportSendMessageRequest) -> str:
            return await self._send_message_handler(session_id, request)

    def set_endpoints(self, endpoints: Dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def _create_session_handler(self) -> str:
        if "create_session" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["create_session"]
        resp = await endpoint.handle()
        return resp.session_id

    async def _send_message_handler(self, session_id: str, request: TransportSendMessageRequest) -> str:
        if "create_session" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["send_message"]
        req = SendMessageRequest(session_id=session_id, content=request.content)
        resp = await endpoint.handle(req)
        return resp.content

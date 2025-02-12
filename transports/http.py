import uvicorn
from fastapi import FastAPI, HTTPException

from endpoint import (
    ListMessagesRequest,
    ListMessagesResponse,
    ListSessionsResponse,
    SendMessageRequest,
)
from kit import EndpointProtocol


class HTTPTransport:
    __slots__ = ["app", "config", "endpoints"]

    def __init__(self, host: str, port: int):
        app = FastAPI(
            title="AI Service",
        )
        self.config = uvicorn.Config(app, host=host, port=port)
        self.endpoints: dict[str, EndpointProtocol] = {}
        self._setup_routes(app)

    def _setup_routes(self, app: FastAPI):
        @app.post("/sessions", tags=["sessions"])
        async def create_session() -> str:
            return await self._create_session_handler()

        @app.get("/sessions", tags=["sessions"])
        async def list_sessions() -> ListSessionsResponse:
            return await self._list_sessions_handler()

        @app.put("/sessions/{session_id}/messages", tags=["messages"])
        async def send_message(session_id: str, request: SendMessageRequest) -> str:
            request.session_id = session_id
            return await self._send_message_handler(request)

        @app.get("/sessions/{session_id}/messages", tags=["messages"])
        async def list_messages(session_id: str) -> ListMessagesResponse:
            request = ListMessagesRequest(session_id=session_id)
            return await self._list_messages_handler(request)

    def set_endpoints(self, endpoints: dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def _create_session_handler(self) -> str:
        if "create_session" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["create_session"]
        resp = await endpoint.handle()
        return resp.session_id

    async def _list_sessions_handler(self) -> ListSessionsResponse:
        if "list_sessions" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["list_sessions"]
        return await endpoint.handle()

    async def _send_message_handler(self, req: SendMessageRequest) -> str:
        if "send_message" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["send_message"]
        resp = await endpoint.handle(req)
        return resp.content

    async def _list_messages_handler(self, req: ListMessagesRequest) -> ListMessagesResponse:
        if "list_messages" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["list_messages"]
        return await endpoint.handle(req)

    async def serve(self):
        server = uvicorn.Server(self.config)
        await server.serve()

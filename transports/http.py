from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import json
import uvicorn

from endpoint import (
    CreateSessionRequest,
    ListAppsResponse,
    ListMessagesRequest,
    ListMessagesResponse,
    ListSessionsResponse,
    SendMessageRequest,
    StreamMessageRequest,
)
from kit import EndpointProtocol
from protocol import MessageChunk


# 可允許的前端來源清單
origins = [
    "http://localhost:4200",
    "http://10.8.0.1:4200",
]


class HTTPTransport:
    __slots__ = ["app", "config", "endpoints"]

    def __init__(self, host: str, port: int):
        app = FastAPI(
            title="AI Service",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.config = uvicorn.Config(app, host=host, port=port)
        self.endpoints: dict[str, EndpointProtocol] = {}
        self._setup_routes(app)

    def _setup_routes(self, app: FastAPI):
        @app.get("/apps", tags=["apps"])
        async def list_apps() -> ListAppsResponse:
            return await self._list_apps_handler()

        @app.post("/sessions", tags=["sessions"])
        async def create_session(request: CreateSessionRequest) -> str:
            return await self._create_session_handler(request)

        @app.get("/sessions", tags=["sessions"])
        async def list_sessions() -> ListSessionsResponse:
            return await self._list_sessions_handler()

        @app.put("/sessions/{session_id}/messages", tags=["messages"])
        async def send_message(session_id: str, request: SendMessageRequest) -> str:
            request.ctx.session_id = session_id
            return await self._send_message_handler(request)

        @app.post("/sessions/{session_id}/messages/stream", tags=["messages"])
        async def stream_message(session_id: str, request: StreamMessageRequest) -> StreamingResponse:
            request.ctx.session_id = session_id
            return await self._stream_message_handler(request)

        @app.get("/sessions/{session_id}/messages", tags=["messages"])
        async def list_messages(session_id: str) -> ListMessagesResponse:
            request = ListMessagesRequest(session_id=session_id)
            return await self._list_messages_handler(request)

    def set_endpoints(self, endpoints: dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def _list_apps_handler(self) -> ListAppsResponse:
        if "list_apps" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["list_apps"]
        return await endpoint.handle()

    async def _create_session_handler(self, req: CreateSessionRequest) -> str:
        if "create_session" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["create_session"]
        resp = await endpoint.handle(req)
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

    async def _stream_message_handler(self, req: StreamMessageRequest):
        if "stream_message" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["stream_message"]

        async def generate_sse():
            try:
                stream: AsyncIterator[MessageChunk] = endpoint.handle(req)
                async for chunk in stream:
                    yield f"data: {chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"
            
            except Exception as e:
                error_data = {"error": str(e), "type": "error"}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_sse(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    async def _list_messages_handler(self, req: ListMessagesRequest) -> ListMessagesResponse:
        if "list_messages" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["list_messages"]
        return await endpoint.handle(req)

    async def serve(self):
        server = uvicorn.Server(self.config)
        await server.serve()

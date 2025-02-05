import uvicorn

from fastapi import FastAPI, HTTPException

from endpoint import SendMessageRequest
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

        @app.put("/sessions/{session_id}/messages", tags=["sessions"])
        async def send_message(session_id: str, request: SendMessageRequest) -> str:
            request.session_id = session_id
            return await self._send_message_handler(request)

    def set_endpoints(self, endpoints: dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def _create_session_handler(self) -> str:
        if "create_session" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["create_session"]
        resp = await endpoint.handle()
        return resp.session_id

    async def _send_message_handler(self, req: SendMessageRequest) -> str:
        if "send_message" not in self.endpoints:
            raise HTTPException(status_code=404, detail="endpoint not found")

        endpoint = self.endpoints["send_message"]
        resp = await endpoint.handle(req)
        return resp.content

    async def serve(self):
        server = uvicorn.Server(self.config)
        await server.serve()

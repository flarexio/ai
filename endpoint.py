from pydantic import BaseModel
from kit import Endpoint

class CreateSessionResponse(BaseModel):
    session_id: str

class CreateSessionEndpoint(Endpoint[None, CreateSessionResponse]):
    async def handle(self) -> CreateSessionResponse:
        session_id = self.service.create_session()
        return CreateSessionResponse(session_id=session_id)

class SendMessageRequest(BaseModel):
    session_id: str
    content: str

class SendMessageResponse(BaseModel):
    content: str

class SendMessageEndpoint(Endpoint[SendMessageRequest, SendMessageResponse]):
    async def handle(self, request: SendMessageRequest) -> SendMessageResponse:
        content = self.service.send_message(request.session_id, request.content)
        return SendMessageResponse(content=content)

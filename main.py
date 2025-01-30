from typing import Dict
from endpoint import CreateSessionEndpoint, SendMessageEndpoint
from kit import EndpointProtocol
from service import ChatService
from transport import HTTPTransport

svc = ChatService()

endpoints: Dict[str, EndpointProtocol] = {
    "create_session": CreateSessionEndpoint(svc),
    "send_message": SendMessageEndpoint(svc),
}

transport = HTTPTransport()
transport.set_endpoints(endpoints)

app = transport.app

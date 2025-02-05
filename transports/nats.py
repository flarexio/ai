import nats
from nats.aio.msg import Msg

from endpoint import SendMessageRequest
from kit import EndpointProtocol

class NATSTransport:
    __slots__ = ["endpoints", "nc", "url", "creds"]

    def __init__(self, url: str, creds: str):
        self.endpoints: dict[str, EndpointProtocol] = {}
        self.url = url
        self.creds = creds

    def set_endpoints(self, endpoints: dict[str, EndpointProtocol]):
        self.endpoints = endpoints

    async def create_session_handler(self, msg: Msg):
        endpoint = self.endpoints["create_session"]
        resp = await endpoint.handle()
        await msg.respond(resp.session_id.encode("utf-8"))

    async def send_message_handler(self, msg: Msg):
        endpoint = self.endpoints["send_message"]
        req = SendMessageRequest(
            session_id=msg.subject.split(".")[2],
            content=msg.data.decode("utf-8"),
        )
        resp = await endpoint.handle(req)
        await msg.respond(resp.content.encode("utf-8"))

    async def serve(self):
        nc = await nats.connect(self.url, user_credentials=self.creds)
        await nc.subscribe("ai.sessions.create", cb=self.create_session_handler)
        await nc.subscribe("ai.sessions.*.messages.send", cb=self.send_message_handler)
        self.nc = nc

    async def shutdown(self):
        await self.nc.drain()
